import os
import pickle
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from pycocotools import mask as coco_mask_utils
import torch.nn.functional as F

def polygon_to_mask(segmentation, height, width):
    """Convert COCO polygon segmentation to a binary mask (H x W bool array)."""
    rles = coco_mask_utils.frPyObjects(segmentation, height, width)
    rle  = coco_mask_utils.merge(rles)
    return coco_mask_utils.decode(rle).astype(bool)

def bboxes_to_mask(bboxes, h, w):
    """Return a boolean (H, W) mask; True where any bbox covers the pixel.

    bboxes: list of (x1, y1, x2, y2) already in [h, w] coordinate space.
    """
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in bboxes:
        r0, r1 = int(y1), int(np.ceil(y2))
        c0, c1 = int(x1), int(np.ceil(x2))
        r0, r1 = max(0, r0), min(h, r1)
        c0, c1 = max(0, c0), min(w, c1)
        if r1 > r0 and c1 > c0:
            mask[r0:r1, c0:c1] = True
    return mask


def scale_bboxes_to_grid(
    bboxes: List[List[float]],
    image_w: int,
    image_h: int,
    grid_w: int,
    grid_h: int,
) -> List[Tuple[float, float, float, float]]:
    """Scale image-pixel bboxes [(x1,y1,x2,y2),...] to patch-grid space.

    image_w/h : original image dimensions in pixels
    grid_w/h  : number of patch columns / rows  (Pw, Ph)
    Returns list of (x1, y1, x2, y2) in grid coordinates.
    """
    sx = grid_w / image_w
    sy = grid_h / image_h
    scaled = []
    for x1, y1, x2, y2 in bboxes:
        scaled.append((x1 * sx, y1 * sy, x2 * sx, y2 * sy))
    return scaled


def compute_iou(pred_binary: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Intersection-over-Union between two boolean/binary 2-D masks."""
    pred = pred_binary.astype(bool)
    gt   = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)

def load_attention_file(path: str) -> Tuple[torch.Tensor, Dict]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "attn" in obj:
        return obj["attn"], obj.get("meta", {})
    # Backward compatibility: raw tensor saved directly
    if torch.is_tensor(obj):
        return obj, {}
    raise ValueError("Unsupported attention file format")


def spatial_entropy(attn_map_2d: torch.Tensor, threshold: float) -> Dict:
    # attn_map_2d: [P, P]
    S = attn_map_2d
    mean_val = torch.mean(S) 
    B = torch.relu(S - mean_val*2)
    B_np = B.detach().cpu().to(torch.float32).numpy()
    binary = (B_np > threshold).astype(np.int32)

    from scipy.ndimage import label
    labeled, num = label(binary, structure=np.ones((3, 3)))

    total = float(B.sum().item())
    if total <= 0:
        return {"spatial_entropy": float("inf"), "labeled_array": labeled, "num_components": 0}

    # Probability mass per component
    probs = []
    for i in range(1, num + 1):
        comp_sum = B_np[labeled == i].sum()
        if comp_sum > 0:
            probs.append(comp_sum / total)
    se = -sum(p * np.log(p) for p in probs if p > 0) if probs else 0.0
    return {"spatial_entropy": float(se), "labeled_array": labeled, "num_components": int(num)}


def elbow_chord(values: List[float]) -> float:
    # Returns threshold value (y), not index
    if len(values) <= 2:
        return min(values) if values else 0.0
    vals = np.array(values, dtype=np.float64)
    order = np.argsort(vals)  # ascending
    y = vals[order]
    x = np.arange(len(y), dtype=np.float64)
    start, end = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    line = end - start
    line_len = np.linalg.norm(line)
    if line_len == 0:
        return y[0]
    unit = line / line_len
    vecs = np.stack([x, y], axis=1) - start
    proj = (vecs @ unit)[:, None] * unit
    d = np.linalg.norm(vecs - proj, axis=1)
    elbow_i = int(np.argmax(d))
    return float(y[elbow_i])


def analyze_heads(cfg, attn: torch.Tensor, meta: Dict, bbox_mask: Optional[List] = None) -> List[Dict]:
    """Analyze heads and return a ranked list.

    attn : [L, H, 1, V]
    meta : includes patch_h, patch_w (or patch_size for square-grid models)
           and image_size (W, H) in pixels.
    bbox_mask : optional list of [x1, y1, x2, y2] boxes in **image pixel** coordinates.
           When provided, AUROC and IoU are computed for each head or a mask.
    """
    L, H, _, V = attn.shape
    # Support non-square grids (e.g. Qwen3-VL): use patch_h/patch_w when available
    Ph = int(meta.get("patch_h", meta.get("patch_size", int(np.sqrt(V)))))
    Pw = int(meta.get("patch_w", meta.get("patch_size", int(np.sqrt(V)))))

    # Criterion 1: head sums over image patches
    sums = []
    for l in range(L):
        for h in range(H):
            s = float(attn[l, h, 0].sum().item())
            sums.append(s)
    
        

    thr_val = elbow_chord(sums) if cfg.logic.threshold.method == "chord" else min(sums)

    # ---- Build ground-truth mask (patch-grid space) if bboxes provided -------
    # BUG FIX: bboxes come in image-pixel coordinates; we must scale them to the
    # patch grid (Ph x Pw) before passing to bboxes_to_mask, otherwise all
    # coordinates are out-of-bounds and the mask stays all zeros.
    
    image_size = meta.get("image_size")  # (W, H) in pixels
    if image_size is not None:
        img_w, img_h = int(image_size[0]), int(image_size[1])
    else:
        # Fallback: assume bboxes are already in grid space
        img_w, img_h = Pw, Ph
    
    if len(bbox_mask[0]) == 4:
        bbox = bbox_mask
        grid_bboxes = scale_bboxes_to_grid(bbox, img_w, img_h, Pw, Ph)
        gt_mask = bboxes_to_mask(grid_bboxes, Ph, Pw)
    else:
        gt_mask = polygon_to_mask(bbox_mask, img_h, img_w)
        
    print(gt_mask.shape)
    flattened_gt_mask = gt_mask.flatten()
    print(flattened_gt_mask.shape)
    print(
        f"GT mask: {int(flattened_gt_mask.sum())} positive patches "
        f"out of {len(flattened_gt_mask)}  "
        f"(grid {Ph}h x {Pw}w, image {img_h}x{img_w}px)"
    )
    # AUROC requires both classes to be present
    gt_has_both_classes = flattened_gt_mask.sum() > 0 and (~flattened_gt_mask).sum() > 0
    

    # ---- Per-head analysis ---------------------------------------------------
    results: List[Dict] = []
#     my_heads = [(19, 4),
#  (19, 5),
#  (14, 14),
#  (21, 1),
#  (18, 11),]
    idx = 0
    for l in range(L):
        for h in range(H):
            # if (l, h) not in my_heads:
            #     continue
            s = sums[idx]
            idx += 1
            a2d     = attn[l, h, 0].reshape(Ph, Pw)
            if len(bbox_mask[0]) != 4:
                a2d = F.interpolate(
                    a2d.unsqueeze(0).unsqueeze(0),      # (1,1,Ph,Pw)
                    size=(img_h, img_w),
                    mode='bilinear',
                    align_corners=False 
                ).squeeze()  
            
            a2d_np  = a2d.detach().cpu().to(torch.float32).numpy()
            auroc = float(roc_auc_score(flattened_gt_mask, a2d_np.flatten()))
            mean_val     = a2d_np.mean()
            pred_binary  = (np.maximum(a2d_np - mean_val * 2, 0) > cfg.logic.entropy.binarize_threshold)
            iou   = compute_iou(pred_binary, gt_mask)

            if s < thr_val:
                se = float("inf")
                bottom_row_focus = False
                n_comp = 0
                
            else:
                a2d     = attn[l, h, 0].reshape(Ph, Pw)
                a2d_np  = a2d.detach().cpu().to(torch.float32).numpy()
                se_res  = spatial_entropy(a2d, cfg.logic.entropy.binarize_threshold)

                bottom_row_focus = bool((a2d.shape[0] > 0) and (a2d[-1, :] > 0.05).any())
                se      = float(se_res["spatial_entropy"])   # lower is better
                n_comp  = int(se_res["num_components"])

            # import matplotlib.pyplot as plt
            # plt.subplot(1,2,1)
            # plt.imshow(gt_mask)
            # plt.title("GT Mask")
            # plt.subplot(1,2,2)
            # plt.imshow(pred_binary)
            # plt.title(f"AUROC: {auroc:.3f} IoU: {iou:.3f}")
            # plt.show()
            results.append({
                "layer":           l,
                "head":            h,
                "attn_sum":        s,
                "spatial_entropy": se,
                "bottom_row_focus": bottom_row_focus,
                "num_components":  n_comp,
                "AUROC":           auroc,
                "IoU":             iou,
            })

    # ---- Filter and sort ----------------------------------------------------
    # Keep heads above threshold, prefer non-bottom-row, skip layer <=1
    # return results
    kept = [
        r for r in results
        if np.isfinite(r["spatial_entropy"])
        and r["attn_sum"] >= thr_val
        and not r["bottom_row_focus"]
        and r["layer"] > 1
    ]
    if len(kept) < cfg.logic.threshold.min_keep:
        # Fallback: take top by sum if too few survive the filter
        by_sum = sorted(results, key=lambda x: x["attn_sum"], reverse=True)
        kept = [x for x in by_sum if not x["bottom_row_focus"]][: cfg.logic.threshold.min_keep]

    if gt_has_both_classes:
        # Sort by AUROC descending (best localisation head first)
        kept = sorted(kept, key=lambda x: x["AUROC"], reverse=True)
    else:
        # Fall back to spatial entropy ascending (most focused head first)
        kept = sorted(kept, key=lambda x: x["spatial_entropy"])

    return kept

