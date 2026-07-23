#!/usr/bin/env python3
"""
run_textvqa_batch.py
====================
Batch inference script that runs the full LocalizationHeads pipeline on every
sample in textvqa_200_samples.jsonl.

Supported backends
------------------
  qwen_vl  – Qwen3-VL / Qwen2-VL  (default)
  llava    – LLaVA-1.5

For each sample the following files are written to <output_dir>/<image_id>/:
  - selected_heads.json    : ranked list of selected heads (layer, head, …)
  - mask.png               : binarised attention mask at original image size
  - analysis.pkl           : full analyze_heads output + meta dict
  - top{K}.png             : original image + attention map grid

The model is loaded only **once** before the loop.

Usage
-----
    # Qwen (default)
    python run_textvqa_batch.py --backend qwen_vl --model Qwen/Qwen3-VL-2B-Instruct

    # LLaVA
    python run_textvqa_batch.py --backend llava --model liuhaotian/llava-v1.5-7b \
        --conv_mode referseg

    # Common options
    python run_textvqa_batch.py --jsonl textvqa_200_samples.jsonl \
        --output textvqa_results --start 0 --end -1 \
        --top_k 5 --device auto --device_id 0 --no_4bit --skip_existing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal config stub
# ---------------------------------------------------------------------------

class _NS:
    """Tiny namespace that supports attribute-style access."""
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __getattr__(self, item):
        return None


def _build_cfg(args: argparse.Namespace) -> _NS:
    """Construct the config namespace expected by the collectors / analyze / bbox_mask."""
    backend = args.backend  # "qwen_vl" | "llava"

    if backend == "qwen_vl":
        model_name = args.model or "Qwen/Qwen3-VL-2B-Instruct"
        model_cfg = _NS(
            name=model_name,
            backend="qwen_vl",
            base=None,
            cache_dir=None,
            use_flash_attn=False,
            use_generate=True,
            load_4bit=(not args.no_4bit),
            min_pixels=256,
            max_pixels=1024,
            max_new_tokens=10,
            do_sample=False,
            num_beams=1,
        )
    else:  # llava
        model_name = args.model or "liuhaotian/llava-v1.5-7b"
        model_cfg = _NS(
            name=model_name,
            backend="llava",
            base=None,
            cache_dir=None,
            conv_mode=args.conv_mode or "referseg",
            use_flash_attn=False,
            use_generate=False,
            load_4bit=(not args.no_4bit),
            max_new_tokens=10,
            do_sample=False,
            num_beams=1,
        )

    logic_cfg = _NS(
        top_k=args.top_k,
        threshold=_NS(method="chord", min_keep=1),
        entropy=_NS(binarize_threshold=0.001),
        smoothing=_NS(sigma=1.0),
        mask=_NS(method="mean_relu"),
    )

    return _NS(
        model=model_cfg,
        logic=logic_cfg,
        device=args.device,
        device_id=args.device_id,
        save_fig=False,
        show_plot=False,
        enable_cfg=args.enable_cfg,
    )


# ---------------------------------------------------------------------------
# Load JSONL
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning(f"  Skipping malformed line {i}: {e}")
    return samples


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _resolve_device(cfg) -> str:
    device = cfg.device
    if device == "auto":
        device = f"cuda:{cfg.device_id}" if cfg.device_id >= 0 and torch.cuda.is_available() else "cpu"
    return device


# ---------------------------------------------------------------------------
# Model loading  (backend-aware)
# ---------------------------------------------------------------------------

def _load_model(cfg) -> Tuple:
    """
    Load model(s) according to cfg.model.backend.

    Returns:
        qwen_vl  →  (model, processor, None, None)
        llava    →  (model, image_processor, tokenizer, context_len)
    """
    backend = cfg.model.backend

    if backend == "qwen_vl":
        log.info(f"[qwen_vl] Loading {cfg.model.name}  (4bit={cfg.model.load_4bit})")
        from collector_qwen import load_qwen_vl_model
        model, processor = load_qwen_vl_model(cfg)
        log.info("Qwen model loaded.")
        return model, processor, None, None

    elif backend == "llava":
        log.info(f"[llava] Loading {cfg.model.name}  (4bit={cfg.model.load_4bit})")
        from collector import load_model_from_cfg
        tokenizer, model, image_processor, context_len, model_name_str = load_model_from_cfg(cfg)
        log.info(f"LLaVA model loaded: {model_name_str}")
        return model, image_processor, tokenizer, context_len

    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose 'qwen_vl' or 'llava'.")


# ---------------------------------------------------------------------------
# Per-sample forward passes  (backend-specific)
# ---------------------------------------------------------------------------

def _run_sample_qwen(cfg, model, processor, image_file: str, query: str) -> Tuple:
    """Run Qwen forward pass; returns (attn_cpu, meta)."""
    from collector_qwen import (
        _load_image,
        _build_messages,
        _find_visual_token_range,
        _forward_collect_qwen,
        _generate_collect_qwen,
    )
    from lab.stations import MetadataStation

    device = _resolve_device(cfg)

    image = _load_image(image_file)
    image_size = image.size  # (W, H)

    messages = _build_messages(query)
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    pixel_values      = inputs["pixel_values"].to(device, dtype=torch.float16)
    input_ids         = inputs["input_ids"].to(device)
    image_grid_thw    = inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        mm_token_type_ids = mm_token_type_ids.to(device)

    begin_pos_vis, vis_len = _find_visual_token_range(input_ids, processor)
    MetadataStation.set_begin_pos("vis", begin_pos_vis)
    MetadataStation.set_vis_len(vis_len)

    with torch.inference_mode():
        if cfg.model.use_generate:
            attn_last_to_vis, gen_text = _generate_collect_qwen(
                model, processor, input_ids, pixel_values, image_grid_thw,
                mm_token_type_ids, begin_pos_vis, vis_len,
                max_new_tokens=cfg.model.max_new_tokens,
                do_sample=cfg.model.do_sample,
                num_beams=cfg.model.num_beams,
            )
            if attn_last_to_vis is None:
                attn_last_to_vis = _forward_collect_qwen(
                    model, input_ids, pixel_values, image_grid_thw,
                    mm_token_type_ids, begin_pos_vis, vis_len
                )
        else:
            gen_text = None
            attn_last_to_vis = _forward_collect_qwen(
                model, input_ids, pixel_values, image_grid_thw,
                mm_token_type_ids, begin_pos_vis, vis_len
            )
        if cfg.enable_cfg:
            messages = _build_messages("")
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = processor(
                text=[text_prompt],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
            if cfg.model.use_generate:
                attn_last_to_vis_uncond, gen_text = _generate_collect_qwen(
                    model, processor, input_ids, pixel_values, image_grid_thw,
                    mm_token_type_ids, begin_pos_vis, vis_len,
                    max_new_tokens=cfg.model.max_new_tokens,
                    do_sample=cfg.model.do_sample,
                    num_beams=cfg.model.num_beams,
                )
                if attn_last_to_vis is None:
                    attn_last_to_vis_uncond = _forward_collect_qwen(
                        model, input_ids, pixel_values, image_grid_thw,
                        mm_token_type_ids, begin_pos_vis, vis_len
                    )
                else:
                    gen_text = None
                    attn_last_to_vis_uncond = _forward_collect_qwen(
                        model, input_ids, pixel_values, image_grid_thw,
                        mm_token_type_ids, begin_pos_vis, vis_len
                    )
            att_cfg = attn_last_to_vis - attn_last_to_vis_uncond
            att_cfg = torch.clamp(att_cfg, min=0)  # kill negatives
            att_cfg = (att_cfg - att_cfg.min()) / (att_cfg.max() - att_cfg.min())  # Then renormalize AFTER clamping
            # import matplotlib.pyplot as plt
            # thw = image_grid_thw[0].cpu().tolist()
            # vcfg = getattr(model.config, "vision_config", model.config)
            # merge = int(getattr(vcfg, "spatial_merge_size", 2))
            # Ph = int(thw[1]) // merge
            # Pw = int(thw[2]) // merge

            # plt.subplot(1, 4, 1)
            # plt.imshow(att_cfg[0, 0, 0].detach().cpu().numpy().reshape(Ph, Pw))
            # plt.title("CFG Attention")
            
            # plt.subplot(1, 4, 2)
            # plt.imshow(attn_last_to_vis[0, 0, 0].detach().cpu().numpy().reshape(Ph, Pw))
            # plt.title("cond Attention")

            # plt.subplot(1, 4, 3)
            # plt.imshow(attn_last_to_vis_uncond[0, 0, 0].detach().cpu().numpy().reshape(Ph, Pw))
            # plt.title("uncond Attention")

            # plt.subplot(1,4, 4)
            # plt.imshow(image)
            # plt.title(query)

            # plt.show()


            attn_last_to_vis = att_cfg

            


    if image_grid_thw is not None:
        thw = image_grid_thw[0].cpu().tolist()
        vcfg = getattr(model.config, "vision_config", model.config)
        merge = int(getattr(vcfg, "spatial_merge_size", 2))
        patch_h = int(thw[1]) // merge
        patch_w = int(thw[2]) // merge
    else:
        patch_h = patch_w = int(np.sqrt(attn_last_to_vis.shape[-1]))

    meta = {
        "image_file": image_file,
        "query":      query,
        "image_size": image_size,
        "model_name": cfg.model.name.split("/")[-1],
        "vis_len":    int(attn_last_to_vis.shape[-1]),
        "patch_size": patch_h,
        "patch_h":    patch_h,
        "patch_w":    patch_w,
        "num_layers": int(attn_last_to_vis.shape[0]),
        "num_heads":  int(attn_last_to_vis.shape[1]),
    }
    if gen_text is not None:
        meta["generated_text"] = gen_text

    return attn_last_to_vis.detach().cpu(), meta


def _run_sample_llava(cfg, model, image_processor, tokenizer, image_file: str, query: str) -> Tuple:
    """Run LLaVA forward pass; returns (attn_cpu, meta)."""
    import re
    from collector import load_image, _forward_collect, _generate_collect
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IMAGE_PLACEHOLDER,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token

    image = load_image(image_file)
    image_size = image.size  # (W, H)

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    image_sizes  = [image.size]

    # Build prompt
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in query:
        qs = (
            re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
            if model.config.mm_use_im_start_end
            else re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
        )
    else:
        qs = (
            (image_token_se + "\n" + query)
            if model.config.mm_use_im_start_end
            else (DEFAULT_IMAGE_TOKEN + "\n" + query)
        )

    conv = conv_templates[cfg.model.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt", conv=conv
    ).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        if cfg.model.use_generate:
            attn_last_to_vis, gen_text = _generate_collect(
                model, tokenizer, image_processor, input_ids, image_tensor, image_sizes,
                max_new_tokens=cfg.model.max_new_tokens,
                do_sample=cfg.model.do_sample,
                num_beams=cfg.model.num_beams,
            )
            if attn_last_to_vis is None:
                attn_last_to_vis = _forward_collect(
                    model, tokenizer, image_processor, input_ids, image_tensor, image_sizes
                )
        else:
            gen_text = None
            attn_last_to_vis = _forward_collect(
                model, tokenizer, image_processor, input_ids, image_tensor, image_sizes
            )

    P = int(np.sqrt(attn_last_to_vis.shape[-1]))
    meta = {
        "image_file": image_file,
        "query":      query,
        "image_size": image_size,
        "model_name": cfg.model.name.split("/")[-1],
        "vis_len":    int(attn_last_to_vis.shape[-1]),
        "patch_size": P,
        "patch_h":    P,
        "patch_w":    P,
        "num_layers": int(attn_last_to_vis.shape[0]),
        "num_heads":  int(attn_last_to_vis.shape[1]),
    }
    if gen_text is not None:
        meta["generated_text"] = gen_text

    return attn_last_to_vis.detach().cpu(), meta


# ---------------------------------------------------------------------------
# Shared post-processing & save
# ---------------------------------------------------------------------------

def _run_sample(cfg, model, aux1, aux2, image_file: str, query: str, save_id: str, out_dir: str, bbox_mask: List[int] = None) -> Dict:
    """
    Dispatch to the correct backend, run analysis, and save all four outputs.
    """
    from analyze import analyze_heads
    from bbox import combine_heads, binarize_mean_relu, upscale_mask, save_mask_png
    from viz import plot_heads_grid

    backend = cfg.model.backend

    # ---- Forward pass (backend-specific) ----
    if backend == "qwen_vl":
        attn_cpu, meta = _run_sample_qwen(cfg, model, aux1, image_file, query)
    else:
        attn_cpu, meta = _run_sample_llava(cfg, model, aux1, aux2, image_file, query)

    # ---- Head analysis ----
    selected = analyze_heads(cfg, attn_cpu, meta, bbox_mask=bbox_mask)  # List[Dict] with layer, head, attn_sum, spatial_entropy, bottom_row_focus, num_components
 

    # ---- Mask ----
    Ph, Pw = meta["patch_h"], meta["patch_w"]
    combo     = combine_heads(attn_cpu, selected[: cfg.logic.top_k], Ph=Ph, Pw=Pw,
                              sigma=cfg.logic.smoothing.sigma)
    mask_grid = binarize_mean_relu(combo)
    mask_img  = upscale_mask(mask_grid, meta["image_size"])

    # ---- Save outputs ----
    os.makedirs(out_dir, exist_ok=True)

    # 1. selected_heads.json
    heads_path = os.path.join(out_dir, "selected_heads.json")
    heads_out = [
        {
            "rank": rank,
            "layer": h["layer"],
            "head": h["head"],
            "attn_sum": float(h["attn_sum"]),
            "spatial_entropy": float(h["spatial_entropy"]),
            "num_components": int(h["num_components"]),
            "bottom_row_focus": bool(h["bottom_row_focus"]),
            "AUROC": float(h.get("AUROC", 0.0)),
            "IoU":   float(h.get("IoU",   0.0)),
        }
        for rank, h in enumerate(selected[: cfg.logic.top_k], start=1)
    ]
    with open(heads_path, "w") as fp:
        json.dump(
            {
                "image_id": save_id,
                "query": query,
                "patch_h": Ph,
                "patch_w": Pw,
                "num_layers": meta["num_layers"],
                "num_heads": meta["num_heads"],
                "selected_heads": heads_out,
            },
            fp, indent=2,
        )

    # 2. mask.png
    mask_path = os.path.join(out_dir, "mask.png")
    save_mask_png(mask_path, mask_img)

    # 3. analysis.pkl
    analysis_path = os.path.join(out_dir, "analysis.pkl")
    with open(analysis_path, "wb") as fp:
        pickle.dump(
            {"image_id": save_id, "query": query, "selected": selected, "meta": meta},
            fp,
        )

    # 4. top{K}.png
    fig_path = os.path.join(out_dir, f"top{cfg.logic.top_k}.png")
    plot_heads_grid(
        attn=attn_cpu,
        selected=selected[: cfg.logic.top_k],
        meta=meta,
        save_path=fig_path,
        show_plot=False,
    )

    return {
        "selected_heads": heads_path,
        "mask": mask_path,
        "analysis": analysis_path,
        "fig": fig_path,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run LocalizationHeads pipeline on all TextVQA samples."
    )
    # Data
    p.add_argument("--jsonl",    default="refcoco_samples/metadata.jsonl",
                   help="Path to the JSONL file")
    p.add_argument("--output",   default="refcoco_results",
                   help="Root output directory")
    p.add_argument("--start",    type=int, default=0,
                   help="First sample index (inclusive)")
    p.add_argument("--end",      type=int, default=-1,
                   help="Last sample index exclusive (-1 = all)")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip samples whose output directory already exists")
    # Model
    p.add_argument("--backend",  default="qwen_vl", choices=["qwen_vl", "llava"],
                   help="Model backend: qwen_vl (default) | llava")
    p.add_argument("--model",    default=None,
                   help="HuggingFace model path (overrides backend default)")
    p.add_argument("--conv_mode", default="referseg",
                   help="[llava only] conversation template (default: referseg)")
    p.add_argument("--no_4bit", action="store_true",
                   help="Disable 4-bit quantisation (fp16 instead)")
    # Analysis
    p.add_argument("--top_k",    type=int, default=5,
                   help="Number of top heads to keep (default: 5)")
    # Device
    p.add_argument("--device",   default="auto",
                   help="Device: 'auto', 'cpu', 'cuda:0', … (default: auto)")
    p.add_argument("--device_id", type=int, default=0,
                   help="CUDA device index when device='auto' (default: 0)")
    
    #cfg
    p.add_argument("--enable_cfg", action="store_true",
                   help="Enable classifier-free guidance (CFG) for Qwen attention backend")
    
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = _build_cfg(args)

    script_dir = Path(__file__).parent.resolve()
    jsonl_path = Path(args.jsonl) if Path(args.jsonl).is_absolute() else script_dir / args.jsonl
    out_root   = Path(args.output) if Path(args.output).is_absolute() else script_dir / args.output
    dataset = ""
    if "metadata" in args.jsonl:
        dataset = "refcoco"
    else:
        dataset = "textvqa"
    
    if not jsonl_path.exists():
        log.error(f"JSONL file not found: {jsonl_path}")
        sys.exit(1)

    out_root.mkdir(parents=True, exist_ok=True)
    log.info(f"Backend : {cfg.model.backend}")
    log.info(f"Model   : {cfg.model.name}")
    log.info(f"Output  : {out_root}")

    samples = load_jsonl(str(jsonl_path))
    log.info(f"Loaded {len(samples)} samples from {jsonl_path}")

    start = max(0, args.start)
    end   = len(samples) if args.end < 0 else min(len(samples), args.end)
    work  = samples[start:end]
    log.info(f"Processing [{start}, {end}) → {len(work)} samples")

    # Load model once
    model, aux1, aux2, _ = _load_model(cfg)
    if dataset == "refcoco":
        template = "Segment the object described below, the first token that you generate should match the object:\n"
        
    elif dataset == "textvqa":
        template = """Directly answer the question based on the image, no explanation is needed.\n"
            "If the image does not contain any relevant evidence, "
            "output \"I cannot answer based on the given image.\"\n"""
    
    failed = 0
    success = 0
    skipped = 0

    for i, sample in enumerate(work, start=start):
        image_id   = sample.get("image_id", f"item_{i}")
        image_file = sample.get("image_file", "")
        if dataset == "textvqa":
            question   = sample.get("question", "")
            bbox_mask = sample.get("bboxs", None)  # Optional ground truth bbox_mask for IoU calculation
        elif dataset == "refcoco":
            question = sample.get("sentences", "")[0].get("raw")
            bbox_mask = sample.get("segmentation", None)  # Optional ground truth bbox_mask for IoU calculation

        if image_file and not Path(image_file).is_absolute():
            image_file = str(script_dir / image_file)

        sample_out_dir = str(out_root / image_id)
        log.info(f"[{i+1}/{end}] {image_id}  q='{question}'")

        if args.skip_existing and Path(sample_out_dir).exists():
            log.info("  → Skipping (already exists)")
            skipped += 1
            continue

        if not image_file or not Path(image_file).exists():
            log.warning(f"  → Image not found: {image_file} — skipping")
            failed += 1
            continue

        try:
            paths = _run_sample(
                cfg        = cfg,
                model      = model,
                aux1       = aux1,
                aux2       = aux2,
                image_file = image_file,
                query      = template + question,
                save_id    = image_id,
                out_dir    = sample_out_dir,
                bbox_mask       = bbox_mask,  # Optional ground truth bbox_mask for IoU calculation
            )
            log.info(f"  ✓ selected_heads → {paths['selected_heads']}")
            log.info(f"    mask           → {paths['mask']}")
            log.info(f"    analysis       → {paths['analysis']}")
            log.info(f"    fig            → {paths['fig']}")
            success += 1

        except Exception as exc:
            log.error(f"  ✗ Failed for {image_id}: {exc}", exc_info=True)
            failed += 1

    log.info("=" * 60)
    log.info(f"Done.  success={success}  skipped={skipped}  failed={failed}")
    log.info(f"Results saved in: {out_root}")


if __name__ == "__main__":
    main()
