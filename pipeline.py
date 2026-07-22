#!/usr/bin/env python3
import os
import json
import pickle
from typing import List, Dict

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from collector import collect_attention
from analyze import load_attention_file, analyze_heads
from bbox import (
    combine_heads,
    binarize_mean_relu,
    upscale_mask,
    bbox_from_mask,
    scale_bbox_to_image,
    save_bbox_json,
    save_mask_png,
)
from viz import plot_heads_grid


def _sanitize(s: str) -> str:
    return s.replace("/", "-").replace(" ", "_")


def _model_dir(cfg: DictConfig) -> str:
    return _sanitize(cfg.model.name)


def _out_root(cfg) -> str:
    return cfg.data.output_dir


def run_single(cfg: DictConfig) -> Dict:
    os.makedirs(_out_root(cfg), exist_ok=True)
    model_dir = _model_dir(cfg)
    attn_root = os.path.join(_out_root(cfg), model_dir)
    os.makedirs(attn_root, exist_ok=True)

    # id for saving
    save_id = cfg.data.get("save_id", "sample")
    query = """Directly answer the question based on the image, no explanation is needed.\n    If the image does not contain any relevant evidence, 
    output "I cannot answer based on the given image. \n""" + cfg.data.query
    # Stage: collect
    attn_file = collect_attention(
        cfg=cfg,
        image_file=cfg.data.image_file,
        query=query,
        save_dir=_out_root(cfg),
        save_id=save_id,
    )

    # Stage: analyze
    attn, meta = load_attention_file(attn_file)
    selected = analyze_heads(cfg, attn, meta)
    

    # Save analysis
    with open(attn_file.replace('.pkl', '_analysis.pkl'), 'wb') as f:
        pickle.dump(selected, f)

    # Visualization
    if cfg.save_fig:
        fig_path = os.path.join(attn_root, f"{save_id}_top{cfg.logic.top_k}.png")
        plot_heads_grid(attn, selected[: cfg.logic.top_k], meta, fig_path, show_plot=cfg.show_plot)

    # Combine to bbox/mask  (Ph/Pw support non-square grids, e.g. Qwen3-VL)
    V = attn.shape[-1]
    Ph = int(meta.get("patch_h", meta.get("patch_size", int(V ** 0.5))))
    Pw = int(meta.get("patch_w", meta.get("patch_size", int(V ** 0.5))))
    combo = combine_heads(attn, selected[: cfg.logic.top_k], Ph=Ph, Pw=Pw, sigma=cfg.logic.smoothing.sigma)
    mask_grid = binarize_mean_relu(combo)
    bbox_grid = bbox_from_mask(mask_grid)
    mask_img = upscale_mask(mask_grid, meta["image_size"])  # [H,W] uint8
    bbox_img = scale_bbox_to_image(bbox_grid, meta["image_size"], Ph=Ph, Pw=Pw)

    # Save bbox/mask
    mask_path = os.path.join(attn_root, f"{save_id}_mask.png")
    save_mask_png(mask_path, mask_img)
    bbox_path = os.path.join(attn_root, f"{save_id}_bbox.json")
    save_bbox_json(bbox_path, bbox_img, meta["image_size"], selected[: cfg.logic.top_k])

    return {"attn_file": attn_file, "analysis": selected, "mask": mask_path, "bbox": bbox_path}


def run_analyze_visualize(cfg: DictConfig) -> None:
    attn, meta = load_attention_file(cfg.data.attention_file)
    selected = analyze_heads(cfg, attn, meta)
    model_dir = _model_dir(cfg)
    out_dir = os.path.join(_out_root(cfg), model_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(cfg.data.attention_file.replace('.pkl', '_analysis.pkl'), 'wb') as f:
        pickle.dump(selected, f)
    if cfg.save_fig:
        fig_path = cfg.data.attention_file.replace('.pkl', f'_top{cfg.logic.top_k}.png')
        plot_heads_grid(attn, selected[: cfg.logic.top_k], meta, fig_path, cfg.show_plot)
    # Optionally bbox/mask
    V = attn.shape[-1]
    Ph = int(meta.get("patch_h", meta.get("patch_size", int(V ** 0.5))))
    Pw = int(meta.get("patch_w", meta.get("patch_size", int(V ** 0.5))))
    combo = combine_heads(attn, selected[: cfg.logic.top_k], Ph=Ph, Pw=Pw, sigma=cfg.logic.smoothing.sigma)
    mask_grid = binarize_mean_relu(combo)
    bbox_grid = bbox_from_mask(mask_grid)
    mask_img = upscale_mask(mask_grid, meta["image_size"])  # [H,W]
    bbox_img = scale_bbox_to_image(bbox_grid, meta["image_size"], Ph=Ph, Pw=Pw)
    save_mask_png(cfg.data.attention_file.replace('.pkl', '_mask.png'), mask_img)
    save_bbox_json(cfg.data.attention_file.replace('.pkl', '_bbox.json'), bbox_img, meta["image_size"], selected[: cfg.logic.top_k])


def run_batch(cfg: DictConfig) -> None:
    # Keep JSONL schema: id, prompt, image_path
    path = cfg.data.data_file
    assert os.path.exists(path), f"Data file not found: {path}"
    lines: List[Dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    lines.append(json.loads(ln))
                except Exception:
                    pass

    start = max(0, cfg.data.start_index)
    end = len(lines) if cfg.data.end_index < 0 else min(len(lines), cfg.data.end_index)
    work = lines if cfg.data.process_all else lines[start:end]

    for i, entry in enumerate(work):
        sid = entry.get('id', f'item_{i}')
        image_file = entry.get('image', '')
        query = entry.get('prompt', '')
        res = run_single(hydra.utils.instantiate(cfg, _convert_="object")) if False else None
        # We cannot deep-copy DictConfig with instantiate easily; call functions directly
        attn_file = collect_attention(cfg, image_file, query, _out_root(cfg), sid)
        attn, meta = load_attention_file(attn_file)
        selected = analyze_heads(cfg, attn, meta)
        model_dir = _model_dir(cfg)
        attn_root = os.path.join(_out_root(cfg), model_dir)
        if cfg.save_fig and cfg.data.visualize_batch:
            fig_path = os.path.join(attn_root, f"{sid}_top{cfg.logic.top_k}.png")
            plot_heads_grid(attn, selected[: cfg.logic.top_k], meta, fig_path, show_plot=cfg.show_plot)
        V = attn.shape[-1]
        Ph = int(meta.get("patch_h", meta.get("patch_size", int(V ** 0.5))))
        Pw = int(meta.get("patch_w", meta.get("patch_size", int(V ** 0.5))))
        combo = combine_heads(attn, selected[: cfg.logic.top_k], Ph=Ph, Pw=Pw, sigma=cfg.logic.smoothing.sigma)
        mask_grid = binarize_mean_relu(combo)
        bbox_grid = bbox_from_mask(mask_grid)
        mask_img = upscale_mask(mask_grid, meta["image_size"])  # [H,W]
        bbox_img = scale_bbox_to_image(bbox_grid, meta["image_size"], Ph=Ph, Pw=Pw)
        mask_path = os.path.join(attn_root, f"{sid}_mask.png")
        bbox_path = os.path.join(attn_root, f"{sid}_bbox.json")
        save_mask_png(mask_path, mask_img)
        save_bbox_json(bbox_path, bbox_img, meta["image_size"], selected[: cfg.logic.top_k])


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if cfg.stage == "collect":
        assert cfg.data.image_file and cfg.data.query, "image_file and query required"
        attn_file = collect_attention(cfg, cfg.data.image_file, cfg.data.query, _out_root(cfg), cfg.data.get("save_id", "sample"))
        print(f"Saved attention to: {attn_file}")
    elif cfg.stage == "analyze":
        assert cfg.data.attention_file, "attention_file required"
        run_analyze_visualize(cfg)
    elif cfg.stage == "visualize":
        assert cfg.data.attention_file, "attention_file required"
        run_analyze_visualize(cfg)
    elif cfg.stage == "pipeline":
        assert cfg.data.image_file and cfg.data.query, "image_file and query required"
        out = run_single(cfg)
        print(json.dumps({k: v for k, v in out.items()}, indent=2))
    elif cfg.stage == "batch":
        run_batch(cfg)
    else:
        raise ValueError(f"Invalid stage: {cfg.stage}")


if __name__ == "__main__":
    main()
