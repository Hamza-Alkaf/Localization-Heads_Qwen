"""
Qwen3-VL attention collector.

This module provides a drop-in replacement for the LLaVA-based collect_attention()
that works with Qwen/Qwen3-VL-* (and Qwen2-VL-*) models via the standard
transformers API.

Key differences from LLaVA:
  - Uses AutoProcessor (handles both tokenisation AND image preprocessing in one step).
  - Uses Qwen2_5_VLForConditionalGeneration (or Qwen2VLForConditionalGeneration).
  - Image tokens are *dynamic*: the number of visual tokens depends on the image
    resolution and the min/max_pixels budget you set in the model config.
  - We locate visual tokens by scanning for the special <|vision_start|> and
    <|vision_end|> tokens in the processed input_ids, not via a fixed index.
"""
from __future__ import annotations

import os
import pickle
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from lab.stations import MetadataStation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(cfg) -> str:
    device = cfg.device
    if device == "auto":
        device = f"cuda:{cfg.device_id}" if cfg.device_id >= 0 and torch.cuda.is_available() else "cpu"
    return device


def _load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        import requests
        from io import BytesIO
        resp = requests.get(path_or_url)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    if not os.path.exists(path_or_url):
        raise FileNotFoundError(f"Image not found: {path_or_url}")
    return Image.open(path_or_url).convert("RGB")


def _sanitize_name(s: str) -> str:
    return s.replace("/", "-").replace(" ", "_")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_qwen_vl_model(cfg) -> Tuple[object, object]:
    """Load Qwen3-VL / Qwen2-VL model + processor.

    Returns: (model, processor)
    """
    # Optional cache override
    if getattr(cfg.model, "cache_dir", None):
        cache_dir = str(cfg.model.cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_HUB_CACHE"] = cache_dir

    device = _resolve_device(cfg)
    model_name = cfg.model.name  # e.g. "Qwen/Qwen3-VL-2B-Instruct"

    # Use Qwen3_VL if available (transformers >= 4.52), fall back to Qwen2VL for older installs
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        ModelClass = Qwen3VLForConditionalGeneration
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        ModelClass = Qwen2VLForConditionalGeneration

    attn_impl = "flash_attention_2" if getattr(cfg.model, "use_flash_attn", False) else "eager"

    min_pixels = getattr(cfg.model, "min_pixels", 256) * 28 * 28
    max_pixels = getattr(cfg.model, "max_pixels", 1024) * 28 * 28

    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=getattr(cfg.model, "cache_dir", None),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    load_kwargs: dict = dict(
        cache_dir=getattr(cfg.model, "cache_dir", None),
        device_map=device,
        attn_implementation=attn_impl,
    )

    if getattr(cfg.model, "load_4bit", False):
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # BitsAndBytesConfig sets the storage dtype; torch_dtype must not conflict
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = ModelClass.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_messages(query: str) -> list:
    """Build the chat-style messages list expected by the Qwen processor.

    The <image> placeholder is inserted automatically by the processor when
    it finds the 'image' key in the content list.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},  # replaced by processor
                {"type": "text", "text": query},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Visual token range detection
# ---------------------------------------------------------------------------

def _find_visual_token_range(input_ids: torch.Tensor, processor) -> Tuple[int, int]:
    """Return (begin_pos, vis_len) for the first image's visual tokens.

    Qwen-VL wraps image tokens between special sentinel tokens:
        <|vision_start|>  ...image patch tokens...  <|vision_end|>

    We locate them by scanning input_ids for those token IDs.
    """
    tokenizer = processor.tokenizer

    # Sentinel token ids
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id   = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    ids = input_ids[0].tolist()  # batch dim → list

    try:
        start_idx = ids.index(vision_start_id) + 1   # first patch token
        end_idx   = ids.index(vision_end_id)          # exclusive
    except ValueError:
        raise RuntimeError(
            "Could not find <|vision_start|>/<|vision_end|> tokens in the "
            "processed input_ids. Make sure you are using an image-capable "
            "Qwen3-VL / Qwen2-VL checkpoint."
        )

    return start_idx, end_idx - start_idx   # begin_pos, vis_len


# ---------------------------------------------------------------------------
# Forward pass with attention collection
# ---------------------------------------------------------------------------

def _forward_collect_qwen(model, input_ids, pixel_values, image_grid_thw,
                           mm_token_type_ids, begin_pos_vis, vis_len):
    """Single forward pass; returns attn [L, H, 1, V]."""
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        output_attentions=True,
        return_dict=True,
    )
    attn_layers = outputs.attentions  # tuple of L tensors, each [B, H, Tq, Tk]
    if not attn_layers:
        raise RuntimeError("No attentions returned from the forward pass.")

    # Guard: filter out None layers (can happen with quantised models)
    valid_layers = [t for t in attn_layers if t is not None]
    if not valid_layers:
        raise RuntimeError("All attention layers returned None from the forward pass.")

    # Stack all layers: [L, H, Tq, Tk] (batch=1 → squeeze)
    attn = torch.stack([t[0] for t in valid_layers], dim=0)

    # Slice: last query token → visual key tokens
    attn_last_to_vis = attn[:, :, -1:, begin_pos_vis: begin_pos_vis + vis_len]
    return attn_last_to_vis


def _generate_collect_qwen(model, processor, input_ids, pixel_values, image_grid_thw,
                            mm_token_type_ids, begin_pos_vis, vis_len,
                            max_new_tokens=10, do_sample=False, num_beams=1):
    """generate() pass; returns (attn [L,H,1,V] or None, generated_text str).

    Returns None to signal the caller to fall back to _forward_collect_qwen when:
      - generate() does not expose attentions (flash-attn backend)
      - the visual-token slice is out-of-range (src too short)
      - the returned attention over visual tokens is degenerate (all zeros)
    """
    # attention_mask is required for correct KV-cache bookkeeping in Qwen-VL.
    # Without it the model may produce garbage / all-zero attention weights.
    attention_mask = torch.ones_like(input_ids)

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        return_dict_in_generate=True,
        output_attentions=True,
    )
    input_len = input_ids.shape[1]
    gen_ids = gen.sequences[:, input_len:]
    generated_text = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

    attn_last_to_vis = None
    if hasattr(gen, "attentions") and gen.attentions:
        # ----------------------------------------------------------------
        # In modern HF transformers (>=4.40), gen.attentions[0] is often
        # the PREFILL step with shape [B, H, T_input, T_input], NOT the
        # first generated token.  The first true decode step has Tq==1.
        #
        # Strategy: iterate through all steps and find the first one where
        # every valid layer has Tq==1.  Fall back to the last row of the
        # first available step if no Tq==1 step exists.
        # ----------------------------------------------------------------

        # Diagnostic: print step count and shapes of first two steps
        num_steps = len(gen.attentions)
        _shapes = []
        for _i, _step in enumerate(gen.attentions[:3]):
            _vl = [t for t in _step if t is not None]
            if _vl:
                _shapes.append(f"step{_i}: [L={len(_vl)}, H={_vl[0][0].shape[0]}, "
                               f"Tq={_vl[0][0].shape[1]}, Tk={_vl[0][0].shape[2]}]")
        print(f"[DEBUG] _generate_collect_qwen: gen.attentions has {num_steps} step(s). "
              f"First steps shapes: {_shapes}")

        # Find the first step whose Tq == 1  (= first decoded token's attention)
        decode_step_attn = None
        prefill_step_attn = None  # fallback: last row of prefill
        for _step in gen.attentions:
            valid_layers = [t for t in _step if t is not None]
            if not valid_layers:
                continue
            _stacked = torch.stack([t[0] for t in valid_layers], dim=0)  # [L, H, Tq, Tk]
            if prefill_step_attn is None:
                prefill_step_attn = _stacked  # keep first non-None step as prefill fallback
            if _stacked.shape[-2] == 1:        # Tq==1 → true decode step
                decode_step_attn = _stacked
                print(f"[INFO] _generate_collect_qwen: using decode-step attention "
                      f"(Tq=1, Tk={_stacked.shape[-1]})")
                break

        if decode_step_attn is None:
            # No Tq==1 step found: transformers exposed only the prefill.
            # Use the last row (last prompt position) of the prefill — semantically
            # equivalent to forward-pass mode; we warn the caller.
            decode_step_attn = prefill_step_attn
            if decode_step_attn is not None:
                print(
                    f"[WARNING] _generate_collect_qwen: no Tq==1 decode step found in "
                    f"gen.attentions ({num_steps} steps). Using last row of prefill step "
                    "(equivalent to forward-pass mode)."
                )

        if decode_step_attn is not None:
            src = decode_step_attn.shape[-1]  # Tk = full context length

            # Guard: visual-token slice must fit within Tk
            if vis_len > 0 and (begin_pos_vis + vis_len) <= src:
                # -1 on Tq axis: works for decode (Tq=1) and prefill (Tq=T, last row)
                candidate = decode_step_attn[:, :, -1:, begin_pos_vis: begin_pos_vis + vis_len]

                # Guard: degenerate all-zero result → fall back to forward pass
                if candidate.sum().item() == 0.0:
                    print(
                        "[WARNING] _generate_collect_qwen: visual attention slice is all-zero "
                        f"(src={src}, begin_pos_vis={begin_pos_vis}, vis_len={vis_len}). "
                        "Falling back to forward-pass attention."
                    )
                else:
                    attn_last_to_vis = candidate
            else:
                print(
                    f"[WARNING] _generate_collect_qwen: src={src} is smaller than "
                    f"begin_pos_vis({begin_pos_vis})+vis_len({vis_len}). "
                    "Falling back to forward-pass attention."
                )

    return attn_last_to_vis, generated_text


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def collect_attention_qwen(cfg, image_file: str, query: str, save_dir: str, save_id: str) -> str:
    """Run one forward pass through Qwen3-VL and save visual attention.

    Saves a pickle at <save_dir>/<model_dir>/<save_id>.pkl with:
        {
          'attn': Tensor[L, H, 1, V],
          'meta': { image_file, query, image_size, model_name,
                    vis_len, patch_size, num_layers, num_heads }
        }

    Returns the saved file path.
    """
    device = _resolve_device(cfg)
    model, processor = load_qwen_vl_model(cfg)

    # ---- Image ----
    image = _load_image(image_file)
    image_size = image.size  # (W, H)

    # ---- Prompt ----
    messages = _build_messages(query)
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # ---- Process inputs ----
    # Only pixel_values should be float16; input_ids and mm_token_type_ids must stay int.
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

    # ---- Locate visual tokens in the sequence ----
    begin_pos_vis, vis_len = _find_visual_token_range(input_ids, processor)

    # Register in MetadataStation so downstream code (collector.py) can access them
    MetadataStation.set_begin_pos("vis", begin_pos_vis)
    MetadataStation.set_vis_len(vis_len)

    # ---- Collect attentions ----
    with torch.inference_mode():
        if getattr(cfg.model, "use_generate", False):
            attn_last_to_vis, gen_text = _generate_collect_qwen(
                model, processor, input_ids, pixel_values, image_grid_thw,
                mm_token_type_ids, begin_pos_vis, vis_len,
                max_new_tokens=getattr(cfg.model, "max_new_tokens", 10),
                do_sample=getattr(cfg.model, "do_sample", False),
                num_beams=getattr(cfg.model, "num_beams", 1),
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

    # ---- Build meta ----
    # image_grid_thw = [num_images, 3] with (T, H_raw, W_raw) in raw patch units BEFORE the
    # spatial merge operation inside the vision encoder. Qwen3-VL merges every
    # (spatial_merge_size × spatial_merge_size) patch block into one token, so the
    # actual attention token grid is (H_raw // merge) × (W_raw // merge).
    if image_grid_thw is not None:
        thw = image_grid_thw[0].cpu().tolist()   # [T, H_raw, W_raw]
        # spatial_merge_size lives in vision_config (Qwen3-VL default = 2)
        vcfg = getattr(model.config, "vision_config", model.config)
        merge = int(getattr(vcfg, "spatial_merge_size", 2))
        patch_h = int(thw[1]) // merge
        patch_w = int(thw[2]) // merge
    else:
        # Fallback: assume square (only reached if processor didn't return grid info)
        patch_h = patch_w = int(np.sqrt(attn_last_to_vis.shape[-1]))

    model_name_str = cfg.model.name.split("/")[-1]

    meta = {
        "image_file":  image_file,
        "query":       query,
        "image_size":  image_size,
        "model_name":  model_name_str,
        "vis_len":     int(attn_last_to_vis.shape[-1]),
        "patch_size":  patch_h,   # kept for LLaVA-compat readers; use patch_h/patch_w for reshape
        "patch_h":     patch_h,
        "patch_w":     patch_w,
        "num_layers":  int(attn_last_to_vis.shape[0]),
        "num_heads":   int(attn_last_to_vis.shape[1]),
    }
    if gen_text is not None:
        meta["generated_text"] = gen_text

    # ---- Save ----
    model_dir = _sanitize_name(cfg.model.name)
    out_dir = os.path.join(save_dir, model_dir)
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, f"{save_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"attn": attn_last_to_vis.detach().cpu(), "meta": meta}, f)

    with open(os.path.join(out_dir, f"{save_id}_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    return save_path
