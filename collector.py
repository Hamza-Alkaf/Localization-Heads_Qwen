import os
import re
import pickle
from typing import Dict, Tuple

import torch
import numpy as np
from PIL import Image

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from lab.stations import MetadataStation


def _sanitize_name(s: str) -> str:
    return s.replace("/", "-").replace(" ", "_")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        import requests
        from io import BytesIO
        resp = requests.get(path_or_url)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    if not os.path.exists(path_or_url):
        raise FileNotFoundError(f"Image not found: {path_or_url}")
    return Image.open(path_or_url).convert("RGB")


def load_model_from_cfg(cfg) -> Tuple[object, object, object, int, str]:
    """Build model/tokenizer/image_processor using Hugging Face path.

    Returns: tokenizer, model, image_processor, context_len, model_name_str
    """
    disable_torch_init()
    # Optional user cache control (force override; must be set before downloads)
    if getattr(cfg.model, "cache_dir", None):
        cache_dir = str(cfg.model.cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_HUB_CACHE"] = cache_dir

    # Choose device string
    device = cfg.device
    if device == "auto":
        device = f"cuda:{cfg.device_id}" if cfg.device_id >= 0 and torch.cuda.is_available() else "cpu"

    model_name_str = get_model_name_from_path(cfg.model.name)
    tok, model, img_proc, context_len = load_pretrained_model(
        model_path=cfg.model.name,
        cache_dir=cfg.model.cache_dir,
        model_base=cfg.model.base,
        model_name=model_name_str,
        device=device,
        use_flash_attn=getattr(cfg.model, "use_flash_attn", False),
        
    )
    return tok, model, img_proc, context_len, model_name_str


def _forward_collect(model, tokenizer, image_processor, input_ids, image_tensor, image_sizes):
    """Collect attentions via a single forward pass.

    Returns attention focused on image tokens with shape [L, H, 1, V].
    """
    outputs = model(
        input_ids=input_ids,
        images=image_tensor.unsqueeze(0),
        image_sizes=image_sizes,
        output_attentions=True,
        return_dict=True,
    )
    attn_layers = outputs.attentions  # tuple length L of [B,H,Tq,Tk]
    if not attn_layers:
        raise RuntimeError("No attentions returned from forward()")

    layers = []
    for t in attn_layers:
        layers.append(t[0])  # [H,Tq,Tk] for batch=1
    attn = torch.stack(layers, dim=0)  # [L,H,Tq,Tk]

    begin_pos_vis = MetadataStation.get_begin_pos('vis')
    vis_len = MetadataStation.get_vis_len()
    if begin_pos_vis is None or vis_len is None:
        raise RuntimeError("Missing visual token segmentation info.")
    attn_last_to_vis = attn[:, :, -1:, begin_pos_vis:begin_pos_vis + vis_len]
    return attn_last_to_vis


def _generate_collect(model, tokenizer, image_processor, input_ids, image_tensor, image_sizes, max_new_tokens=10, do_sample=False, num_beams=1):
    """Run generate to obtain output tokens, and try to collect attentions from the first generation step.

    Returns None to signal the caller to fall back to _forward_collect when:
      - generate() does not expose attentions (flash-attn backend)
      - the visual-token slice is out-of-range (src too short)
      - the returned attention over visual tokens is degenerate (all zeros)

    Returns: (attn [L,H,1,V] or None, generated_text str)
    """
    # attention_mask is required for correct KV-cache bookkeeping.
    # Without it the model may produce garbage / all-zero attention weights.
    attention_mask = torch.ones_like(input_ids)

    gen = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        images=image_tensor.unsqueeze(0),
        image_sizes=image_sizes,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        return_dict_in_generate=True,
        output_attentions=True,
    )
    sequences = gen.sequences
    input_len = input_ids.shape[1]
    gen_ids = sequences[:, input_len:]
    generated_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

    attn_last_to_vis = None
    if hasattr(gen, 'attentions') and gen.attentions:
        # ----------------------------------------------------------------
        # In modern HF transformers (>=4.40), gen.attentions[0] is often
        # the PREFILL step with shape [B, H, T_input, T_input], NOT the
        # first generated token.  The first true decode step has Tq==1.
        #
        # Strategy: iterate through all steps and find the first one where
        # every valid layer has Tq==1.  Fall back to the last row of the
        # first available step if no Tq==1 step exists.
        # ----------------------------------------------------------------

        begin_pos_vis = MetadataStation.get_begin_pos('vis')
        vis_len = MetadataStation.get_vis_len()
        if begin_pos_vis is None or vis_len is None:
            raise RuntimeError("Missing visual token segmentation info.")

        # Diagnostic: print step count and shapes of first two steps
        num_steps = len(gen.attentions)
        _shapes = []
        for _i, _step in enumerate(gen.attentions[:3]):
            _vl = [t for t in _step if t is not None]
            if _vl:
                _shapes.append(f"step{_i}: [L={len(_vl)}, H={_vl[0][0].shape[0]}, "
                               f"Tq={_vl[0][0].shape[1]}, Tk={_vl[0][0].shape[2]}]")
        print(f"[DEBUG] _generate_collect: gen.attentions has {num_steps} step(s). "
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
                print(f"[INFO] _generate_collect: using decode-step attention "
                      f"(Tq=1, Tk={_stacked.shape[-1]})")
                break

        if decode_step_attn is None:
            # No Tq==1 step found: transformers exposed only the prefill.
            # Use the last row (last prompt position) of the prefill — semantically
            # equivalent to forward-pass mode; we warn the caller.
            decode_step_attn = prefill_step_attn
            if decode_step_attn is not None:
                print(
                    f"[WARNING] _generate_collect: no Tq==1 decode step found in "
                    f"gen.attentions ({num_steps} steps). Using last row of prefill step "
                    "(equivalent to forward-pass mode)."
                )

        if decode_step_attn is not None:
            src = decode_step_attn.shape[-1]  # Tk = full context length

            # Guard: visual-token slice must fit within Tk
            if vis_len > 0 and (begin_pos_vis + vis_len) <= src:
                # -1 on Tq axis: works for decode (Tq=1) and prefill (Tq=T, last row)
                candidate = decode_step_attn[:, :, -1:, begin_pos_vis:begin_pos_vis + vis_len]

                # Guard: degenerate all-zero result → fall back to forward pass
                if candidate.sum().item() == 0.0:
                    print(
                        "[WARNING] _generate_collect: visual attention slice is all-zero "
                        f"(src={src}, begin_pos_vis={begin_pos_vis}, vis_len={vis_len}). "
                        "Falling back to forward-pass attention."
                    )
                else:
                    attn_last_to_vis = candidate
            else:
                print(
                    f"[WARNING] _generate_collect: src={src} is smaller than "
                    f"begin_pos_vis({begin_pos_vis})+vis_len({vis_len}). "
                    "Falling back to forward-pass attention."
                )

    return attn_last_to_vis, generated_text


def collect_attention(cfg, image_file: str, query: str, save_dir: str, save_id: str) -> str:
    """Run one forward pass and save attention focused on image tokens.

    Dispatches to backend-specific collector based on cfg.model.backend.
    Supported backends: 'llava' (default), 'qwen_vl'.

    Saves a pickle with dict: {
      'attn': Tensor[L, H, 1, V],
      'meta': {image_file, query, image_size, model_name, vis_len, patch_size, num_layers, num_heads}
    }
    Returns the saved file path.
    """
    backend = getattr(cfg.model, "backend", "llava")
    if backend == "qwen_vl":
        from collector_qwen import collect_attention_qwen
        return collect_attention_qwen(cfg, image_file, query, save_dir, save_id)

    # --- LLaVA backend (original code below) ---
    tokenizer, model, image_processor, _, model_name_str = load_model_from_cfg(cfg)

    # Prepare image
    image = load_image(image_file)
    image_size = image.size  # (W, H)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    image_sizes = [image.size]

    # Prepare prompt
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in query:
        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, query) if model.config.mm_use_im_start_end else re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
    else:
        qs = (image_token_se + "\n" + query) if model.config.mm_use_im_start_end else (DEFAULT_IMAGE_TOKEN + "\n" + query)

    conv = conv_templates[cfg.model.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
        conv=conv,
    ).unsqueeze(0).to(model.device)

    # Collect attentions (and optional generated text)
    with torch.inference_mode():
        if getattr(cfg.model, 'use_generate', False):
            attn_last_to_vis, gen_text = _generate_collect(
                model, tokenizer, image_processor, input_ids, image_tensor, image_sizes,
                max_new_tokens=getattr(cfg.model, 'max_new_tokens', 10),
                do_sample=getattr(cfg.model, 'do_sample', False),
                num_beams=getattr(cfg.model, 'num_beams', 1),
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
        "query": query,
        "image_size": image_size,
        "model_name": model_name_str,
        "vis_len": int(attn_last_to_vis.shape[-1]),
        "patch_size": int(P),
        "num_layers": int(attn_last_to_vis.shape[0]),
        "num_heads": int(attn_last_to_vis.shape[1]),
    }
    if getattr(cfg.model, 'use_generate', False) and gen_text is not None:
        meta["generated_text"] = gen_text

    model_dir = _sanitize_name(cfg.model.name)
    out_dir = os.path.join(save_dir, model_dir)
    _ensure_dir(out_dir)

    save_path = os.path.join(out_dir, f"{save_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"attn": attn_last_to_vis.detach().cpu(), "meta": meta}, f)

    # Save a small side metadata for convenience
    with open(os.path.join(out_dir, f"{save_id}_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    return save_path
