"""
policy_loader.py — Shared utility to load the INTACT Pi-0 checkpoint into lerobot 0.4.x.

The INTACT checkpoint was saved with an older lerobot config schema.
This module bridges the gap by:
  1. Downloading and parsing the original config.json
  2. Mapping deprecated / renamed fields to the current PI0Config fields
  3. Constructing PI0Config and calling from_pretrained() with it
"""

import json
from pathlib import Path

import torch


def _parse_input_features(raw_features: dict) -> dict:
    """Convert the old input_features dict into lerobot 0.4.x PolicyFeature objects."""
    from lerobot.configs.types import PolicyFeature, FeatureType

    name_map = {
        "VISUAL": FeatureType.VISUAL,
        "STATE":  FeatureType.STATE,
        "ACTION": FeatureType.ACTION,
    }

    result = {}
    for key, val in raw_features.items():
        ftype = name_map[val["type"]]
        shape = tuple(val["shape"])
        result[key] = PolicyFeature(type=ftype, shape=shape)
    return result


def load_intact_pi0(
    model_id: str = "juexzz/INTACT-pi0-finetune-rephrase-bridge",
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.float16,
) -> tuple:
    """
    Load the INTACT Pi-0 checkpoint, patching the config schema.

    Returns:
        (policy, image_key, image_resolution)
          policy            — loaded PI0Policy, in eval mode, on `device`
          image_key         — the observation key to use for images (e.g. 'observation.images.top')
          image_resolution  — (H, W) tuple the model expects
    """
    from huggingface_hub import hf_hub_download
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    # ── 1. Load and patch config ───────────────────────────────────────────────
    config_path = hf_hub_download(model_id, "config.json")
    with open(config_path) as f:
        raw = json.load(f)

    # Fields that were renamed or restructured in lerobot 0.4.x
    field_renames = {
        "num_steps":              "num_inference_steps",
        "tokenizer_max_len":      "tokenizer_max_length",
    }
    # Fields that no longer exist in PI0Config and should be dropped
    fields_to_drop = {
        "type", "resize_imgs_with_padding", "proj_width", "use_cache",
        "adapt_to_pi_aloha", "use_delta_joint_actions_aloha",
        "attention_implementation", "train_state_proj", "paligemma_pretrained_path",
    }

    patched = {}
    for k, v in raw.items():
        if k in fields_to_drop:
            continue
        new_k = field_renames.get(k, k)
        patched[new_k] = v

    # resolve image_resolution from the (now dropped) resize_imgs_with_padding
    if "image_resolution" not in patched:
        rir = raw.get("resize_imgs_with_padding")
        if rir:
            patched["image_resolution"] = tuple(rir)

    # Parse input/output features into typed objects
    input_features  = _parse_input_features(patched.pop("input_features",  {}))
    output_features = _parse_input_features(patched.pop("output_features", {}))

    # Build config (only pass known PI0Config fields)
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(PI0Config)}
    safe_kwargs  = {k: v for k, v in patched.items() if k in valid_fields}

    config = PI0Config(
        input_features=input_features,
        output_features=output_features,
        **safe_kwargs,
    )
    config.device = device

    # ── 2. Load weights ────────────────────────────────────────────────────────
    print(f"Loading INTACT Pi-0 from: {model_id}")
    policy = PI0Policy.from_pretrained(model_id, config=config, torch_dtype=torch_dtype)
    policy = policy.to(device)
    policy.eval()
    print("✓ Policy loaded successfully")

    # ── 3. Determine image key and resolution ─────────────────────────────────
    image_keys = [k for k in input_features if "image" in k.lower() or "visual" in input_features[k].type.name.lower()]
    image_key  = image_keys[0] if image_keys else "observation.image"
    res = config.image_resolution  # (H, W)

    return policy, image_key, res


def tokenize(instructions: list[str], device: str, max_length: int = 72) -> tuple:
    """Tokenise a list of instruction strings using the PaliGemma tokenizer.

    The INTACT checkpoint has no tokenizer_config.json, so we load the
    tokenizer from the lerobot/pi0 base model which uses the same tokenizer.

    Args:
        instructions:  list of raw instruction strings
        device:        torch device string
        max_length:    pad/truncate to this fixed length (must match config.tokenizer_max_length)
    """
    from transformers import AutoTokenizer

    TOKENIZER_ID = "lerobot/pi0"
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    # PaliGemma expects a trailing newline
    texts = [t if t.endswith("\n") else t + "\n" for t in instructions]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",   # pad every sequence to exactly max_length
        max_length=max_length,
        truncation=True,
    )
    # lerobot's modeling_pi0 uses lang_masks as a boolean pad-mask (True = valid)
    return enc["input_ids"].to(device), enc["attention_mask"].bool().to(device)


