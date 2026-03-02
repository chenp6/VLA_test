#!/usr/bin/env python3
"""
test_sanity.py — Minimal inference sanity check for INTACT Pi-0 BridgeV2 finetuned checkpoint.

Loads the policy from HuggingFace and runs it on dummy inputs to verify shapes and
that the checkpoint is usable — no simulator required.

Usage:
    python test_sanity.py [--device cuda|cpu] [--dtype float16|bfloat16|float32]
"""

import argparse
import sys
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity test for INTACT Pi-0 checkpoint")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype",  type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size",  type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if args.device not in ("auto",):
        device = args.device

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("INTACT Pi-0 BridgeV2 — Sanity Test")
    print("=" * 60)
    print(f"  Device : {device}")
    print(f"  Dtype  : {args.dtype}")
    print(f"  Batch  : {args.batch_size}")
    if device == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # ── 1. Load model ──────────────────────────────────────────────────────────
    MODEL_ID = "juexzz/INTACT-pi0-finetune-rephrase-bridge"
    print(f"Loading policy: {MODEL_ID}")
    print("  (First run downloads ~several GB — please wait…)\n")

    from policy_loader import load_intact_pi0, tokenize
    import dataclasses

    t0 = time.time()
    policy, image_key, (H, W) = load_intact_pi0(MODEL_ID, device=device, torch_dtype=torch_dtype)
    tok_max_len = policy.config.tokenizer_max_length
    print(f"\n✓ Model loaded in {time.time() - t0:.1f}s")
    print(f"  Image key       : {image_key}")
    print(f"  Image resolution: {H}x{W}\n")

    # ── 2. Build dummy batch ───────────────────────────────────────────────────
    B = args.batch_size
    instructions = ["put the carrot on the plate",
                    "place the eggplant in the basket"][:B]
    while len(instructions) < B:
        instructions.append("pick up the object and place it in the target")

    lang_tokens, lang_masks = tokenize(instructions, device, max_length=tok_max_len)

    images = torch.rand(B, 3, H, W, dtype=torch.float32, device=device)
    state  = torch.randn(B, 7, dtype=torch.float32, device=device)

    batch = {
        image_key:                              images,
        "observation.state":                    state,
        "observation.language.tokens":          lang_tokens,
        "observation.language.attention_mask":  lang_masks,
    }

    print("Dummy batch:")
    for k, v in batch.items():
        print(f"  {k:<50s}: {tuple(v.shape)}")
    print()

    # ── 3. Forward pass ───────────────────────────────────────────────────────
    print("Running policy.select_action(batch) …")
    t1 = time.time()
    with torch.no_grad():
        actions = policy.select_action(batch)
    infer_ms = (time.time() - t1) * 1000

    # ── 4. Report ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Action shape  : {tuple(actions.shape)}")
    print(f"  Action dtype  : {actions.dtype}")
    print(f"  Inference time: {infer_ms:.1f} ms")
    print()
    print(f"  First 7 values (batch[0]): {actions[0].cpu().float().tolist()[:7]}")
    print()
    assert actions.shape[0] == B, f"Batch size mismatch: {actions.shape[0]} != {B}"
    print("✓ All assertions passed — checkpoint is loadable and producing valid outputs.")


if __name__ == "__main__":
    main()
