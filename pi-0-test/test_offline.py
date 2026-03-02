#!/usr/bin/env python3
"""
test_offline.py — Offline action prediction on saved BridgeV2 trajectory data.

Expected .pt file format (dict):
    "images"      : Tensor(T, 3, H, W),  float32 in [0, 1]
    "states"      : Tensor(T, S),         float32
    "instruction" : str                   (single episode instruction)
    "actions"     : Tensor(T, A)          (optional — ground truth)

Usage:
    python test_offline.py --data bridge_sample.pt [--max-steps 50]
"""

import argparse
import sys
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      type=str, required=True)
    parser.add_argument("--device",    type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype",     type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def load_trajectory(path):
    import os
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".pt":
        return torch.load(path, map_location="cpu")
    if ext == ".npz":
        import numpy as np
        raw = np.load(path, allow_pickle=True)
        return {k: torch.from_numpy(raw[k]) if hasattr(raw[k], "__array__") else raw[k]
                for k in raw.files}
    raise ValueError(f"Unsupported format: {ext}")


def main():
    args  = parse_args()
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if args.device not in ("auto",):
        device = args.device
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    MODEL_ID = "juexzz/INTACT-pi0-finetune-rephrase-bridge"

    print("=" * 60)
    print("INTACT Pi-0 — Offline Trajectory Test")
    print("=" * 60)

    print(f"Loading trajectory: {args.data}")
    try:
        data = load_trajectory(args.data)
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.data}")
        sys.exit(1)

    images     = data["images"]
    states     = data["states"]
    instruction = data["instruction"]
    if torch.is_tensor(instruction):
        instruction = str(instruction)
    gt_actions = data.get("actions")

    T         = len(images)
    max_steps = min(args.max_steps, T) if args.max_steps else T
    print(f"  Length: {T}  (running {max_steps}), state-dim={states.shape[1]}")
    print(f"  Instruction: \"{instruction}\"\n")

    from policy_loader import load_intact_pi0, tokenize

    print(f"Loading model: {MODEL_ID} …")
    policy, image_key, (H, W) = load_intact_pi0(MODEL_ID, device=device, torch_dtype=torch_dtype)
    tok_max_len = policy.config.tokenizer_max_length
    print(f"  Image key: {image_key}  resolution: {H}x{W}\n")

    # Tokenise once for the whole episode
    lang_tokens, lang_masks = tokenize([instruction], device, max_length=tok_max_len)

    predicted_actions = []
    total_ms = 0.0

    for t in range(max_steps):
        img_t   = images[t:t+1].to(device=device, dtype=torch.float32)
        state_t = states[t:t+1].to(device=device, dtype=torch.float32)

        # Resize image to model's expected resolution if needed
        if img_t.shape[-2:] != (H, W):
            import torch.nn.functional as F
            img_t = F.interpolate(img_t, size=(H, W), mode="bilinear", align_corners=False)

        batch = {
            image_key:                             img_t,
            "observation.state":                   state_t,
            "observation.language.tokens":         lang_tokens,
            "observation.language.attention_mask": lang_masks,
        }

        t0 = time.time()
        with torch.no_grad():
            action = policy.select_action(batch)
        total_ms += (time.time() - t0) * 1000
        predicted_actions.append(action.cpu())

        if t % 10 == 0 or t == max_steps - 1:
            vals   = [f"{v:.3f}" for v in action[0].cpu().float().tolist()[:7]]
            gt_str = ""
            if gt_actions is not None:
                gv = [f"{v:.3f}" for v in gt_actions[t].float().tolist()[:7]]
                gt_str = f"  GT: [{', '.join(gv)}]"
            print(f"  Step {t:4d}/{max_steps-1} | pred: [{', '.join(vals)}]{gt_str}")

    print()
    print(f"  Steps run: {max_steps},  avg inference: {total_ms/max_steps:.1f} ms/step")
    all_preds = torch.cat(predicted_actions, dim=0)
    print(f"  Predicted actions shape: {tuple(all_preds.shape)}")
    if gt_actions is not None:
        gt_sl   = gt_actions[:max_steps].float()
        pred_sl = all_preds.float()
        if pred_sl.shape == gt_sl.shape:
            print(f"  MAE vs GT: {(pred_sl - gt_sl).abs().mean().item():.4f}")
    print("\n✓ Offline test complete.")


if __name__ == "__main__":
    main()
