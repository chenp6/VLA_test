#!/usr/bin/env python3
"""
test_simplerenv.py — Full SimplerEnv evaluation for INTACT Pi-0 BridgeV2 checkpoint.

Prerequisites:
    pip install git+https://github.com/simpler-env/SimplerEnv.git

Usage:
    python test_simplerenv.py [--task carrot_on_plate] [--episodes 20]
"""

import argparse
import sys
import time

import torch

# BridgeV2 task names as defined in SimplerEnv (widowx robot)
BRIDGE_TASKS = [
    "widowx_carrot_on_plate",
    "widowx_put_eggplant_in_basket",
    "widowx_stack_cube",
    "widowx_spoon_on_towel",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="widowx_carrot_on_plate",
                        choices=BRIDGE_TASKS + ["all"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--env-id", type=str, default="Bridge-v0")
    return parser.parse_args()


def tokenize(text, tokenizer, device, max_length=72):
    text = text if text.endswith("\n") else text + "\n"
    enc = tokenizer([text], return_tensors="pt", padding="max_length",
                    max_length=max_length, truncation=True)
    return enc["input_ids"].to(device), enc["attention_mask"].bool().to(device)


def evaluate_task(policy, env, tokenizer, episodes, device, task_name, tok_max_len=72):
    successes = 0
    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        step = 0

        instruction = info.get("instruction", task_name.replace("_", " "))
        lang_tokens, lang_masks = tokenize(instruction, tokenizer, device, max_length=tok_max_len)

        while not done:
            rgb = torch.from_numpy(obs["image"]["overhead_camera"]["rgb"]).permute(2, 0, 1).unsqueeze(0)
            rgb = rgb.to(device=device, dtype=torch.float32) / 255.0
            # robot state: flatten whatever is available
            robot_state_np = obs.get("agent", {}).get("qpos", None)
            if robot_state_np is None:
                # fallback for older obs format
                robot_state_np = obs.get("robot_state", None)
            if robot_state_np is None:
                robot_state_np = obs.get("extra", {}).get("tcp_pose", None)
            robot_state = torch.from_numpy(robot_state_np.flatten()).unsqueeze(0).to(device=device, dtype=torch.float32)

            batch = {
                image_key:                             rgb,
                "observation.state":                   robot_state,
                "observation.language.tokens":         lang_tokens,
                "observation.language.attention_mask": lang_masks,
            }
            with torch.no_grad():
                action = policy.select_action(batch)

            action_np = action[0].cpu().float().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            step += 1

        success = info.get("success", False)
        successes += int(success)
        print(f"    Episode {ep+1:3d}/{episodes} {'✓' if success else '✗'}  (steps={step})")

    return successes / episodes


def main():
    args = parse_args()
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if args.device not in ("auto",):
        device = args.device
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("INTACT Pi-0 — SimplerEnv Evaluation")
    print("=" * 60)
    print(f"  Task: {args.task}  |  Episodes: {args.episodes}  |  Device: {device}\n")

    try:
        import simpler_env
    except ImportError:
        print("ERROR: SimplerEnv not installed.")
        print("  pip install git+https://github.com/simpler-env/SimplerEnv.git")
        sys.exit(1)

    MODEL_ID = "juexzz/INTACT-pi0-finetune-rephrase-bridge"

    from policy_loader import load_intact_pi0
    from transformers import AutoTokenizer

    # Load tokenizer from base model (INTACT checkpoint has no tokenizer files)
    TOKENIZER_ID = "lerobot/pi0"
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    print(f"Loading model: {MODEL_ID} …")
    policy, image_key, (H, W) = load_intact_pi0(MODEL_ID, torch_dtype=torch_dtype, device=device)
    tok_max_len = policy.config.tokenizer_max_length
    print("✓ Model loaded\n")

    tasks = BRIDGE_TASKS if args.task == "all" else [args.task]
    results = {}

    for task_name in tasks:
        print(f"─── Task: {task_name} ───")
        try:
            env = simpler_env.make(args.env_id, task_name=task_name)
        except Exception as e:
            print(f"  ERROR: {e}\n  Skipping.")
            continue

        t0 = time.time()
        rate = evaluate_task(policy, env, tokenizer, args.episodes, device, task_name, tok_max_len)
        env.close()
        results[task_name] = rate
        print(f"  → Success rate: {rate*100:.1f}%  ({time.time()-t0:.0f}s)\n")

    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for t, r in results.items():
        print(f"  {t:<30s}  {r*100:.1f}%")
    if results:
        print(f"  {'Average':<30s}  {sum(results.values())/len(results)*100:.1f}%")
    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()
