#!/usr/bin/env python3
"""
test_robosuite.py — Native ARM64 evaluation for INTACT Pi-0 BridgeV2.

Uses robosuite as the simulation backend (MuJoCo-based), providing a 
high-performance alternative to SimplerEnv on aarch64.

Research-backed mappings for BridgeV2:
- State: [pos_3, euler_xyz_3, gripper_normalized_1] (World Frame)
- Action: 5Hz deltas scaled by 0.05 (pos) and 0.2 (rot)
- Frequency: 5Hz Policy / 20Hz Sim (4 steps per action)
"""

import argparse
import sys
import time
import numpy as np
import torch
import robosuite as suite
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from policy_loader import load_intact_pi0, tokenize

ROBOSUITE_ENVS = [
    "Lift",
    "Stack",
    "PickPlaceMilk",
    "PickPlaceBread",
    "PickPlaceCereal",
    "PickPlaceCan",
    "Door",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PickPlaceMilk", choices=ROBOSUITE_ENVS)
    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()

def get_observation(obs, device, image_key, H, W, cam_name='agentview'):
    """Convert robosuite observation to model input batch (BridgeV2 schema)."""
    # 1. Image processing: [0, 255] uint8 (H, W, 3) -> [0, 1] float32 (1, 3, H, W)
    img = obs[f'{cam_name}_image'] 
    img = np.flipud(img) # Correct for MuJoCo offscreen flip
    
    img_torch = torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0)
    img_torch = img_torch.to(device=device, dtype=torch.float32) / 255.0
    
    if img_torch.shape[-2:] != (H, W):
        img_torch = F.interpolate(img_torch, size=(H, W), mode="bilinear", align_corners=False)

    # 2. State mapping: [x, y, z, roll, pitch, yaw, gripper] in World Frame
    pos = obs['robot0_eef_pos']
    quat = obs['robot0_eef_quat'] # [x, y, z, w]
    # BridgeV2 uses Euler XYZ
    euler = R.from_quat(quat).as_euler('xyz')
    
    # Gripper: robosuite Panda [0.04, 0.04] -> [open, open]
    # Map to [0, 1] where 1 is fully closed.
    gripper_qpos = np.mean(obs['robot0_gripper_qpos'])
    gripper_normalized = 1.0 - (gripper_qpos / 0.04) 
    gripper_normalized = np.clip(gripper_normalized, 0.0, 1.0)
    
    state_np = np.concatenate([pos, euler, [gripper_normalized]]).astype(np.float32)
    state_torch = torch.from_numpy(state_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    
    return img_torch, state_torch

def main():
    args = parse_args()
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if args.device not in ("auto",):
        device = args.device
    
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    MODEL_ID = "juexzz/INTACT-pi0-finetune-rephrase-bridge"

    print("=" * 60)
    print("INTACT Pi-0 — Robosuite Evaluation (ARM64 Native)")
    print("=" * 60)
    print(f"  Env: {args.env}  |  Robot: {args.robot}  |  Episodes: {args.episodes}\n")

    # 1. Load Model
    policy, image_key, (H, W) = load_intact_pi0(MODEL_ID, device=device, torch_dtype=torch_dtype)
    tok_max_len = policy.config.tokenizer_max_length
    print(f"✓ Model loaded (Img: {H}x{W}, Tok: {tok_max_len})\n")

    # 2. Setup Environment
    print(f"Initializing robosuite environment: {args.env} ...")
    # In RC 1.5, we just specify the defaults and it works
    env = suite.make(
        env_name=args.env,
        robots=args.robot,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names='agentview',
        camera_heights=H,
        camera_widths=W,
        control_freq=20,    # Simulation runs at 20Hz
        horizon=400,
    )
    
    # Custom camera adjustment: zoom in to match BridgeV2 "over-the-shoulder" scale.
    cam_id = env.sim.model.camera_name2id('agentview')
    env.sim.model.cam_pos[cam_id] = [0.55, 0.0, 1.45]
    env.sim.model.cam_quat[cam_id] = [0.45, 0.45, 0.55, 0.55]

    # Task instructions
    task_instructions = {
        "Lift": "lift the cube",
        "Stack": "stack the red cube on the green cube",
        "PickPlaceMilk": "pick up the milk carton and place it in the bin",
        "PickPlaceBread": "pick up the loaf of bread and place it in the bin",
        "PickPlaceCereal": "pick up the cereal box and place it in the bin",
        "PickPlaceCan": "pick up the can and place it in the bin",
        "Door": "open the door",
    }
    instruction = task_instructions.get(args.env, "pick up the object and place it in the target")
    lang_tokens, lang_masks = tokenize([instruction], device, max_length=tok_max_len)
    print(f"  Instruction: \"{instruction}\"\n")

    # 3. Evaluation Loop
    successes = 0
    policy_freq = 5.0    # Hz
    sim_freq = 20.0      # Hz
    steps_per_action = int(sim_freq / policy_freq) # 4

    for ep in range(args.episodes):
        ret = env.reset()
        obs = ret[0] if isinstance(ret, tuple) else ret
        done = False
        step = 0
        total_reward = 0.0
        
        print(f"  Episode {ep+1:3d}/{args.episodes} start...")
        
        # Save first frame
        img_torch, state_torch = get_observation(obs, device, image_key, H, W)
        from torchvision.utils import save_image
        save_image(img_torch[0], f"debug_ep{ep+1}_step0.png")

        while not done and step < 400:
            # Policy inference @ 5Hz
            img_torch, state_torch = get_observation(obs, device, image_key, H, W)
            
            batch = {
                image_key:                             img_torch,
                "observation.state":                    state_torch,
                "observation.language.tokens":          lang_tokens,
                "observation.language.attention_mask":  lang_masks,
            }
            
            with torch.no_grad():
                # Policy outputs chunk of actions; we take the first one.
                action_chunk = policy.select_action(batch) 
            
            # Action: 7D relative delta [dx, dy, dz, drx, dry, drz, g]
            # BridgeV2 deltas are small. Scaling to robosuite units:
            action_raw = action_chunk[0].cpu().float().numpy()
            
            # Apply scaling
            scaled_action = action_raw.copy()
            scaled_action[0:3] *= 0.05  # Position scale (m)
            scaled_action[3:6] *= 0.20  # Rotation scale (rad)
            # Gripper: model [0, 1] (0=open, 1=closed) -> robosuite [-1, 1] (1=closed)
            scaled_action[6] = 2.0 * action_raw[6] - 1.0 
            
            # Repetitive stepping for 5Hz consistency
            for _ in range(steps_per_action):
                ret = env.step(scaled_action)
                obs, reward, terminated, truncated = ret[:4]
                info = ret[4] if len(ret) > 4 else {}
                total_reward = max(total_reward, reward)
                
                step += 1
                done = terminated or truncated
                if done: break
            
            if step % 80 == 0:
                print(f"    Step {step:3d} | Reward: {total_reward:.4f}")

        # Check success
        is_success = total_reward > 0.9 or info.get('success', False)
        successes += int(is_success)
        print(f"  Episode {ep+1:3d} finished | {'✓ SUCCESS' if is_success else '✗ FAILED'} | max_reward={total_reward:.4f}\n")

    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Success Rate: {successes / args.episodes * 100:.1f}% ({successes}/{args.episodes})")
    print("\n✓ Evaluation complete.")
    env.close()

if __name__ == "__main__":
    main()
