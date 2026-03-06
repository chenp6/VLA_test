#!/usr/bin/env python3
"""
Evaluate the original INTACT Pi-0 policy on a custom ManiSkill3 task:
PickCube with widowx250s robot.
"""

import os

os.environ["SAPIEN_DISABLE_VULKAN"] = "1"
os.environ["SAPIEN_RENDER_SYSTEM"] = "egl"
os.environ["EGL_PLATFORM"] = "surfaceless"
os.environ.pop("DISPLAY", None)

import argparse

import gymnasium as gym
import mani_skill
import mani_skill.envs
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from policy_loader import load_intact_pi0, tokenize

# Register custom env id: PickCubeWidowX250S-v1
import pickcube_widowx250s_env  # noqa: F401

MANISKILL3_ENVS = ["PickCubeWidowX250S-v1"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="PickCubeWidowX250S-v1", choices=MANISKILL3_ENVS)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--obs_mode", type=str, default="rgbd")
    p.add_argument("--control_mode", type=str, default="pd_ee_delta_pose")
    p.add_argument("--cam", type=str, default="base_camera")
    p.add_argument("--instruction", type=str, default="pick up the cube")
    return p.parse_args()


def get_observation(obs, device, image_h, image_w, cam_name="base_camera"):
    rgb = None
    if isinstance(obs, dict) and "image" in obs and isinstance(obs["image"], dict):
        img_block = obs["image"]
        cam_block = img_block.get(cam_name)
        if cam_block is None:
            cam_block = next(iter(img_block.values()))

        if isinstance(cam_block, dict):
            rgb = cam_block.get("rgb", None)
            if rgb is None:
                rgb = cam_block.get("color", None)
        else:
            rgb = cam_block

    if rgb is None:
        raise KeyError(f"Cannot find RGB image, available cameras: {list(obs.get('image', {}).keys())}")

    rgb = np.asarray(rgb)
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.shape[-1] != 3:
        raise ValueError(f"RGB should be (H,W,3), got {rgb.shape}")

    img_torch = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device=device) / 255.0
    if img_torch.shape[-2:] != (image_h, image_w):
        img_torch = F.interpolate(img_torch, size=(image_h, image_w), mode="bilinear", align_corners=False)

    def parse_pose7(x):
        x = np.asarray(x).reshape(-1)
        if x.shape[0] >= 7:
            return x[:3], x[3:7]
        return None, None

    eef_pos, eef_quat = None, None
    gripper_norm = None

    extra = obs.get("extra", None)
    if isinstance(extra, dict):
        for k in ["tcp_pose", "ee_pose", "eef_pose"]:
            if k in extra:
                eef_pos, eef_quat = parse_pose7(extra[k])
                if eef_pos is not None:
                    break

        if eef_pos is None:
            if "tcp_pos" in extra and "tcp_quat" in extra:
                eef_pos = np.asarray(extra["tcp_pos"]).reshape(-1)[:3]
                eef_quat = np.asarray(extra["tcp_quat"]).reshape(-1)[:4]

        for k in ["gripper_open", "gripper", "gripper_qpos", "gripper_state"]:
            if k in extra:
                g = np.asarray(extra[k]).reshape(-1)
                gv = float(np.mean(g))
                gripper_norm = gv if 0.0 <= gv <= 1.0 else float(np.clip(1.0 - (gv / 0.04), 0.0, 1.0))
                break

    agent = obs.get("agent", None)
    if (eef_pos is None or eef_quat is None) and isinstance(agent, dict):
        for k in ["tcp_pose", "ee_pose", "eef_pose"]:
            if k in agent:
                eef_pos, eef_quat = parse_pose7(agent[k])
                if eef_pos is not None:
                    break

        if gripper_norm is None:
            for k in ["gripper_open", "gripper", "gripper_qpos", "gripper_state"]:
                if k in agent:
                    g = np.asarray(agent[k]).reshape(-1)
                    gv = float(np.mean(g))
                    gripper_norm = gv if 0.0 <= gv <= 1.0 else float(np.clip(1.0 - (gv / 0.04), 0.0, 1.0))
                    break

    if eef_pos is None or eef_quat is None:
        raise KeyError("Cannot find EEF/TCP pose in observation.")

    euler = R.from_quat(eef_quat).as_euler("xyz")
    if gripper_norm is None:
        gripper_norm = 0.0

    state_np = np.concatenate([eef_pos, euler, [gripper_norm]]).astype(np.float32)
    state_torch = torch.from_numpy(state_np).unsqueeze(0).to(device=device, dtype=torch.float32)

    return img_torch, state_torch


def map_action_to_env(action_raw):
    a = np.asarray(action_raw, dtype=np.float32).copy()
    a = a[:7]
    a[0:3] *= 0.05
    a[3:6] *= 0.20
    a[6] = 2.0 * float(a[6]) - 1.0
    return a.astype(np.float32)


def main():
    args = parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if args.device != "auto":
        device = args.device

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    model_id = "juexzz/INTACT-pi0-finetune-rephrase-bridge"

    print("=" * 60)
    print("INTACT Pi-0 — ManiSkill3 CustomTask (widowx250s PickCube)")
    print("=" * 60)

    policy, image_key, (image_h, image_w) = load_intact_pi0(model_id, device=device, torch_dtype=torch_dtype)
    tok_max_len = policy.config.tokenizer_max_length
    lang_tokens, lang_masks = tokenize([args.instruction], device, max_length=tok_max_len)

    env = gym.make(
        args.env,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode="rgb_array",
    )

    successes = 0
    policy_freq = 5.0
    sim_freq = 20.0
    steps_per_action = int(sim_freq / policy_freq)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        step = 0
        max_reward = -1e9

        while not done and step < 400:
            img_torch, state_torch = get_observation(obs, device, image_h, image_w, cam_name=args.cam)
            batch = {
                image_key: img_torch,
                "observation.state": state_torch,
                "observation.language.tokens": lang_tokens,
                "observation.language.attention_mask": lang_masks,
            }

            with torch.no_grad():
                action_chunk = policy.select_action(batch)

            if isinstance(action_chunk, torch.Tensor) and action_chunk.ndim == 3:
                action_raw = action_chunk[0, 0].detach().cpu().float().numpy()
            else:
                action_raw = action_chunk[0].detach().cpu().float().numpy()

            env_action = map_action_to_env(action_raw)
            for _ in range(steps_per_action):
                obs, reward, terminated, truncated, info = env.step(env_action)
                done = bool(terminated or truncated)
                max_reward = max(max_reward, float(reward))
                step += 1
                if done or step >= 400:
                    break

        is_success = bool(info.get("success", False) or info.get("is_success", False) or (max_reward > 0.9))
        successes += int(is_success)
        print(f"Episode {ep + 1}/{args.episodes}: {'SUCCESS' if is_success else 'FAILED'} (max_reward={max_reward:.4f})")

    print(f"Success rate: {successes / args.episodes * 100:.1f}% ({successes}/{args.episodes})")
    env.close()


if __name__ == "__main__":
    main()
