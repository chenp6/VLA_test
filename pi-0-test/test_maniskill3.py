#!/usr/bin/env python3
"""
test_maniskill3.py — Evaluation for INTACT Pi-0 BridgeV2 on ManiSkill2.

Keeps the same high-level assumptions as the robosuite script:
- State: [eef_pos(3), euler_xyz(3), gripper_norm(1)]
- Action: 7D delta [dx,dy,dz, drx,dry,drz, g] with scaling pos*0.05, rot*0.20
- Policy @ 5Hz, Sim @ 20Hz (4 env steps per policy action)
"""

# ---- IMPORTANT: force EGL + disable Vulkan BEFORE importing sapien / mani_skill2 ----
import os
os.environ["SAPIEN_DISABLE_VULKAN"] = "1"
os.environ["SAPIEN_RENDER_SYSTEM"] = "egl"
os.environ["EGL_PLATFORM"] = "surfaceless"
os.environ.pop("DISPLAY", None)

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from policy_loader import load_intact_pi0, tokenize

import gymnasium as gym
import mani_skill
import mani_skill.envs
import mani_skill.agents

MANISKILL_ENVS = [
    "PickCubeWidowXAI-v1",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="PickCubeWidowXAI-v1", choices=MANISKILL_ENVS)
    p.add_argument("--robot_uids", type=str, default="widowx250s", choices=["widowx250s","widowx250s_bridgedataset_flat_table","widowx250s_bridgedataset_sink"])
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--obs_mode", type=str, default="rgbd")              # rgbd 常見
    p.add_argument("--control_mode", type=str, default="pd_ee_delta_pose")  # 最接近 delta-EEF
    p.add_argument("--cam", type=str, default="base_camera")            # 常見 camera name
    return p.parse_args()


def _first_dict_value(d):
    return next(iter(d.values()))

	
def get_observation(obs, device, H, W, cam_name="base_camera"):
    # ---- RGB ----
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
        raise KeyError(f"Cannot find rgb. image keys={list(obs.get('image', {}).keys())}")

    rgb = np.asarray(rgb)
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.shape[-1] != 3:
        raise ValueError(f"RGB must be (H,W,3). Got {rgb.shape}")

    img_torch = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    img_torch = img_torch.to(device=device, dtype=torch.float32) / 255.0
    if img_torch.shape[-2:] != (H, W):
        img_torch = F.interpolate(img_torch, size=(H, W), mode="bilinear", align_corners=False)

    # ---- State (EEF/TCP pose + gripper) ----
    def parse_pose7(x):
        x = np.asarray(x).reshape(-1)
        if x.shape[0] >= 7:
            pos = x[:3]
            quat = x[3:7]
            return pos, quat
        return None, None

    eef_pos = None
    eef_quat = None
    gripper_norm = None

    # 1) Prefer obs["extra"]
    extra = obs.get("extra", None)
    if isinstance(extra, dict):
        # pose as 7D
        for k in ["tcp_pose", "ee_pose", "eef_pose"]:
            if k in extra:
                eef_pos, eef_quat = parse_pose7(extra[k])
                if eef_pos is not None:
                    break

        # pose split
        if eef_pos is None:
            if "tcp_pos" in extra and "tcp_quat" in extra:
                eef_pos = np.asarray(extra["tcp_pos"]).reshape(-1)[:3]
                eef_quat = np.asarray(extra["tcp_quat"]).reshape(-1)[:4]
            elif "ee_pos" in extra and "ee_quat" in extra:
                eef_pos = np.asarray(extra["ee_pos"]).reshape(-1)[:3]
                eef_quat = np.asarray(extra["ee_quat"]).reshape(-1)[:4]

        # gripper (best-effort)
        for k in ["gripper_open", "gripper", "gripper_qpos", "gripper_state"]:
            if k in extra:
                g = np.asarray(extra[k]).reshape(-1)
                gv = float(np.mean(g))
                gripper_norm = gv if 0.0 <= gv <= 1.0 else float(np.clip(1.0 - (gv / 0.04), 0.0, 1.0))
                break

    # 2) Fallback obs["agent"]
    if (eef_pos is None or eef_quat is None):
        agent = obs.get("agent", None)
        if isinstance(agent, dict):
            for k in ["tcp_pose", "ee_pose", "eef_pose"]:
                if k in agent:
                    eef_pos, eef_quat = parse_pose7(agent[k])
                    if eef_pos is not None:
                        break

            for k in ["gripper_open", "gripper", "gripper_qpos", "gripper_state"]:
                if k in agent and gripper_norm is None:
                    g = np.asarray(agent[k]).reshape(-1)
                    gv = float(np.mean(g))
                    gripper_norm = gv if 0.0 <= gv <= 1.0 else float(np.clip(1.0 - (gv / 0.04), 0.0, 1.0))
                    break

    if eef_pos is None or eef_quat is None:
        agent_keys = list(obs["agent"].keys()) if isinstance(obs.get("agent"), dict) else type(obs.get("agent")).__name__
        extra_keys = list(obs["extra"].keys()) if isinstance(obs.get("extra"), dict) else type(obs.get("extra")).__name__
        raise KeyError(f"Cannot find EEF/TCP pose. agent_keys={agent_keys}, extra_keys={extra_keys}")

    euler = R.from_quat(eef_quat).as_euler("xyz")
    if gripper_norm is None:
        gripper_norm = 0.0

    state_np = np.concatenate([eef_pos, euler, [gripper_norm]]).astype(np.float32)
    state_torch = torch.from_numpy(state_np).unsqueeze(0).to(device=device, dtype=torch.float32)

    return img_torch, state_torch

def map_action_to_env(action_raw):
    """
    Pi0 -> ManiSkill2 controller action
    action_raw: (7,) [dx,dy,dz, drx,dry,drz, g] with g in [0,1]
    return: (7,) by default for pd_ee_delta_pose style controllers
    """
    a = np.asarray(action_raw, dtype=np.float32).copy()
    a[0:7] *= 0.25
    #a[3:6] *= 0.20
    #a[6] = 2.0 * float(a[6]) - 1.0  # [-1,1] closed=+1
    return a.astype(np.float32)


def main():
    args = parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if args.device != "auto":
        device = args.device

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    MODEL_ID = "juexzz/INTACT-pi0-finetune-rephrase-bridge"

    print("=" * 60)
    print("INTACT Pi-0 — ManiSkill2 Evaluation")
    print("=" * 60)
    print(f"  Env: {args.env}  |  Episodes: {args.episodes}")
    print(f"  Obs: {args.obs_mode} | Control: {args.control_mode} | Cam: {args.cam}\n")
    print("controller:", args.control_mode)
    
	  # 1) Load model
    policy, image_key, (H, W) = load_intact_pi0(MODEL_ID, device=device, torch_dtype=torch_dtype)
    tok_max_len = policy.config.tokenizer_max_length
    print(f"✓ Model loaded (Img: {H}x{W}, Tok: {tok_max_len})")
    print(f"  image_key: {image_key}\n")

    # 2) Make env (render_mode rgb_array to avoid window)
    env = gym.make(
        args.env,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode="rgb_array",
        #robot_uids=args.robot_uids
    )
    print("available control modes:", getattr(env, "SUPPORTED_CONTROL_MODES", None))

		
    # Instruction (simple default)
    instruction = "pick up the cube"
    lang_tokens, lang_masks = tokenize([instruction], device, max_length=tok_max_len)
    print(f'  Instruction: "{instruction}"\n')

    # 3) Evaluation loop
    successes = 0
    policy_freq = 5.0
    sim_freq = 20.0
    steps_per_action = int(sim_freq / policy_freq)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        step = 0
        max_reward = -1e9

        print(f"  Episode {ep+1:3d}/{args.episodes} start...")

        # Save first frame (if image exists)
        img_torch, state_torch = get_observation(obs, device, H, W, cam_name=args.cam)
        try:
            from torchvision.utils import save_image
            save_image(img_torch[0], f"debug_ep{ep+1}_step0.png")
        except Exception:
            pass

        while not done and step < 400:
            img_torch, state_torch = get_observation(obs, device, H, W, cam_name=args.cam)

            batch = {
                image_key:img_torch,
                "observation.state":state_torch,
                "observation.language.tokens":lang_tokens,
                "observation.language.attention_mask": lang_masks,
            }

            with torch.no_grad():
                action_chunk = policy.select_action(batch)
            if isinstance(action_chunk, torch.Tensor) and action_chunk.ndim == 3:
                action_raw = action_chunk[0, 0].detach().cpu().float().numpy()
            else:
                action_raw = action_chunk[0].detach().cpu().float().numpy()

			# 如果模型輸出是 32 維，只取 WidowX 真正需要的前 7 維
            if action_raw.shape[-1] >= 7:
                action_raw = action_raw[:7]
            else:
                raise ValueError(f"Unexpected action dim: {action_raw.shape}")
            env_action = map_action_to_env(action_raw)

            for _ in range(steps_per_action):
                obs, reward, terminated, truncated, info = env.step(env_action)
                done = bool(terminated or truncated)
                max_reward = max(max_reward, float(reward))
                step += 1
                if done or step >= 400:
                    break

            if step % 80 == 0:
                print(f"    Step {step:3d} | max_reward: {max_reward:.4f}")

        is_success = bool(info.get("success", False) or info.get("is_success", False) or (max_reward > 0.9))
        successes += int(is_success)
        print(f"  Episode {ep+1:3d} finished | {'✓ SUCCESS' if is_success else '✗ FAILED'} | max_reward={max_reward:.4f}\n")

    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Success Rate: {successes / args.episodes * 100:.1f}% ({successes}/{args.episodes})")
    env.close()


if __name__ == "__main__":
    main()
