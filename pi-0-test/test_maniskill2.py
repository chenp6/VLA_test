#!/usr/bin/env python3
"""
test_maniskill2.py — Evaluation for INTACT Pi-0 BridgeV2 on ManiSkill2.

Keeps the same high-level assumptions as the robosuite script:
- State: [eef_pos(3), euler_xyz(3), gripper_norm(1)]
- Action: 7D delta [dx,dy,dz, drx,dry,drz, g] with scaling pos*0.05, rot*0.20
- Policy @ 5Hz, Sim @ 20Hz (4 env steps per policy action)
"""

import os
import sys

# ---- IMPORTANT: configure render backend BEFORE importing sapien / mani_skill ----
# We parse only --render_mode from argv so "human" can keep DISPLAY-based viewer.
def _get_cli_render_mode(default="rgb_array"):
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg.startswith("--render_mode="):
            return arg.split("=", 1)[1].strip()
        if arg.startswith("--render-mode="):
            return arg.split("=", 1)[1].strip()
        if arg in ("--render_mode", "--render-mode") and i + 1 < len(argv):
            return argv[i + 1].strip()
    return default


_cli_render_mode = _get_cli_render_mode(default="rgb_array")
_has_display = bool(os.environ.get("DISPLAY"))
if _cli_render_mode == "human" and not _has_display:
    print("[WARN] --render_mode human requested but DISPLAY is not set. Fallback to rgb_array.")
    _cli_render_mode = "rgb_array"

os.environ["SAPIEN_DISABLE_VULKAN"] = "1"
if _cli_render_mode == "human":
    # Keep DISPLAY for interactive viewer.
    os.environ.pop("SAPIEN_RENDER_SYSTEM", None)
    os.environ.pop("EGL_PLATFORM", None)
else:
    # Headless rendering path (rgb_array).
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
import mani_skill.envs  # registers envs


MANISKILL_ENVS = [
    "PickCube-v1",
    # 你 registry 目前只看到 PickCube-v0；之後註冊更多再加進來
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="PickCube-v1", choices=MANISKILL_ENVS)
    p.add_argument("--robot", type=str, default="panda")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--obs_mode", type=str, default="rgbd")              # rgbd 常見
    p.add_argument("--control_mode", type=str, default="pd_ee_delta_pose")  # 最接近 delta-EEF
    p.add_argument("--cam", type=str, default="base_camera")            # 常見 camera name
    p.add_argument("--render_mode", type=str, default="rgb_array", choices=["human", "rgb_array"])  
    p.add_argument(
        "--shader",
        type=str,
        default="minimal",
        choices=["minimal", "default", "rt", "rt-med", "rt-fast"],
    )


    return p.parse_args()


def _first_dict_value(d):
    return next(iter(d.values()))


def _find_rgb_anywhere(obj):
    """Best-effort recursive RGB lookup in nested observation dicts."""
    if isinstance(obj, dict):
        for key in ("rgb", "color", "Color"):
            if key in obj:
                return obj[key]
        for value in obj.values():
            rgb = _find_rgb_anywhere(value)
            if rgb is not None:
                return rgb
    return None


def _to_numpy(x):
    """Convert torch / array-like data to numpy safely."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_observation(obs, device, H, W, cam_name="base_camera"):
    # ---- RGB ----
    rgb = None
    img_block = None

    if isinstance(obs, dict):
        if isinstance(obs.get("image"), dict):
            img_block = obs["image"]
        elif isinstance(obs.get("sensor_data"), dict):
            img_block = obs["sensor_data"]

    if isinstance(img_block, dict) and len(img_block) > 0:
        cam_block = img_block.get(cam_name)
        if cam_block is None:
            cam_block = next(iter(img_block.values()))

        if isinstance(cam_block, dict):
            rgb = cam_block.get("rgb", None)
            if rgb is None:
                rgb = cam_block.get("color", None)
            if rgb is None:
                rgb = cam_block.get("Color", None)
        else:
            rgb = cam_block

        if rgb is None:
            rgb = _find_rgb_anywhere(img_block)

    if rgb is None:
        image_keys = list(obs.get("image", {}).keys()) if isinstance(obs.get("image"), dict) else None
        sensor_keys = list(obs.get("sensor_data", {}).keys()) if isinstance(obs.get("sensor_data"), dict) else None
        obs_keys = list(obs.keys()) if isinstance(obs, dict) else type(obs).__name__
        raise KeyError(
            f"Cannot find RGB. obs_keys={obs_keys}, image_keys={image_keys}, sensor_data_keys={sensor_keys}"
        )

    rgb = _to_numpy(rgb)
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
        x = _to_numpy(x).reshape(-1)
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
                eef_pos = _to_numpy(extra["tcp_pos"]).reshape(-1)[:3]
                eef_quat = _to_numpy(extra["tcp_quat"]).reshape(-1)[:4]
            elif "ee_pos" in extra and "ee_quat" in extra:
                eef_pos = _to_numpy(extra["ee_pos"]).reshape(-1)[:3]
                eef_quat = _to_numpy(extra["ee_quat"]).reshape(-1)[:4]

        # gripper (best-effort)
        for k in ["gripper_open", "gripper", "gripper_qpos", "gripper_state"]:
            if k in extra:
                g = _to_numpy(extra[k]).reshape(-1)
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
                    g = _to_numpy(agent[k]).reshape(-1)
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
    a[0:3] *= 0.05
    a[3:6] *= 0.20
    a[6] = 2.0 * float(a[6]) - 1.0  # [-1,1] closed=+1
    return a.astype(np.float32)


def main():
    args = parse_args()
    if args.render_mode == "human" and not os.environ.get("DISPLAY"):
        print("[WARN] render_mode=human but DISPLAY is not available. Using rgb_array instead.")
        args.render_mode = "rgb_array"

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if args.device != "auto":
        device = args.device

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    MODEL_ID = "juexzz/INTACT-pi0-finetune-rephrase-bridge"

    print("=" * 60)
    print("INTACT Pi-0 — ManiSkill2 Evaluation")
    print("=" * 60)
    print(f"  Env: {args.env} | Robot: {args.robot} | Episodes: {args.episodes}")
    print(
        f"  Obs: {args.obs_mode} | Control: {args.control_mode} | "
        f"Cam: {args.cam} | Render: {args.render_mode} | Shader: {args.shader}\n"
    )

    # 1) Load model
    policy, image_key, (H, W) = load_intact_pi0(MODEL_ID, device=device, torch_dtype=torch_dtype)
    tok_max_len = policy.config.tokenizer_max_length
    print(f"✓ Model loaded (Img: {H}x{W}, Tok: {tok_max_len})")
    print(f"  image_key: {image_key}\n")

    # 2) Make env (render_mode rgb_array to avoid window)
    env = gym.make(
        args.env,
        robot_uids=args.robot,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sensor_configs={"shader_pack": args.shader},
        human_render_camera_configs={"shader_pack": args.shader},
        viewer_camera_configs={"shader_pack": args.shader},
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
                image_key:                             img_torch,
                "observation.state":                    state_torch,
                "observation.language.tokens":          lang_tokens,
                "observation.language.attention_mask":  lang_masks,
            }

            with torch.no_grad():
                action_chunk = policy.select_action(batch)

            # IMPORTANT: if output is (B, chunk, act_dim), take [0,0]
            if isinstance(action_chunk, torch.Tensor) and action_chunk.ndim == 3:
                action_raw = action_chunk[0, 0].detach().cpu().float().numpy()
            else:
                action_raw = action_chunk[0].detach().cpu().float().numpy()

            env_action = map_action_to_env(action_raw)

            for _ in range(steps_per_action):
                obs, reward, terminated, truncated, info = env.step(env_action)
                reward_val = float(_to_numpy(reward).reshape(-1)[0])
                terminated_flag = bool(_to_numpy(terminated).reshape(-1)[0])
                truncated_flag = bool(_to_numpy(truncated).reshape(-1)[0])
                if args.render_mode == "human":
                    env.render()
                done = bool(terminated_flag or truncated_flag)
                max_reward = max(max_reward, reward_val)
                step += 1
                if step % 1 == 0:
                    print(f"    Step {step:3d} | max_reward: {max_reward:.4f}")
                if terminated_flag or truncated_flag:
                    print(f"[END] terminated={terminated_flag}, truncated={truncated_flag}, reward={reward_val:.4f}")
                    print(f"[END] info keys: {list(info.keys())}")
                    print(f"[END] info: {info}")

                if done or step >= 400:
                    break


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
