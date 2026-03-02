# VLA Test

Testing suite for Vision-Language-Action (VLA) model checkpoints.

---

## `pi-0-test/` — INTACT Pi-0 BridgeV2

Tests the finetuned **INTACT π₀ BridgeV2** checkpoint:
[`juexzz/INTACT-pi0-finetune-rephrase-bridge`](https://huggingface.co/juexzz/INTACT-pi0-finetune-rephrase-bridge)

- Base model: [lerobot/pi0](https://huggingface.co/lerobot/pi0) (PaliGemma vision + Gemma action expert)
- Fine-tuned on [BridgeV2](https://rail-berkeley.github.io/bridgedata/) with task-rephrasing augmentation
- Output: delta EEF actions, chunk size 4, 7-DoF
- Paper: [From Intention to Execution (arXiv 2506.09930)](https://arxiv.org/abs/2506.09930)

### Setup

```bash
cd pi-0-test
bash setup_env.sh        # creates .venv, installs all deps, patches lerobot
source .venv/bin/activate
```

> **Note:** `setup_env.sh` automatically installs the lerobot-patched `transformers` fork
> (`fix/lerobot_openpi` branch) required for pi0 — standard PyPI `transformers` is incompatible.

### Tests

| Script | What it does |
|--------|-------------|
| `test_sanity.py` | Dummy-input forward pass — verifies model loads and produces valid actions |
| `test_offline.py` | Runs policy over a saved `.pt` trajectory file |
| `test_simplerenv.py` | Full SimplerEnv rollouts (requires **x86_64** or Docker) |
| `test_robosuite.py` | **Native ARM64** evaluation using MuJoCo and robosuite |

### Simulation (ARM64 Native)

The recommended simulation environment for **ARM64 (aarch64)** is **robosuite** + **MuJoCo**. This provides high-performance, GPU-accelerated evaluation without emulation overhead.

```bash
# 1. Sanity check (no simulator)
python test_sanity.py

# 2. Robosuite evaluation (PickPlaceMilk, Lift, Stack, etc.)
python test_robosuite.py --env Lift --episodes 10
```

### Simulation (x86_64 / SimplerEnv)

`SimplerEnv` relies on `SAPIEN 2.2`, which does not have ARM64 wheels. On ARM64 hosts, this requires x86_64 Docker emulation (not recommended for performance).

```bash
# SimplerEnv evaluation
pip install git+https://github.com/simpler-env/SimplerEnv.git
python test_simplerenv.py --task carrot_on_plate --episodes 20
```

### Hardware

Tested on **NVIDIA GB10** (128.5 GB VRAM, CUDA 13.0).
Inference runs at ~700 ms/step on a single GPU with `float16`.
