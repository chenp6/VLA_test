# INTACT Pi-0 BridgeV2 Finetuned Checkpoint — Test Suite

This folder contains scripts for testing the **INTACT π₀ BridgeV2 finetuned checkpoint**:

> **Model**: [`juexzz/INTACT-pi0-finetune-rephrase-bridge`](https://huggingface.co/juexzz/INTACT-pi0-finetune-rephrase-bridge)  
> **Best checkpoint**: Step 7565 (epoch 5), selected by default in the HF repo.

---

## Files

| File | Purpose |
|------|---------|
| `setup_env.sh` | Create and configure the `pi0_intact` conda environment |
| `test_sanity.py` | Minimal inference test with dummy inputs (no simulator needed) |
| `test_offline.py` | Offline action prediction on saved BridgeV2 trajectory data (`.pt` file) |
| `test_simplerenv.py` | Full SimplerEnv evaluation (optional, closest to INTACT paper) |

---

## Quick Start

### Step 1 — Set up the environment

```bash
bash setup_env.sh
conda activate pi0_intact
```

### Step 2 — Check GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Step 3 — Run sanity test (no simulator)

```bash
python test_sanity.py
```

Expected output: action tensor shape printed, e.g. `torch.Size([2, 4, 7])`.

### Step 4 (optional) — Offline trajectory test

Place a BridgeV2 `.pt` trajectory file at `bridge_sample.pt`, then:

```bash
python test_offline.py --data bridge_sample.pt
```

### Step 5 (optional) — SimplerEnv evaluation

Install SimplerEnv first:

```bash
pip install git+https://github.com/simpler-env/SimplerEnv.git
```

Then run:

```bash
python test_simplerenv.py --task carrot_on_plate --episodes 20
```

---

## Model Details

- **Input**: Single RGB image (256×256), 1 language instruction, 1 robot state (7-dim EEF)
- **Output**: Delta EEF actions, chunk size 4
- **Training**: 15 epochs (~22,695 steps), batch size 1024, lr 1e-5, 4×H100/A100
- **Paper**: [From Intention to Execution: Probing the Generalization Boundaries of VLA Models](https://arxiv.org/abs/2506.09930)
