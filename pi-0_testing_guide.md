<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# can you write a guide to testing the intact finetuned checkpoint for a coding agent?

Below is a concrete, step‑by‑step guide your coding agent can follow to test the **INTACT π₀ BridgeV2 finetuned checkpoint** on a DGX Spark, with and without a simulator.[^1][^2]

***

## 1. Choose the checkpoint

Use the **rephrase variant**, which has slightly better SimplerEnv success rates and a clearly specified best checkpoint:[^3][^2]

- Model ID: **`juexzz/INTACT-pi0-finetune-rephrase-bridge`**
- Best checkpoint: **step 7565 (epoch 5)**, already selected as the default in the Hugging Face repo.[^2][^3]

Your agent only needs this model ID for loading.

***

## 2. Environment setup on DGX Spark

### 2.1. Create and activate env

Your agent can run:

```bash
conda create -n pi0_intact python=3.11 -y
conda activate pi0_intact
pip install "torch>=2.3" --index-url https://download.pytorch.org/whl/cu121
pip install lerobot transformers accelerate huggingface_hub
```

LeRobot is required because INTACT uses the **LeRobot π₀ implementation**.[^4][^1]

### 2.2. Check GPU

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The agent should assert that CUDA is available and a GPU name is printed.

***

## 3. Minimal inference test (no simulator)

This is the simplest sanity check: run the policy on dummy inputs to verify shapes and performance.

### 3.1. Load the policy

```python
import torch
from lerobot.common.policies.pi0.modeling_pi0 import Pi0Policy  # HF implementation used by INTACT
from huggingface_hub import login

# Optionally: login() if using private HF access token

device = "cuda" if torch.cuda.is_available() else "cpu"

policy = Pi0Policy.from_pretrained(
    "juexzz/INTACT-pi0-finetune-rephrase-bridge",
    torch_dtype=torch.float16,
).to(device)
```

This matches the usage snippet in the INTACT model card.[^1][^2]

### 3.2. Construct a fake batch

According to the INTACT config, inputs are: **single RGB image, one language instruction, one robot state**.[^3][^2]

Your agent can do:

```python
B = 2  # batch size
H, W = 256, 256  # or use the model’s expected resolution

batch = {
    "observation.images": torch.randn(B, 3, H, W, device=device, dtype=torch.float16),
    "observation.state": torch.randn(B, 10, device=device, dtype=torch.float16),  # placeholder size
    "instruction": ["put the carrot on the plate", "place the eggplant in the basket"],
}
```

The exact state dimension will depend on the LeRobot π₀ config (EEF pos, orientation, gripper, etc.), but for a shape‑check this placeholder is sufficient.[^4][^1]

### 3.3. Run action selection

```python
with torch.no_grad():
    actions = policy.select_action(batch)
print(actions.shape)
```

The agent should verify that `actions` has shape `[B, action_dim]` or `[B, T, action_dim]` depending on the chunking (INTACT uses **delta EEF with chunk size 4**).[^2][^3]

***

## 4. Testing on logged BridgeV2 trajectories (no simulator needed)

If you have BridgeV2 logs or can download a small subset, you can do **offline action prediction**:

1. Convert a few trajectories into a simple `.pt` or `.npz` with:
    - `images`: `(T, 3, H, W)`
    - `states`: `(T, S)`
    - `instructions`: single string per episode.[^5]
2. Your agent loads them and iterates over timesteps:
```python
data = torch.load("bridge_sample.pt")  # dict with images, states, instruction

images = data["images"].to(device)      # (T, 3, H, W)
states = data["states"].to(device)      # (T, S)
instr = data["instruction"]

for t in range(len(images)):
    batch = {
        "observation.images": images[t:t+1],
        "observation.state": states[t:t+1],
        "instruction": [instr],
    }
    with torch.no_grad():
        action = policy.select_action(batch)
    # store / visualize action here
```

This lets your agent verify numerical stability and basic behavior without setting up SimplerEnv.[^1][^2]

***

## 5. Full SimplerEnv evaluation (optional, closer to INTACT paper)

For a more faithful test, your coding agent can replicate the INTACT evaluation protocol:[^3][^1]

### 5.1. Install SimplerEnv

```bash
pip install git+https://github.com/simpler-env/SimplerEnv.git
```

SimplerEnv wraps the four original Bridge tasks (**carrot_on_plate, eggplant_in_basket, stack_cube, spoon_on_towel**).[^2]

### 5.2. Create a small evaluation script

Pseudocode your agent can implement:

```python
import simpler_env
import torch
from lerobot.common.policies.pi0.modeling_pi0 import Pi0Policy

env = simpler_env.make("Bridge-v0", task_name="carrot_on_plate")  # example
policy = Pi0Policy.from_pretrained("juexzz/INTACT-pi0-finetune-rephrase-bridge").cuda()
policy.eval()

successes = 0
episodes = 20

for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        batch = {
            "observation.images": torch.from_numpy(obs["rgb"]).permute(2,0,1).unsqueeze(0).cuda().float()/255.0,
            "observation.state": torch.from_numpy(obs["robot_state"]).unsqueeze(0).cuda().float(),
            "instruction": [info["instruction"]],
        }
        with torch.no_grad():
            action = policy.select_action(batch)
        obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        done = terminated or truncated
    successes += info.get("success", False)

print("Success rate:", successes / episodes)
```

You’ll need to match the exact observation keys and shapes from SimplerEnv; your agent can introspect `obs` and `info` at runtime and adapt.[^1][^2]

***

## 6. Resource settings for DGX Spark

- Use **`torch_dtype=torch.float16`** and `batch_size` 1–4 during evaluation to keep memory low on a single GPU; the model was trained on 4×H100/A100 with large batches, but inference is light.[^3][^2]
- You don’t need distributed training or multi‑GPU; a single GPU is enough for all tests above.

***

If you’d like, I can turn this into a single self‑contained `test_intact_pi0.py` script that your coding agent can just drop into a repo and run.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://huggingface.co/juexzz/INTACT-pi0-finetune-bridge

[^2]: https://huggingface.co/juexzz/INTACT-pi0-finetune-rephrase-bridge/blob/main/README.md

[^3]: https://huggingface.co/juexzz/INTACT-pi0-finetune-rephrase-bridge/raw/main/README.md

[^4]: https://github.com/huggingface/lerobot/blob/2b71789e15c35418b1ccecbceb81f4a598bfd883/lerobot/common/policies/pi0/modeling_pi0.py

[^5]: https://proceedings.mlr.press/v229/walke23a/walke23a.pdf

[^6]: https://huggingface.co/juexzz/INTACT-pi0-finetune-bridge/resolve/d5991f89f4d421c8985412ec57f588de2655babe/README.md?download=true

[^7]: https://github.com/peek-robot/openpi/blob/main/README.md

[^8]: https://github.com/Physical-Intelligence/openpi

[^9]: https://huggingface.co/api/resolve-cache/models/juexzz/INTACT-pi0-finetune-bridge/d5991f89f4d421c8985412ec57f588de2655babe/README.md?download=true\&etag="d3123e444ad28f52ad981c69a3b9a6437d3bce57"

[^10]: https://huggingface.co/juexzz

[^11]: https://huggingface.co/juexzz/INTACT-pi0-scratch-bridge/resolve/main/README.md?download=true

[^12]: https://arxiv.org/html/2602.12281v2

[^13]: https://github.com/Physical-Intelligence/openpi/issues/799

[^14]: https://www.pi.website/download/pi0.pdf

[^15]: https://huggingface.co/juexzz/INTACT-pi0-finetune-bridge/commit/d5991f89f4d421c8985412ec57f588de2655babe

[^16]: https://arxiv.org/html/2511.19859v1

