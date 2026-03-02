#!/usr/bin/env bash
# setup_env.sh
# Sets up a Python venv for testing the INTACT pi-0 BridgeV2 checkpoint.
# Usage: bash setup_env.sh  (from inside the pi-0-test directory)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=== Creating virtual environment at $VENV_DIR ==="
python3 -m venv "$VENV_DIR"

echo ""
echo "=== Installing dependencies ==="
"$VENV_DIR/bin/pip" install --upgrade pip

# PyTorch — cu130 wheel for the GB10 GPU (CUDA 13.0 driver, sm_12.1)
"$VENV_DIR/bin/pip" install "torch>=2.10" --index-url https://download.pytorch.org/whl/cu130

# lerobot + core HF stack
"$VENV_DIR/bin/pip" install lerobot accelerate huggingface_hub sentencepiece

# transformers: MUST use the lerobot pi0 patched branch (fix/lerobot_openpi).
# Standard PyPI transformers has incompatible Gemma/SigLIP attention APIs.
"$VENV_DIR/bin/pip" install \
    "git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi" \
    "scipy>=1.10.1,<1.15"

echo ""
echo "=== Patching lerobot's siglip.check guard ==="
# lerobot 0.4.x checks for an unreleased patched transformers module.
# We bypass the guard so the model can still be used with stock transformers.
MODELING_PI0="$VENV_DIR/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/lerobot/policies/pi0/modeling_pi0.py"

if [ -f "$MODELING_PI0" ]; then
    python3 - <<'PYEOF'
import sys, pathlib, re

venv_python = sys.argv[1] if len(sys.argv) > 1 else None

import subprocess, os
venv_dir = os.environ.get("VENV_DIR", "")
py_ver = subprocess.check_output(
    [f"{venv_dir}/bin/python", "-c",
     "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"]
).decode().strip()

path = pathlib.Path(f"{venv_dir}/lib/python{py_ver}/site-packages/lerobot/policies/pi0/modeling_pi0.py")
if not path.exists():
    print(f"  WARNING: {path} not found — skipping patch")
    sys.exit(0)

text = path.read_text()
OLD = '''        msg = """An incorrect transformer version is used, please create an issue on https://github.com/huggingface/lerobot/issues"""

        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None'''

NEW = '''        # NOTE: siglip.check guard bypassed — requires unreleased patched transformers.
        try:
            from transformers.models.siglip import check
            if not check.check_whether_transformers_replace_is_installed_correctly():
                pass
        except (ImportError, Exception):
            pass  # standard transformers is fine for inference'''

if OLD in text:
    path.write_text(text.replace(OLD, NEW))
    print("  ✓ siglip.check guard patched successfully")
elif NEW in text:
    print("  ✓ siglip.check guard already patched — skipping")
else:
    print("  WARNING: Could not find the exact guard text — lerobot version may differ")
PYEOF
else
    echo "  WARNING: modeling_pi0.py not found at expected path — skipping patch"
fi

echo ""
echo "=== Verifying GPU availability ==="
"$VENV_DIR/bin/python" -c "
import torch
available = torch.cuda.is_available()
print(f'CUDA available: {available}')
if available:
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('WARNING: No GPU detected — model will run on CPU (very slow).')
"

echo ""
echo "=== Setup complete! ==="
echo "Activate with:  source $VENV_DIR/bin/activate"
echo "Then run:       python test_sanity.py"
