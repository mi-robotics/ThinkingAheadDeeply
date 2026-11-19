#!/usr/bin/env bash
# Install recurrent_vla's dependencies + LIBERO into the `rdvla` conda env (after torch 2.2.0 is present).
# recurrent_vla itself needs no install: scripts/*.sh put the repo root on PYTHONPATH.
set -euo pipefail
#   ENV=<conda env dir> LIBERO=<LIBERO clone> scripts/install.sh
ENV="${ENV:-$HOME/miniconda3/envs/rdvla}"
PIP="$ENV/bin/pip"
PY="$ENV/bin/python"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIBERO="${LIBERO:-$ROOT/third_party/LIBERO}"     # git clone https://github.com/Lifelong-Robot-Learning/LIBERO

"$PY" -c "import torch; assert torch.__version__.startswith('2.2.0'), torch.__version__; print('torch ok', torch.__version__)"

"$PIP" install -r "$ROOT/requirements.txt" 2>&1 | tail -3
# flash-attn is not required: the checkpoint does not request flash_attention_2, so the LLM uses PyTorch attention.

# LIBERO's legacy setup.py breaks pip's PEP 660 editable install ("No module named libero") and pins old
# torch: install without deps and expose the source tree with a .pth file.
"$PIP" install --no-deps -e "$LIBERO" 2>&1 | tail -1
SP=$("$PY" -c "import site; print(site.getsitepackages()[0])")
echo "$LIBERO" > "$SP/libero_local.pth"
# LIBERO prompts for dataset paths on first import; pre-write its config for non-interactive runs.
if [ ! -f ~/.libero/config.yaml ]; then
  L="$LIBERO/libero"; mkdir -p ~/.libero
  printf 'benchmark_root: %s/libero\nbddl_files: %s/libero/bddl_files\ninit_states: %s/libero/init_files\ndatasets: %s/datasets\nassets: %s/libero/assets\n' "$L" "$L" "$L" "$L" "$L" > ~/.libero/config.yaml
fi
# Version pins found necessary on this machine (2026-09-14):
#   numpy<2              torch 2.2 / tensorflow 2.15 are built against numpy 1.x
#   tensorflow-metadata  1.21 needs protobuf>=5 (runtime_version); TF 2.15 needs protobuf<5
#   wandb 0.16.6         wandb 0.28 needs protobuf>4.21, incompatible with tensorflow-metadata 1.14
#   mujoco 3.1.6         robosuite 1.4.1 asserts on joint types returned by mujoco 3.13
"$PIP" install "numpy<2" "tensorflow-metadata==1.14.0" "wandb==0.16.6" "mujoco==3.1.6" 2>&1 | tail -1

"$PY" - <<'EOF'
import torch, transformers, peft
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__, "peft", peft.__version__)
try:
    import flash_attn; print("flash_attn", flash_attn.__version__)
except Exception as e:
    print("flash_attn import FAILED:", e)
import robosuite, libero; print("robosuite", robosuite.__version__, "libero ok")
EOF
echo "install complete"
