#!/usr/bin/env bash
# Train one HVLA model on LIBERO.
#
#   scripts/train.sh CONFIG [--seed N] [--gpu G] [--out DIR] [--preset NAME] [-- --key=value ...]
#
#   CONFIG    a yaml under recurrent_vla/configs/libero/ (a path, or a name such as hvla_recurrent_long)
#   --seed    training seed (default 42); also names the default output dir
#   --gpu     CUDA device (default 0)
#   --out     output dir, relative to the repo root (default outputs/libero/<config>/seed<seed>)
#   --preset  memhvla  -> RDVLA_IMAGE_HISTORY=10 RDVLA_HIST_VIT=memhvla RDVLA_ENCODER_LR_SCALE=0.1 RDVLA_EMA=1
#                         RDVLA_RAM_CACHE=encoded   (anything already set in the environment wins)
#             prop10   -> RDVLA_PROPRIO_HISTORY=10
#             img10    -> RDVLA_IMAGE_HISTORY=10 RDVLA_RAM_CACHE=encoded   (history CLS, no gradient)
#             act10    -> RDVLA_ACTION_HISTORY=10                         (actions t-10 .. t-1)
#             actgap   -> RDVLA_ACTION_HISTORY=10 RDVLA_ACTION_GAP=2      (actions t-10 .. t-3: a_{t-1} not copyable)
#             actdrop  -> RDVLA_ACTION_HISTORY=10 RDVLA_ACTION_HIST_DROP=0.5
#   after --   extra config overrides, e.g. -- --max_steps=25000 --save_freq=2000
#
# History / architecture switches are environment variables (all default off). Set them for training only:
# this script writes the ones in effect to OUT/run.env and scripts/eval_libero.sh restores them, so a
# checkpoint is always evaluated with the inputs it was trained on.
#   RDVLA_PROPRIO_HISTORY  RDVLA_IMAGE_HISTORY  RDVLA_HIST_VIT  RDVLA_SHORT_DROPOUT  RDVLA_ENCODER_LR_SCALE
#   RDVLA_EMA  RDVLA_NUM_ACTIONS_CHUNK  RDVLA_RAM_CACHE  RDVLA_ACTION_HISTORY  RDVLA_ACTION_GAP  RDVLA_ACTION_HIST_DROP
#
# Restart-safe: rerunning the same command resumes from OUT's train_state (optimiser, scheduler, RNG, EMA).
# Long runs must outlive the shell:   setsid nohup scripts/train.sh ... > train.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$HOME/miniconda3/envs/rdvla/bin/python}"
TORCHRUN="$(dirname "$PY")/torchrun"

[ $# -ge 1 ] || { sed -n '2,27p' "$0"; exit 2; }
CONFIG="$1"; shift
SEED=42; GPU=0; OUT=""; PRESET=""; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --seed) SEED="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --preset) PRESET="$2"; shift 2;;
    --) shift; EXTRA=("$@"); break;;
    *) echo "unknown argument: $1"; exit 2;;
  esac
done
case "$CONFIG" in *.yaml) ;; *) CONFIG="recurrent_vla/configs/libero/$CONFIG.yaml";; esac
cd "$ROOT"
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG"; exit 2; }
NAME="$(basename "$CONFIG" .yaml)"
OUT="${OUT:-outputs/libero/$NAME/seed$SEED}"

case "$PRESET" in
  "") ;;
  memhvla) : "${RDVLA_IMAGE_HISTORY:=10}" "${RDVLA_HIST_VIT:=memhvla}" "${RDVLA_ENCODER_LR_SCALE:=0.1}" \
             "${RDVLA_EMA:=1}" "${RDVLA_RAM_CACHE:=encoded}";;
  prop10) : "${RDVLA_PROPRIO_HISTORY:=10}";;
  img10) : "${RDVLA_IMAGE_HISTORY:=10}" "${RDVLA_RAM_CACHE:=encoded}";;
  act10) : "${RDVLA_ACTION_HISTORY:=10}";;
  actgap) : "${RDVLA_ACTION_HISTORY:=10}" "${RDVLA_ACTION_GAP:=2}";;
  actdrop) : "${RDVLA_ACTION_HISTORY:=10}" "${RDVLA_ACTION_HIST_DROP:=0.5}";;
  *) echo "unknown preset: $PRESET"; exit 2;;
esac
case "$NAME" in memhvla_*)
  [ "${RDVLA_HIST_VIT:-}" = memhvla ] || { echo "$NAME needs --preset memhvla (or RDVLA_HIST_VIT=memhvla ...)"; exit 2; };;
esac

mkdir -p "$OUT"
ENVF="$OUT/run.env"
VARS="RDVLA_PROPRIO_HISTORY RDVLA_IMAGE_HISTORY RDVLA_HIST_VIT RDVLA_SHORT_DROPOUT RDVLA_ENCODER_LR_SCALE RDVLA_EMA RDVLA_NUM_ACTIONS_CHUNK RDVLA_RAM_CACHE RDVLA_ACTION_HISTORY RDVLA_ACTION_GAP RDVLA_ACTION_HIST_DROP"
NEW="$(for v in $VARS; do [ -n "${!v:-}" ] && echo "export $v=${!v}"; done; echo "# config=$CONFIG seed=$SEED")"
if [ -f "$ENVF" ] && [ "$(grep -v '^#' "$ENVF")" != "$(echo "$NEW" | grep -v '^#')" ]; then
  echo "refusing to resume $OUT with different settings:"; diff <(grep -v '^#' "$ENVF") <(echo "$NEW" | grep -v '^#'); exit 2
fi
echo "$NEW" > "$ENVF"
for v in $VARS; do [ -n "${!v:-}" ] && export "$v"; done

export TOKENIZERS_PARALLELISM=false WANDB_MODE="${WANDB_MODE:-disabled}" RDVLA_PREFETCH="${RDVLA_PREFETCH:-4}" \
       RDVLA_HEALTH_EVERY="${RDVLA_HEALTH_EVERY:-500}" RDVLA_DEPTH_EVAL_EVERY="${RDVLA_DEPTH_EVAL_EVERY:-5000}" \
       RDVLA_DEPTH_EVAL_BATCHES="${RDVLA_DEPTH_EVAL_BATCHES:-6}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       RDVLA_METRICS_FILE="$ROOT/$OUT/metrics.jsonl" CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "$(date '+%F %T') train $CONFIG seed=$SEED gpu=$GPU out=$OUT $(grep '^export' "$ENVF" | sed 's/export //' | tr '\n' ' ')"
"$TORCHRUN" --standalone --nnodes 1 --nproc-per-node 1 -m recurrent_vla.train --config "$CONFIG" \
  --seed="$SEED" --output_dir="$OUT" "${EXTRA[@]}" < /dev/null
rc=$?
echo "$(date '+%F %T') train exit $rc"
exit $rc
