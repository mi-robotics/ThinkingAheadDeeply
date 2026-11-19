#!/usr/bin/env bash
# Evaluate one HVLA checkpoint on LIBERO, at one or more evaluation seeds.
#
#   scripts/eval_libero.sh CKPT [--seeds "7 8 9"] [--k 12] [--exec 1] [--trials 10] [--tasks "0 1 ..."]
#                               [--workers 4] [--gpu 0] [--out DIR] [--suite libero_10]
#
#   CKPT       a checkpoint dir, e.g. outputs/libero/hvla_recurrent_long/seed42/<run>--99999_chkpt
#   --seeds    evaluation seeds (default 7, the protocol seed). initial_states=DEFAULT for every seed, so episode i
#              of task t starts from the same state in every run: paired McNemar (scripts/paired_test.py) is valid
#   --k        recurrent iterations, or Euler steps (NFE) for the flow head (default 12)
#   --exec     actions executed per policy query (default 1 = closed loop; 8 = the whole chunk)
#   --trials   episodes per task (default 10 -> 100 episodes per suite)
#   --tasks    task ids (default all 10)
#   --workers  parallel task processes on the GPU (default 4)
#   --suite    default: from the run's config.yaml (dataset_name)
#   --out      default: <run dir>/evals/<ckpt name>/k<k>_exec<exec>
#
# Restores the run's history / architecture switches from <run dir>/run.env (written by scripts/train.sh).
# One process per task (LIBERO stalls after ~19 resets in one process). Restart-safe: finished tasks are skipped.
# Writes <out>/seed<s>/task<t>.json, <out>/seed<s>/eval.json and <out>/summary.json (mean +- sd over seeds).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$HOME/miniconda3/envs/rdvla/bin/python}"

[ $# -ge 1 ] || { sed -n '2,21p' "$0"; exit 2; }
CKPT="$(cd "$1" && pwd)" || exit 2; shift
SEEDS="7"; K=12; EXEC=1; TRIALS=10; TASKS="0 1 2 3 4 5 6 7 8 9"; W=4; GPU=0; OUT=""; SUITE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --seeds) SEEDS="$2"; shift 2;;  --k) K="$2"; shift 2;;  --exec) EXEC="$2"; shift 2;;
    --trials) TRIALS="$2"; shift 2;;  --tasks) TASKS="$2"; shift 2;;  --workers) W="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;  --out) OUT="$2"; shift 2;;  --suite) SUITE="$2"; shift 2;;
    *) echo "unknown argument: $1"; exit 2;;
  esac
done
[ -f "$CKPT/model.safetensors" ] || { echo "not a checkpoint (no model.safetensors): $CKPT"; exit 2; }
RUN="$(dirname "$CKPT")"
if [ -f "$RUN/run.env" ]; then source "$RUN/run.env"; else echo "warning: no $RUN/run.env; evaluating with no history switches"; fi
if [ -z "$SUITE" ]; then
  DS="$(grep -m1 'dataset_name:' "$RUN/config.yaml" 2>/dev/null | awk '{print $2}')"
  SUITE="${DS%_no_noops}"
  [ -n "$SUITE" ] || { echo "cannot infer the suite; pass --suite"; exit 2; }
fi
CHUNK="${RDVLA_NUM_ACTIONS_CHUNK:-8}"
OUT="${OUT:-$RUN/evals/$(basename "$CKPT" | sed 's/.*--//')/k${K}_exec${EXEC}}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="$GPU" CUDA_VISIBLE_DEVICES="$GPU" \
       TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 TOKENIZERS_PARALLELISM=false \
       PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "$(date '+%F %T') eval $CKPT suite=$SUITE seeds=[$SEEDS] k=$K exec=$EXEC trials=$TRIALS out=$OUT" \
     "$(env | grep '^RDVLA_' | sort | tr '\n' ' ')"

cd "$ROOT"
for s in $SEEDS; do
  D="$OUT/seed$s"; mkdir -p "$D/logs"
  for t in $TASKS; do
    grep -qs overall_success_rate "$D/task$t.json" && continue
    timeout 7200 "$PY" -m recurrent_vla.eval.run_libero_eval \
      --pretrained_checkpoint "$CKPT" --task_suite_name "$SUITE" --task_id "$t" --num_trials_per_task "$TRIALS" \
      --seed "$s" --initial_states_path DEFAULT --use_recurrent True --recurrence_strategy fixed \
      --recurrent_num_iter "$K" --num_exec_actions "$EXEC" --num_open_loop_steps "$CHUNK" \
      --save_rollout_videos False --local_log_dir "$D/logs" --json_log_file "$D/task$t.json" \
      < /dev/null > "$D/task$t.log" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge "$W" ]; do sleep 10; done
  done
done
wait

"$PY" - "$OUT" "$SEEDS" "$TASKS" <<'MERGE'
import json, os, statistics, sys
out, seeds, tasks = sys.argv[1], sys.argv[2].split(), [int(t) for t in sys.argv[3].split()]
rates, missing = {}, []
for s in seeds:
    d = f"{out}/seed{s}"
    per = {}
    for t in tasks:
        f = f"{d}/task{t}.json"
        j = json.load(open(f)) if os.path.exists(f) else {}
        if "overall_success_rate" not in j:
            missing.append(f"seed{s}/task{t}")
            continue
        per[t] = j
    if len(per) < len(tasks):
        continue
    n = sum(j["total_episodes"] for j in per.values()); k = sum(j["total_successes"] for j in per.values())
    json.dump({"overall_success_rate": k / n, "total_episodes": n, "total_successes": k,
               "per_task_success": {t: j["overall_success_rate"] for t, j in per.items()}, "per_task": per},
              open(f"{d}/eval.json", "w"), indent=1)
    rates[s] = k / n
summary = {"seeds": rates, "mean": statistics.mean(rates.values()) if rates else None,
           "sd": statistics.stdev(rates.values()) if len(rates) > 1 else None, "incomplete": missing}
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
txt = " ".join(f"seed{s}={100 * r:.1f}%" for s, r in rates.items())
if rates:
    txt += f"  mean {100 * summary['mean']:.1f}%" + (f" +- {100 * summary['sd']:.1f}" if summary["sd"] is not None else "")
print(txt or "no complete seed", ("  INCOMPLETE: " + " ".join(missing)) if missing else "")
sys.exit(1 if missing else 0)
MERGE
rc=$?
echo "$(date '+%F %T') eval exit $rc -> $OUT/summary.json"
exit $rc
