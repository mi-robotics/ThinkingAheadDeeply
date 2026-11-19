# Ablation evaluation protocol

**Every ablation in this project is evaluated this way. No exceptions without recording why.**

This exists because on 2026-09-22 a single-run ablation produced **+10 pp at p = 0.0414** — a result
that would have entered the paper — and it evaporated when run at two more seeds (+10, +6, −1 pp,
sign flipping). The protocol below is what that failure costs us in compute, and it is cheaper than
retracting a claim.

---

## Fixed parameters

| parameter | value | why fixed |
|---|---|---|
| suite | **LIBERO-Long** (`libero_10_no_noops`) | discriminates. LIBERO-Spatial saturates (99–100%) and cannot separate arms |
| training seeds | **3** (0, 1, 2) | one run per arm cannot separate "the change helped" from "this run was better" |
| eval seed | **7**, identical across all conditions | fixed per-episode initial states are what make paired McNemar valid |
| initial states | `initial_states_path=DEFAULT` | `get_task_init_states(task_id)[episode_idx]` — episode *i* of task *t* starts identically in every run, regardless of seed |
| episodes | **100** (10 tasks × 10 trials) | see resolution limit below |
| depths (k) | **1, 2** | the rectified operating points |
| action steps | **5** (`num_exec_actions=5`) | matches the teacher policy and every result before 2026-09-22, so ablations stay comparable to existing numbers |
| adaptive paths | **all off** | `adaptive_exec`, `dynamic_exec`, `use_linear_decay_horizon` = False |

## Action steps: fixed at 5

The head predicts an **8-action chunk**. `num_exec_actions` controls how many are executed before
the policy is re-queried; the remaining 3 are discarded. **Ablations use 5**, matching the teacher
policy and every result produced before 2026-09-22, so ablation numbers are directly comparable to
the existing teacher and rectified baselines.

`num_open_loop_steps=8` does **not** control this, despite its docstring saying it does. It is the
chunk length. Only `num_exec_actions` governs execution.

**A 1/5/8 sweep was considered and dropped.** It would have tripled every condition to 18 evals
(~6–7 h each), and the expensive arm is 1 action-step, which queries the policy ~5× more often. The
axis is genuinely interesting — the project's claim is about *compute per decision*, and a head
needing k=12 at 8 action-steps may cost less per episode than one needing k=1 at 1 action-step — so
it is worth one dedicated study later rather than a tax on every ablation. `eval_ablation.sh` still
supports it: `ACTION_STEPS="1 5 8"`.

## Statistics

- **Within a seed:** paired McNemar (exact binomial) on `(task, episode)` keys — `experiments/final/paired_test.py`.
- **Across seeds:** mean ± sd of the per-seed deltas. **Report the sd.** A mean without it is what made
  the +10 pp look real.
- **Sign consistency across seeds is the headline test.** If the sign flips, the result is a null
  regardless of any individual p-value.
- A single seed reaching p < 0.05 while others do not is **one marginal hit in three** — noise.

## Resolution limit — read before designing a condition

**Run-to-run spread on this setup is ~5–6 pp at 100 episodes** (measured: deltas +10, +6, −1 pp
across three seeds of an arm pair that is genuinely null).

An ablation whose expected effect is smaller than ~6 pp **cannot be resolved** at 100 episodes, no
matter how many seeds. Either raise the episode count or drop the condition — running it and
reporting a null we could never have detected is worse than not running it.

## Offline metrics are a filter, never the verdict

Offline L1 / `noise_std` disagreed with closed-loop success **twice** on 2026-09-22, in both
directions. Use them to triage which conditions earn rollouts. Never conclude from them.

## Cost model

Per condition (measured on the local RTX 4090, LIBERO-Long, 100 episodes):

| item | cost | basis |
|---|---|---|
| rectification | ~12 min × 3 seeds = **36 min** | measured |
| eval, 100 episodes, 5 action-steps | **~14 min** per (seed, depth) | measured |

Grid = 3 seeds × 2 depths = **6 evals** ≈ **84 min**, plus 36 min rectification ≈ **2 h per
condition**.

So **N conditions ≈ 2N hours** on one GPU. That is the number to size the ablation list against.
If the list grows beyond what the budget allows, screen offline first and pay for rollouts only on
conditions that survive — accepting that offline is a filter and never the verdict.

## Recording

Each condition writes `runs/ablation_<name>/`:

- `result.json` — the per-seed deltas, mean, sd, sign consistency, verdict
- `seeds.json` — per-seed contingency tables and p-values
- one `eval_<arm>_s<seed>/eval_k<k>_a<actions>.json` per cell
- an HTML report whose **headline is computed from the data**, never asserted — see
  `experiments/final/ablation_report.py`

**State the prediction before running.** If it turns out wrong, keep it in the report. The
noise-coupling report records two wrong predictions and is more useful for it.
