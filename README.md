# ThinkingAheadDeeply

Built on the recurrent-depth HVLA architecture from
[ResponsiveVLA](https://github.com/mi-robotics/RepsonsiveVLA/tree/main/recurrent_vla) (imported at commit `9d438d3`).
That code in turn derives from RD-VLA (github.com/rd-vla/rd-vla, MIT; see `recurrent_vla/LICENSE-rd-vla`).

## Layout

```
recurrent_vla/                  all model, training and evaluation code
  train.py                      training loop, probes, auto-resume, EMA (python -m recurrent_vla.train)
  config.py                     training-config schema + YAML loading;  configs/libero/*.yaml
  constants.py                  LIBERO constants and the RDVLA_* history / chunk switches
  models/                       hvla_context.py (shared S1 stack), hvla_recurrent.py, hvla_flow.py,
                                recurrent_core.py (looped core + config), projectors.py
  vlm/                          the S2 VLM (Prism / Qwen2.5-0.5B) and base-weight loading
  data/                         RLDS pipeline, prompting, tokenizer, collator, action history
  eval/                         LIBERO evaluation
  rectify/                      distillation so 1-2 recurrent iterations reach the converged latent
scripts/                        train.sh, eval_libero.sh, paired_test.py (McNemar), install.sh
ablation_eval_protocol.md       how every comparison is run and reported
```

## Setup

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO third_party/LIBERO
ENV=$HOME/miniconda3/envs/rdvla LIBERO=$PWD/third_party/LIBERO scripts/install.sh   # torch 2.2.0 first
ln -s <dir with prism-qwen25-extra-dinosiglip-224px-0_5b/ and configs/> pretrained_models
ln -s <dir with libero/<suite>_no_noops RLDS datasets> datasets
```

No install needed: `scripts/*.sh` put the repo root on `PYTHONPATH`. `pretrained_models/`, `datasets/` and
`outputs/` are not tracked in git.

## Train / evaluate

```bash
setsid nohup scripts/train.sh hvla_recurrent_long --seed 1 --gpu 0 > train_s1.log 2>&1 < /dev/null &
scripts/eval_libero.sh outputs/libero/hvla_recurrent_long/seed1/<run>--<step>_chkpt --seeds "7" --k 12 --exec 1
```

`--k` is the number of recurrent iterations (or NFE for the flow head); `--exec` is how many actions run per query.
