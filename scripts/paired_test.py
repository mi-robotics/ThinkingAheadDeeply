#!/usr/bin/env python3
"""Paired McNemar test between two LIBERO eval JSONs, plus Wilson intervals.

    python experiments/final/paired_test.py A.json B.json [--labels teacher rectified]

Why paired rather than comparing success rates
----------------------------------------------
Two eval runs of the same suite roll out the SAME initial states: run_libero_eval.py does
`initial_states = task_suite.get_task_init_states(task_id)` and then indexes
`initial_states[episode_idx]`, so episode i of task t starts identically in every run
regardless of seed. That makes (task, episode) a valid pairing key.

Pairing matters here because the unpaired comparison is badly underpowered at these rates.
With 100 episodes, a 95% vs 96% difference has a Wilson interval far wider than the gap, so an
unpaired read says "no difference" almost regardless of the truth. McNemar throws away the
episodes both arms agree on and tests only the discordant pairs (b = A-only successes,
c = B-only), which is where the information actually is.

It does NOT control for rollout non-determinism (renderer/physics), which has been measured in
this project to move scores by a few episodes. So a small discordant count should be read as
"indistinguishable", not "equal". The exact binomial version is used rather than the chi-square
approximation, because b + c is typically small (< 25) and the approximation is poor there.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load_episodes(path: Path) -> dict[tuple[str, int], bool]:
    """-> {(task, episode_idx): success}."""
    d = json.loads(Path(path).read_text())
    out = {}
    for task, eps in d.get("tasks", {}).items():
        for e in eps:
            out[(task, int(e["episode"]))] = bool(e.get("success"))
    return out


def binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial p-value for McNemar (H0: p = 0.5 among discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X <= k) under Binom(n, 0.5), doubled and clamped.
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    args = ap.parse_args()
    la, lb = args.labels

    A, B = load_episodes(Path(args.a)), load_episodes(Path(args.b))
    shared = sorted(set(A) & set(B))
    only_a, only_b = len(set(A) - set(B)), len(set(B) - set(A))
    if only_a or only_b:
        print(f"WARNING: unpaired episodes dropped -- {only_a} only in {la}, {only_b} only in {lb}")
    if not shared:
        raise SystemExit("no shared (task, episode) keys: these runs are not comparable")

    ka = sum(A[k] for k in shared)
    kb = sum(B[k] for k in shared)
    n = len(shared)
    both = sum(1 for k in shared if A[k] and B[k])
    neither = sum(1 for k in shared if not A[k] and not B[k])
    b_only_a = sum(1 for k in shared if A[k] and not B[k])      # A succeeds, B fails
    c_only_b = sum(1 for k in shared if not A[k] and B[k])      # B succeeds, A fails
    p = binom_two_sided(b_only_a, c_only_b)

    lo_a, hi_a = wilson(ka, n)
    lo_b, hi_b = wilson(kb, n)
    print(f"paired episodes: {n}")
    print(f"  {la:>12s}: {ka}/{n} = {100*ka/n:.1f}%   95% CI [{100*lo_a:.1f}, {100*hi_a:.1f}]")
    print(f"  {lb:>12s}: {kb}/{n} = {100*kb/n:.1f}%   95% CI [{100*lo_b:.1f}, {100*hi_b:.1f}]")
    print(f"  difference : {100*(kb-ka)/n:+.1f} pp")
    print()
    print("contingency (paired):")
    print(f"  both succeed      : {both}")
    print(f"  both fail         : {neither}")
    print(f"  {la} only         : {b_only_a}")
    print(f"  {lb} only         : {c_only_b}")
    print(f"  discordant total  : {b_only_a + c_only_b}")
    print()
    print(f"McNemar exact two-sided p = {p:.4f}")
    if p < 0.05:
        better = lb if c_only_b > b_only_a else la
        print(f"  -> significant at 0.05; {better} is better on the paired episodes")
    else:
        print("  -> NOT significant at 0.05: the two arms are statistically indistinguishable")
        print("     on these episodes. Report them as equivalent, not as one beating the other.")


if __name__ == "__main__":
    main()
