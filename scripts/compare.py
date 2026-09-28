"""Compare a candidate run with a frozen baseline run (same dataset, same clips).

Prints metrics side by side, VLM calls/cost/latency, and every clip whose prediction changed,
so a change can be judged as "better on X, no degradation on Y" before it is adopted.
See docs/benchmarking.md.

Usage:
    python scripts/compare.py benchmarks/baseline_v1/results/test_b_mnnit_qwen3.6-plus_paza_n50.csv \
                              outputs/test_b_mnnit_episode_n50.csv [--out report.md]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401  (sets import path)
import pandas as pd

from src.metrics import binary_metrics


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["verdict"] != "ERROR"].drop_duplicates("clip_id", keep="last").set_index("clip_id")


def _stats(df: pd.DataFrame) -> dict:
    m = binary_metrics(df["label"].astype(int).tolist(), df["pred"].astype(int).tolist())
    m["vlm_calls"] = int(df["n_vlm_calls"].sum())
    m["cost_usd"] = float(df["cost_usd"].sum())
    m["latency_per_call_s"] = float(df["latency_s"].sum()) / max(m["vlm_calls"], 1)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--out", default=None, help="also write the report to this .md file")
    args = ap.parse_args()

    base, cand = _load(args.baseline), _load(args.candidate)
    common = base.index.intersection(cand.index)
    only_b, only_c = base.index.difference(cand.index), cand.index.difference(base.index)
    b, c = _stats(base.loc[common]), _stats(cand.loc[common])

    lines = [f"## Compare: {Path(args.baseline).stem}  ->  {Path(args.candidate).stem}", "",
             f"Clips compared: {len(common)}"]
    if len(only_b) or len(only_c):
        lines.append(f"WARNING: {len(only_b)} clips only in baseline, {len(only_c)} only in candidate "
                     "(not the same clip set; metrics use the common clips only)")
    lines += ["", "| metric | baseline | candidate | change |", "|---|---:|---:|---:|"]
    for k in ("precision", "recall", "specificity", "f1"):
        lines.append(f"| {k} | {b[k]:.1%} | {c[k]:.1%} | {c[k] - b[k]:+.1%} |")
    for k in ("tp", "fn", "fp", "tn", "vlm_calls"):
        lines.append(f"| {k} | {b[k]} | {c[k]} | {c[k] - b[k]:+d} |")
    lines.append(f"| cost_usd | {b['cost_usd']:.4f} | {c['cost_usd']:.4f} | {c['cost_usd'] - b['cost_usd']:+.4f} |")
    lines.append(f"| latency per call (s) | {b['latency_per_call_s']:.1f} | {c['latency_per_call_s']:.1f} | "
                 f"{c['latency_per_call_s'] - b['latency_per_call_s']:+.1f} |")

    changed = [cid for cid in common if int(base.at[cid, "pred"]) != int(cand.at[cid, "pred"])]
    fixed = [cid for cid in changed if int(cand.at[cid, "pred"]) == int(cand.at[cid, "label"])]
    broken = [cid for cid in changed if cid not in fixed]
    lines += ["", f"### Prediction changed on {len(changed)} clips: {len(fixed)} fixed, {len(broken)} broken"]
    for title, ids in (("Fixed (now correct)", fixed), ("Broken (now wrong) - regressions", broken)):
        if ids:
            lines += ["", f"**{title}**", ""]
            for cid in ids:
                lines.append(f"- `{cid}` (label {base.at[cid, 'label']}): "
                             f"{base.at[cid, 'verdict']} ({base.at[cid, 'confidence']}) -> "
                             f"{cand.at[cid, 'verdict']} ({cand.at[cid, 'confidence']}), "
                             f"calls {base.at[cid, 'n_vlm_calls']} -> {cand.at[cid, 'n_vlm_calls']}")
    same_pred_diff_verdict = [cid for cid in common if cid not in changed
                              and base.at[cid, "verdict"] != cand.at[cid, "verdict"]]
    if same_pred_diff_verdict:
        lines += ["", f"Same prediction, different verdict (e.g. CONFIRMED <-> UNCERTAIN): "
                      + ", ".join(f"`{x}`" for x in same_pred_diff_verdict)]

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
