"""Metrics + confusion matrix from a verdict CSV. Writes <csv>.report.md next to it.

Usage:
    python scripts/evaluate.py outputs/test_a_mnnit_<model>.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401  (sets import path)
import pandas as pd

from src.metrics import binary_metrics, format_report

PAZA = {"precision": 0.895, "recall": 0.593, "specificity": 0.928, "f1": 0.713}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    args = ap.parse_args()
    path = Path(args.csv)

    df = pd.read_csv(path)
    errors = df[df["verdict"] == "ERROR"]
    df = df[df["verdict"] != "ERROR"].drop_duplicates("clip_id", keep="last")
    y_true = df["label"].astype(int).tolist()
    y_pred = df["pred"].astype(int).tolist()
    m = binary_metrics(y_true, y_pred)

    parts = [format_report(m, f"Results: {path.stem}")]
    parts += ["", "### vs. Paza (reported)", "", "| metric | ours | Paza |", "|---|---:|---:|"]
    parts += [f"| {k} | {m[k]:.3f} | {v:.3f} |" for k, v in PAZA.items()]
    parts += ["", "### Verdict breakdown", "", df.groupby(["label", "verdict"]).size().to_string()]
    if "cost_usd" in df:
        parts += ["", f"Total API cost: ${df['cost_usd'].sum():.4f}",
                  f"Mean latency: {df['latency_s'].mean():.1f}s"]
    if len(errors):
        parts += ["", f"**{len(errors)} ERROR rows excluded** (rerun run_test_a.py to retry them)."]

    for title, sel in (("False alarms (normal clip flagged)", (df.label == 0) & (df.pred == 1)),
                       ("Misses (theft clip passed as normal)", (df.label == 1) & (df.pred == 0))):
        sub = df[sel]
        parts += ["", f"### {title}: {len(sub)}", ""]
        parts += [f"- `{r.clip_id}` {r.verdict} ({r.confidence}): {r.explanation}" for r in sub.itertuples()]

    report = "\n".join(parts)
    print(report)
    out = path.with_suffix(".report.md")
    out.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
