"""Binary classification metrics. Label 1 = shoplifting."""
from __future__ import annotations


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    def div(a, b):
        return a / b if b else 0.0

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    return {
        "n": len(y_true), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": div(tn, tn + fp),
        "f1": div(2 * precision * recall, precision + recall),
        "accuracy": div(tp + tn, len(y_true)),
    }


def format_report(m: dict, title: str = "") -> str:
    lines = [f"## {title}"] if title else []
    lines += [
        f"Clips scored: {m['n']}",
        "",
        "|                | pred theft | pred normal |",
        "|----------------|-----------:|------------:|",
        f"| **true theft** | {m['tp']:>10} | {m['fn']:>11} |",
        f"| **true normal**| {m['fp']:>10} | {m['tn']:>11} |",
        "",
        f"- Precision:   {m['precision']:.1%}",
        f"- Recall:      {m['recall']:.1%}",
        f"- Specificity: {m['specificity']:.1%}",
        f"- F1:          {m['f1']:.3f}",
        f"- Accuracy:    {m['accuracy']:.1%}",
    ]
    return "\n".join(lines)
