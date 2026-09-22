"""Per-attribute scoring with explicit denominators."""
from collections import Counter

from .prompting import ATTRIBUTES


def _macro_f1(pairs: list[tuple[str, str | None]]) -> float:
    labels = {g for g, _ in pairs}
    tp, fp, fn = Counter(), Counter(), Counter()
    for gold, pred in pairs:
        if pred == gold:
            tp[gold] += 1
        else:
            fn[gold] += 1
            if pred is not None:
                fp[pred] += 1
    f1s = []
    for lab in labels:
        p = tp[lab] / (tp[lab] + fp[lab]) if tp[lab] + fp[lab] else 0.0
        r = tp[lab] / (tp[lab] + fn[lab]) if tp[lab] + fn[lab] else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def score(rows: list[dict], preds: list[dict | None]) -> dict:
    """rows carry gold labels (None = unlabeled); preds are parsed outputs or None.

    An unparseable output counts as wrong for every labeled attribute of that row.
    Macro-F1 averages over gold classes present in the evaluated rows.
    """
    assert len(rows) == len(preds)
    out = {}
    for attr in ATTRIBUTES:
        pairs = []
        for row, pred in zip(rows, preds):
            gold = row.get(attr)
            if gold is None:
                continue
            value = pred.get(attr) if isinstance(pred, dict) else None
            pairs.append((gold, value if isinstance(value, str) else None))
        n = len(pairs)
        correct = sum(1 for g, p in pairs if g == p)
        out[attr] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else None,
            "macro_f1": _macro_f1(pairs) if n else None,
        }
    return out
