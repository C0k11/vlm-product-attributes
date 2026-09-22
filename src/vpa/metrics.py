"""Per-attribute scoring with explicit denominators."""
from collections import Counter

from .prompting import ATTRIBUTES


def _macro_f1(pairs: list[tuple[str, str | None]]) -> float:
    """Macro-F1 over the union of gold and predicted classes (sklearn's default).

    A missing prediction (None) counts against recall of the gold class but is
    not a class of its own.
    """
    labels = {g for g, _ in pairs} | {p for _, p in pairs if p is not None}
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


def unique_image_mask(rows: list[dict]) -> list[bool]:
    """True for the first row of each image_id in listing_key order.

    Variants of one product (sizes, marketplaces) often share a main image, so
    the row-level score weights popular products several times. Scoring only
    the masked rows counts each image once.
    """
    first = {}
    for i, r in sorted(enumerate(rows), key=lambda x: x[1]["listing_key"]):
        first.setdefault(r["image_id"], i)
    keep = set(first.values())
    return [i in keep for i in range(len(rows))]


def score_both(rows: list[dict], preds: list[dict | None]) -> dict:
    mask = unique_image_mask(rows)
    return {
        "rows": score(rows, preds),
        "unique_images": score([r for r, m in zip(rows, mask) if m], [p for p, m in zip(preds, mask) if m]),
    }


def score(rows: list[dict], preds: list[dict | None]) -> dict:
    """rows carry gold labels (None = unlabeled); preds are parsed outputs or None.

    An unparseable output counts as wrong for every labeled attribute of that row.
    Macro-F1 averages over gold and predicted classes in the evaluated rows.
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


def fmt(m: dict) -> str:
    if not m["n"]:
        return "n=0"
    return f"{m['correct']}/{m['n']} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}"
