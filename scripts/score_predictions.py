"""Score a predictions file against a split, row level and unique-image level.

Predictions: JSONL with listing_key and any of product_type / color / material.
Every row of the split must have a prediction; a missing key counts as wrong.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vpa.metrics import fmt, score_both  # noqa: E402
from vpa.splits import load_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None, help="write scores JSON here")
    args = ap.parse_args()

    rows = load_split(args.split)
    by_key = {}
    with open(args.predictions, encoding="utf-8") as fh:
        for line in fh:
            p = json.loads(line)
            by_key[p["listing_key"]] = p
    missing = sum(1 for r in rows if r["listing_key"] not in by_key)
    print(f"split {args.split}: rows {len(rows)}, predictions {len(by_key)}, rows without prediction {missing}")
    preds = [by_key.get(r["listing_key"]) for r in rows]
    res = score_both(rows, preds)
    for view, scores in res.items():
        for attr, m in scores.items():
            print(f"  [{view}] {attr}: {fmt(m)}")
    if args.out:
        Path(args.out).write_text(json.dumps({"split": args.split, "predictions": args.predictions,
                                              "rows_without_prediction": missing, **res}, indent=2))


if __name__ == "__main__":
    main()
