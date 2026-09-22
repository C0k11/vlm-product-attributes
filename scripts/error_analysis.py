"""Error breakdown for a predictions file: top confusions per attribute and
accuracy per product type, with denominators."""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vpa.prompting import ATTRIBUTES  # noqa: E402
from vpa.splits import load_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--split", default="test")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--min-n", type=int, default=20, help="min rows for a per-type line")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_split(args.split)
    preds = {}
    with open(args.predictions, encoding="utf-8") as fh:
        for line in fh:
            p = json.loads(line)
            preds[p["listing_key"]] = p
    report = {"predictions": args.predictions, "split": args.split, "attributes": {}}
    for attr in ATTRIBUTES:
        pairs = [(r[attr], (preds.get(r["listing_key"]) or {}).get(attr), r["product_type"]) for r in rows if r[attr]]
        conf = collections.Counter((g, p) for g, p, _ in pairs if g != p)
        per_type = collections.defaultdict(lambda: [0, 0])
        for g, p, t in pairs:
            per_type[t][0] += g == p
            per_type[t][1] += 1
        by_gold = collections.defaultdict(lambda: [0, 0])
        for g, p, _ in pairs:
            by_gold[g][0] += g == p
            by_gold[g][1] += 1
        n_err = sum(conf.values())
        print(f"\n== {attr}: errors {n_err}/{len(pairs)}")
        for (g, p), c in conf.most_common(args.top):
            print(f"   {g:28} -> {str(p):28} {c:4}  ({c / n_err:.1%} of errors)")
        print(f"   recall by gold class (n >= {args.min_n}):")
        for g, (ok, n) in sorted(by_gold.items(), key=lambda x: x[1][0] / x[1][1]):
            if n >= args.min_n:
                print(f"     {g:28} {ok}/{n} = {ok / n:.3f}")
        if attr != "product_type":
            print(f"   accuracy by product type (n >= {args.min_n}):")
            for t, (ok, n) in sorted(per_type.items(), key=lambda x: x[1][0] / x[1][1]):
                if n >= args.min_n:
                    print(f"     {t:28} {ok}/{n} = {ok / n:.3f}")
        report["attributes"][attr] = {
            "n": len(pairs), "errors": n_err,
            "top_confusions": [{"gold": g, "pred": p, "count": c} for (g, p), c in conf.most_common(args.top)],
            "recall_by_gold": {g: {"correct": ok, "n": n} for g, (ok, n) in by_gold.items()},
            "accuracy_by_product_type": {t: {"correct": ok, "n": n} for t, (ok, n) in per_type.items()},
        }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
