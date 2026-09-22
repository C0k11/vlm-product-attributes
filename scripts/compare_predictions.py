"""Paired comparison of two prediction files on the same split: exact McNemar
test on per-row correctness and a paired bootstrap interval for the accuracy
difference, per attribute."""
import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vpa.prompting import ATTRIBUTES  # noqa: E402
from vpa.splits import load_split  # noqa: E402


def load(path):
    with open(path, encoding="utf-8") as fh:
        return {p["listing_key"]: p for p in map(json.loads, fh)}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--split", default="test")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    rows = load_split(args.split)
    pa, pb = load(args.a), load(args.b)
    rng = random.Random(0)
    for attr in ATTRIBUTES:
        pairs = [((pa.get(r["listing_key"]) or {}).get(attr) == r[attr], (pb.get(r["listing_key"]) or {}).get(attr) == r[attr])
                 for r in rows if r[attr]]
        n = len(pairs)
        acc_a = sum(x for x, _ in pairs) / n
        acc_b = sum(y for _, y in pairs) / n
        only_a = sum(1 for x, y in pairs if x and not y)
        only_b = sum(1 for x, y in pairs if y and not x)
        diffs = []
        for _ in range(args.boot):
            s = [pairs[rng.randrange(n)] for _ in range(n)]
            diffs.append(sum(x for x, _ in s) / n - sum(y for _, y in s) / n)
        diffs.sort()
        lo, hi = diffs[int(0.025 * args.boot)], diffs[int(0.975 * args.boot) - 1]
        print(f"{attr:12} n={n:5}  A={acc_a:.4f}  B={acc_b:.4f}  diff={acc_a - acc_b:+.4f} "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  only A right={only_a} only B right={only_b} McNemar p={mcnemar_exact(only_a, only_b):.4g}")


if __name__ == "__main__":
    main()
