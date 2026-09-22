"""Majority-class baseline: predict the most frequent train label for every row."""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vpa.prompting import ATTRIBUTES  # noqa: E402
from vpa.splits import load_split  # noqa: E402

train = load_split("train")
majority = {a: collections.Counter(r[a] for r in train if r[a]).most_common(1)[0][0] for a in ATTRIBUTES}
print("majority classes:", majority)
out = Path("outputs/baselines/majority")
out.mkdir(parents=True, exist_ok=True)
for split in ["val", "test"]:
    with open(out / f"{split}_predictions.jsonl", "w", encoding="utf-8") as fh:
        for r in load_split(split):
            fh.write(json.dumps({"listing_key": r["listing_key"], **majority}) + "\n")
