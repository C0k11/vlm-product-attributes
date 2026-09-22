"""Sample a stratified pilot set from the val split and fetch its images.

Only val rows with a color label are eligible, so every pilot row can be scored
on product_type and color; material is scored where present. Images are saved
twice: the original resized to --long-side, and the official 256 px version.
"""
import argparse
import collections
import io
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

BASE = "https://amazon-berkeley-objects.s3.amazonaws.com/images"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def save_pair(row: dict, out_dir: Path, long_side: int) -> int:
    raw = fetch(f"{BASE}/original/{row['image_path']}")
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
    img.save(out_dir / f"img{long_side}" / f"{row['image_id']}.jpg", quality=95)
    small = fetch(f"{BASE}/small/{row['image_path']}")
    Image.open(io.BytesIO(small)).convert("RGB").save(out_dir / "img256" / f"{row['image_id']}.jpg", quality=95)
    return len(raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="data/processed/val.jsonl")
    ap.add_argument("--out", default="data/pilot")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--long-side", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.val, encoding="utf-8")]
    eligible = [r for r in rows if r["color"]]
    print(f"val rows {len(rows)}, eligible (has color) {len(eligible)}")

    rng = random.Random(args.seed)
    by_type = collections.defaultdict(list)
    for r in eligible:
        by_type[r["product_type"]].append(r)
    for v in by_type.values():
        rng.shuffle(v)
    # Round-robin over types, at most one row per group, until n rows.
    picked, used_groups = [], set()
    types = sorted(by_type)
    while len(picked) < args.n and any(by_type.values()):
        for t in types:
            while by_type[t]:
                r = by_type[t].pop()
                if r["group"] not in used_groups:
                    picked.append(r)
                    used_groups.add(r["group"])
                    break
            if len(picked) == args.n:
                break
    print(f"picked {len(picked)} rows from {len({r['product_type'] for r in picked})} types")

    out = Path(args.out)
    (out / f"img{args.long_side}").mkdir(parents=True, exist_ok=True)
    (out / "img256").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        sizes = list(pool.map(lambda r: save_pair(r, out, args.long_side), picked))
    print(f"downloaded {len(sizes)} originals, {sum(sizes) / 1e6:.1f} MB before resizing")

    with open(out / "pilot.jsonl", "w", encoding="utf-8") as fh:
        for r in picked:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {k: sum(1 for r in picked if r[k]) for k in ["product_type", "color", "material"]}
    print(f"label counts in pilot: {counts}")
    print("types:", collections.Counter(r["product_type"] for r in picked).most_common())
    print("colors:", collections.Counter(r["color"] for r in picked).most_common())
    print("materials:", collections.Counter(r["material"] for r in picked if r["material"]).most_common())


if __name__ == "__main__":
    main()
