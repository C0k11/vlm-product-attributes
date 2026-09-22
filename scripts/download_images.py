"""Fetch images for the processed splits.

small:    download abo-images-small.tar (longest side 256 px) and extract only the
          images referenced by train/val/test into data/images/img256/<image_id>.jpg
original: download originals for the given splits and save them resized to
          --long-side into data/images/img<long_side>/<image_id>.jpg
"""
import argparse
import io
import json
import tarfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

BASE = "https://amazon-berkeley-objects.s3.amazonaws.com"


def split_rows(splits: list[str]) -> list[dict]:
    rows = []
    for s in splits:
        with open(f"data/processed/{s}.jsonl", encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh)
    return rows


def fetch_small(out_root: Path, tar_path: Path) -> None:
    if not tar_path.exists():
        tmp = tar_path.with_suffix(".part")
        print(f"downloading {BASE}/archives/abo-images-small.tar -> {tar_path}", flush=True)
        urllib.request.urlretrieve(f"{BASE}/archives/abo-images-small.tar", tmp)
        tmp.rename(tar_path)
    wanted = {}
    for r in split_rows(["train", "val", "test"]):
        wanted[f"images/small/{r['image_path']}"] = r["image_id"]
    out_dir = out_root / "img256"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(tar_path, "r|") as tar:
        for member in tar:
            image_id = wanted.get(member.name)
            if image_id is None or not member.isfile():
                continue
            data = tar.extractfile(member).read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.save(out_dir / f"{image_id}.jpg", quality=95)
            written += 1
    missing = len(wanted) - written
    print(f"small images: wanted {len(wanted)}, written {written}, missing {missing}", flush=True)


def _fetch_one(args: tuple) -> tuple[str, int]:
    image_id, path, out_dir, long_side = args
    dest = Path(out_dir) / f"{image_id}.jpg"
    if dest.exists():
        return image_id, 0
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"{BASE}/images/original/{path}", timeout=60) as resp:
                raw = resp.read()
            break
        except Exception:
            if attempt == 2:
                return image_id, -1
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
    img.save(dest, quality=95)
    return image_id, len(raw)


def fetch_original(out_root: Path, splits: list[str], long_side: int, workers: int) -> None:
    out_dir = out_root / f"img{long_side}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = split_rows(splits)
    jobs = sorted({(r["image_id"], r["image_path"]) for r in rows})
    tasks = [(i, p, str(out_dir), long_side) for i, p in jobs]
    total_bytes, failed, done = 0, [], 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for image_id, n in pool.map(_fetch_one, tasks, chunksize=16):
            done += 1
            if n < 0:
                failed.append(image_id)
            else:
                total_bytes += n
            if done % 1000 == 0:
                print(f"  {done}/{len(tasks)}", flush=True)
    print(f"originals for {splits}: images {len(tasks)}, downloaded {total_bytes / 1e9:.2f} GB, "
          f"failed {len(failed)} {failed[:5]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["small", "original"])
    ap.add_argument("--out", default="data/images")
    ap.add_argument("--tar", default="data/abo/abo-images-small.tar")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--long-side", type=int, default=768)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    if args.mode == "small":
        fetch_small(Path(args.out), Path(args.tar))
    else:
        fetch_original(Path(args.out), args.splits.split(","), args.long_side, args.workers)


if __name__ == "__main__":
    main()
