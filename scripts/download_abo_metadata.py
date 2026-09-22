"""Download ABO listing shards and the image metadata table."""
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://amazon-berkeley-objects.s3.amazonaws.com"
FILES = [f"listings/metadata/listings_{i:x}.json.gz" for i in range(16)] + [
    "images/metadata/images.csv.gz",
    "LICENSE-CC-BY-4.0.txt",
]


def fetch(rel: str, out_dir: Path) -> tuple[str, int]:
    dest = out_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{BASE}/{rel}", timeout=120) as resp:
        expected = int(resp.headers["Content-Length"])
        if dest.exists() and dest.stat().st_size == expected:
            return rel, expected
        data = resp.read()
    if len(data) != expected:
        raise IOError(f"{rel}: got {len(data)} bytes, expected {expected}")
    dest.write_bytes(data)
    return rel, expected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/abo")
    args = ap.parse_args()
    out_dir = Path(args.out)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda r: fetch(r, out_dir), FILES))
    total = sum(n for _, n in results)
    print(f"downloaded {len(results)}/{len(FILES)} files, {total / 1e6:.1f} MB into {out_dir}")


if __name__ == "__main__":
    main()
