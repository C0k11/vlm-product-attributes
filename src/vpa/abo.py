"""Loading helpers for the Amazon Berkeley Objects listing metadata."""
import gzip
import json
from multiprocessing import Pool
from pathlib import Path

LISTINGS_GLOB = "listings/metadata/listings_*.json.gz"


def en_values(listing: dict, key: str, sub: str = "value") -> list[str]:
    """Return values of a multilingual field whose language tag starts with 'en'."""
    out = []
    for entry in listing.get(key, []):
        if not entry.get("language_tag", "").startswith("en"):
            continue
        val = entry.get(sub)
        if isinstance(val, list):
            out.extend(v for v in val if v)
        elif val:
            out.append(val)
    return out


def product_type(listing: dict) -> str | None:
    pt = listing.get("product_type")
    return pt[0]["value"] if pt else None


def listing_key(listing: dict) -> str:
    return f"{listing['item_id']}|{listing.get('domain_name', '')}"


def _read_shard(path: str) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def load_listings(abo_root: str | Path, workers: int = 16) -> list[dict]:
    paths = sorted(str(p) for p in Path(abo_root).glob(LISTINGS_GLOB))
    if not paths:
        raise FileNotFoundError(f"no listing shards under {abo_root}")
    with Pool(min(workers, len(paths))) as pool:
        shards = pool.map(_read_shard, paths)
    return [row for shard in shards for row in shard]
