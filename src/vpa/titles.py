"""Product titles in any language, keyed by listing.

The processed splits only store the English title. About 29% of rows come
from non-English marketplaces, so text baselines look titles up here and fall
back to the first non-English item_name. The lookup is cached to a small JSON
file because parsing all listing shards takes several GB of RAM.
"""
import json
from pathlib import Path

from .abo import listing_key, load_listings

CACHE = Path("data/processed/titles_cache.json")


def load_titles(abo_root: str | Path = "data/abo", workers: int = 4) -> dict[str, tuple[str, str]]:
    """Map listing_key -> (title, language_tag), preferring an en_* title."""
    if CACHE.exists():
        return {k: tuple(v) for k, v in json.loads(CACHE.read_text(encoding="utf-8")).items()}
    out = {}
    for lst in load_listings(abo_root, workers=workers):
        names = lst.get("item_name", [])
        if not names:
            continue
        en = [n for n in names if n.get("language_tag", "").startswith("en")]
        pick = en[0] if en else names[0]
        if pick.get("value"):
            out[listing_key(lst)] = (pick["value"], pick.get("language_tag", ""))
    CACHE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out
