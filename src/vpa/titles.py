"""Product titles in any language, keyed by listing.

The processed splits only store the English title. About 29% of rows come
from non-English marketplaces, so text baselines look titles up here and fall
back to the first non-English item_name.
"""
from pathlib import Path

from .abo import listing_key, load_listings


def load_titles(abo_root: str | Path = "data/abo") -> dict[str, tuple[str, str]]:
    """Map listing_key -> (title, language_tag), preferring an en_* title."""
    out = {}
    for lst in load_listings(abo_root):
        names = lst.get("item_name", [])
        if not names:
            continue
        en = [n for n in names if n.get("language_tag", "").startswith("en")]
        pick = en[0] if en else names[0]
        if pick.get("value"):
            out[listing_key(lst)] = (pick["value"], pick.get("language_tag", ""))
    return out
