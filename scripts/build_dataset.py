"""Build the attribute dataset from ABO listings with a leakage-safe grouped split.

Grouping: listings are linked when they share an item_id (same item in several
marketplaces) or any image. Images attached to more than --generic-threshold
listings (size charts, brand banners) are not used for linking; listings whose
main image is such a generic image are dropped instead.

Split: each group is assigned to train / val / test by a SHA-1 hash of its
smallest listing key, so the assignment does not depend on row order. Groups
larger than --max-eval-group listings always go to train; otherwise one chained
brand family (furniture linked through shared room photos) can end up as a
third of the test set.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vpa.abo import listing_key, load_listings, product_type  # noqa: E402
from vpa.labels import extract_labels  # noqa: E402

EXCLUDED_TYPES = {"CELLULAR_PHONE_CASE"}


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def split_of(group_key: str, val_pct: int, test_pct: int) -> str:
    bucket = int(hashlib.sha1(group_key.encode()).hexdigest(), 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "val"
    return "train"


def image_ids(listing: dict) -> set[str]:
    ids = set(listing.get("other_image_id", []))
    if listing.get("main_image_id"):
        ids.add(listing["main_image_id"])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abo-root", default="data/abo")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--generic-threshold", type=int, default=50)
    ap.add_argument("--min-type-count", type=int, default=300)
    ap.add_argument("--val-pct", type=int, default=10)
    ap.add_argument("--test-pct", type=int, default=10)
    ap.add_argument("--max-eval-group", type=int, default=500)
    args = ap.parse_args()

    listings = load_listings(args.abo_root)
    print(f"listings loaded: {len(listings)}")

    with gzip.open(Path(args.abo_root) / "images/metadata/images.csv.gz", "rt") as fh:
        images = {row["image_id"]: row for row in csv.DictReader(fh)}
    print(f"image metadata rows: {len(images)}")

    # Grouping over all listings, including excluded types, so that links through
    # phone cases or other types are not lost.
    img_users = collections.defaultdict(list)
    item_users = collections.defaultdict(list)
    for idx, lst in enumerate(listings):
        item_users[lst["item_id"]].append(idx)
        for img in image_ids(lst):
            img_users[img].append(idx)
    generic = {img for img, users in img_users.items() if len(users) > args.generic_threshold}
    print(f"images shared by > {args.generic_threshold} listings (not used for linking): {len(generic)}")

    uf = UnionFind(len(listings))
    for users in item_users.values():
        for other in users[1:]:
            uf.union(users[0], other)
    for img, users in img_users.items():
        if img in generic:
            continue
        for other in users[1:]:
            uf.union(users[0], other)

    members = collections.defaultdict(list)
    for idx in range(len(listings)):
        members[uf.find(idx)].append(idx)
    sizes = sorted((len(m) for m in members.values()), reverse=True)
    print(f"groups: {len(members)}; largest 5 sizes: {sizes[:5]}; singletons: {sum(1 for s in sizes if s == 1)}")

    group_split = {}
    forced = 0
    for root, idxs in members.items():
        gkey = min(listing_key(listings[i]) for i in idxs)
        if len(idxs) > args.max_eval_group:
            group_split[root] = (gkey, "train")
            forced += 1
        else:
            group_split[root] = (gkey, split_of(gkey, args.val_pct, args.test_pct))
    print(f"groups larger than {args.max_eval_group} listings forced to train: {forced}")

    # Candidate rows: one per listing, main image only.
    drop = collections.Counter()
    rows = []
    for idx, lst in enumerate(listings):
        pt = product_type(lst)
        main = lst.get("main_image_id")
        if pt is None:
            drop["no product_type"] += 1
            continue
        if pt in EXCLUDED_TYPES:
            drop["excluded type"] += 1
            continue
        if not main or main not in images:
            drop["no main image"] += 1
            continue
        if main in generic:
            drop["main image is generic"] += 1
            continue
        gkey, split = group_split[uf.find(idx)]
        names = [x["value"] for x in lst.get("item_name", []) if x.get("language_tag", "").startswith("en")]
        rows.append({
            "listing_key": listing_key(lst),
            "item_id": lst["item_id"],
            "group": gkey,
            "split": split,
            "image_id": main,
            "image_path": images[main]["path"],
            "image_hw": [int(images[main]["height"]), int(images[main]["width"])],
            "title": names[0] if names else None,
            **extract_labels(lst),
        })
    print(f"dropped: {dict(drop)}")

    # Product type vocabulary from the train split only.
    train_types = collections.Counter(r["product_type"] for r in rows if r["split"] == "train")
    vocab = sorted(t for t, c in train_types.items() if c >= args.min_type_count)
    before = len(rows)
    rows = [r for r in rows if r["product_type"] in vocab]
    print(f"product types with >= {args.min_type_count} train listings: {len(vocab)}; rows kept {len(rows)} of {before}")

    # Leakage checks on the final rows.
    by_split = collections.defaultdict(list)
    for r in rows:
        by_split[r["split"]].append(r)
    all_images_by_split = collections.defaultdict(set)
    items_by_split = collections.defaultdict(set)
    for idx, lst in enumerate(listings):
        split = group_split[uf.find(idx)][1]
        all_images_by_split[split] |= image_ids(lst) - generic
        items_by_split[split].add(lst["item_id"])
    for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:
        shared_img = all_images_by_split[a] & all_images_by_split[b]
        shared_item = items_by_split[a] & items_by_split[b]
        print(f"leak check {a}/{b}: shared non-generic images {len(shared_img)}, shared item_ids {len(shared_item)}")
        assert not shared_img and not shared_item

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"args": vars(args), "product_types": vocab, "splits": {}}
    for split in ["train", "val", "test"]:
        part = sorted(by_split[split], key=lambda r: r["listing_key"])
        part = [r for r in part if r["product_type"] in vocab]
        path = out / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for r in part:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        n_color = sum(1 for r in part if r["color"])
        n_material = sum(1 for r in part if r["material"])
        manifest["splits"][split] = {
            "rows": len(part), "groups": len({r["group"] for r in part}),
            "with_color": n_color, "with_material": n_material, "sha256": digest,
        }
        print(f"{split}: rows {len(part)}, groups {manifest['splits'][split]['groups']}, "
              f"with color {n_color}, with material {n_material}, sha256 {digest[:16]}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
