"""CLIP baselines on image embeddings.

zero-shot:    cosine similarity between the image and a text prompt per class
              (averaged over a few templates)
linear probe: logistic regression per attribute on frozen train embeddings,
              regularization picked on val macro-F1

Embeddings are cached per split so the two baselines share one forward pass.
"""
import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vpa.labels import COLORS, MATERIALS  # noqa: E402
from vpa.metrics import fmt, score, score_both  # noqa: E402
from vpa.prompting import ATTRIBUTES  # noqa: E402
from vpa.splits import load_split  # noqa: E402

TYPE_NAMES = {
    "ACCESSORY": "fashion accessory", "BEAUTY": "beauty product", "BOOT": "boot",
    "CHAIR": "chair", "DRINKING_CUP": "drinking cup", "EARRING": "pair of earrings",
    "FINEEARRING": "pair of fine jewelry earrings", "FINENECKLACEBRACELETANKLET": "fine jewelry necklace, bracelet or anklet",
    "FINERING": "fine jewelry ring", "FLAT_SHEET": "bed sheet", "GROCERY": "grocery item",
    "HANDBAG": "handbag", "HARDWARE_HANDLE": "cabinet or door handle", "HAT": "hat",
    "HEALTH_PERSONAL_CARE": "health and personal care product", "HOME": "home product",
    "HOME_BED_AND_BATH": "bed and bath product", "HOME_FURNITURE_AND_DECOR": "piece of furniture or home decor",
    "JANITORIAL_SUPPLY": "cleaning supply", "KITCHEN": "kitchen item", "LAMP": "lamp",
    "LIGHT_BULB": "light bulb", "LIGHT_FIXTURE": "light fixture", "NECKLACE": "necklace",
    "OFFICE_PRODUCTS": "office product", "OTTOMAN": "ottoman", "OUTDOOR_LIVING": "outdoor living product",
    "PET_SUPPLIES": "pet supply", "RUG": "rug", "SANDAL": "sandal", "SHOES": "pair of shoes",
    "SOFA": "sofa", "SPORTING_GOODS": "sporting goods item", "STOOL_SEATING": "stool",
    "SUITCASE": "suitcase", "TABLE": "table", "WALL_ART": "piece of wall art",
}
TEMPLATES = {
    "product_type": ["a photo of a {}.", "a product photo of a {}.", "an e-commerce catalog image of a {}."],
    "color": ["a photo of a {} product.", "a product photo of something that is {}.", "a {} colored item."],
    "material": ["a photo of a product made of {}.", "a product photo of an item made of {}.", "a {} item."],
}
COLOR_WORDS = {"multicolor": "multicolored", "clear": "transparent"}


class ImageSet(Dataset):
    def __init__(self, rows, image_dir, processor):
        self.ids = [r["image_id"] for r in rows]
        self.image_dir, self.processor = Path(image_dir), processor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(self.image_dir / f"{self.ids[i]}.jpg").convert("RGB")
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


@torch.inference_mode()
def embed_images(model, processor, rows, image_dir, cache: Path) -> np.ndarray:
    if cache.exists():
        emb = np.load(cache)
        assert emb.shape[0] == len(rows), f"stale cache {cache}"
        return emb
    loader = DataLoader(ImageSet(rows, image_dir, processor), batch_size=128, num_workers=12)
    out = []
    for batch in loader:
        feats = model.get_image_features(pixel_values=batch.to("cuda", torch.float16))
        out.append(torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy())
    emb = np.concatenate(out)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    return emb


@torch.inference_mode()
def class_text_embeddings(model, processor, attr, classes) -> np.ndarray:
    if attr == "product_type":
        names = [TYPE_NAMES[c] for c in classes]
    elif attr == "color":
        names = [COLOR_WORDS.get(c, c) for c in classes]
    else:
        names = list(classes)
    embs = []
    for name in names:
        prompts = [t.format(name) for t in TEMPLATES[attr]]
        tok = processor(text=prompts, return_tensors="pt", padding=True).to("cuda")
        f = torch.nn.functional.normalize(model.get_text_features(**tok).float(), dim=-1).mean(0)
        embs.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    return np.stack(embs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="D:/Project/ml_cache/models/vlm/openai__clip-vit-large-patch14")
    ap.add_argument("--images", default="data/images/img256")
    ap.add_argument("--out", default="outputs/baselines/clip_vitl14")
    ap.add_argument("--cs", default="0.3,1,3,10,30")
    args = ap.parse_args()

    out = Path(args.out)
    model = CLIPModel.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda").eval()
    processor = CLIPProcessor.from_pretrained(args.model)
    product_types = json.load(open("data/processed/manifest.json"))["product_types"]
    classes = {"product_type": product_types, "color": COLORS, "material": MATERIALS}

    splits = {s: load_split(s) for s in ["train", "val", "test"]}
    emb = {s: embed_images(model, processor, rows, args.images, out / f"emb_{s}.npy") for s, rows in splits.items()}
    for s in emb:
        print(f"{s}: embeddings {emb[s].shape}")

    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    report = {"model": args.model, "images": args.images, "git_commit": commit, "zero_shot": {}, "linear_probe": {}}

    # Zero-shot.
    zs = {s: [{} for _ in splits[s]] for s in ["val", "test"]}
    for attr in ATTRIBUTES:
        text = class_text_embeddings(model, processor, attr, classes[attr])
        for s in zs:
            best = (emb[s] @ text.T).argmax(1)
            for i, k in enumerate(best):
                zs[s][i][attr] = classes[attr][k]
    for s in zs:
        report["zero_shot"][s] = score_both(splits[s], zs[s])

    # Linear probe.
    lp = {s: [{} for _ in splits[s]] for s in ["val", "test"]}
    chosen = {}
    for attr in ATTRIBUTES:
        idx = [i for i, r in enumerate(splits["train"]) if r[attr]]
        y = [splits["train"][i][attr] for i in idx]
        cs = [float(x) for x in args.cs.split(",")]
        fitted = Parallel(n_jobs=len(cs))(
            delayed(LogisticRegression(C=c, max_iter=3000).fit)(emb["train"][idx], y) for c in cs)
        best = None
        for c, clf in zip(cs, fitted):
            vp = [{attr: p} for p in clf.predict(emb["val"])]
            f1 = score(splits["val"], vp)[attr]["macro_f1"]
            print(f"  probe {attr}: C={c} train n={len(idx)} val macro_f1={f1:.4f}", flush=True)
            if best is None or f1 > best[0]:
                best = (f1, c, clf)
        chosen[attr] = {"C": best[1], "val_macro_f1": best[0], "train_n": len(idx)}
        for s in lp:
            for i, p in enumerate(best[2].predict(emb[s])):
                lp[s][i][attr] = p
    report["linear_probe_chosen"] = chosen
    for s in lp:
        report["linear_probe"][s] = score_both(splits[s], lp[s])

    for name, preds in [("zero_shot", zs), ("linear_probe", lp)]:
        for s in preds:
            with open(out / f"{name}_{s}_predictions.jsonl", "w", encoding="utf-8") as fh:
                for r, p in zip(splits[s], preds[s]):
                    fh.write(json.dumps({"listing_key": r["listing_key"], **p}) + "\n")
            for view, scores in report[name][s].items():
                for attr, m in scores.items():
                    print(f"[clip {name}] {s} [{view}] {attr}: "
                          f"{fmt(m)}")
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
