"""Image + title late-fusion baseline: title TF-IDF features concatenated with
frozen CLIP ViT-L/14 image embeddings, logistic regression per attribute.

Reuses the CLIP embeddings cached by baseline_clip.py. The image block is scaled
by --img-weight so the dense 768-d embedding is not swamped by the sparse text
block; the weight and C are picked on val macro-F1.
"""
import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vpa.metrics import fmt, score, score_both  # noqa: E402
from vpa.prompting import ATTRIBUTES  # noqa: E402
from vpa.splits import load_split  # noqa: E402
from vpa.titles import load_titles  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="outputs/baselines/clip_vitl14")
    ap.add_argument("--out", default="outputs/baselines/fusion_tfidf_clip")
    ap.add_argument("--cs", default="3,10,30")
    ap.add_argument("--img-weights", default="0.5,1,2")
    args = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()

    titles = load_titles()
    splits = {s: load_split(s) for s in ["train", "val", "test"]}
    text = {s: [titles.get(r["listing_key"], ("",))[0] or "" for r in rows] for s, rows in splits.items()}
    emb = {s: np.load(Path(args.emb) / f"emb_{s}.npy") for s in splits}
    for s in splits:
        assert emb[s].shape[0] == len(splits[s])

    word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(text["train"])
    char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                           max_features=200_000).fit(text["train"])
    T = {s: hstack([word.transform(text[s]), char.transform(text[s])]).tocsr() for s in splits}

    def feats(s, w):
        return hstack([T[s], csr_matrix(emb[s] * w)]).tocsr()

    preds = {s: [{} for _ in splits[s]] for s in ["val", "test"]}
    chosen = {}
    grid = [(float(c), float(w)) for c in args.cs.split(",") for w in args.img_weights.split(",")]
    for attr in ATTRIBUTES:
        idx = [i for i, r in enumerate(splits["train"]) if r[attr]]
        y = [splits["train"][i][attr] for i in idx]
        Xtr = {w: feats("train", w)[idx] for _, w in grid}
        fitted = Parallel(n_jobs=3, prefer="threads")(
            delayed(LogisticRegression(C=c, max_iter=3000).fit)(Xtr[w], y) for c, w in grid)
        best = None
        for (c, w), clf in zip(grid, fitted):
            vp = [{attr: p} for p in clf.predict(feats("val", w))]
            f1 = score(splits["val"], vp)[attr]["macro_f1"]
            print(f"  {attr}: C={c} img_weight={w} train n={len(idx)} val macro_f1={f1:.4f}", flush=True)
            if best is None or f1 > best[0]:
                best = (f1, c, w, clf)
        chosen[attr] = {"C": best[1], "img_weight": best[2], "val_macro_f1": best[0], "train_n": len(idx)}
        for s in preds:
            for i, p in enumerate(best[3].predict(feats(s, best[2]))):
                preds[s][i][attr] = p

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"method": "tfidf(title) + CLIP ViT-L/14 image embedding, logistic regression", "git_commit": commit,
              "chosen": chosen, "results": {}}
    for s in preds:
        report["results"][s] = score_both(splits[s], preds[s])
        with open(out / f"{s}_predictions.jsonl", "w", encoding="utf-8") as fh:
            for r, p in zip(splits[s], preds[s]):
                fh.write(json.dumps({"listing_key": r["listing_key"], **p}) + "\n")
        for attr, m in report["results"][s]["rows"].items():
            print(f"[fusion] {s} [rows] {attr}: {fmt(m)}")
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
