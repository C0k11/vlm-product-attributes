"""Title-only baseline: TF-IDF (word and character n-grams) + logistic regression.

One classifier per attribute, trained on train rows that carry that label. The
regularization strength is picked on val macro-F1; test is scored once with
the chosen setting. Rows without any title get the train majority class.
"""
import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vpa.metrics import fmt, score, score_both  # noqa: E402
from vpa.prompting import ATTRIBUTES  # noqa: E402
from vpa.splits import load_split  # noqa: E402
from vpa.titles import load_titles  # noqa: E402


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/baselines/text_tfidf_lr")
    ap.add_argument("--cs", default="0.3,1,3,10,30")
    args = ap.parse_args()

    titles = load_titles()
    splits = {s: load_split(s) for s in ["train", "val", "test"]}
    text = {s: [titles.get(r["listing_key"], (None, None))[0] for r in rows] for s, rows in splits.items()}
    for s, rows in splits.items():
        n_title = sum(1 for t in text[s] if t)
        print(f"{s}: rows {len(rows)}, with a title in any language {n_title}")

    word = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                           sublinear_tf=True, max_features=200_000)
    train_text = [t or "" for t in text["train"]]
    word.fit(train_text)
    char.fit(train_text)

    def feats(texts):
        texts = [t or "" for t in texts]
        return hstack([word.transform(texts), char.transform(texts)]).tocsr()

    X = {s: feats(text[s]) for s in splits}
    preds = {s: [{} for _ in splits[s]] for s in ["val", "test"]}
    chosen = {}
    for attr in ATTRIBUTES:
        idx = [i for i, r in enumerate(splits["train"]) if r[attr] and text["train"][i]]
        y = [splits["train"][i][attr] for i in idx]
        majority = collections.Counter(r[attr] for r in splits["train"] if r[attr]).most_common(1)[0][0]
        cs = [float(x) for x in args.cs.split(",")]
        fitted = Parallel(n_jobs=min(3, len(cs)), prefer="threads")(
            delayed(LogisticRegression(C=c, max_iter=3000).fit)(X["train"][idx], y) for c in cs)
        best = None
        for c, clf in zip(cs, fitted):
            val_p = [{attr: p if text["val"][i] else majority} for i, p in enumerate(clf.predict(X["val"]))]
            f1 = score(splits["val"], val_p)[attr]["macro_f1"]
            print(f"  {attr}: C={c} train n={len(idx)} val macro_f1={f1:.4f}", flush=True)
            if best is None or f1 > best[0]:
                best = (f1, c, clf)
        chosen[attr] = {"C": best[1], "val_macro_f1": best[0], "train_n": len(idx), "majority": majority}
        for s in ["val", "test"]:
            for i, p in enumerate(best[2].predict(X[s])):
                preds[s][i][attr] = p if text[s][i] else majority

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"method": "tfidf word(1-2)+char_wb(2-5) + logistic regression, title only",
              "git_commit": git_commit(), "chosen": chosen, "results": {}}
    for s in ["val", "test"]:
        both = score_both(splits[s], preds[s])
        report["results"][s] = both
        res = both["rows"]
        with open(out / f"{s}_predictions.jsonl", "w", encoding="utf-8") as fh:
            for r, p in zip(splits[s], preds[s]):
                fh.write(json.dumps({"listing_key": r["listing_key"], **p}) + "\n")
        for attr, m in res.items():
            print(f"[text baseline] {s} [rows] {attr}: {fmt(m)}")
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
