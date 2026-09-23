"""Build the README result tables straight from prediction files and reports,
so no number is copied by hand. Test scores are recomputed from each method's
predictions with the same scoring code; the inference table reads the val
benchmark reports and the vLLM logs (weights memory)."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vpa.metrics import score_both  # noqa: E402
from vpa.splits import load_split  # noqa: E402

TEST_METHODS = [
    ("Majority class", "-", "outputs/baselines/majority/test_predictions.jsonl"),
    ("CLIP ViT-L/14 zero-shot", "image", "outputs/baselines/clip_vitl14/zero_shot_test_predictions.jsonl"),
    ("CLIP ViT-L/14 linear probe", "image", "outputs/baselines/clip_vitl14/linear_probe_test_predictions.jsonl"),
    ("Qwen3.5-4B zero-shot", "image", "outputs/eval/qwen3.5-4b_zeroshot_img256/constrained_predictions.jsonl"),
    ("Qwen3.5-4B + QLoRA", "image", "outputs/eval/qlora_r16_short_test/constrained_predictions.jsonl"),
    ("Qwen3.5-4B + LoRA", "image", "outputs/eval/lora_r16_short_test/constrained_predictions.jsonl"),
    ("Title TF-IDF + logistic regression", "title", "outputs/baselines/text_tfidf_lr/test_predictions.jsonl"),
    ("Title TF-IDF + CLIP embedding, logistic regression", "image + title", "outputs/baselines/fusion_tfidf_clip/test_predictions.jsonl"),
    ("Qwen3.5-4B zero-shot", "image + title", "outputs/eval/zeroshot_title_test/constrained_predictions.jsonl"),
    ("Qwen3.5-4B + LoRA", "image + title", "outputs/eval/lora_title_test/constrained_predictions.jsonl"),
]

BENCH = [
    ("zs_full_768", "zero-shot, label lists in prompt, 768 px"),
    ("zs_full_256", "zero-shot, label lists in prompt, 256 px"),
    ("lora_adapter_256", "LoRA adapter served by vLLM, short prompt, 256 px"),
    ("lora_merged_256", "LoRA merged into bf16 weights, short prompt, 256 px"),
    ("lora_merged_256_fp8", "merged, FP8 weight quantization, short prompt, 256 px"),
]
LATENCY_RUN = ("lora_merged_256_seq1", "merged bf16, short prompt, 256 px, one request at a time (max_num_seqs=1)")


def load_preds(path):
    with open(path, encoding="utf-8") as fh:
        return {p["listing_key"]: p for p in map(json.loads, fh)}


def cell(m):
    return f"{m['accuracy']:.3f} ({m['macro_f1']:.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-logs", default=None, help="directory with one vLLM log per benchmark run")
    ap.add_argument("--out", default="docs/results.md")
    args = ap.parse_args()

    rows = load_split("test")
    lines = ["## Test results", "",
             "Frozen test split. Accuracy with macro-F1 in brackets, counted per row. Denominators: "
             "product_type 5,437 rows, color 1,610, material 888. VLM rows use JSON-schema constrained decoding.", "",
             "| Method | Input | product_type | color | material |", "|---|---|---|---|---|"]
    for name, inp, path in TEST_METHODS:
        if not Path(path).exists():
            lines.append(f"| {name} | {inp} | missing | missing | missing |")
            continue
        preds = load_preds(path)
        missing = sum(1 for r in rows if r["listing_key"] not in preds)
        assert missing == 0, f"{path}: {missing} rows without prediction"
        s = score_both(rows, [preds[r["listing_key"]] for r in rows])["rows"]
        lines.append(f"| {name} | {inp} | {cell(s['product_type'])} | {cell(s['color'])} | {cell(s['material'])} |")

    lines += ["", "## Inference trade-offs (val, 5,713 rows, one RTX 4090, vLLM 0.30.0)", "",
              "Accuracy per row on val. Weights memory is vLLM's reported model load size. "
              "Throughput is images per second over the whole split, with up to 128 concurrent requests, "
              "submitted in chunks of 512.", "",
              "| Setting | Prompt tokens | Weights memory | Images/s | product_type | color | material |",
              "|---|---|---|---|---|---|---|"]
    for key, label in BENCH:
        rep = Path("outputs/bench") / key / "report.json"
        if not rep.exists():
            lines.append(f"| {label} | missing | | | | | |")
            continue
        d = json.load(open(rep))
        ps = d["passes"]["constrained"]
        s = ps["scores"]["rows"]
        mem = ""
        if args.bench_logs:
            log = Path(args.bench_logs) / f"{key}.log"
            if log.exists():
                m = re.search(r"Model loading took ([0-9.]+) GiB", log.read_text(errors="ignore"))
                mem = f"{float(m.group(1)):.2f} GiB" if m else ""
        extra = f" (n={d['n_rows']})" if d["n_rows"] != 5713 else ""
        lines.append(f"| {label}{extra} | {ps['mean_prompt_tokens']:.0f} | {mem} | {ps['images_per_s']:.1f} | "
                     f"{s['product_type']['accuracy']:.3f} | {s['color']['accuracy']:.3f} | {s['material']['accuracy']:.3f} |")
    key, label = LATENCY_RUN
    rep = Path("outputs/bench") / key / "report.json"
    if rep.exists():
        d = json.load(open(rep))
        ps = d["passes"]["constrained"]
        lines += ["", f"Single-request latency, {label}: {1000 * ps['seconds'] / d['n_rows']:.0f} ms per image on average "
                      f"over the first {d['n_rows']} val images, including image preprocessing."]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
