"""Zero-shot pilot with Hugging Face transformers (bf16, greedy, batched).

Used when a model does not fit under vLLM next to the memory other Windows
applications hold on the GPU. Same prompt, images and scoring as run_pilot.py.
Throughput from this script is not comparable to vLLM numbers.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vpa.metrics import score  # noqa: E402
from vpa.prompting import build_instruction, parse_output  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def model_revision(model_dir: Path) -> str | None:
    meta = model_dir / ".cache/huggingface/download/config.json.metadata"
    return meta.read_text().splitlines()[0] if meta.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--pilot", default="data/pilot/pilot.jsonl")
    ap.add_argument("--images", default="data/pilot/img768")
    ap.add_argument("--manifest", default="data/processed/manifest.json")
    ap.add_argument("--out", default="outputs/pilot_hf")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--chat-kwargs", default="{}")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.pilot, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    product_types = json.load(open(args.manifest))["product_types"]
    instruction = build_instruction(product_types)
    chat_kwargs = json.loads(args.chat_kwargs)
    image_dir = Path(args.images)

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    t_load = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    load_s = time.perf_counter() - t_load
    weights_gib = torch.cuda.memory_allocated() / 2**30
    print(f"loaded {args.name} in {load_s:.0f}s, weights on GPU {weights_gib:.2f} GiB, rows {len(rows)}")

    records, parsed = [], []
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start: start + args.batch_size]
        conversations = [
            [{"role": "user", "content": [
                {"type": "image", "image": Image.open(image_dir / f"{r['image_id']}.jpg").convert("RGB")},
                {"type": "text", "text": instruction},
            ]}]
            for r in batch
        ]
        inputs = processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", padding=True, **chat_kwargs,
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        texts = processor.batch_decode(new_tokens, skip_special_tokens=True)
        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
        for r, text, plen, toks in zip(batch, texts, prompt_lens, new_tokens):
            obj, status = parse_output(text)
            parsed.append(obj)
            pad_id = processor.tokenizer.pad_token_id
            records.append({
                "listing_key": r["listing_key"], "text": text, "status": status,
                "prompt_tokens": int(plen), "output_tokens": int((toks != pad_id).sum()),
            })
        print(f"  {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    elapsed = time.perf_counter() - t0
    peak_gib = torch.cuda.max_memory_allocated() / 2**30

    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "free.jsonl", "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n = len(rows)
    statuses = [r["status"] for r in records]
    summary = {
        "model": args.name, "model_dir": args.model, "model_revision": model_revision(Path(args.model)),
        "git_commit": git_commit(), "framework": "transformers", "dtype": "bfloat16",
        "pilot_file": args.pilot, "images": str(image_dir), "n": n, "batch_size": args.batch_size,
        "chat_template_kwargs": chat_kwargs,
        "json_strict": statuses.count("strict"), "json_fenced": statuses.count("fenced"),
        "json_invalid": statuses.count("invalid"),
        "seconds": round(elapsed, 2), "images_per_s_hf": round(n / elapsed, 2),
        "weights_gib": round(weights_gib, 2), "peak_allocated_gib": round(peak_gib, 2),
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in records) / n, 1),
        "scores": score(rows, parsed),
    }
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2))
    print(f"[{args.name}] hf n={n} strict={summary['json_strict']} fenced={summary['json_fenced']} "
          f"invalid={summary['json_invalid']} img/s(hf)={summary['images_per_s_hf']} "
          f"peak_alloc={summary['peak_allocated_gib']} GiB prompt_tok={summary['mean_prompt_tokens']}")
    for attr, s in summary["scores"].items():
        acc = f"{s['accuracy']:.3f}" if s["accuracy"] is not None else "n/a"
        f1 = f"{s['macro_f1']:.3f}" if s["macro_f1"] is not None else "n/a"
        print(f"    {attr}: {s['correct']}/{s['n']} acc={acc} macro_f1={f1}")


if __name__ == "__main__":
    main()
