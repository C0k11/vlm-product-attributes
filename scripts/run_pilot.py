"""Zero-shot pilot for one VLM under vLLM (run inside WSL).

Two passes over the same rows:
  free:        no decoding constraint, measures how often the model emits valid JSON
  constrained: JSON-schema constrained decoding, measures accuracy and throughput
The multimodal preprocessing cache is disabled so the timed pass does not reuse
work from the first pass.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vpa.metrics import score  # noqa: E402
from vpa.prompting import build_instruction, json_schema, parse_output  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def model_revision(model_dir: Path) -> str | None:
    meta = model_dir / ".cache/huggingface/download/config.json.metadata"
    return meta.read_text().splitlines()[0] if meta.exists() else None


class GpuSampler(threading.Thread):
    """Samples total used GPU memory (MiB) from nvidia-smi."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval, self.peak, self._halt = interval, 0, threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout.strip()
            if out:
                self.peak = max(self.peak, int(out.splitlines()[0]))
            time.sleep(self.interval)

    def stop(self) -> int:
        self._halt.set()
        self.join()
        return self.peak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local model directory")
    ap.add_argument("--name", required=True)
    ap.add_argument("--pilot", default="data/pilot/pilot.jsonl")
    ap.add_argument("--images", default="data/pilot/img768")
    ap.add_argument("--manifest", default="data/processed/manifest.json")
    ap.add_argument("--out", default="outputs/pilot")
    ap.add_argument("--gpu-mem", type=float, default=0.72,
                    help="fraction of total GPU memory for vLLM; Windows apps hold part of the card")
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--passes", default="free,constrained")
    ap.add_argument("--chat-kwargs", default="{}", help="JSON dict passed as chat_template_kwargs")
    ap.add_argument("--mm-limits", default='{"image": 1}')
    ap.add_argument("--load-strategy", default="eager",
                    help="safetensors load strategy; eager reads each shard sequentially, "
                         "much faster than mmap when weights sit on a Windows drive under WSL")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    rows = [json.loads(line) for line in open(args.pilot, encoding="utf-8")]
    product_types = json.load(open(args.manifest))["product_types"]
    image_dir = Path(args.images).resolve()
    print(f"rows: {len(rows)}; product types in prompt: {len(product_types)}; images: {image_dir}")
    baseline_mib = int(subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.split()[0])
    print(f"GPU memory in use before loading (other processes): {baseline_mib} MiB")

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        limit_mm_per_prompt=json.loads(args.mm_limits),
        allowed_local_media_path=str(image_dir.parent.parent),
        mm_processor_cache_gb=0,
        safetensors_load_strategy=args.load_strategy,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        quantization=args.quantization,
        seed=0,
    )

    instruction = build_instruction(product_types)
    conversations = [
        [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"file://{image_dir / (r['image_id'] + '.jpg')}"}},
                {"type": "text", "text": instruction},
            ],
        }]
        for r in rows
    ]
    chat_kwargs = json.loads(args.chat_kwargs)

    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.name,
        "model_dir": args.model,
        "model_revision": model_revision(Path(args.model)),
        "git_commit": git_commit(),
        "pilot_file": args.pilot,
        "images": str(image_dir),
        "n_rows": len(rows),
        "gpu_memory_utilization": args.gpu_mem,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enforce_eager": args.enforce_eager,
        "quantization": args.quantization,
        "gpu_used_by_others_mib": baseline_mib,
        "chat_template_kwargs": chat_kwargs,
        "passes": {},
    }

    # Warm-up on a few rows so CUDA graph capture and compilation are not timed.
    llm.chat(conversations[:4], SamplingParams(temperature=0, max_tokens=8),
             chat_template_kwargs=chat_kwargs, use_tqdm=False)

    for name in args.passes.split(","):
        params = SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=0)
        if name == "constrained":
            params = SamplingParams(
                temperature=0, max_tokens=args.max_tokens, seed=0,
                structured_outputs=StructuredOutputsParams(json=json_schema(product_types)),
            )
        sampler = GpuSampler()
        sampler.start()
        t0 = time.perf_counter()
        results = llm.chat(conversations, params, chat_template_kwargs=chat_kwargs, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        peak = sampler.stop()

        parsed, statuses, records = [], [], []
        for r, res in zip(rows, results):
            text = res.outputs[0].text
            obj, status = parse_output(text)
            parsed.append(obj)
            statuses.append(status)
            records.append({
                "listing_key": r["listing_key"], "text": text, "status": status,
                "prompt_tokens": len(res.prompt_token_ids), "output_tokens": len(res.outputs[0].token_ids),
                "finish_reason": res.outputs[0].finish_reason,
            })
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        n = len(rows)
        summary = {
            "n": n,
            "json_strict": sum(s == "strict" for s in statuses),
            "json_fenced": sum(s == "fenced" for s in statuses),
            "json_invalid": sum(s == "invalid" for s in statuses),
            "seconds": round(elapsed, 2),
            "images_per_s": round(n / elapsed, 2),
            "mean_prompt_tokens": round(sum(x["prompt_tokens"] for x in records) / n, 1),
            "mean_output_tokens": round(sum(x["output_tokens"] for x in records) / n, 1),
            "finish_length": sum(x["finish_reason"] == "length" for x in records),
            "peak_gpu_used_mib": peak,
            "scores": score(rows, parsed),
        }
        report["passes"][name] = summary
        print(f"[{args.name}] pass={name} n={n} strict={summary['json_strict']} fenced={summary['json_fenced']} "
              f"invalid={summary['json_invalid']} img/s={summary['images_per_s']} "
              f"prompt_tok={summary['mean_prompt_tokens']} out_tok={summary['mean_output_tokens']} peak_mib={peak}")
        for attr, s in summary["scores"].items():
            acc = f"{s['accuracy']:.3f}" if s["accuracy"] is not None else "n/a"
            f1 = f"{s['macro_f1']:.3f}" if s["macro_f1"] is not None else "n/a"
            print(f"    {attr}: {s['correct']}/{s['n']} acc={acc} macro_f1={f1}")

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
