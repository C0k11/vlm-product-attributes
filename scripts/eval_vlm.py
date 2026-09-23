"""Evaluate a VLM (optionally with a LoRA adapter) on a split under vLLM.

Writes raw outputs and a unified predictions file per pass, then scores at row
level and unique-image level. Run inside WSL.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vpa.metrics import fmt, score_both  # noqa: E402
from vpa.prompting import ATTRIBUTES, build_prompt, json_schema, parse_output  # noqa: E402
from vpa.splits import load_split  # noqa: E402


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "unknown"


class GpuSampler(threading.Thread):
    """Peak total GPU memory in use (MiB), sampled from nvidia-smi."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval, self.peak, self._halt = interval, 0, threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            out = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            if out.split()[0].isdigit():
                self.peak = max(self.peak, int(out.split()[0]))
            time.sleep(self.interval)

    def stop(self) -> int:
        self._halt.set()
        self.join()
        return self.peak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--images", default="data/images/img768")
    ap.add_argument("--lora", default=None, help="LoRA adapter directory")
    ap.add_argument("--with-title", action="store_true")
    ap.add_argument("--prompt", choices=["full", "short"], default="full")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--passes", default="constrained")
    ap.add_argument("--gpu-mem", type=float, default=0.72)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--load-strategy", default=None)
    ap.add_argument("--chat-kwargs", default="{}")
    ap.add_argument("--mm-limits", default='{"image": 1}')
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quantization", default=None, help="e.g. fp8 (online weight quantization)")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--chunk", type=int, default=512,
                    help="requests submitted per llm.chat call; bounds the preprocessed image tensors held in RAM")
    args = ap.parse_args()
    commit = sh(["git", "rev-parse", "--short", "HEAD"])
    gpu_before = sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

    rows = load_split(args.split)
    if args.limit:
        rows = rows[: args.limit]
    product_types = json.load(open("data/processed/manifest.json"))["product_types"]
    image_dir = Path(args.images).resolve()
    missing = [r["image_id"] for r in rows if not (image_dir / f"{r['image_id']}.jpg").exists()]
    print(f"split {args.split}: rows {len(rows)}, missing images {len(missing)}")
    assert not missing, missing[:5]

    titles = {}
    if args.with_title:
        from vpa.titles import load_titles
        titles = {k: v[0] for k, v in load_titles().items()}

    llm_kwargs = dict(
        model=args.model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
        limit_mm_per_prompt=json.loads(args.mm_limits), allowed_local_media_path=str(image_dir.parent.parent),
        mm_processor_cache_gb=0, max_num_seqs=args.max_num_seqs, seed=0,
        quantization=args.quantization, enforce_eager=args.enforce_eager,
    )
    if args.load_strategy:
        llm_kwargs["safetensors_load_strategy"] = args.load_strategy
    lora_req = None
    if args.lora:
        cfg = json.load(open(Path(args.lora) / "adapter_config.json"))
        llm_kwargs.update(enable_lora=True, max_lora_rank=cfg["r"], max_loras=1)
        lora_req = LoRARequest("adapter", 1, str(Path(args.lora).resolve()))
    llm = LLM(**llm_kwargs)

    conversations = [
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"file://{image_dir / (r['image_id'] + '.jpg')}"}},
            {"type": "text", "text": build_prompt(args.prompt, product_types,
                                                  titles.get(r["listing_key"]) if args.with_title else None)},
        ]}]
        for r in rows
    ]
    chat_kwargs = json.loads(args.chat_kwargs)
    llm.chat(conversations[:4], SamplingParams(temperature=0, max_tokens=8), chat_template_kwargs=chat_kwargs,
             use_tqdm=False, lora_request=lora_req)

    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "name": args.name, "model": args.model, "lora": args.lora, "split": args.split, "n_rows": len(rows),
        "images": str(image_dir), "with_title": args.with_title, "prompt": args.prompt, "git_commit": commit, "quantization": args.quantization,
        "gpu_used_by_others_mib": gpu_before,
        "vllm_config": {"gpu_memory_utilization": args.gpu_mem, "max_num_seqs": args.max_num_seqs,
                        "max_model_len": args.max_model_len, "chunk": args.chunk},
        "chat_template_kwargs": chat_kwargs, "passes": {},
    }
    for name in args.passes.split(","):
        extra = {}
        if name == "constrained":
            extra["structured_outputs"] = StructuredOutputsParams(json=json_schema(product_types))
        params = SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=0, **extra)
        sampler = GpuSampler()
        sampler.start()
        t0 = time.perf_counter()
        results = []
        for start in range(0, len(conversations), args.chunk):
            results += llm.chat(conversations[start:start + args.chunk], params, chat_template_kwargs=chat_kwargs,
                                use_tqdm=False, lora_request=lora_req)
        elapsed = time.perf_counter() - t0
        peak_mib = sampler.stop()
        preds, statuses = [], []
        with open(out_dir / f"{name}_raw.jsonl", "w", encoding="utf-8") as raw_fh, \
                open(out_dir / f"{name}_predictions.jsonl", "w", encoding="utf-8") as pred_fh:
            for r, res in zip(rows, results):
                text = res.outputs[0].text
                obj, status = parse_output(text)
                preds.append(obj)
                statuses.append(status)
                raw_fh.write(json.dumps({"listing_key": r["listing_key"], "text": text, "status": status,
                                         "prompt_tokens": len(res.prompt_token_ids),
                                         "output_tokens": len(res.outputs[0].token_ids)}) + "\n")
                pred_fh.write(json.dumps({"listing_key": r["listing_key"],
                                          **{a: (obj or {}).get(a) for a in ATTRIBUTES}}) + "\n")
        n = len(rows)
        summary = {
            "json_strict": statuses.count("strict"), "json_fenced": statuses.count("fenced"),
            "json_invalid": statuses.count("invalid"), "seconds": round(elapsed, 1),
            "images_per_s": round(n / elapsed, 2), "peak_gpu_used_mib": peak_mib,
            "mean_prompt_tokens": round(sum(len(r.prompt_token_ids) for r in results) / n, 1),
            "scores": score_both(rows, preds),
        }
        report["passes"][name] = summary
        print(f"[{args.name}] {args.split} pass={name} n={n} strict={summary['json_strict']} "
              f"fenced={summary['json_fenced']} invalid={summary['json_invalid']} img/s={summary['images_per_s']} "
              f"prompt_tok={summary['mean_prompt_tokens']} peak_gpu_mib={peak_mib}")
        for view, scores in summary["scores"].items():
            for attr, m in scores.items():
                print(f"    [{view}] {attr}: {fmt(m)}")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
