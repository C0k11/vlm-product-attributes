"""Merge a LoRA adapter into the base weights and save a standalone bf16 model,
so it can be served without per-request LoRA computation (run inside WSL)."""
import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    merged = model.merge_and_unload()
    out = Path(args.out)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(args.base).save_pretrained(out)
    # keep the chat template and preprocessor files exactly as shipped with the base model
    for name in ["chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json",
                 "tokenizer_config.json", "tokenizer.json"]:
        src = Path(args.base) / name
        if src.exists():
            shutil.copy(src, out / name)
    meta = json.load(open(Path(args.adapter) / "train_meta.json"))
    (out / "merge_meta.json").write_text(json.dumps({"base": args.base, "adapter": args.adapter,
                                                     "adapter_git_commit": meta.get("git_commit")}, indent=2))
    print(f"merged model saved to {out}")


if __name__ == "__main__":
    main()
