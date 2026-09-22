"""LoRA / QLoRA fine-tuning of Qwen3.5-4B for attribute extraction (run inside WSL).

Each example is the evaluation prompt (image + instruction) followed by the
target JSON. Many rows lack a color or material label; those fields are still
written (with a placeholder value) so the output format stays fixed, but their
value tokens are masked out of the loss. Prompt tokens are masked as well.

Train rows are deduplicated on (image, labels): variants that share a main
image and labels would otherwise repeat the same example.
"""
import argparse
import collections
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, Trainer,
                          TrainerCallback, TrainingArguments)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vpa.prompting import ATTRIBUTES, build_prompt  # noqa: E402
from vpa.splits import load_split  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "out_proj"]


def sh(cmd):
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "unknown"


class AttrDataset(Dataset):
    def __init__(self, rows, image_dir, processor, instruction, placeholders, chat_kwargs):
        self.rows, self.image_dir, self.processor = rows, Path(image_dir), processor
        self.instruction, self.placeholders, self.chat_kwargs = instruction, placeholders, chat_kwargs
        tok = processor.tokenizer
        self.end_ids = tok("<|im_end|>\n", add_special_tokens=False)["input_ids"]

    def __len__(self):
        return len(self.rows)

    def _answer(self, row):
        """Token ids and labels for the target JSON, masking unlabeled values."""
        tok = self.processor.tokenizer
        pieces = []
        for i, attr in enumerate(ATTRIBUTES):
            key = ('{"' if i == 0 else '", "') + attr + '": "'
            value = row[attr] if row[attr] else self.placeholders[attr]
            pieces.append((key, True))
            pieces.append((value, row[attr] is not None))
        pieces.append(('"}', True))
        ids, labels = [], []
        for text, train_on in pieces:
            t = tok(text, add_special_tokens=False)["input_ids"]
            ids += t
            labels += t if train_on else [-100] * len(t)
        ids += self.end_ids
        labels += self.end_ids
        return ids, labels

    def __getitem__(self, i):
        row = self.rows[i]
        img = Image.open(self.image_dir / f"{row['image_id']}.jpg").convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.instruction}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                    **self.chat_kwargs)
        enc = self.processor(text=[prompt], images=[img], return_tensors="pt")
        ans_ids, ans_labels = self._answer(row)
        prompt_ids = enc["input_ids"][0].tolist()
        item = {
            "input_ids": torch.tensor(prompt_ids + ans_ids),
            "labels": torch.tensor([-100] * len(prompt_ids) + ans_labels),
        }
        for k, v in enc.items():
            if k in ("input_ids", "attention_mask"):
                continue
            if v.dim() == 2 and v.shape[0] == 1 and v.shape[1] == len(prompt_ids):
                # per-token side inputs (e.g. mm_token_type_ids): extend over the answer with zeros
                item[k] = torch.cat([v[0], torch.zeros(len(ans_ids), dtype=v.dtype)])
            else:
                item[k] = v
        return item


def collate(batch, pad_id):
    """Left padding, so every answer ends at the last position and the loss can
    be computed from the last few logits only."""
    n = max(len(b["input_ids"]) for b in batch)
    out = {
        "input_ids": torch.full((len(batch), n), pad_id, dtype=torch.long),
        "labels": torch.full((len(batch), n), -100, dtype=torch.long),
        "attention_mask": torch.zeros((len(batch), n), dtype=torch.long),
    }
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        out["input_ids"][i, n - L:] = b["input_ids"]
        out["labels"][i, n - L:] = b["labels"]
        out["attention_mask"][i, n - L:] = 1
    seq_keys = [k for k in batch[0] if k not in out and batch[0][k].dim() == 1
                and len(batch[0][k]) == len(batch[0]["input_ids"])]
    for k in seq_keys:
        t = torch.zeros((len(batch), n), dtype=batch[0][k].dtype)
        for i, b in enumerate(batch):
            t[i, n - len(b[k]):] = b[k]
        out[k] = t
    for k in batch[0]:
        if k not in out:
            out[k] = torch.cat([b[k] for b in batch], dim=0)
    return out


class AnswerLossTrainer(Trainer):
    """Cross-entropy on the answer span only.

    The vocabulary has about 248k entries, so full-sequence logits for a batch of
    8 x ~470 tokens cost several GB. With left padding the answers sit at the end,
    so only the last k positions are projected through lm_head.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # the loss below is normalized by num_items_in_batch over the whole
        # accumulated batch, so the Trainer must not divide it again
        self.model_accepts_loss_kwargs = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        has_label = labels != -100
        first = torch.where(has_label.any(1), has_label.float().argmax(1), labels.shape[1]).min().item()
        k = labels.shape[1] - int(first) + 1
        outputs = model(**inputs, logits_to_keep=k, use_cache=False)
        logits = outputs.logits[:, :-1, :].float()
        target = labels[:, -(k - 1):]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                                                 ignore_index=-100, reduction="sum")
        denom = num_items_in_batch if num_items_in_batch is not None else (target != -100).sum()
        loss = loss / denom
        return (loss, outputs) if return_outputs else loss


class ThroughputLog(TrainerCallback):
    def __init__(self):
        self.t0 = None

    def on_train_begin(self, args, state, control, **kw):
        self.t0 = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is not None and self.t0 and state.global_step:
            logs["examples_per_s"] = round(state.global_step * args.train_batch_size
                                           * args.gradient_accumulation_steps / (time.perf_counter() - self.t0), 2)
            logs["max_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quant", choices=["none", "4bit"], default="none")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=None, help="cap on train examples (for smoke tests)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chat-kwargs", default='{"enable_thinking": false}')
    ap.add_argument("--prompt", choices=["full", "short"], default="short")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_split("train")
    seen, uniq = set(), []
    for r in rows:
        key = (r["image_id"],) + tuple(r[a] for a in ATTRIBUTES)
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    random.shuffle(uniq)
    if args.max_train:
        uniq = uniq[: args.max_train]
    placeholders = {a: collections.Counter(r[a] for r in rows if r[a]).most_common(1)[0][0] for a in ATTRIBUTES}
    print(f"train rows {len(rows)}, unique (image, labels) used {len(uniq)}; "
          f"labeled color {sum(1 for r in uniq if r['color'])}, material {sum(1 for r in uniq if r['material'])}; "
          f"placeholders for unlabeled values {placeholders}", flush=True)

    product_types = json.load(open("data/processed/manifest.json"))["product_types"]
    processor = AutoProcessor.from_pretrained(args.model)
    ds = AttrDataset(uniq, args.images, processor, build_prompt(args.prompt, product_types), placeholders,
                     json.loads(args.chat_kwargs))

    quant_cfg = None
    if args.quant == "4bit":
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                       bnb_4bit_compute_dtype=torch.bfloat16,
                                       llm_int8_skip_modules=["visual", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                        quantization_config=quant_cfg)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if quant_cfg is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, target_modules=LORA_TARGETS,
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert not any(".visual." in n for n in trainable), "LoRA landed on the vision tower"
    model.print_trainable_parameters()

    steps_per_epoch = math.ceil(len(ds) / (args.batch_size * args.grad_accum))
    targs = TrainingArguments(
        output_dir=args.out, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, num_train_epochs=args.epochs,
        max_steps=args.max_steps, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * steps_per_epoch)),
        bf16=True, logging_steps=20, save_strategy="steps", save_steps=500, save_total_limit=2,
        report_to=[], seed=args.seed,
        gradient_checkpointing=not args.no_grad_ckpt, gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=args.workers, remove_unused_columns=False, optim="adamw_torch",
    )
    trainer = AnswerLossTrainer(model=model, args=targs, train_dataset=ds,
                      data_collator=lambda b: collate(b, processor.tokenizer.pad_token_id),
                      callbacks=[ThroughputLog()])
    t0 = time.perf_counter()
    result = trainer.train()
    train_s = time.perf_counter() - t0

    final = Path(args.out) / "final"
    model.save_pretrained(final)
    meta = {
        "base_model": args.model, "quant": args.quant, "lora": {"r": args.rank, "alpha": args.alpha,
        "dropout": args.dropout, "targets": LORA_TARGETS}, "lr": args.lr, "epochs": args.epochs,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum, "train_examples": len(ds),
        "prompt": args.prompt, "gradient_checkpointing": not args.no_grad_ckpt, "images": args.images,
        "steps": result.global_step, "train_loss": result.training_loss, "train_seconds": round(train_s),
        "examples_per_s": round(len(ds) * args.epochs / train_s, 2) if args.max_steps < 0 else None,
        "max_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "placeholders": placeholders, "git_commit": sh(["git", "rev-parse", "--short", "HEAD"]),
        "log_history": trainer.state.log_history,
    }
    (final / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "log_history"}, indent=2))


if __name__ == "__main__":
    main()
