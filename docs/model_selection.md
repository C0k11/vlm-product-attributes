# Step 0: dataset check and model survey

Date: 2026-09-22. Sections 1 and 2 come from official pages (links inline); items marked "measured" were counted by a script on the listed files. Section 4 holds the measured pilot results.

## 1. Dataset: Amazon Berkeley Objects (ABO)

Sources: [ABO index](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), [README](https://amazon-berkeley-objects.s3.amazonaws.com/README.md), [listings README](https://amazon-berkeley-objects.s3.amazonaws.com/listings/README.md), [images README](https://amazon-berkeley-objects.s3.amazonaws.com/images/README.md), [paper](https://arxiv.org/abs/2110.06199).

- License: the LICENSE file shipped with the data is CC BY 4.0 (`LICENSE-CC-BY-4.0.txt`). The paper text says CC BY-NC 4.0. We follow the shipped file, credit "Amazon.com", and cite the CVPR 2022 paper. A portfolio project is fine under either reading.
- Scale (official): 147,702 listings, 398,212 unique catalog images, 576 product types (paper).
- Files (HTTP Content-Length):
  - `archives/abo-listings.tar`: 87,480,320 bytes (16 NDJSON shards, `listings/metadata/listings_<0-f>.json.gz`)
  - `images/metadata/images.csv.gz`: about 6.4 MB (`image_id,height,width,path`)
  - `archives/abo-images-small.tar`: 3,253,381,120 bytes, longest side at most 256 px
  - `archives/abo-images-original.tar`: about 118 GB. Individual originals can be fetched from `images/original/<path>`, so the full archive is not needed.
  - Measured on a random sample of 30 images: mean original file 378 KB, mean small file 7.9 KB.
- Linking: `main_image_id` / `other_image_id` in a listing point to `image_id` in `images.csv.gz`. A listing is keyed by (`item_id`, `domain_name`). There is no parent or variation id.

### Label coverage (measured, shards 0, 7, f; n = 27,686 listings)

| Field | Present (any language) | Present (en_* tag) |
|---|---|---|
| product_type | 100% | n/a |
| color | 78.8% | 66.7% |
| material | 36.2% | 31.5% |
| style | 29.3% | 20.6% |
| fabric_type | 5.6% | 3.7% |
| item_shape | 3.5% | 2.7% |
| pattern | 3.0% | 2.2% |

- `CELLULAR_PHONE_CASE` is 44.0% of these listings (12,190 / 27,686). It has to be excluded or capped, otherwise one class dominates every metric.
- Color is seller free text. The `standardized_values` field has 66 distinct English values in these shards, with case duplicates (Black / black, Grey / Gray) and a large "multi-colored" bucket. It needs mapping to a closed set.
- `style`, `pattern`, and `item_shape` are either sparse or not what the name says (top `style` values include "Running Shoes" and "Quotes"). Proposed label set: product_type, color, material.

### Leakage

Images are shared across listings (the same shoe photo reused for several sizes; generic secondary images reused across many phone cases). Splitting by `item_id` alone is not enough. Plan: union-find over listings that share any `image_id` or `item_id`, then split by group. The test split is frozen once written.

## 2. Small VLM survey (10B params or less)

Params are safetensors totals from the Hugging Face API. Download size is the sum of repo files. vLLM column is from the [supported models source](https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md) (checked 2026-09-22). None of the repos below are gated.

| Model | Released | Params | Download | License | vLLM (LoRA serving) | Fine-tune support | Notes |
|---|---|---|---|---|---|---|---|
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 2026-03 | 4.66B | 9.34 GB | Apache-2.0 | yes (yes) | LLaMA-Factory, ms-swift, Unsloth | Thinking on by default, disable with `enable_thinking=False`. [Unsloth](https://unsloth.ai/docs/models/qwen3.5/fine-tune) advises against 4-bit QLoRA on Qwen3.5; bf16 LoRA about 10 GB. Needs transformers >= 5.2, vLLM >= 0.17. |
| [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 2026-03 | 2.27B | 4.57 GB | Apache-2.0 | yes (yes) | same | Non-thinking by default. Speed option. |
| [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 2025-10 | 8.77B | 17.55 GB | Apache-2.0 | yes (yes) | official scripts, LLaMA-Factory, ms-swift, Unsloth (4-bit QLoRA notebook) | Established QLoRA path. |
| [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | 2026-04 | 8.00B total (4.5B effective) | 16.02 GB | Apache-2.0 | yes (yes) | LLaMA-Factory, ms-swift, Unsloth (QLoRA about 10 GB) | Google states native structured JSON output. Image token budget 70 to 1120. |
| [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | 2026-04 | 5.12B total | about 10 GB (computed) | Apache-2.0 | yes (yes) | same | |
| [OpenGVLab/InternVL3_5-4B / 8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) | 2025-08 | 4.7B / 8.5B | about 9.5 / 17 GB (computed) | Apache-2.0 | yes (yes) | LLaMA-Factory (-HF ids), ms-swift | Lower MMBench than Qwen3.5 at similar size (paper). |
| [openbmb/MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) | 2026-05 | 1.30B | about 2.6 GB (computed) | Apache-2.0 | yes (yes) | LLaMA-Factory, ms-swift | Very cheap; needs transformers >= 5.7. |
| [LiquidAI/LFM2.5-VL-1.6B-Extract](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-Extract) | 2026-05 | 1.60B | 3.33 GB | LFM Open License 1.0 (free under $10M revenue) | yes (yes) | LLaMA-Factory, TRL, Unsloth | Trained specifically for image to JSON field extraction. |
| [HuggingFaceTB/SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | 2025-02 | 2.25B | about 4.5 GB (computed) | Apache-2.0 | yes (no) | TRL / QLoRA notebooks | No stated JSON ability. |
| [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | 2025-02 | 5.57B | about 11 GB (computed) | MIT | yes (yes) | official scripts, ms-swift (pins old transformers) | Weaker tooling. |
| [microsoft/Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) | 2024-06 | 0.78B | already cached locally (1.5 GB) | MIT | plugin only, no LoRA serving | PEFT, ms-swift, maestro; not TRL or LLaMA-Factory | Task-prompt encoder-decoder, not a chat model. See below. |

Every vLLM-served model can be forced to emit schema-valid JSON through vLLM [structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html), so raw JSON validity (without constraints) and constrained validity are reported separately.

### Florence-2: what it is and is not

Florence-2 is a sequence-to-sequence model (DaViT encoder plus BART-style decoder) driven by fixed task tokens such as `<CAPTION>`, `<MORE_DETAILED_CAPTION>`, `<OD>`, `<OCR>`, `<CAPTION_TO_PHRASE_GROUNDING>` ([transformers docs](https://huggingface.co/docs/transformers/main/en/model_doc/florence2)). It does not follow free-form instructions: the released checkpoints have no VQA ability ([HF blog](https://huggingface.co/blog/finetune-florence2)), so it cannot produce attribute JSON zero-shot. It can be fine-tuned with a new task prompt; [Fashion Florence (arXiv 2605.09827)](https://arxiv.org/abs/2605.09827) applied LoRA to Florence-2-large for fashion attribute JSON. For this project it fits as an optional small fine-tuned comparison point, not as a zero-shot candidate. There is no Florence-3.

### Platform notes

- vLLM runs on Linux, so batch inference and the pilot run in WSL2 (Ubuntu distro stored at `D:\wsl\Ubuntu`). Plain transformers inference also works on Windows.
- Windows Python env currently has transformers 4.55.4, too old for Qwen3.5 (5.2+) and MiniCPM-V 4.6 (5.7+); a separate env is needed either way.

## 3. Pilot plan

Candidates: Qwen3.5-4B, Gemma-4-E4B-it, Qwen3-VL-8B-Instruct.

- Sample: 200 listings, main image only, drawn from the val split only (so test stays untouched), requiring a color label; material is scored where present. Stratified by product type, one row per group. Labels normalized to closed sets (37 product types, 16 colors, 9 materials; see `src/vpa/labels.py`).
- Per model, same prompt with the candidate label lists:
  - per-attribute accuracy, with denominators printed
  - raw JSON validity (unconstrained) and validity under vLLM JSON-schema decoding
  - throughput (images/s, vLLM offline batch, fixed image size) and GPU memory
- Image input: originals resized to a fixed longest side; the 256 px small version is run once for the best model to see how much resolution matters.
- Downloads: listing shards 87 MB, images.csv.gz 6.4 MB, 200 original images (about 76 MB at the sampled mean), model weights 9.34 + 16.02 + 17.55 = 42.9 GB, plus the vLLM environment in WSL.

## 4. Pilot results (2026-09-22)

Setup: 200 val rows (`data/pilot/pilot.jsonl`, stratified over 34 product types, one row per group; material labeled on 84). Image only, no title, original resized to 768 px long side. Zero-shot, greedy decoding, same prompt for every model (37 product types, 16 colors, 9 materials listed in the prompt). All three models in bf16 through the same transformers script (`scripts/run_pilot_hf.py`, batch 4, transformers 5.17.0), commit 39d34d8.

| Model (revision) | JSON valid, no constraint | product_type | color | material | Weights on GPU | Peak allocated |
|---|---|---|---|---|---|---|
| Qwen3.5-4B (851bf6e) | 200/200 | 145/200 (0.725) | 112/200 (0.560) | 66/84 (0.786) | 8.46 GiB | 9.75 GiB |
| Gemma-4-E4B-it (ee0ef60) | 200/200 | 139/200 (0.695) | 113/200 (0.565) | 57/84 (0.679) | 14.79 GiB | 15.30 GiB |
| Qwen3-VL-8B-Instruct (0c351dd) | 200/200 | 138/200 (0.690) | 114/200 (0.570) | 63/84 (0.750) | 16.33 GiB | 17.39 GiB |

Paired exact McNemar tests, Qwen3.5-4B against each other model: product_type p = 0.286 (Gemma) and 0.248 (Qwen3-VL); color p = 1.000 and 0.774; material p = 0.012 (Gemma, 10 vs 1 discordant) and 0.453. With six comparisons, none passes a Bonferroni threshold of 0.0083. On this sample the three models are not separable on accuracy.

Cross-check: Qwen3.5-4B under vLLM 0.30.0 on the same rows gave 144/200, 111/200, 66/84 with JSON-schema constrained decoding and 143/200, 111/200, 65/84 without, all 200 outputs valid JSON. Mean prompt length was 823 tokens under both frameworks.

Serving under vLLM on this machine (RTX 4090, 23,028 MiB, with about 4.3 to 4.6 GB held by other Windows applications):
- Qwen3.5-4B: runs with gpu_memory_utilization 0.72, CUDA graphs on, 5.34 images/s with schema-constrained output (200 images, 823 prompt tokens and 32 output tokens on average).
- Gemma-4-E4B-it: does not start. Non-KV memory about 17.8 GiB leaves no room for KV cache inside the roughly 18 GiB that is free.
- Qwen3-VL-8B-Instruct: not attempted under vLLM; its weights alone are 16.33 GiB.

Transformers throughput (batch 4, not comparable to vLLM): 1.60, 1.06 and 0.54 images/s. Qwen3.5-4B ran with the reference PyTorch implementation of its linear-attention layers because `flash-linear-attention` and `causal_conv1d` are not installed.

Error pattern for Qwen3.5-4B (vLLM, constrained): color mistakes are neighbouring shades (black to grey 6, white to silver 6, beige to brown 4); product_type mistakes follow the catalog taxonomy (FINENECKLACEBRACELETANKLET to NECKLACE 4, HOME to FLAT_SHEET or HOME_BED_AND_BATH 6). Both are the kind of convention fine-tuning can learn.
