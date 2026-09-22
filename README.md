# Product attribute extraction with a small VLM

Predict structured attributes (product type, color, material) from an e-commerce product image, as JSON, with Qwen3.5-4B fine-tuned by LoRA on Amazon Berkeley Objects, and compare it against text, CLIP and zero-shot baselines on a frozen, leakage-checked test split.

Results are filled in only from runs on the frozen test split with the saved weights; see "Results".

## Data

[Amazon Berkeley Objects (ABO)](https://amazon-berkeley-objects.s3.amazonaws.com/index.html), CC BY 4.0 (per the LICENSE file shipped with the data). Credit: Amazon.com; Collins et al., "ABO: Dataset and Benchmarks for Real-World 3D Object Understanding", CVPR 2022.

- 147,702 listings, 398,212 catalog images. One example per listing: its main image and its seller-entered attributes.
- Labels
  - `product_type`: 37 classes (types with at least 300 train listings)
  - `color`: ABO `standardized_values`, mapped to 16 classes
  - `material`: free text mapped to 9 classes; unmappable values (for example "stone" on sofas and rugs, "synthetic" on shoes) are left unlabeled
  - Mapping tables: `src/vpa/labels.py`
- `CELLULAR_PHONE_CASE` is excluded; it is 44% of all listings and would dominate every metric.
- Many rows have only some labels. Each attribute is scored on the rows that carry it, and every metric reports its denominator.

### Split and leakage control

Product variants (sizes, colors, marketplaces) often reuse the same photos, so splitting by `item_id` leaks images across splits.

1. Listings are linked when they share an `item_id` or any image. Images attached to more than 50 listings, such as size charts and banners, are not used for linking. Rows whose main image is one of them are dropped.
2. Connected components (union-find) are assigned to train / val / test by a hash of the component, 80 / 10 / 10. Components larger than 500 listings go to train. One chain of furniture listings linked through shared room photos would otherwise make up a third of the test set.
3. A check asserts that no non-generic image and no `item_id` appears in two splits.
4. The test file's SHA-256 is pinned in `src/vpa/splits.py` and verified on every load.

| Split | Rows | Groups | color labeled | material labeled |
|---|---|---|---|---|
| train | 46,906 | 23,656 | 14,578 | 7,910 |
| val | 5,713 | 3,080 | 1,695 | 864 |
| test | 5,437 | 3,005 | 1,610 | 888 |

Many test rows share a main image: 5,437 rows point to 3,659 unique images. Scores are reported both per row and per unique image (first row per image).

## Methods

- **Majority class**: most frequent train label per attribute.
- **Title only**: TF-IDF over word 1-2 grams and character 2-5 grams, with logistic regression. The regularization strength is picked on val. ABO titles are written by sellers and often state the type, color and material directly, which makes this baseline strong.
- **CLIP ViT-L/14**: zero-shot with prompt templates, and a linear probe (logistic regression on frozen image embeddings).
- **Qwen3.5-4B zero-shot**: the image plus an instruction listing the allowed labels. vLLM decodes under a JSON schema whose enums are the label sets.
- **Qwen3.5-4B + LoRA**: bf16 LoRA (r=16) on the language model's attention, linear-attention and MLP projections; the vision tower is frozen.
  - The loss covers only the answer tokens. Values of unlabeled attributes are masked out of the loss, so rows with partial labels can still be used.
  - The fine-tuned model gets a short prompt without the label lists, which cuts the tokens per example from 468 to 124. The JSON schema still restricts outputs to valid labels.

## Layout

```
scripts/
  download_abo_metadata.py   listing shards and image metadata
  build_dataset.py           labels, grouped split, leakage check
  download_images.py         256 px images, and 768 px originals for val/test
  baseline_majority.py, baseline_text.py, baseline_clip.py
  eval_vlm.py                vLLM evaluation, optional LoRA adapter
  train_lora.py              LoRA / QLoRA fine-tuning
  score_predictions.py, error_analysis.py
src/vpa/                     loaders, label maps, prompts, metrics, frozen split
```

## Reproducing

Training and vLLM inference run in WSL2 (Ubuntu 24.04) on one RTX 4090. Environment: vLLM 0.30.0, torch 2.13.0+cu130, transformers 5.17.0, peft 0.21.0, flash-linear-attention 0.5.2.

```
python scripts/download_abo_metadata.py
python scripts/build_dataset.py
python scripts/download_images.py small
python scripts/download_images.py original --splits val,test
python scripts/baseline_majority.py
python scripts/baseline_text.py
python scripts/baseline_clip.py
python scripts/eval_vlm.py --model <Qwen3.5-4B dir> --name zeroshot --split test --images data/images/img256 --chat-kwargs '{"enable_thinking": false}'
python scripts/train_lora.py --model <Qwen3.5-4B dir> --images data/images/img256 --out outputs/train/lora --prompt short --lr 2e-4
python scripts/eval_vlm.py --model <Qwen3.5-4B dir> --lora outputs/train/lora/final --prompt short --name lora --split test --images data/images/img256 --chat-kwargs '{"enable_thinking": false}'
```

## Results

Pending: filled from the frozen-test runs once fine-tuning is evaluated.
