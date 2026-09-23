## Test results

Frozen test split. Accuracy with macro-F1 in brackets, counted per row. Denominators: product_type 5,437 rows, color 1,610, material 888. VLM rows use 256 px images and JSON-schema constrained decoding.

| Method | Input | product_type | color | material |
|---|---|---|---|---|
| Majority class | - | 0.257 (0.011) | 0.249 (0.025) | 0.343 (0.057) |
| CLIP ViT-L/14 zero-shot | image | 0.638 (0.559) | 0.651 (0.532) | 0.584 (0.442) |
| CLIP ViT-L/14 linear probe | image | 0.854 (0.810) | 0.755 (0.704) | 0.884 (0.767) |
| Qwen3.5-4B zero-shot | image | 0.703 (0.621) | 0.716 (0.627) | 0.797 (0.617) |
| Qwen3.5-4B + QLoRA | image | 0.856 (0.818) | 0.784 (0.719) | 0.868 (0.805) |
| Qwen3.5-4B + LoRA | image | 0.862 (0.829) | 0.786 (0.722) | 0.880 (0.816) |
| Title TF-IDF + logistic regression | title | 0.912 (0.895) | 0.901 (0.876) | 0.937 (0.878) |
| Title TF-IDF + CLIP embedding, logistic regression | image + title | 0.928 (0.908) | 0.912 (0.868) | 0.938 (0.859) |
| Qwen3.5-4B zero-shot | image + title | 0.738 (0.665) | 0.791 (0.723) | 0.833 (0.694) |
| Qwen3.5-4B + LoRA | image + title | 0.921 (0.903) | 0.930 (0.903) | 0.920 (0.864) |

## Inference trade-offs (val, 5,713 rows, one RTX 4090, vLLM 0.30.0)

Accuracy per row on val. Weights memory is vLLM's reported model load size. Throughput is images per second over the whole split, with up to 128 concurrent requests, submitted in chunks of 512.

| Setting | Prompt tokens | Weights memory | Images/s | product_type | color | material |
|---|---|---|---|---|---|---|
| zero-shot, label lists in prompt, 768 px | 807 | 8.61 GiB | 13.3 | 0.683 | 0.719 | 0.804 |
| zero-shot, label lists in prompt, 256 px | 444 | 8.61 GiB | 23.2 | 0.678 | 0.716 | 0.773 |
| LoRA adapter served by vLLM, short prompt, 256 px | 100 | 8.70 GiB | 45.0 | 0.871 | 0.799 | 0.877 |
| LoRA merged into bf16 weights, short prompt, 256 px | 100 | 8.61 GiB | 53.1 | 0.870 | 0.799 | 0.876 |
| merged, FP8 weight quantization, short prompt, 256 px | 100 | 5.30 GiB | 64.3 | 0.872 | 0.792 | 0.877 |

Single-request latency, merged bf16, short prompt, 256 px, one request at a time (max_num_seqs=1): 372 ms per image on average over the first 300 val images, including image preprocessing.
