# Error analysis: Qwen3.5-4B + LoRA on the frozen test split

Predictions: `outputs/eval/lora_r16_short_test/constrained_predictions.jsonl` (adapter `outputs/train/lora_r16_short/final`, image only, 256 px, JSON-schema constrained). Breakdown produced by `scripts/error_analysis.py`. Counts are rows; denominators in brackets.

## product_type: 748 errors (5,437 rows)

| Gold | Predicted | Rows | Share of errors |
|---|---|---|---|
| BOOT | SHOES | 76 | 10.2% |
| SHOES | SANDAL | 73 | 9.8% |
| HOME_BED_AND_BATH | HOME | 46 | 6.1% |
| SHOES | BOOT | 43 | 5.7% |
| NECKLACE | FINENECKLACEBRACELETANKLET | 37 | 4.9% |
| EARRING | FINEEARRING | 34 | 4.5% |
| HEALTH_PERSONAL_CARE | GROCERY | 33 | 4.4% |

Three kinds of error:

1. **Footwear boundaries.** BOOT, SHOES and SANDAL confusions account for 192 of 748 errors. SHOES dominates the class counts (1,396 test rows against 254 BOOT), and most mistakes on ankle boots go toward SHOES. These are genuine model errors.
2. **Catalog taxonomy rather than appearance.** NECKLACE has a recall of 13/50 = 0.26, and most misses are predicted as FINENECKLACEBRACELETANKLET. A sample of these rows shows titles such as "Lab Grown Diamond Bezel-Set Necklace in 14k Yellow Gold" and "Sterling Silver Open Circle Pendant Necklace". By the catalog's own convention those are fine jewelry, so here the label is the doubtful side. HOME against HOME_BED_AND_BATH, and HEALTH_PERSONAL_CARE against GROCERY, are similar category-assignment choices.
3. **Label conflicts that no model can resolve.** 262 test rows sit on 83 main images that carry more than one product type (for example SANDAL and SHOES on the same photo, or HOME and HOME_BED_AND_BATH). Any single answer per image must miss at least 93 of these rows. That floor is 12.4% of the 748 errors and caps product_type row accuracy at 1 - 93/5,437 = 0.983.

## color: 345 errors (1,610 rows)

- The biggest confusion is between neighbouring shades: brown against beige in both directions (32 + 28 rows, 17% of errors), then blue against grey and black against blue.
- Recall is lowest for green (46/82), beige (72/124), silver (16/27) and multicolor (56/87).
- By product type, color accuracy is lowest on SOFA (32/48), ACCESSORY (32/48) and CHAIR (45/64).
- Some labels are loose. A kennel whose title says "Khaki" is labeled brown, and the model says beige. Others are plain misses: a pillowcase titled "Chocolate" was predicted beige.

## material: 107 errors (888 rows)

- **wood** has the lowest recall (42/73 = 0.575), and the largest error is wood predicted as fabric (19 rows).
- Every sampled row in that group is upholstered furniture: wingback chairs, loveseats, bar stools and ottomans, with titles saying "Upholstered". The seller's material label names the frame. The photo shows fabric. For multi-material objects, "main material" is not defined in the data.
- Material accuracy is lowest on CHAIR (30/50) and SOFA (25/39), the categories with this problem.

## What this suggests

- A sizeable share of residual errors come from the labels, not the model: taxonomy choices, frame-versus-cover materials, and conflicting labels on shared images. More training on the same labels will not remove them.
- Remaining model-side errors: footwear subtype boundaries (possibly helped by higher resolution or class re-weighting), and fine colour distinctions between neighbouring shades.
