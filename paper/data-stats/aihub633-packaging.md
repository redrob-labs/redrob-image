# aihub633-packaging

Source: AI Hub 633 (의약품, 화장품 패키징 OCR), labels + validation images + Training images 1; max 8 crops per image. Numbers only; AI Hub text and images stay in Seoul.

| Item | Value |
|---|---:|
| Source images kept | 62,613 |
| ...with a full-scene sample | 40,718 |
| Samples train/crop | 223,807 |
| Samples train/full | 27,065 |
| Samples val/crop | 102,528 |
| Samples val/full | 13,653 |
| Crop share | 88.9% |
| Distinct syllables | 1,263 |
| Syllables outside KS X 1001 | 42 |
| KS X 1001 covered (of 2,350) | 1,221 |
| KS X 1001 with >= 1 | 1,221 |
| KS X 1001 with >= 5 | 1,049 |
| KS X 1001 with >= 20 | 886 |
| KS X 1001 with >= 100 | 685 |

## Samples by category

| Category/kind | Samples |
|---|---:|
| cosmetics/crop | 47,008 |
| cosmetics/full | 6,897 |
| medicine/crop | 279,327 |
| medicine/full | 33,821 |

## Train text blocks by Hangul syllables

| Syllables | Blocks |
|---|---:|
| 0 | 558,998 |
| 1-2 | 671,964 |
| 3-5 | 869,489 |
| 6-10 | 165,365 |
| 11+ | 19,316 |

## Drops

| Reason | Count |
|---|---:|
| crops_skipped_bad_bbox | 895 |
| duplicate_dhash | 281 |
| duplicate_md5 | 322 |
| full_skipped_bad_bbox | 895 |
| full_skipped_unlabeled_text | 22,333 |
| image_not_downloaded | 40,607 |
| no_readable_text | 13 |
| no_usable_sample | 438 |
| size_mismatch | 4 |
| unseen | 43 |

## Settings

| Setting | Value |
|---|---|
| train_res | 512 |
| min_text_px | 20 |
| min_crop | 384 |
| crop_margin | 1.3 |
| ignore_text_px | 8 |
| crop_max_side | 1024 |
| full_max_side | 1600 |
| full_check_res | 1024 |
