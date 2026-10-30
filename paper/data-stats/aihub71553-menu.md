# aihub71553-menu

Source: AI Hub 71553 (관광 음식메뉴판), validation + Training subsets downloaded so far; unlabeled text from text detection. Numbers only; AI Hub text and images stay in Seoul.

| Item | Value |
|---|---:|
| Source images kept | 6,839 |
| ...with a full-scene sample | 1 |
| Samples val/crop | 22,146 |
| Samples val/full | 1 |
| Crop share | 100.0% |
| Distinct syllables | 0 |
| Syllables outside KS X 1001 | 0 |
| KS X 1001 covered (of 2,350) | 0 |
| KS X 1001 with >= 1 | 0 |
| KS X 1001 with >= 5 | 0 |
| KS X 1001 with >= 20 | 0 |
| KS X 1001 with >= 100 | 0 |

## Samples by category

| Category/kind | Samples |
|---|---:|
| menu/crop | 22,146 |
| menu/full | 1 |

## Train text blocks by Hangul syllables

| Syllables | Blocks |
|---|---:|
| 0 | 0 |
| 1-2 | 0 |
| 3-5 | 0 |
| 6-10 | 0 |
| 11+ | 0 |

## Drops

| Reason | Count |
|---|---:|
| det_unlabeled_boxes | 345,443 |
| detection_orientation_mismatch | 546 |
| duplicate_dhash | 5 |
| full_skipped_unlabeled_text | 9,400 |
| image_not_downloaded | 72,098 |
| no_detection | 8,005 |
| no_readable_text | 1 |
| no_usable_sample | 2,562 |
| unseen | 52 |

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
