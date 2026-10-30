# aihub105-pilot

Source: AI Hub 105 (야외 실제 촬영 한글 이미지), pilot subset. Numbers only; AI Hub text and images stay in Seoul.

| Item | Value |
|---|---:|
| Source images kept | 32,140 |
| ...with a full-scene sample | 17,882 |
| Samples train/crop | 37,216 |
| Samples train/full | 17,882 |
| Crop share | 67.5% |
| Distinct syllables | 1,341 |
| Syllables outside KS X 1001 | 24 |
| KS X 1001 covered (of 2,350) | 1,317 |
| KS X 1001 with >= 1 | 1,317 |
| KS X 1001 with >= 5 | 962 |
| KS X 1001 with >= 20 | 655 |
| KS X 1001 with >= 100 | 332 |

## Samples by category

| Category/kind | Samples |
|---|---:|
| 간판/crop | 24,417 |
| 간판/full | 10,136 |
| 책표지/crop | 12,799 |
| 책표지/full | 7,746 |

## Train text blocks by Hangul syllables

| Syllables | Blocks |
|---|---:|
| 0 | 18 |
| 1-2 | 27,017 |
| 3-5 | 43,854 |
| 6-10 | 10,806 |
| 11+ | 140 |

## Drops

| Reason | Count |
|---|---:|
| crops_skipped_bad_bbox | 69 |
| duplicate_dhash | 81 |
| duplicate_md5 | 243 |
| full_skipped_bad_bbox | 69 |
| full_skipped_unlabeled_text | 25,994 |
| no_readable_text | 6 |
| no_usable_sample | 11,736 |
| unseen | 30 |

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
