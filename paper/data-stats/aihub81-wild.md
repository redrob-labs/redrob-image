# aihub81-wild

Source: AI Hub 81 (한국어 글자체 이미지), Text in the wild (book, product, sign, traffic sign). Numbers only; AI Hub text and images stay in Seoul.

| Item | Value |
|---|---:|
| Source images kept | 85,857 |
| ...with a full-scene sample | 29,016 |
| Samples train/crop | 177,915 |
| Samples train/full | 29,016 |
| Crop share | 86.0% |
| Distinct syllables | 1,523 |
| Syllables outside KS X 1001 | 34 |
| KS X 1001 covered (of 2,350) | 1,489 |
| KS X 1001 with >= 1 | 1,489 |
| KS X 1001 with >= 5 | 1,178 |
| KS X 1001 with >= 20 | 883 |
| KS X 1001 with >= 100 | 512 |

## Samples by category

| Category/kind | Samples |
|---|---:|
| book/crop | 34,012 |
| book/full | 1,301 |
| product/crop | 40,194 |
| product/full | 8,889 |
| sign/crop | 55,566 |
| sign/full | 10,777 |
| traffic_sign/crop | 48,143 |
| traffic_sign/full | 8,049 |

## Train text blocks by Hangul syllables

| Syllables | Blocks |
|---|---:|
| 0 | 249,545 |
| 1-2 | 118,195 |
| 3-5 | 134,959 |
| 6-10 | 9,456 |
| 11+ | 57 |

## Drops

| Reason | Count |
|---|---:|
| crops_skipped_bad_bbox | 408 |
| duplicate_dhash | 222 |
| duplicate_md5 | 709 |
| full_skipped_bad_bbox | 408 |
| full_skipped_unlabeled_text | 69,807 |
| no_readable_text | 10 |
| no_usable_sample | 12,966 |
| unseen | 44 |

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
