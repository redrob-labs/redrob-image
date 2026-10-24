# aihub71725-tourism

Source: AI Hub 71725 (관광분야 이미지-텍스트 쌍), posters/banners + tables (지도·티켓·안내책자 제외); max 8 crops per image. Numbers only; AI Hub text and images stay in Seoul.

| Item | Value |
|---|---:|
| Source images kept | 65,564 |
| ...with a full-scene sample | 65,035 |
| Samples train/crop | 104,889 |
| Samples train/full | 58,019 |
| Samples val/crop | 13,156 |
| Samples val/full | 7,016 |
| Crop share | 64.5% |
| Distinct syllables | 1,597 |
| Syllables outside KS X 1001 | 15 |
| KS X 1001 covered (of 2,350) | 1,582 |
| KS X 1001 with >= 1 | 1,582 |
| KS X 1001 with >= 5 | 1,291 |
| KS X 1001 with >= 20 | 1,053 |
| KS X 1001 with >= 100 | 734 |

## Samples by category

| Category/kind | Samples |
|---|---:|
| 관광정보_및_프로그램표/crop | 6,510 |
| 관광정보_및_프로그램표/full | 9,464 |
| 기타(포스터,_배너,_현수막)/crop | 83,489 |
| 기타(포스터,_배너,_현수막)/full | 23,985 |
| 시간표/crop | 11,235 |
| 시간표/full | 17,468 |
| 요금/시간_통합표/crop | 5,320 |
| 요금/시간_통합표/full | 3,992 |
| 요금표/crop | 11,491 |
| 요금표/full | 10,126 |

## Train text blocks by Hangul syllables

| Syllables | Blocks |
|---|---:|
| 0 | 1,352,087 |
| 1-2 | 1,774,200 |
| 3-5 | 471,779 |
| 6-10 | 872 |
| 11+ | 0 |

## Drops

| Reason | Count |
|---|---:|
| crops_skipped_bad_bbox | 1,084 |
| duplicate_dhash | 7,568 |
| duplicate_md5 | 17 |
| full_skipped_bad_bbox | 1,084 |
| full_skipped_unlabeled_text | 834 |
| image_not_downloaded | 177,354 |
| no_readable_text | 7 |
| no_usable_sample | 305 |
| unseen | 101 |

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
