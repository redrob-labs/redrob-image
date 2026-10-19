# Z-Image-Turbo steps=9 cfg=0.0 1024px lang=ko OCR=qwen3vl-8b

| set | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| all | 220 | 0.442 | 0.439 | 0.127 |

## by split

| split | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| seen | 100 | 0.356 | 0.369 | 0.220 |
| unseen | 100 | 0.451 | 0.436 | 0.060 |
| unseen_syllable | 20 | 0.825 | 0.808 | 0.000 |

## by split_bucket

| split_bucket | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| seen/1-2 | 25 | 0.480 | 0.480 | 0.320 |
| seen/11+ | 25 | 0.334 | 0.383 | 0.000 |
| seen/3-5 | 25 | 0.293 | 0.283 | 0.440 |
| seen/6-10 | 25 | 0.319 | 0.331 | 0.120 |
| unseen/1-2 | 25 | 0.740 | 0.740 | 0.000 |
| unseen/11+ | 25 | 0.252 | 0.284 | 0.000 |
| unseen/3-5 | 25 | 0.382 | 0.352 | 0.240 |
| unseen/6-10 | 25 | 0.429 | 0.367 | 0.000 |
| unseen_syllable/1-2 | 7 | 0.929 | 0.929 | 0.000 |
| unseen_syllable/3-5 | 13 | 0.769 | 0.744 | 0.000 |

## by bucket

| bucket | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| 1-2 | 57 | 0.649 | 0.649 | 0.140 |
| 11+ | 50 | 0.293 | 0.334 | 0.000 |
| 3-5 | 63 | 0.426 | 0.405 | 0.270 |
| 6-10 | 50 | 0.374 | 0.349 | 0.060 |

## by scene

| scene | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| book_cover | 44 | 0.368 | 0.369 | 0.182 |
| menu | 44 | 0.473 | 0.476 | 0.114 |
| poster | 44 | 0.435 | 0.432 | 0.136 |
| product_label | 44 | 0.449 | 0.453 | 0.114 |
| signboard | 44 | 0.484 | 0.466 | 0.091 |

## by has_unseen_syllable

| has_unseen_syllable | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| False | 200 | 0.404 | 0.402 | 0.140 |
| True | 20 | 0.825 | 0.808 | 0.000 |

