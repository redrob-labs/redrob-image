# Qwen-Image-2512 steps=50 cfg=4.0 1024px lang=ko OCR=paddleocr

| set | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| all | 8 | 0.673 | 0.678 | 0.000 |

## by split

| split | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| seen | 4 | 0.664 | 0.675 | 0.000 |
| unseen | 4 | 0.682 | 0.681 | 0.000 |

## by split_bucket

| split_bucket | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| seen/1-2 | 1 | 1.000 | 1.000 | 0.000 |
| seen/11+ | 1 | 0.545 | 0.556 | 0.000 |
| seen/3-5 | 1 | 1.000 | 1.000 | 0.000 |
| seen/6-10 | 1 | 0.111 | 0.143 | 0.000 |
| unseen/1-2 | 1 | 1.000 | 1.000 | 0.000 |
| unseen/11+ | 1 | 0.312 | 0.308 | 0.000 |
| unseen/3-5 | 1 | 0.667 | 0.667 | 0.000 |
| unseen/6-10 | 1 | 0.750 | 0.750 | 0.000 |

## by bucket

| bucket | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| 1-2 | 2 | 1.000 | 1.000 | 0.000 |
| 11+ | 2 | 0.429 | 0.432 | 0.000 |
| 3-5 | 2 | 0.833 | 0.833 | 0.000 |
| 6-10 | 2 | 0.431 | 0.446 | 0.000 |

## by scene

| scene | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| signboard | 8 | 0.673 | 0.678 | 0.000 |

## by has_unseen_syllable

| has_unseen_syllable | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| False | 8 | 0.673 | 0.678 | 0.000 |

