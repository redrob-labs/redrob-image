# glyph-react-base OCR=dual(paddleocr+qwen3vl-8b)

| set | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| all | 10 | 0.832 | 0.789 | 0.000 |

## by split

| split | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| seen | 6 | 0.876 | 0.804 | 0.000 |
| unseen | 4 | 0.766 | 0.767 | 0.000 |

## by split_bucket

| split_bucket | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| seen/1-2 | 2 | 1.000 | 0.750 | 0.000 |
| seen/11+ | 2 | 0.739 | 0.806 | 0.000 |
| seen/3-5 | 1 | 1.000 | 1.000 | 0.000 |
| seen/6-10 | 1 | 0.778 | 0.714 | 0.000 |
| unseen/1-2 | 1 | 1.000 | 1.000 | 0.000 |
| unseen/11+ | 1 | 0.562 | 0.692 | 0.000 |
| unseen/3-5 | 1 | 1.000 | 1.000 | 0.000 |
| unseen/6-10 | 1 | 0.500 | 0.375 | 0.000 |

## by bucket

| bucket | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| 1-2 | 3 | 1.000 | 0.833 | 0.000 |
| 11+ | 3 | 0.680 | 0.768 | 0.000 |
| 3-5 | 2 | 1.000 | 1.000 | 0.000 |
| 6-10 | 2 | 0.639 | 0.545 | 0.000 |

## by scene

| scene | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| poster | 2 | 0.967 | 1.000 | 0.000 |
| signboard | 8 | 0.798 | 0.737 | 0.000 |

## by has_unseen_syllable

| has_unseen_syllable | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| False | 10 | 0.832 | 0.789 | 0.000 |

