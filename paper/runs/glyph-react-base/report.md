# Z-Image steps=28 cfg=4.0 1024px lang=ko OCR=qwen3vl-8b

| set | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| all | 10 | 0.703 | 0.717 | 0.100 |

## by split

| split | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| seen | 6 | 0.672 | 0.697 | 0.167 |
| unseen | 4 | 0.750 | 0.748 | 0.000 |

## by split_bucket

| split_bucket | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| seen/1-2 | 2 | 0.500 | 0.500 | 0.500 |
| seen/11+ | 2 | 0.683 | 0.806 | 0.000 |
| seen/3-5 | 1 | 1.000 | 1.000 | 0.000 |
| seen/6-10 | 1 | 0.667 | 0.571 | 0.000 |
| unseen/1-2 | 1 | 1.000 | 1.000 | 0.000 |
| unseen/11+ | 1 | 0.500 | 0.615 | 0.000 |
| unseen/3-5 | 1 | 1.000 | 1.000 | 0.000 |
| unseen/6-10 | 1 | 0.500 | 0.375 | 0.000 |

## by bucket

| bucket | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| 1-2 | 3 | 0.667 | 0.667 | 0.333 |
| 11+ | 3 | 0.622 | 0.742 | 0.000 |
| 3-5 | 2 | 1.000 | 1.000 | 0.000 |
| 6-10 | 2 | 0.583 | 0.473 | 0.000 |

## by scene

| scene | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| poster | 2 | 0.933 | 1.000 | 0.000 |
| signboard | 8 | 0.646 | 0.647 | 0.125 |

## by has_unseen_syllable

| has_unseen_syllable | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| False | 10 | 0.703 | 0.717 | 0.100 |

