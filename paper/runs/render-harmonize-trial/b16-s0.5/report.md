# Z-Image steps=28 cfg=4.0 1024px lang=ko OCR=dual

| set | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| all | 20 | 0.305 | 0.303 | 0.250 |

## by split

| split | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| seen | 8 | 0.248 | 0.244 | 0.250 |
| unseen | 8 | 0.244 | 0.244 | 0.375 |
| unseen_syllable | 4 | 0.542 | 0.542 | 0.000 |

## by split_bucket

| split_bucket | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| seen/1-2 | 2 | 0.500 | 0.500 | 0.500 |
| seen/11+ | 2 | 0.102 | 0.094 | 0.000 |
| seen/3-5 | 2 | 0.167 | 0.167 | 0.500 |
| seen/6-10 | 2 | 0.222 | 0.214 | 0.000 |
| unseen/1-2 | 2 | 0.500 | 0.500 | 0.000 |
| unseen/11+ | 2 | 0.156 | 0.154 | 0.500 |
| unseen/3-5 | 2 | 0.250 | 0.250 | 0.500 |
| unseen/6-10 | 2 | 0.071 | 0.071 | 0.500 |
| unseen_syllable/1-2 | 2 | 0.750 | 0.750 | 0.000 |
| unseen_syllable/3-5 | 2 | 0.333 | 0.333 | 0.000 |

## by bucket

| bucket | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| 1-2 | 6 | 0.583 | 0.583 | 0.167 |
| 11+ | 4 | 0.129 | 0.124 | 0.250 |
| 3-5 | 6 | 0.250 | 0.250 | 0.333 |
| 6-10 | 4 | 0.147 | 0.143 | 0.250 |

## by scene

| scene | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| book_cover | 1 | 0.333 | 0.333 | 0.000 |
| poster | 9 | 0.196 | 0.188 | 0.333 |
| signboard | 10 | 0.400 | 0.404 | 0.200 |

## by has_unseen_syllable

| has_unseen_syllable | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| False | 16 | 0.246 | 0.244 | 0.312 |
| True | 4 | 0.542 | 0.542 | 0.000 |

