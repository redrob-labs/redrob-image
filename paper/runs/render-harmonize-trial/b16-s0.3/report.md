# Z-Image steps=28 cfg=4.0 1024px lang=ko OCR=dual

| set | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| all | 20 | 0.294 | 0.292 | 0.400 |

## by split

| split | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| seen | 8 | 0.167 | 0.161 | 0.625 |
| unseen | 8 | 0.213 | 0.215 | 0.375 |
| unseen_syllable | 4 | 0.708 | 0.708 | 0.000 |

## by split_bucket

| split_bucket | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| seen/1-2 | 2 | 0.500 | 0.500 | 0.500 |
| seen/11+ | 2 | 0.000 | 0.000 | 1.000 |
| seen/3-5 | 2 | 0.000 | 0.000 | 1.000 |
| seen/6-10 | 2 | 0.167 | 0.143 | 0.000 |
| unseen/1-2 | 2 | 0.500 | 0.500 | 0.000 |
| unseen/11+ | 2 | 0.031 | 0.038 | 0.500 |
| unseen/3-5 | 2 | 0.250 | 0.250 | 0.500 |
| unseen/6-10 | 2 | 0.071 | 0.071 | 0.500 |
| unseen_syllable/1-2 | 2 | 0.750 | 0.750 | 0.000 |
| unseen_syllable/3-5 | 2 | 0.667 | 0.667 | 0.000 |

## by bucket

| bucket | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| 1-2 | 6 | 0.583 | 0.583 | 0.167 |
| 11+ | 4 | 0.016 | 0.019 | 0.750 |
| 3-5 | 6 | 0.306 | 0.306 | 0.500 |
| 6-10 | 4 | 0.119 | 0.107 | 0.250 |

## by scene

| scene | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| book_cover | 1 | 0.333 | 0.333 | 0.000 |
| poster | 9 | 0.263 | 0.254 | 0.444 |
| signboard | 10 | 0.317 | 0.322 | 0.400 |

## by has_unseen_syllable

| has_unseen_syllable | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| False | 16 | 0.190 | 0.188 | 0.500 |
| True | 4 | 0.708 | 0.708 | 0.000 |

