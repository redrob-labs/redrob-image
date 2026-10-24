# Z-Image steps=28 cfg=4.0 1024px lang=ko OCR=dual

| set | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| all | 20 | 0.332 | 0.322 | 0.300 |

## by split

| split | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| seen | 8 | 0.170 | 0.162 | 0.500 |
| unseen | 8 | 0.307 | 0.289 | 0.250 |
| unseen_syllable | 4 | 0.708 | 0.708 | 0.000 |

## by split_bucket

| split_bucket | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| seen/1-2 | 2 | 0.000 | 0.000 | 1.000 |
| seen/11+ | 2 | 0.292 | 0.293 | 0.000 |
| seen/3-5 | 2 | 0.000 | 0.000 | 1.000 |
| seen/6-10 | 2 | 0.389 | 0.357 | 0.000 |
| unseen/1-2 | 2 | 0.750 | 0.750 | 0.000 |
| unseen/11+ | 2 | 0.146 | 0.077 | 0.500 |
| unseen/3-5 | 2 | 0.125 | 0.125 | 0.500 |
| unseen/6-10 | 2 | 0.205 | 0.205 | 0.000 |
| unseen_syllable/1-2 | 2 | 0.750 | 0.750 | 0.000 |
| unseen_syllable/3-5 | 2 | 0.667 | 0.667 | 0.000 |

## by bucket

| bucket | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| 1-2 | 6 | 0.500 | 0.500 | 0.333 |
| 11+ | 4 | 0.219 | 0.185 | 0.250 |
| 3-5 | 6 | 0.264 | 0.264 | 0.500 |
| 6-10 | 4 | 0.297 | 0.281 | 0.000 |

## by scene

| scene | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| book_cover | 1 | 0.333 | 0.333 | 0.000 |
| poster | 9 | 0.305 | 0.276 | 0.333 |
| signboard | 10 | 0.358 | 0.363 | 0.300 |

## by has_unseen_syllable

| has_unseen_syllable | n | CER | CER (no space) | EM |
|---|---:|---:|---:|---:|
| False | 16 | 0.239 | 0.226 | 0.375 |
| True | 4 | 0.708 | 0.708 | 0.000 |

