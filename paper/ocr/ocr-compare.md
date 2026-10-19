# OCR comparison on the rendered reference set

600 crops. CER clipped at 1.0 per crop; EM = exact match ignoring spaces.

| engine | CER | CER (no space) | EM | syl. recall KS X 1001 | syl. recall non-KS X 1001 | s/crop |
|---|---:|---:|---:|---:|---:|---:|
| paddleocr | 0.458 | 0.457 | 0.482 | 0.816 | 0.005 | 0.028 |
| easyocr | 0.506 | 0.507 | 0.302 | 0.757 | 0.000 | 0.021 |
| qwen3vl-8b | 0.440 | 0.445 | 0.507 | 0.853 | 0.000 | 0.356 |

## kind (CER no space / EM)

| kind | paddleocr | easyocr | qwen3vl-8b |
|---|---:|---:|---:|
| seen | 0.157 / 0.76 | 0.203 / 0.57 | 0.119 / 0.85 |
| unseen | 0.224 / 0.68 | 0.316 / 0.34 | 0.215 / 0.67 |
| unseen_syllable | 0.988 / 0.01 | 1.000 / 0.00 | 1.000 / 0.00 |

## aug (CER no space / EM)

| aug | paddleocr | easyocr | qwen3vl-8b |
|---|---:|---:|---:|
| blur | 0.417 / 0.51 | 0.561 / 0.25 | 0.464 / 0.51 |
| clean | 0.380 / 0.54 | 0.382 / 0.37 | 0.330 / 0.62 |
| jpeg | 0.445 / 0.40 | 0.542 / 0.17 | 0.445 / 0.47 |
| noise | 0.368 / 0.60 | 0.447 / 0.36 | 0.368 / 0.59 |
| perspective | 1.000 / 0.00 | 1.000 / 0.00 | 0.964 / 0.01 |
| rotate | 0.277 / 0.67 | 0.326 / 0.41 | 0.255 / 0.69 |
| texture | 0.372 / 0.55 | 0.394 / 0.39 | 0.366 / 0.56 |
| wrap | 0.435 / 0.52 | 0.445 / 0.42 | 0.406 / 0.57 |

## font (CER no space / EM)

| font | paddleocr | easyocr | qwen3vl-8b |
|---|---:|---:|---:|
| blackhansans/BlackHanSans-Regular.ttf | 0.150 / 0.71 | 0.212 / 0.43 | 0.166 / 0.71 |
| dohyeon/DoHyeon-Regular.ttf | 0.196 / 0.72 | 0.311 / 0.38 | 0.186 / 0.72 |
| gaegu/Gaegu-Bold.ttf | 0.222 / 0.65 | 0.258 / 0.53 | 0.142 / 0.76 |
| gaegu/Gaegu-Regular.ttf | 0.403 / 0.33 | 0.274 / 0.30 | 0.149 / 0.73 |
| gowundodum/GowunDodum-Regular.ttf | 0.587 / 0.37 | 0.611 / 0.25 | 0.574 / 0.39 |
| jua/Jua-Regular.ttf | 0.174 / 0.78 | 0.326 / 0.50 | 0.247 / 0.66 |
| nanumgothic/NanumGothic-ExtraBold.ttf | 0.556 / 0.44 | 0.594 / 0.25 | 0.556 / 0.44 |
| nanumgothic/NanumGothic-Regular.ttf | 0.474 / 0.50 | 0.548 / 0.29 | 0.498 / 0.45 |
| nanummyeongjo/NanumMyeongjo-Regular.ttf | 0.507 / 0.45 | 0.578 / 0.27 | 0.488 / 0.49 |
| nanumpenscript/NanumPenScript-Regular.ttf | 0.614 / 0.29 | 0.657 / 0.17 | 0.601 / 0.35 |
| notosanskr/NotoSansKR[wght].ttf | 0.555 / 0.42 | 0.607 / 0.24 | 0.567 / 0.43 |
| notoserifkr/NotoSerifKR[wght].ttf | 0.481 / 0.50 | 0.535 / 0.35 | 0.503 / 0.46 |
