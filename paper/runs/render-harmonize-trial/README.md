# Plan B trial: render -> harmonize, 20 prompts

`scripts/render_harmonize.py` + `scripts/run_render_harmonize_box.sh`, Z-Image Base 28 steps, guidance 4, 1024px,
seed 42, 20 stratified prompts from `eval/prompts.jsonl` + `eval/prompts_unseen_syllables.jsonl`. Dual OCR
(PaddleOCR AND qwen3vl-8b). Images stay on the gpu-box; `sheet.png` shows 8 prompts.

| config | CER (no space) | EM | seen CER / EM | unseen CER / EM | unseen-syllable CER / EM |
|---|---:|---:|---|---|---|
| plain (no harmonize) | 0.306 | 0.35 | 0.186 / 0.50 | 0.225 / 0.38 | 0.708 / 0.00 |
| b16-s0.3 | 0.292 | 0.40 | 0.161 / 0.62 | 0.215 / 0.38 | 0.708 / 0.00 |
| b16-s0.5 | 0.303 | 0.25 | 0.244 / 0.25 | 0.244 / 0.38 | 0.542 / 0.00 |
| b16-s0.7 | 0.322 | 0.30 | 0.162 / 0.50 | 0.289 / 0.25 | 0.708 / 0.00 |
| b48-s0.3 | 0.315 | 0.40 | 0.161 / 0.62 | 0.273 / 0.38 | 0.708 / 0.00 |
| b48-s0.5 | 0.331 | 0.30 | 0.230 / 0.38 | 0.326 / 0.38 | 0.542 / 0.00 |
| b48-s0.7 | 0.281 | 0.35 | 0.164 / 0.50 | 0.184 / 0.38 | 0.708 / 0.00 |
| r64 v2 (final, 220 prompts) | 0.281 | 0.30 | 0.151 | 0.312 / 0.19 | - |

Group sizes: seen 8, unseen 8, unseen-syllable 4.

## What limits the score

The strokes are the font's, pixel-exact, in every config (they are outside the inpaint mask and pasted back).
The misses come from the measurement and the scene, not from the text:

- The scene still has text. The blank-surface prompt and the negative prompt did not stop Base from drawing
  signs: e.g. seen-11+-00 reads `가져 파 개몸 미믿 든 팁 번 이곳은 금연 구역입나다 ...`, the target plus scene
  text on the same OCR line.
- The OCR misreads clean font text: `퉁소` -> qwen `통소`, `넙치` -> paddle `덥치`, `편의점`-class words pass.
- No OCR reads the unseen syllables (`뇱`, `꾔`, `줵쮲`, `툟`), so that group is 0 EM whatever is drawn.

So the 90% EM gate in `docs/render-harmonize-plan.md` cannot be measured with this OCR, and harmonize strength
barely moves accuracy. The open question for plan B is naturalness, judged by a person on `sheet.png`.

Sheet columns: scene, plain, b16-s0.3, b16-s0.5, b16-s0.7, b48-s0.5. Rows: seen-3-5-00, seen-1-2-00,
unseen-6-10-00, unseen-1-2-01, seen-1-2-01, seen-11+-01, unseen-3-5-01, usyl-02.
