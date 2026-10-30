#!/usr/bin/env python3
"""Plan C reward (docs/reward-training-plan.md): is the text in the image stroke-for-stroke right?

Three layers, all read from the text box only (score_crops.crop):
  glyph  per syllable, Qwen3-VL sees the crop AND the target rendered in a clean font, and is asked whether
         syllable k in the photo has exactly the same jamo as syllable k in the reference. The score is
         P(yes) from the yes/no logits, so it is continuous and ranks two near-identical images.
  word   dual OCR of the crop (PaddleOCR AND Qwen3-VL agree), as eval_baseline --ocr dual
  extra  hallucinated text: Hangul syllables PaddleOCR reads on the WHOLE image beyond the target's own

  total = syl_mean + 2 * lik   (v5; v1 was 0.6 * glyph_mean + 0.3 * word_em - 0.1 * extra)
  syl    per syllable, the cell alone, log P(target syllable) when Qwen3-VL transcribes it (no word prior)
  lik    whole crop, mean token log P(target text) when Qwen3-VL transcribes it

Stage 0 check (must pass before any training): rank the one-jamo pairs from make_jamo_pairs.py.
  python scripts/reward.py --check ~/runs/reward-check
writes <dir>/rewards.jsonl and <dir>/check.json; pass = total ranks the right image higher in >= 95% of pairs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))
import metrics  # noqa: E402
from score_crops import crop  # noqa: E402

GLYPH_Q = (
    "Image 1 is a photo of some Korean text. Image 2 is the correct text \"{t}\" printed in a clean font. "
    "Look at Korean syllable number {k} of {n} (counting left to right, top line first, spaces not counted). "
    "In the reference it is \"{c}\". Does that syllable in the photo have exactly the same initial consonant, "
    "the same vowel and the same final consonant as \"{c}\", stroke for stroke? A single missing or extra "
    "stroke, a different vowel, a single instead of a double consonant, or a missing 받침 means no. "
    "Answer only yes or no."
)
LIK_Q = "Transcribe the Korean text in this image exactly, letter by letter, as it is drawn. Answer with the text only."
SYL_Q = ("This image shows one Korean character (a single Hangul syllable). Write exactly that character as it "
         "is drawn, stroke for stroke. Answer with the one character only.")
# v1 (0.6 glyph + 0.3 word) ranked one-jamo pairs right in 78.5% (check.json); whole-crop likelihood alone
# 83.1%, per-syllable cells alone 84.3%. v5 total = syl_mean + 2 * lik (log-probs) = 89.4%, weights picked on
# these same 498 pairs, so that number is optimistic. glyph/word/extra are still recorded, not in the total.
W_SYL, W_LIK, W_GLYPH, W_WORD, W_EXTRA = 1.0, 2.0, 0.0, 0.0, 0.0


def syllable_cells(img, r: dict):
    """(cell image, syllable) for each Hangul syllable, from the placement the composite was drawn with."""
    import glyph as G
    from render_harmonize import FONT_DIR

    W, H = img.size
    bold = "굵은" in r.get("prompt_ko", "") or "bold" in r.get("prompt_ko", "")  # as render_harmonize.render
    path = str(FONT_DIR / ("NotoSansKR-Bold.ttf" if bold else "NotoSansKR-Regular.ttf"))
    ref = G._font(64, path)
    for b in G.layout(r["target_text"], r.get("scene", ""), W, H):
        x0, y0, x1, y1 = (int(round(v)) for v in b["bbox_xyxy"])
        w, h, text = x1 - x0, y1 - y0, b["text"]
        bx0, by0, bx1, by1 = ref.getbbox(text)
        s = G.FILL * min(w / max(bx1 - bx0, 1), h / max(by1 - by0, 1))
        f = G._font(max(4, int(64 * s)), path)
        _, t, _, bt = f.getbbox(text, anchor="mm")
        cx, cy = x0 + w / 2, y0 + h / 2
        left = cx - f.getlength(text) / 2  # anchor "mm" centres the advance width
        for i, ch in enumerate(text):
            if not 0xAC00 <= ord(ch) <= 0xD7A3:
                continue
            a = left + f.getlength(text[:i])
            e = a + f.getlength(ch)
            px, py = 0.12 * (e - a), 0.12 * (bt - t)
            yield img.crop((max(0, int(a - px)), max(0, int(cy + t - py)),
                            min(W, int(e + px)), min(H, int(cy + bt + py)))), ch


def hangul(s: str) -> list[str]:
    return [c for c in s if 0xAC00 <= ord(c) <= 0xD7A3]


def reference(text: str, size) -> "Image":
    """Target text, black on white, in the crop's size (one line per layout line is not needed here)."""
    from PIL import Image, ImageOps
    from render_harmonize import text_alpha

    w, h = size
    return ImageOps.invert(text_alpha(text, w, h, bold=False)).convert("RGB")


def zoom(img, min_h: int):
    from PIL import Image

    if img.height >= min_h:
        return img
    s = min_h / img.height
    return img.resize((round(img.width * s), min_h), Image.BICUBIC)


class Reward:
    def __init__(self, min_h: int = 192, margin: float = 0.15):
        import ocr_engines

        pp = Path.home() / "paddle-venv/bin/python"
        if pp.exists():
            os.environ.setdefault("REDROB_PADDLE_PY", str(pp))
        self.paddle = ocr_engines.make("paddleocr")
        self.qwen = ocr_engines.make("qwen3vl-8b")
        self.min_h, self.margin = min_h, margin
        tok = self.qwen.proc.tokenizer
        self.yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")})
        self.no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")})

    def p_yes(self, photo, ref, q: str) -> float:
        torch = self.qwen.torch
        msgs = [{"role": "user", "content": [{"type": "image", "image": photo}, {"type": "image", "image": ref},
                                             {"type": "text", "text": q}]}]
        inp = self.qwen.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                 return_tensors="pt").to(self.qwen.model.device)
        with torch.inference_mode():
            lg = self.qwen.model(**inp).logits[0, -1].float()
        y, n = torch.logsumexp(lg[self.yes], 0), torch.logsumexp(lg[self.no], 0)
        return torch.sigmoid(y - n).item()

    def likelihood(self, photo, text: str) -> float:
        """Mean log-prob Qwen3-VL gives the target text when asked to transcribe the crop (teacher forced).

        Unlike free OCR this cannot drift to a nearby plausible word: it prices the exact target, so one
        wrong jamo in the photo lowers the probability of the right syllable's tokens."""
        torch = self.qwen.torch
        q = [{"role": "user", "content": [{"type": "image", "image": photo}, {"type": "text", "text": LIK_Q}]}]
        full = q + [{"role": "assistant", "content": [{"type": "text", "text": text}]}]
        kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
        pre = self.qwen.proc.apply_chat_template(q, add_generation_prompt=True, **kw)
        inp = self.qwen.proc.apply_chat_template(full, **kw).to(self.qwen.model.device)
        n = len(self.qwen.proc.tokenizer.encode(text, add_special_tokens=False))
        s = pre["input_ids"].shape[1]
        with torch.inference_mode():
            lg = self.qwen.model(**inp).logits[0].float()
        ids = inp["input_ids"][0]
        lp = torch.log_softmax(lg[s - 1:s - 1 + n], -1).gather(1, ids[s:s + n, None])[:, 0]
        return lp.tolist()

    def syllables(self, img, r: dict) -> list[float]:
        """Log-prob of each target syllable read ALONE from its own cell (teacher forced, no word around it).

        The whole-crop likelihood fails on vowels (72%) because Qwen3-VL reads the word, and the word prior
        ("신선한") outvotes the one wrong stroke in the photo. A single cell has no word to lean on.
        Cells come from the known placement (glyph.layout + render_harmonize.text_alpha geometry): fine for the
        stage-0 composites; model-drawn images need a detector to find the cells first."""
        torch = self.qwen.torch
        tok = self.qwen.proc.tokenizer
        out = []
        for c, s in syllable_cells(img, r):
            c = zoom(c, self.min_h // 2)
            q = [{"role": "user", "content": [{"type": "image", "image": c}, {"type": "text", "text": SYL_Q}]}]
            full = q + [{"role": "assistant", "content": [{"type": "text", "text": s}]}]
            kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
            pre = self.qwen.proc.apply_chat_template(q, add_generation_prompt=True, **kw)
            inp = self.qwen.proc.apply_chat_template(full, **kw).to(self.qwen.model.device)
            n = len(tok.encode(s, add_special_tokens=False))
            k = pre["input_ids"].shape[1]
            with torch.inference_mode():
                lg = self.qwen.model(**inp).logits[0].float()
            ids = inp["input_ids"][0]
            out.append(torch.log_softmax(lg[k - 1:k - 1 + n], -1).gather(1, ids[k:k + n, None])[:, 0].sum().item())
        return out

    def __call__(self, img, r: dict, old: dict | None = None) -> dict:
        """old: a previous rewards.jsonl row for this image; its glyph/word/extra layers are reused."""
        t = r["target_text"]
        c = zoom(crop(img, r, self.margin), self.min_h)
        toks = self.likelihood(c, t)
        lik = sum(toks) / len(toks)  # min(token) was tried: 74.5% vs 83.1% for the mean
        syl = self.syllables(img, r)
        sm = sum(syl) / len(syl) if syl else lik
        if old is not None:
            base = {k: old[k] for k in ("glyph", "glyph_mean", "glyph_min", "word_em", "word_cer", "word_best",
                                        "extra")}
        else:
            base = self._layers(img, c, t)
        return dict(base, lik=lik, lik_tokens=toks, syl=syl, syl_mean=sm, syl_min=min(syl) if syl else lik,
                    total=W_SYL * sm + W_LIK * lik)

    def _layers(self, img, c, t: str) -> dict:
        ref = reference(t, c.size)
        syl = hangul(t)
        g = [self.p_yes(c, ref, GLYPH_Q.format(t=t, k=k + 1, n=len(syl), c=s)) for k, s in enumerate(syl)]
        per = {"paddleocr": self.paddle(c), "qwen3vl-8b": self.qwen(c)}
        word = metrics.combine(*(metrics.score_one(t, ls) for ls in per.values()))
        full = self.paddle(img)
        extra_n = max(0, len(hangul("".join(l["text"] for l in full))) - len(syl))
        extra = min(1.0, extra_n / max(1, len(syl)))
        gm = sum(g) / len(g) if g else 0.0
        return dict(glyph=g, glyph_mean=gm, glyph_min=min(g) if g else 0.0, word_em=bool(word["em"]),
                    word_cer=word["cer"], word_best=word["best"], extra=extra)


def check(d: Path, min_h: int, tag: str = "v2", reuse: Path | None = None) -> dict:
    from PIL import Image

    pairs = [json.loads(l) for l in (d / "pairs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rw = Reward(min_h=min_h)
    old: dict[str, dict] = {}
    if reuse is not None and reuse.exists():
        for l in reuse.read_text(encoding="utf-8").splitlines():
            if l.strip():
                x = json.loads(l)
                old[x["key"]] = x
    cache: dict[str, dict] = {}
    out = d / f"rewards-{tag}.jsonl"
    if out.exists():  # resume
        for l in out.read_text(encoding="utf-8").splitlines():
            if l.strip():
                x = json.loads(l)
                cache[x["key"]] = x
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        def score(key, path, r):
            if key not in cache:
                cache[key] = dict(key=key, **rw(Image.open(path).convert("RGB"), r, old.get(key)))
                f.write(json.dumps(cache[key], ensure_ascii=False) + "\n")
                f.flush()
            return cache[key]

        rows = []
        for i, p in enumerate(pairs):
            good = score("good/" + p["id"], d / "images" / "good" / f"{p['id']}.png", p)
            bad = score("bad/" + p["pair_id"], d / "images" / "bad" / f"{p['pair_id']}.png", p)
            k = len(hangul(p["target_text"][:p["pos"]]))
            rows.append(dict(p, good=good, bad=bad,
                             glyph_at_pos=(good["glyph"][k], bad["glyph"][k])))
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(pairs)} {time.time() - t0:.0f}s", flush=True)

    def acc(rs, f):
        return sum(f(x) for x in rs) / max(1, len(rs))

    tests = {
        "total": lambda x: x["good"]["total"] > x["bad"]["total"],
        "lik": lambda x: x["good"]["lik"] > x["bad"]["lik"],
        "syl_mean": lambda x: x["good"]["syl_mean"] > x["bad"]["syl_mean"],
        "syl_min": lambda x: x["good"]["syl_min"] > x["bad"]["syl_min"],
        "glyph_mean": lambda x: x["good"]["glyph_mean"] > x["bad"]["glyph_mean"],
        "glyph_min": lambda x: x["good"]["glyph_min"] > x["bad"]["glyph_min"],
        "glyph_at_pos": lambda x: x["glyph_at_pos"][0] > x["glyph_at_pos"][1],
        "word_em": lambda x: x["good"]["word_em"] and not x["bad"]["word_em"],
        "word_cer": lambda x: x["good"]["word_cer"] < x["bad"]["word_cer"],
    }

    def table(rs):
        return {name: round(acc(rs, f), 3) for name, f in tests.items()} | {"n": len(rs)}

    res = {"overall": table(rows)}
    for key in ("kind", "split", "has_unseen_syllable"):
        for v in sorted({str(x.get(key)) for x in rows}):
            res[f"{key}={v}"] = table([x for x in rows if str(x.get(key)) == v])
    res["mean_good_total"] = round(acc(rows, lambda x: x["good"]["total"]), 3)
    res["mean_bad_total"] = round(acc(rows, lambda x: x["bad"]["total"]), 3)
    res["pass"] = res["overall"]["total"] >= 0.95
    res["weights"] = dict(syl=W_SYL, lik=W_LIK, glyph=W_GLYPH, word=W_WORD, extra=W_EXTRA)
    res["min_h"], res["wall_sec"] = min_h, time.time() - t0
    (d / f"check-{tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=Path, required=True, help="make_jamo_pairs.py output dir")
    ap.add_argument("--min-h", type=int, default=384, help="upscale the text crop to at least this height")
    ap.add_argument("--tag", default="v2", help="writes rewards-<tag>.jsonl and check-<tag>.json")
    ap.add_argument("--reuse", type=Path, help="earlier rewards.jsonl: reuse its glyph/word/extra layers")
    a = ap.parse_args()
    res = check(a.check.expanduser(), a.min_h, a.tag, a.reuse.expanduser() if a.reuse else None)
    print(json.dumps(res["overall"]), "PASS" if res["pass"] else "FAIL", flush=True)
    print("REWARD CHECK DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
