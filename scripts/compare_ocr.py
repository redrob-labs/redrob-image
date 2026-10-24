#!/usr/bin/env python3
"""Compare OCR engines on the rendered reference set.

  python scripts/compare_ocr.py --refset data/ocr_refset --engines paddleocr easyocr qwen3vl-8b --out paper/ocr

Writes <out>/ocr-compare.json (summary + per-engine breakdowns), <out>/ocr-compare.md, and
<out>/raw/<engine>.jsonl (per-crop output). Rare-syllable failure is measured as syllable recall:
the share of target syllables aligned as 'equal' with the best candidate, split by KS X 1001
membership (the 2,350 common syllables) vs. the other 8,822.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import metrics  # noqa: E402
import ocr_engines  # noqa: E402


def ksx(ch: str) -> bool:
    try:
        return len(ch.encode("euc-kr")) == 2
    except UnicodeEncodeError:
        return False


def syllable_hits(target: str, cand: str) -> list[tuple[str, bool]]:
    from rapidfuzz.distance import Levenshtein

    t, c = metrics.nospace(target), metrics.nospace(cand)
    hit = [False] * len(t)
    for op in Levenshtein.opcodes(t, c):
        if op.tag == "equal":
            for i in range(op.src_start, op.src_end):
                hit[i] = True
    return [(ch, h) for ch, h in zip(t, hit) if 0xAC00 <= ord(ch) <= 0xD7A3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refset", default="data/ocr_refset")
    ap.add_argument("--engines", nargs="+", default=["paddleocr", "easyocr", "qwen3vl-8b"])
    ap.add_argument("--out", default="paper/ocr")
    a = ap.parse_args()
    ref = Path(a.refset)
    items = [json.loads(l) for l in (ref / "manifest.jsonl").open(encoding="utf-8")]
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in a.engines:
        t0 = time.time()
        eng = ocr_engines.make(name)
        load_s = time.time() - t0
        eng(Image.open(ref / items[0]["file"]))  # warm-up
        rows, secs = [], []
        syl = defaultdict(lambda: [0, 0])
        with (out / "raw" / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for it in items:
                img = Image.open(ref / it["file"])
                t = time.time()
                lines = eng(img)
                secs.append(time.time() - t)
                s = metrics.score_one(it["text"], lines)
                for ch, h in syllable_hits(it["text"], s["best"]):
                    k = "ksx1001" if ksx(ch) else "non_ksx1001"
                    syl[k][0] += h
                    syl[k][1] += 1
                r = dict(it, **s, lines=[l["text"] for l in lines], sec=secs[-1])
                rows.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        agg = metrics.aggregate(rows, keys=("kind", "aug", "font"))
        agg.pop("split_bucket", None)
        summary[name] = dict(
            overall=agg["overall"], by_kind=agg["kind"], by_aug=agg["aug"], by_font=agg["font"],
            syllable_recall={k: dict(hit=v[0], n=v[1], recall=v[0] / max(1, v[1])) for k, v in syl.items()},
            sec_per_crop_mean=sum(secs) / len(secs), sec_per_crop_p50=sorted(secs)[len(secs) // 2],
            load_sec=load_s,
        )
        print(name, json.dumps(summary[name]["overall"]), summary[name]["syllable_recall"], flush=True)
        del eng
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
    meta = dict(n_crops=len(items), refset=str(ref), engines=a.engines)
    (out / "ocr-compare.json").write_text(json.dumps(dict(meta=meta, engines=summary), indent=1, ensure_ascii=False))
    md = ["# OCR comparison on the rendered reference set", "",
          f"{len(items)} crops. CER clipped at 1.0 per crop; EM = exact match ignoring spaces.", "",
          "| engine | CER | CER (no space) | EM | syl. recall KS X 1001 | syl. recall non-KS X 1001 | s/crop |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for n, s in summary.items():
        o, sr = s["overall"], s["syllable_recall"]
        md.append(f"| {n} | {o['cer']:.3f} | {o['cer_nospace']:.3f} | {o['em']:.3f} | "
                  f"{sr.get('ksx1001', {}).get('recall', 0):.3f} | {sr.get('non_ksx1001', {}).get('recall', 0):.3f} | "
                  f"{s['sec_per_crop_mean']:.3f} |")
    for key in ("by_kind", "by_aug", "by_font"):
        md += ["", f"## {key[3:]} (CER no space / EM)", "", "| " + key[3:] + " | " + " | ".join(summary) + " |",
               "|---|" + "---:|" * len(summary)]
        for k in next(iter(summary.values()))[key]:
            md.append(f"| {k} | " + " | ".join(f"{s[key][k]['cer_nospace']:.3f} / {s[key][k]['em']:.2f}"
                                             for s in summary.values()) + " |")
    (out / "ocr-compare.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
