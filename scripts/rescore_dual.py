"""Re-score finished eval runs with the strict two-engine rule (paddleocr AND qwen3vl-8b must agree).

Reuses each run's existing scores.jsonl and runs only the engine it is missing, on the saved images.
Writes scores_dual.jsonl, report_dual.json and report_dual.md next to the original report (untouched).

  python scripts/rescore_dual.py ~/runs/pilot-r32/final-eval-Z-Image ~/runs/pilot-r64v2/final-eval-Z-Image
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import metrics  # noqa: E402

DUAL = ["paddleocr", "qwen3vl-8b"]


def main(runs: list[str]) -> None:
    pp = Path.home() / "paddle-venv/bin/python"
    if pp.exists():
        os.environ.setdefault("REDROB_PADDLE_PY", str(pp))
    import ocr_engines
    from PIL import Image

    engs: dict = {}
    for run in map(Path, runs):
        if (run / "report_dual.json").exists():
            print("skip (done)", run)
            continue
        meta = json.loads((run / "report.json").read_text())["meta"]
        have = meta["ocr"]
        if have not in DUAL:
            print("skip (ocr", have, ")", run)
            continue
        need = DUAL[1] if have == DUAL[0] else DUAL[0]
        if need not in engs:
            engs[need] = ocr_engines.make(need)
        rows = [json.loads(l) for l in (run / "scores.jsonl").open(encoding="utf-8") if l.strip()]
        out = []
        for r in rows:
            other = engs[need](Image.open(run / "images" / f"{r['image_id']}.png"))
            per = {have: r["lines"], need: other}
            sc = {n: metrics.score_one(r["target_text"], per[n]) for n in DUAL}
            row = {k: v for k, v in r.items() if k not in ("cer", "cer_raw", "cer_nospace", "em", "best", "ocr_concat")}
            row.update(metrics.combine(sc[DUAL[0]], sc[DUAL[1]]))
            for n in DUAL:
                row[f"lines_{n}"], row[f"score_{n}"] = per[n], sc[n]
            out.append(row)
        with (run / "scores_dual.jsonl").open("w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        agg = metrics.aggregate(out)
        meta = dict(meta, ocr="dual", ocr_engines=DUAL)
        title = f"{run.name} OCR=dual({'+'.join(DUAL)})"
        (run / "report_dual.json").write_text(json.dumps(dict(meta=meta, metrics=agg), indent=1, ensure_ascii=False))
        (run / "report_dual.md").write_text(metrics.markdown(agg, title) + "\n")
        print(run, json.dumps(agg["overall"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
