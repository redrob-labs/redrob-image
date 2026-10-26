"""Compare the classifier reward with human labels on the r64 v2 human-check sheet.

    python scripts/human_check_compare.py --eval ~/runs/pilot-r64v2/final-eval-Z-Image \
        --key paper/runs/human-check-r64v2/key.json --right 1,2,4,5,9,11,18 --out <dir>/compare.json

Classifier verdict "right" = every target syllable is aligned to a cell whose argmax is that syllable
(no missing, no wrong cell); extra cells are reported but do not decide the verdict, as the human label
only judges the target text. Also reports how well the reward alone separates the two groups (AUC).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--right", required=True, help="sheet numbers the human marked right, comma separated")
    ap.add_argument("--model", type=Path, default=Path("~/runs/syl-cls-v2/model.pt"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import torch
    from PIL import Image
    import reward_cls as R

    key = json.loads(a.key.expanduser().read_text(encoding="utf-8"))
    right = {int(x) for x in a.right.split(",") if x.strip()}
    rows = {r["image_id"]: r for r in R.load_rows(a.eval.expanduser())}
    rw = R.ClsReward(a.model.expanduser())
    out = []
    for k in key:
        r = rows[k["image_id"]]
        img = Image.open(a.eval.expanduser() / "images" / f"{k['image_id']}.png").convert("RGB")
        lines = r.get("lines_paddleocr") or []
        tgt = [c for c in k["target_text"] if R.is_syl(c)]
        cells = [c for l in R.order([dict(l) for l in lines]) for c in R.line_cells(img, l)][: 4 * len(tgt) + 8]
        if cells and tgt:
            lp_all = rw._logp_all(cells)
            idx = torch.tensor([ord(c) - 0xAC00 for c in tgt])
            tot, path = R.align(lp_all[:, idx].tolist())
            arg = lp_all.argmax(1).tolist()
            ok = all(a_ == "m" and arg[i] == ord(tgt[j]) - 0xAC00 for a_, i, j in path if a_ != "x")
            read = "".join(chr(0xAC00 + int(arg[i])) for a_, i, j in path if a_ == "m")
            extra = sum(1 for a_, _, _ in path if a_ == "x")
            reward = tot / len(tgt)
        else:
            ok, read, extra, reward = False, "", 0, R.MISS
        out.append(dict(n=k["n"], target=k["target_text"], split=k["split"], human=k["n"] in right,
                        cls=ok, reward=round(reward, 3), read_aligned=read, extra_cells=extra,
                        dual_em=bool(r.get("em"))))
    agree = sum(o["human"] == o["cls"] for o in out)
    pos = [o["reward"] for o in out if o["human"]]
    neg = [o["reward"] for o in out if not o["human"]]
    auc = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / max(len(pos) * len(neg), 1)
    dual_agree = sum(o["human"] == o["dual_em"] for o in out)
    summ = dict(n=len(out), human_right=len(pos), cls_right=sum(o["cls"] for o in out), agree=agree,
                both_right=sum(o["human"] and o["cls"] for o in out),
                cls_right_human_wrong=sum(o["cls"] and not o["human"] for o in out),
                human_right_cls_wrong=sum(o["human"] and not o["cls"] for o in out),
                reward_auc=round(auc, 3), dual_ocr_agree=dual_agree)
    a.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    a.out.expanduser().write_text(json.dumps({"summary": summ, "items": out}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))
    for o in out:
        print(o["n"], o["target"], "H" if o["human"] else "-", "C" if o["cls"] else "-", o["reward"],
              o["read_aligned"], o["extra_cells"], "D" if o["dual_em"] else "-")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
