"""Text-rendering metrics: CER and word exact match against OCR output.

CER for one image = min over OCR candidates of edit_distance(target, candidate) / len(target),
where target is the whitespace-normalized target_text and the candidates are
  (a) all detected lines concatenated in reading order (rows top-to-bottom, left-to-right), and
  (b) every contiguous span of those lines (includes every single line).
Two variants are reported:
  cer         - spaces kept (single spaces), as specified;
  cer_nospace - all whitespace removed from both sides (OCR spacing is unreliable; this is the
                variant that measures glyph correctness only).
Per-image CER is clipped to 1.0 before averaging (an empty or junk read costs 1.0, not more).
Exact match (em) = the nospace target equals the nospace string of some candidate.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def nospace(s: str) -> str:
    return re.sub(r"\s+", "", s)


def edit_distance(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein

        return Levenshtein.distance(a, b)
    except ImportError:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def reading_order(lines: list[dict]) -> list[dict]:
    """Group boxed lines into rows by vertical overlap, then sort rows top-down and lines left-right.
    Lines without boxes (VLM output) keep their given order."""
    if not lines or any(l.get("box") is None for l in lines):
        return list(lines)
    ls = sorted(lines, key=lambda l: (l["box"][1] + l["box"][3]) / 2)
    rows: list[list[dict]] = []
    for l in ls:
        y0, y1 = l["box"][1], l["box"][3]
        if rows:
            r = rows[-1]
            ry0 = min(x["box"][1] for x in r)
            ry1 = max(x["box"][3] for x in r)
            ov = min(y1, ry1) - max(y0, ry0)
            if ov > 0.5 * min(y1 - y0, ry1 - ry0):
                r.append(l)
                continue
        rows.append([l])
    return [l for r in rows for l in sorted(r, key=lambda x: x["box"][0])]


def candidates(lines: list[dict]) -> list[str]:
    """[full concat, every contiguous span of >=1 lines in reading order]. Spans cover the two
    specified candidates (full concat, best single line) and also drop unrelated scene text
    before/after the target (e.g. an extra 'OPEN' sign)."""
    ordered = reading_order(lines)[:30]
    t = [l["text"] for l in ordered]
    c = [" ".join(t)]
    c += [" ".join(t[i:j]) for i in range(len(t)) for j in range(i + 1, len(t) + 1)]
    return [norm(x) for x in c]


def score_one(target: str, lines: list[dict]) -> dict:
    t = norm(target)
    tn = nospace(t)
    cands = candidates(lines) or [""]
    best = min(cands, key=lambda c: edit_distance(t, c))
    cer = edit_distance(t, best) / max(1, len(t))
    cer_ns = min(edit_distance(tn, nospace(c)) for c in cands) / max(1, len(tn))
    em = any(nospace(c) == tn for c in cands)
    return dict(cer=min(1.0, cer), cer_raw=cer, cer_nospace=min(1.0, cer_ns), em=em, best=best,
                ocr_concat=cands[0])


def aggregate(rows: list[dict], keys=("split", "bucket", "scene", "has_unseen_syllable")) -> dict:
    """rows: prompt fields + score_one fields. Returns overall and per-key / per-(split,bucket) means."""

    def summ(rs):
        n = len(rs)
        return dict(
            n=n,
            cer=sum(r["cer"] for r in rs) / n,
            cer_nospace=sum(r["cer_nospace"] for r in rs) / n,
            em=sum(r["em"] for r in rs) / n,
        )

    out = {"overall": summ(rows)}
    for k in keys:
        g = defaultdict(list)
        for r in rows:
            g[str(r.get(k))].append(r)
        out[k] = {v: summ(rs) for v, rs in sorted(g.items())}
    if all("split" in r and "bucket" in r for r in rows):
        g = defaultdict(list)
        for r in rows:
            g[f"{r['split']}/{r['bucket']}"].append(r)
        out["split_bucket"] = {v: summ(rs) for v, rs in sorted(g.items())}
    return out


def markdown(agg: dict, title: str) -> str:
    def table(d: dict, head: str) -> list[str]:
        lines = [f"| {head} | n | CER | CER (no space) | EM |", "|---|---:|---:|---:|---:|"]
        for k, v in d.items():
            lines.append(f"| {k} | {v['n']} | {v['cer']:.3f} | {v['cer_nospace']:.3f} | {v['em']:.3f} |")
        return lines

    md = [f"# {title}", ""]
    md += table({"all": agg["overall"]}, "set") + [""]
    for k in ("split", "split_bucket", "bucket", "scene", "has_unseen_syllable"):
        if k in agg:
            md += [f"## by {k}", ""] + table(agg[k], k) + [""]
    return "\n".join(md)


if __name__ == "__main__":
    import sys

    # python eval/metrics.py scores.jsonl  -> prints markdown of an existing per-image score file
    rs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    print(markdown(aggregate(rs), sys.argv[1]))
