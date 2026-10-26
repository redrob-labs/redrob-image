#!/usr/bin/env python3
"""Plan C stage 1 training prompts (deterministic): 2,000 texts that never touch the eval set.

  1-2: 400   3-5: 600   from data/corpus/words.txt (frequency ranks 500-80,000, Hangul only)
  6-10: 600  11+: 400   word spans cut from data/corpus/sentences.txt (Hangul and spaces only)
At least 40% of each bucket has a hard syllable (double consonant or compound vowel ㅘㅙㅚㅝㅞㅟㅢㅒㅖ),
since those are where r64 v2 still fails.

Excluded: any eval target or word in it (eval/prompts.jsonl, prompts_unseen_syllables.jsonl) and any text
with a syllable from eval/unseen_syllables.txt. Scenes and colours as make_prompts.render.

  python eval/make_train_prompts.py  ->  eval/train_prompts_c1.jsonl
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from make_prompts import BUCKETS, COLORS, ROOT, SCENES, bucket_of, is_syl, read_list, render

SEED = 20261026
QUOTA = {"1-2": 400, "3-5": 600, "6-10": 600, "11+": 400}
HARD_V = {9, 10, 11, 14, 15, 16, 19, 3, 7}  # ㅘㅙㅚㅝㅞㅟㅢㅒㅖ (jungseong index)
HARD_C = {1, 4, 8, 10, 13}  # ㄲㄸㅃㅆㅉ (choseong index)


def hard(t: str) -> bool:
    for c in t:
        if is_syl(c):
            k = ord(c) - 0xAC00
            if k // 588 in HARD_C or (k % 588) // 28 in HARD_V:
                return True
    return False


def main() -> None:
    rng = random.Random(SEED)
    corpus = ROOT.parent / "data/corpus"
    ev = [json.loads(l) for f in ("prompts.jsonl", "prompts_unseen_syllables.jsonl")
          for l in (ROOT / f).read_text(encoding="utf-8").splitlines() if l.strip()]
    banned_text = {r["target_text"] for r in ev} | {w for r in ev for w in r["target_text"].split()}
    banned_text |= {w for _, w in read_list(ROOT / "unseen_words.txt")}
    banned_syl = set("".join(s for _, s in read_list(ROOT / "unseen_syllables.txt")))

    def ok(t: str) -> bool:
        return (t not in banned_text and not any(w in banned_text for w in t.split())
                and not any(c in banned_syl for c in t))

    words = []
    for k, line in enumerate((corpus / "words.txt").read_text(encoding="utf-8").splitlines()):
        if not 500 <= k < 80000:
            continue
        w = line.split("\t")[0]
        if w and all(is_syl(c) for c in w) and ok(w):
            words.append(w)
    spans = []
    for line in (corpus / "sentences.txt").read_text(encoding="utf-8").splitlines()[:200000]:
        ws = [w for w in re.split(r"\s+", re.sub(r"[^가-힣\s]", " ", line)) if w]
        if len(ws) < 2:
            continue
        i = rng.randrange(len(ws))
        for j in range(i + 1, min(len(ws), i + 8) + 1):
            t = " ".join(ws[i:j])
            if bucket_of(t) in ("6-10", "11+") and ok(t) and sum(is_syl(c) for c in t) <= 18:
                spans.append(t)
    pools = {b: [] for b in BUCKETS}
    for t in set(words) | set(spans):
        b = bucket_of(t)
        if (b in ("1-2", "3-5")) == (" " not in t):
            pools[b].append(t)
    rows = []
    for b in BUCKETS:
        pool = sorted(pools[b])
        rng.shuffle(pool)
        h = [t for t in pool if hard(t)]
        e = [t for t in pool if not hard(t)]
        n_h = int(QUOTA[b] * 0.4)
        chosen = h[:n_h] + e[:QUOTA[b] - n_h]
        assert len(chosen) == QUOTA[b], (b, len(h), len(e))
        rng.shuffle(chosen)
        for i, t in enumerate(chosen):
            scene = SCENES[i % 5]
            ko, en = render(scene, t, COLORS[rng.randrange(len(COLORS))])
            rows.append(dict(id=f"c1-{b}-{i:04d}", split="train_c1", bucket=b, scene=scene, target_text=t,
                             has_unseen_syllable=False, hard=hard(t), prompt_ko=ko, prompt_en=en))
    out = ROOT / "train_prompts_c1.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(len(rows), "->", out, {b: sum(r["hard"] for r in rows if r["bucket"] == b) for b in BUCKETS})


if __name__ == "__main__":
    main()
