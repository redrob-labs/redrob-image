#!/usr/bin/env python3
"""Statistics for an aihub105_parse.py run. Numbers only: no AI Hub text, file name or image leaves the box.

Reads <out>/index.jsonl, drops.json, run.json and writes paper/data-stats/<name>.json and .md.

Usage:
    python scripts/aihub105_report.py --out ~/aihub/105/shards --name aihub105-pilot
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
from text_source import KS_SET  # noqa: E402

STATS = ROOT / "paper" / "data-stats"
BUCKETS = [(1, 2), (3, 5), (6, 10), (11, None)]


def hangul(ch: str) -> bool:
    return "\uac00" <= ch <= "\ud7a3"


def bucket(n: int) -> str:
    for lo, hi in BUCKETS:
        if n >= lo and (hi is None or n <= hi):
            return f"{lo}-{hi}" if hi else f"{lo}+"
    return "0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", default="aihub105-pilot")
    a = ap.parse_args()
    out = a.out.expanduser()
    rows = [json.loads(ln) for ln in open(out / "index.jsonl", encoding="utf-8")]
    drops = json.loads((out / "drops.json").read_text(encoding="utf-8"))
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))

    kinds = collections.Counter((r["split"], r["kind"]) for r in rows)
    cats = collections.Counter((r["category"], r["kind"]) for r in rows)
    src_imgs = {(r["split"], r["set"], r["src_stem"]) for r in rows}
    full_imgs = {(r["split"], r["set"], r["src_stem"]) for r in rows if r["kind"] == "full"}
    lengths = collections.Counter()
    syl = collections.Counter()
    blocks_per = collections.Counter()
    nonhangul = 0
    per_image: dict = {}  # source image -> {text: max count in any one sample}; crops repeat full-scene text
    for r in rows:
        if r["split"] != "train":
            continue
        blocks_per[min(len(r["texts"]), 10)] += 1
        for t in r["texts"]:
            h = [c for c in t if hangul(c)]
            lengths[bucket(len(h))] += 1
            nonhangul += not h
        cur = per_image.setdefault((r["set"], r["src_stem"]), {})
        for t, n in collections.Counter(r["texts"]).items():
            cur[t] = max(cur.get(t, 0), n)
    for texts in per_image.values():
        for t, n in texts.items():
            for c in t:
                if hangul(c):
                    syl[c] += n
    ks_hit = sum(1 for c in KS_SET if syl[c] > 0)
    ks_ge = {n: sum(1 for c in KS_SET if syl[c] >= n) for n in (1, 5, 20, 100)}
    stats = {
        "name": a.name,
        "source": "AI Hub 105 (야외 실제 촬영 한글 이미지), pilot subset",
        "settings": {k: run[k] for k in ("train_res", "min_text_px", "min_crop", "crop_margin", "ignore_text_px",
                                         "crop_max_side", "full_max_side", "full_check_res")},
        "source_images_kept": len(src_imgs),
        "source_images_with_full_sample": len(full_imgs),
        "samples": {f"{s}/{k}": n for (s, k), n in sorted(kinds.items())},
        "samples_by_category": {f"{c}/{k}": n for (c, k), n in sorted(cats.items())},
        "crop_share": round(sum(n for (s, k), n in kinds.items() if k == "crop") / max(len(rows), 1), 3),
        "drops": drops,
        "train_text_blocks_by_syllables": {b: lengths.get(b, 0) for b in ["0"] + [bucket(lo) for lo, _ in BUCKETS]},
        "train_text_blocks_without_hangul": nonhangul,
        "train_blocks_per_sample": {str(k) + ("+" if k == 10 else ""): v for k, v in sorted(blocks_per.items())},
        "syllables_distinct": len(syl),
        "syllables_outside_ks": sum(1 for c in syl if c not in KS_SET),
        "ks2350_covered": ks_hit,
        "ks2350_with_at_least": ks_ge,
        "syllable_occurrences": sum(syl.values()),
    }
    STATS.mkdir(parents=True, exist_ok=True)
    (STATS / f"{a.name}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    md = [f"# {a.name}", "", f"Source: {stats['source']}. Numbers only; AI Hub text and images stay in Seoul.", "",
          "| Item | Value |", "|---|---:|",
          f"| Source images kept | {stats['source_images_kept']:,} |",
          f"| ...with a full-scene sample | {stats['source_images_with_full_sample']:,} |"]
    md += [f"| Samples {k} | {v:,} |" for k, v in stats["samples"].items()]
    md += [f"| Crop share | {stats['crop_share']:.1%} |",
           f"| Distinct syllables | {stats['syllables_distinct']:,} |",
           f"| Syllables outside KS X 1001 | {stats['syllables_outside_ks']:,} |",
           f"| KS X 1001 covered (of 2,350) | {ks_hit:,} |"]
    md += [f"| KS X 1001 with >= {n} | {v:,} |" for n, v in ks_ge.items()]
    md += ["", "## Samples by category", "", "| Category/kind | Samples |", "|---|---:|"]
    md += [f"| {k} | {v:,} |" for k, v in stats["samples_by_category"].items()]
    md += ["", "## Train text blocks by Hangul syllables", "", "| Syllables | Blocks |", "|---|---:|"]
    md += [f"| {k} | {v:,} |" for k, v in stats["train_text_blocks_by_syllables"].items()]
    md += ["", "## Drops", "", "| Reason | Count |", "|---|---:|"]
    md += [f"| {k} | {v:,} |" for k, v in sorted(drops.items())]
    md += ["", "## Settings", "", "| Setting | Value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in stats["settings"].items()]
    (STATS / f"{a.name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
