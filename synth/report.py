#!/usr/bin/env python3
"""Coverage + statistics report for a synthetic run, read back from the webdataset shards.

Re-scans EVERY json in synth/out/shards/*.tar (block texts, wrapped lines and all three captions)
with the frozen unseen filter, so the leak count is measured on what was written, not on the plan.
Writes paper/data-stats/<name>.json and .md (numbers only) and synth/out/preview.html.

Usage:
    .venv/bin/python synth/report.py --name synth-10k --n 20 --n2 2 --preview 100
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import shutil
import statistics
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_source import KS, KS_SET  # noqa: E402
from unseen import UnseenFilter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "synth" / "out"
STATS = ROOT / "paper" / "data-stats"


def hist(values, edges):
    out = collections.OrderedDict()
    for lo, hi in zip(edges, edges[1:] + [None]):
        lab = f"{lo}-{hi - 1}" if hi else f"{lo}+"
        out[lab] = sum(1 for v in values if v >= lo and (hi is None or v < hi))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="synth-10k")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n2", type=int, default=2)
    ap.add_argument("--preview", type=int, default=100)
    a = ap.parse_args()
    flt = UnseenFilter.load()
    run = json.loads((OUT / "run.json").read_text())
    counts = collections.Counter()
    scenes, fams, styles, bgk, bglic, fontlic = (collections.Counter() for _ in range(6))
    block_len, img_len, nblocks, sizes = [], [], [], []
    leaks, missing_in_caption, poly_out, n_img, n_jpg, cov_blocks, total_blocks = [], 0, 0, 0, 0, 0, 0
    bg_files = set()
    metas_for_preview = {}
    shards = sorted((OUT / "shards").glob("*.tar"))
    want = set(range(0, run["count"], max(1, run["count"] // a.preview)))
    for sh in shards:
        with tarfile.open(sh) as tf:
            for m in tf.getmembers():
                if m.name.endswith(".jpg"):
                    n_jpg += 1
                    continue
                if not m.name.endswith(".json"):
                    continue
                d = json.loads(tf.extractfile(m).read().decode("utf-8"))
                n_img += 1
                W, H = d["width"], d["height"]
                scenes[d["scene"]] += 1
                bg = d["background"]
                bgk[f'{bg["type"]}:{bg["kind"]}'] += 1
                bglic[bg["license"]] += 1
                if bg["type"] == "photo":
                    bg_files.add(bg["file"])
                caps = d["captions"]
                for c in caps.values():
                    r = flt.reason(c)
                    if r:
                        leaks.append((d["key"], "caption", r))
                tot = 0
                nblocks.append(len(d["text_blocks"]))
                for b in d["text_blocks"]:
                    total_blocks += 1
                    cov_blocks += bool(b["coverage"])
                    for t in [b["text"]] + b["lines_text"]:
                        r = flt.reason(t)
                        if r:
                            leaks.append((d["key"], "text", r))
                    for ch in b["text"]:
                        if "가" <= ch <= "힣":
                            counts[ch] += 1
                    L = len(b["text"].replace("\n", ""))
                    block_len.append(L)
                    tot += L
                    fams[b["font_family"]] += 1
                    styles[b["font_style"]] += 1
                    fontlic[b["font_license"]] += 1
                    sizes.append(b["size"])
                    for ln in b["lines_text"]:
                        if any(f'"{ln}"' not in c for c in caps.values()):
                            missing_in_caption += 1
                    if any(not (0 <= x <= W and 0 <= y <= H) for x, y in b["polygon"]):
                        poly_out += 1
                img_len.append(tot)
                if int(d["key"]) in want:
                    metas_for_preview[d["key"]] = d
    ks_counts = [counts.get(s, 0) for s in KS]
    held_word_ks = [s for s in KS if not flt.ok(s)]
    ks_req = [s for s in KS if flt.ok(s)]
    other = [chr(c) for c in range(0xAC00, 0xD7A4) if chr(c) not in KS_SET]
    other_allowed = [s for s in other if not flt.bad_syllable(s)]
    rep = {
        "name": a.name, "images": n_img, "jpgs": n_jpg, "shards": len(shards),
        "images_per_sec": run["images_per_sec"], "render_seconds": run["render_seconds"], "workers": run["workers"],
        "seed": run["seed"], "stage": run["stage"], "n": a.n, "n2": a.n2, "render_errors": run["error_count"],
        "ksx1001": {
            "syllables": 2350, "min": min(counts.get(s, 0) for s in ks_req), "median": statistics.median(ks_counts),
            "max": max(ks_counts), "below_n": sum(1 for s in ks_req if counts.get(s, 0) < a.n),
            "held_out_as_unseen_words": {s: counts.get(s, 0) for s in held_word_ks},
            "min_including_held_out": min(ks_counts),
        },
        "hangul_11172": {
            "covered_at_least_1": sum(1 for c in range(0xAC00, 0xD7A4) if counts.get(chr(c), 0) > 0),
            "allowed_syllables": len(ks_req) + len(other_allowed),
            "non_ks_allowed": len(other_allowed),
            "non_ks_at_least_n2": sum(1 for s in other_allowed if counts.get(s, 0) >= a.n2),
            "non_ks_min": min(counts.get(s, 0) for s in other_allowed),
        },
        "unseen_filter": {
            "leaked_unseen_syllable_occurrences": sum(counts.get(s, 0) for s in flt.syllables),
            "leaked_texts_or_captions": len(leaks), "leak_examples": leaks[:10],
            "corpus_sentences_rejected_at_build": "see data/corpus build log (build_corpus.py prints unseen_rejected)",
            "render_time_rejects": run["render_time_leaks"],
        },
        "captions": {"lines_missing_from_any_caption": missing_in_caption},
        "polygons_outside_image": poly_out,
        "text_blocks": total_blocks, "coverage_blocks": cov_blocks,
        "blocks_per_image": dict(sorted(collections.Counter(nblocks).items())),
        "block_length_chars": {"mean": round(statistics.mean(block_len), 2), "median": statistics.median(block_len),
                               "hist": hist(block_len, [1, 3, 5, 9, 17, 33, 65])},
        "image_text_chars": {"mean": round(statistics.mean(img_len), 2), "median": statistics.median(img_len),
                             "hist": hist(img_len, [1, 11, 21, 41, 81, 121])},
        "font_size_px": {"min": min(sizes), "median": statistics.median(sizes), "max": max(sizes),
                         "hist": hist(sizes, [12, 18, 24, 32, 48, 72, 120, 200])},
        "scenes": dict(scenes.most_common()), "font_styles": dict(styles.most_common()),
        "font_families": dict(fams.most_common()), "font_families_used": len(fams),
        "font_licenses": dict(fontlic), "backgrounds": dict(bgk.most_common()), "background_licenses": dict(bglic),
        "distinct_photo_backgrounds_used": len(bg_files),
        "plan_info": run["plan_info"],
    }
    STATS.mkdir(parents=True, exist_ok=True)
    (STATS / f"{a.name}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    ks = rep["ksx1001"]
    hg = rep["hangul_11172"]
    lines = [
        f"# {a.name}: synthetic Korean text images", "",
        f"Generated by `synth/generate.py` (seed {rep['seed']}, stage {rep['stage']}, N={a.n}, N2={a.n2}). "
        "Numbers re-measured from the written shards by `synth/report.py`.", "",
        "| Item | Value |", "|---|---|",
        f"| Images / shards | {rep['images']} / {rep['shards']} |",
        f"| Throughput | {rep['images_per_sec']} images/s ({rep['workers']} CPU workers) |",
        f"| KS X 1001 count per syllable min / median / max | {ks['min']} / {ks['median']} / {ks['max']} |",
        f"| KS X 1001 syllables below N={a.n} | {ks['below_n']} |",
        f"| KS X 1001 syllables that are themselves unseen words (never rendered) | {', '.join(f'{k}={v}' for k, v in ks['held_out_as_unseen_words'].items())} |",
        f"| Hangul syllables covered (of 11,172) | {hg['covered_at_least_1']} (allowed: {hg['allowed_syllables']}) |",
        f"| Non-KS allowed syllables with >= N2={a.n2} | {hg['non_ks_at_least_n2']} / {hg['non_ks_allowed']} |",
        f"| Unseen syllable occurrences in output | {rep['unseen_filter']['leaked_unseen_syllable_occurrences']} |",
        f"| Texts/captions containing an unseen word or syllable | {rep['unseen_filter']['leaked_texts_or_captions']} |",
        f"| Rendered lines missing from a caption | {missing_in_caption} |",
        f"| Polygons outside the image | {poly_out} |",
        f"| Text blocks (coverage blocks) | {total_blocks} ({cov_blocks}) |",
        f"| Block length chars mean / median | {rep['block_length_chars']['mean']} / {rep['block_length_chars']['median']} |",
        f"| Text chars per image mean / median | {rep['image_text_chars']['mean']} / {rep['image_text_chars']['median']} |",
        f"| Font size px min / median / max | {rep['font_size_px']['min']} / {rep['font_size_px']['median']} / {rep['font_size_px']['max']} |",
        f"| Font families used | {rep['font_families_used']} (licenses: {rep['font_licenses']}) |",
        f"| Distinct photo/texture backgrounds used | {rep['distinct_photo_backgrounds_used']} (licenses: {rep['background_licenses']}) |",
        "", "## Distributions", "",
        "| Scene | Images |", "|---|---|", *[f"| {k} | {v} |" for k, v in rep["scenes"].items()], "",
        "| Font style | Blocks |", "|---|---|", *[f"| {k} | {v} |" for k, v in rep["font_styles"].items()], "",
        "| Font family | Blocks |", "|---|---|", *[f"| {k} | {v} |" for k, v in rep["font_families"].items()], "",
        "| Background | Images |", "|---|---|", *[f"| {k} | {v} |" for k, v in rep["backgrounds"].items()], "",
        "| Block length (chars) | Blocks |", "|---|---|", *[f"| {k} | {v} |" for k, v in rep["block_length_chars"]["hist"].items()], "",
        "| Text per image (chars) | Images |", "|---|---|", *[f"| {k} | {v} |" for k, v in rep["image_text_chars"]["hist"].items()], "",
        "| Blocks per image | Images |", "|---|---|", *[f"| {k} | {v} |" for k, v in rep["blocks_per_image"].items()], "",
    ]
    (STATS / f"{a.name}.md").write_text("\n".join(lines), encoding="utf-8")
    # ---- preview
    pdir = OUT / "preview"
    pdir.mkdir(exist_ok=True)
    cards = []
    for key in sorted(metas_for_preview):
        d = metas_for_preview[key]
        shutil.copy(OUT / "flat" / f"{key}.jpg", pdir / f"{key}.jpg")
        c = d["captions"]
        bg = d["background"]
        cards.append(
            f'<div class="c"><img src="preview/{key}.jpg" loading="lazy" width="{d["width"] // 3}" height="{d["height"] // 3}">'
            f'<p><b>{key}</b> · {d["scene"]} · {html.escape(bg["kind"])} ({html.escape(bg["license"])})</p>'
            f'<p class="ko">{html.escape(c["ko_long"])}</p><p class="ko">{html.escape(c["ko_short"])}</p><p>{html.escape(c["en"])}</p>'
            f'<p class="f">{html.escape(", ".join(b["font_family"] for b in d["text_blocks"]))}</p></div>')
    (OUT / "preview.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>synth preview</title><style>body{font-family:sans-serif;margin:16px}"
        ".c{display:inline-block;vertical-align:top;width:360px;margin:8px;font-size:12px}.c img{width:341px;height:auto;border:1px solid #ccc}"
        ".ko{color:#234}.f{color:#888}</style>"
        f"<h1>{a.name} preview ({len(cards)} of {n_img})</h1>" + "".join(cards), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("images", "images_per_sec", "ksx1001", "hangul_11172", "unseen_filter", "captions", "polygons_outside_image")}, ensure_ascii=False))
    bad = leaks or rep["unseen_filter"]["leaked_unseen_syllable_occurrences"] or missing_in_caption or ks["below_n"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
