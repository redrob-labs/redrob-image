#!/usr/bin/env python3
"""AI Hub 105 (야외 실제 촬영 한글 이미지) -> webdataset shards of text crops and clean full scenes.

Layout (as extracted by infra/aihub_extract.py):
    <root>/030.*/01.데이터/{1.Training,2.Validation}/[원천]<zip name>/.../<stem>.jpg
    <root>/030.*/01.데이터/{1.Training,2.Validation}/[라벨]<split>/.../<stem>.json
    <root>/030.*/01.데이터/<split>/{원천,라벨링}데이터_230216_add/...   (add-on set, paired only with each other)
Images are paired with labels of the SAME split and the SAME set (base or _add) by file stem. Stems repeat
across the base and add-on sets with different content, so never pair across them.

Label JSON: images[0].{width,height}, annotations[].{text, bbox=[x, y, w, h]}. text "xxx" = unreadable.

Per image:
  - drop the whole image if any readable text fails the frozen unseen filter (synth/unseen.py),
    if it is a byte or dHash duplicate of an earlier image, or if no box is usable;
  - "full" sample when every box is in bounds and any unreadable text is too small to read at
    --full-check-res (so the caption can name all legible text);
  - "crop" samples: one window per readable box, searched over size, aspect (trainer buckets) and position,
    big enough that the text stays legible at --train-res, fully containing every readable box it touches
    and showing no unreadable text that would be legible at --train-res.
Mixing full vs crop is a training-config choice; every sample carries "kind".

Outputs (never commit -- AI Hub data): <out>/<split>/aihub105-<split>-XXXXX.tar (<key>.jpg + <key>.json),
<out>/index.jsonl (one line per sample, metadata only), <out>/drops.json.
Stats (numbers only, safe to commit): scripts/aihub105_report.py.

Usage (on the box, data stays in Seoul):
    python scripts/aihub105_parse.py --root ~/aihub/105/extracted --out ~/aihub/105/shards --workers 16
    # optional, once a private bucket exists (infra/s3_bucket.py):
    python scripts/aihub105_parse.py ... --s3 s3://<bucket>/aihub105/v1
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
from unseen import UnseenFilter  # noqa: E402

UNREADABLE = {"", "xxx", "XXX", "###"}
IMG_EXT = {".jpg", ".jpeg", ".png"}

_F: UnseenFilter | None = None
_A: argparse.Namespace | None = None


def readable(text: str) -> bool:
    t = text.strip()
    return t not in UNREADABLE and set(t.lower()) != {"x"}


def pair_key(path: Path, data_root: Path) -> tuple[str, str]:
    """(split, set) for a file: split = 1.Training / 2.Validation, set = base / add."""
    rel = path.relative_to(data_root).parts
    return rel[0], ("add" if "_add" in rel[1] else "base")


def collect81(root: Path) -> tuple[list[dict], collections.Counter]:
    """AI Hub 81 text-in-the-wild: one textinthewild_data_info.json (COCO-like) for every image.
    Text blocks are the 'word' boxes; a 'character' box outside every word box becomes its own block;
    'ignored' boxes are unreadable text. Labels travel inline in the job."""
    infos = sorted(root.rglob("textinthewild_data_info.json"))
    if not infos:
        raise SystemExit(f"no textinthewild_data_info.json under {root}")
    d = json.load(open(infos[0], encoding="utf-8"))
    files = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            if Path(f).suffix.lower() in IMG_EXT:
                files[f] = os.path.join(dp, f)
    by_img = collections.defaultdict(list)
    for an in d["annotations"]:
        by_img[an["image_id"]].append(an)
    jobs, drops = [], collections.Counter()
    for im in d["images"]:
        p = files.get(im["file_name"])
        if p is None:
            drops["image_not_downloaded"] += 1
            continue
        anns = by_img.get(im["id"], [])
        words = [a for a in anns if (a.get("attributes") or {}).get("class") == "word"]
        wboxes = [(a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3])
                  for a in words if isinstance(a.get("bbox"), list) and len(a["bbox"]) == 4]
        out = [{"text": a.get("text") or "", "bbox": a.get("bbox")} for a in words]
        for a in anns:
            c = (a.get("attributes") or {}).get("class")
            b = a.get("bbox")
            if c == "ignored":
                out.append({"text": "xxx", "bbox": b})
            elif c == "character" and isinstance(b, list) and len(b) == 4:
                cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
                if not any(w[0] <= cx <= w[2] and w[1] <= cy <= w[3] for w in wboxes):
                    out.append({"text": a.get("text") or "", "bbox": b})
        jobs.append({"img": p, "label": "", "ann": out, "size": [im["width"], im["height"]], "split": "train",
                     "set": "wild", "category": im.get("type", "").replace(" ", "_"), "subclass": ""})
    return jobs, drops


def collect_nvidia(root: Path) -> tuple[list[dict], collections.Counter]:
    """nvidia/OCR-Synthetic-Multilingual-v1 HDF5 files (CC BY 4.0). Every rendered line is labeled, so line
    boxes become text blocks and full pages are clean. Images stay inside the .h5; jobs point at (file, index)."""
    import h5py

    jobs, drops = [], collections.Counter()
    for h in sorted(root.rglob("*.h5")):
        split = "val" if "/validation/" in str(h) or "/test/" in str(h) else "train"
        with h5py.File(h, "r") as f:
            dims = f["dimensions"][:]
            for i, raw in enumerate(f["annotations"]):
                ann = json.loads(raw)
                lines = [{"text": ln.get("text") or "", "bbox": ln.get("bbox")} for ln in ann.get("line_bboxes", [])]
                if not lines:
                    drops["no_lines"] += 1
                    continue
                jobs.append({"img": f"{h}#{i}", "h5": str(h), "idx": i, "stem": f"{h.stem}-{i:06d}", "label": "", "ann": lines,
                             "size": [int(dims[i][1]), int(dims[i][0])], "split": split, "set": "ocrsyn",
                             "category": "document", "subclass": h.stem})
    return jobs, drops


_H5: dict = {}


def read_raw(job: dict) -> bytes:
    if "h5" not in job:
        return Path(job["img"]).read_bytes()
    import h5py

    if job["h5"] not in _H5:
        _H5[job["h5"]] = h5py.File(job["h5"], "r")
    return _H5[job["h5"]]["images"][job["idx"]].tobytes()


def collect(root: Path) -> tuple[list[dict], collections.Counter]:
    data_roots = sorted(root.glob("*/01.데이터"))
    if not data_roots:
        raise SystemExit(f"no */01.데이터 under {root}")
    jobs, drops = [], collections.Counter()
    for dr in data_roots:
        labels: dict[tuple, dict[str, Path]] = collections.defaultdict(dict)
        images: list[tuple[tuple, Path]] = []
        for dp, _, fs in os.walk(dr):
            for f in fs:
                p = Path(dp) / f
                ext = p.suffix.lower()
                if ext == ".json":
                    k = pair_key(p, dr)
                    if p.stem in labels[k]:
                        drops["label_stem_collision"] += 1
                    labels[k][p.stem] = p
                elif ext in IMG_EXT:
                    images.append((pair_key(p, dr), p))
        for k, p in sorted(images, key=lambda t: str(t[1])):
            lp = labels[k].get(p.stem)
            if lp is None:
                drops["no_label"] += 1
                continue
            jobs.append({"img": str(p), "label": str(lp), "split": "train" if k[0].startswith("1") else "val",
                         "set": k[1], "category": p.stem.split("_")[0], "subclass": p.stem.split("_")[1] if "_" in p.stem else ""})
    return jobs, drops


def dhash(im: Image.Image) -> int:
    g = im.convert("L").resize((9, 8), Image.BILINEAR)
    px = g.tobytes()
    return sum(1 << i for i in range(64) if px[(i // 8) * 9 + i % 8] > px[(i // 8) * 9 + i % 8 + 1])


def clamp_box(b, W, H):
    if not isinstance(b, (list, tuple)) or len(b) != 4 or any(not isinstance(v, (int, float)) for v in b):
        return None, 0.0
    x, y, w, h = (float(v) for v in b)
    x0, y0, x1, y1 = max(0.0, x), max(0.0, y), min(float(W), x + w), min(float(H), y + h)
    if x1 <= x0 or y1 <= y0:
        return None, 0.0
    kept = (x1 - x0) * (y1 - y0) / max(w * h, 1e-6)
    return (x0, y0, x1, y1), kept


def overlap(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def contains(c, b, tol=1.0) -> bool:
    return b[0] >= c[0] - tol and b[1] >= c[1] - tol and b[2] <= c[2] + tol and b[3] <= c[3] + tol


def encode(im: Image.Image, max_side: int) -> tuple[bytes, float, tuple[int, int]]:
    s = min(1.0, max_side / max(im.size))
    if s < 1.0:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95)
    return buf.getvalue(), s, im.size


def block(i, text, box, ox, oy, s):
    x0, y0, x1, y1 = ((box[0] - ox) * s, (box[1] - oy) * s, (box[2] - ox) * s, (box[3] - oy) * s)
    r = lambda v: round(v, 1)  # noqa: E731
    return {"index": i, "text": text, "bbox_xyxy": [r(x0), r(y0), r(x1), r(y1)],
            "polygon": [[r(x0), r(y0)], [r(x1), r(y0)], [r(x1), r(y1)], [r(x0), r(y1)]]}


ASPECTS = (1.0, 4 / 3, 3 / 4, 16 / 9, 9 / 16, 2.0, 0.5)


def crop_windows(boxes, W, H, a):
    """One window per readable box (w/h in ASPECTS, the trainer's buckets), searched over size, aspect and
    position so that it fully contains every readable box it touches and shows no unreadable text that would
    still be legible at --train-res. boxes: list of (xyxy, text|None)."""
    out, seen = [], set()
    for i, (b, t) in enumerate(boxes):
        if t is None:
            continue
        bw, bh = b[2] - b[0], b[3] - b[1]
        best = None
        for m in (a.crop_margin, 1.15, 1.6, 2.2):
            for ar in ASPECTS:
                # smallest window of this aspect that holds the box with margin m, at least min_crop on its short side
                w = max(bw * m, bh * m * ar, a.min_crop * max(ar, 1.0))
                h = w / ar
                if w > W or h > H:
                    continue
                sc = a.train_res / (w * h) ** 0.5
                if min(bw, bh) * sc < a.min_text_px:
                    continue
                # slide the window over every position that keeps the box inside it
                xs = _positions(b[0], b[2], w, W)
                ys = _positions(b[1], b[3], h, H)
                for y0 in ys:
                    for x0 in xs:
                        c = (x0, y0, x0 + w, y0 + h)
                        inside = _fits(c, boxes, sc, a)
                        if inside is None:
                            continue
                        cand = (len(inside), -abs(m - a.crop_margin), -abs(ar - 1.0), c, inside)
                        if best is None or cand[:3] > best[:3]:
                            best = cand
            if best is not None:
                break
        if best is None:
            continue
        c, inside = best[3], best[4]
        sig = tuple(inside)
        if sig in seen:
            continue
        seen.add(sig)
        out.append((c, inside))
    return out


def _positions(lo, hi, size, limit, steps=5):
    a0, a1 = max(0.0, hi - size), min(lo, limit - size)
    if a1 < a0:
        return []
    if a1 - a0 < 1:
        return [a0]
    mid = (a0 + a1) / 2
    pts = [mid] + [a0 + (a1 - a0) * k / (steps - 1) for k in range(steps)]
    return pts


def _fits(c, boxes, sc, a):
    inside = []
    for j, (o, ot) in enumerate(boxes):
        if overlap(c, o) <= 0:
            continue
        if ot is None:
            # unreadable text is allowed only while it is too small to read at the trained resolution
            if min(o[2] - o[0], o[3] - o[1]) * sc >= a.ignore_text_px:
                return None
            continue
        if not contains(c, o):
            return None
        if a.min_inner_text_px and min(o[2] - o[0], o[3] - o[1]) * sc < a.min_inner_text_px:
            return None  # labeled text too small to draw at this resolution would teach a caption it cannot follow
        inside.append(j)
    return inside


def init(a):
    global _F, _A
    _F, _A = UnseenFilter.load(ROOT), a


def work(args) -> dict:
    chunk_id, split, jobs = args
    a, F = _A, _F
    shard = Path(a.out) / split / f"{a.source}-{split}-{chunk_id:05d}.tar"
    shard.parent.mkdir(parents=True, exist_ok=True)
    tmp = shard.with_suffix(".tar.part")
    index, drops = [], collections.Counter()
    with tarfile.open(tmp, "w") as tf:
        def put(key, jpg, meta):
            for name, data in ((f"{key}.jpg", jpg), (f"{key}.json", json.dumps(meta, ensure_ascii=False).encode())):
                ti = tarfile.TarInfo(name)
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
            index.append({k: v for k, v in meta.items() if k != "text_blocks"} |
                         {"texts": [b["text"] for b in meta["text_blocks"]], "shard": shard.name})

        for job in jobs:
            try:
                if "ann" in job:
                    d = {"images": [{"width": job["size"][0], "height": job["size"][1]}], "annotations": job["ann"]}
                else:
                    d = json.load(open(job["label"], encoding="utf-8"))
                im = load_image(job, d)
            except Exception:
                drops["unreadable_file"] += 1
                continue
            if im is None:
                drops["size_mismatch"] += 1
                continue
            W, H = im.size
            boxes, bad_label, texts = [], False, []
            for an in d.get("annotations") or []:
                b, kept = clamp_box(an.get("bbox"), W, H)
                t = an.get("text") or ""
                if b is None or kept < 0.5:
                    bad_label = True
                    continue
                if readable(t):
                    t = " ".join(t.split())
                    texts.append(t)
                    boxes.append((b, t))
                else:
                    boxes.append((b, None))
            if not any(t for _, t in boxes):
                drops["no_readable_text"] += 1
                continue
            if any(F.reason(t) for t in texts):
                drops["unseen"] += 1
                continue
            src_stem = job.get("stem") or Path(job["img"]).stem
            base = {"source": a.source.replace("aihub", "aihub-"), "split": split, "set": job["set"], "category": job["category"],
                    "subclass": job["subclass"], "src_stem": src_stem, "src_size": [W, H],
                    "md5": job["md5"], "dhash": job["dhash"]}
            stem = f"{job['set']}-{src_stem}"
            before = len(index)
            sc_full = a.full_check_res / (W * H) ** 0.5
            unlabeled_legible = any(t is None and min(b[2] - b[0], b[3] - b[1]) * sc_full >= a.ignore_text_px
                                    for b, t in boxes)
            if not bad_label and not unlabeled_legible:
                jpg, s, (w2, h2) = encode(im, a.full_max_side)
                put(f"{stem}-full", jpg, base | {"kind": "full", "width": w2, "height": h2, "window": [0, 0, W, H],
                                               "text_blocks": [block(i, t, b, 0, 0, s)
                                                               for i, (b, t) in enumerate(boxes) if t is not None]})
            else:
                drops["full_skipped_unlabeled_text" if not bad_label else "full_skipped_bad_bbox"] += 1
            if bad_label:  # a box we could not place may sit inside any window
                drops["crops_skipped_bad_bbox"] += 1
                continue
            for n, (c, inside) in enumerate(crop_windows(boxes, W, H, a)):
                ci = tuple(int(round(v)) for v in c)
                sub = im.crop(ci)
                jpg, s, (w2, h2) = encode(sub, a.crop_max_side)
                put(f"{stem}-c{n}", jpg, base | {"kind": "crop", "width": w2, "height": h2, "window": list(ci),
                                                 "text_blocks": [block(k, boxes[j][1], boxes[j][0], ci[0], ci[1], s)
                                                                 for k, j in enumerate(inside)]})
            if len(index) == before:
                drops["no_usable_sample"] += 1
    tmp.rename(shard)
    return {"shard": str(shard), "index": index, "drops": drops}


def load_image(job: dict, d: dict) -> Image.Image | None:
    """Image in the orientation its label was drawn on, or None if neither orientation matches."""
    im = Image.open(io.BytesIO(read_raw(job)))
    want = (d["images"][0]["width"], d["images"][0]["height"])
    if im.size == want:
        return im.convert("RGB")
    t = ImageOps.exif_transpose(im)
    return t.convert("RGB") if t.size == want else None


def hash_job(job: dict) -> dict:
    raw = read_raw(job)
    try:
        h = f"{dhash(Image.open(io.BytesIO(raw))):016x}"
    except Exception:
        h = "unreadable"
    return job | {"md5": hashlib.md5(raw).hexdigest(), "dhash": h}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--dataset", choices=["105", "81", "nvidia"], default="105", help="dataset layout of --root")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--per-shard", type=int, default=300, help="source images per shard")
    ap.add_argument("--train-res", type=int, default=512, help="resolution the crop is trained at")
    ap.add_argument("--min-text-px", type=float, default=20, help="min text short side at --train-res")
    ap.add_argument("--min-crop", type=float, default=384, help="min crop side in source pixels")
    ap.add_argument("--crop-margin", type=float, default=1.3, help="preferred window side / box side")
    ap.add_argument("--ignore-text-px", type=float, default=8,
                    help="unreadable text whose short side is below this at --train-res may sit inside a crop")
    ap.add_argument("--crop-max-side", type=int, default=1024)
    ap.add_argument("--min-inner-text-px", type=float, default=0,
                    help="reject a crop if any labeled text in it is below this at --train-res (0 = off, as in v1)")
    ap.add_argument("--full-max-side", type=int, default=1600)
    ap.add_argument("--full-check-res", type=int, default=1024,
                    help="highest resolution full scenes are trained at; unreadable text must be illegible there")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (dry run)")
    ap.add_argument("--s3", default="", help="s3://private-bucket/prefix to sync shards to (ap-northeast-2)")
    a = ap.parse_args()
    a.root, a.out = a.root.expanduser(), a.out.expanduser()
    a.source = {"105": "aihub105", "81": "aihub81", "nvidia": "nvidia_ocr_ko"}[a.dataset]

    jobs, drops = {"105": collect, "81": collect81, "nvidia": collect_nvidia}[a.dataset](a.root)
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} paired images; hashing for duplicates", flush=True)
    with mp.Pool(a.workers) as pool:
        hashed = pool.map(hash_job, jobs, chunksize=64)
    # Keep the first copy (by path) of byte-identical or same-dHash images; train copies win over val.
    seen_md5, seen_dh, jobs = set(), set(), []
    for j in sorted(hashed, key=lambda j: (j["split"] != "train", j["img"])):
        if j["md5"] in seen_md5:
            drops["duplicate_md5"] += 1
            continue
        if j["dhash"] != "unreadable" and j["dhash"] in seen_dh:
            drops["duplicate_dhash"] += 1
            continue
        seen_md5.add(j["md5"])
        seen_dh.add(j["dhash"])
        jobs.append(j)
    by_split = collections.defaultdict(list)
    for j in jobs:
        by_split[j["split"]].append(j)
    tasks, cid = [], 0
    for split, js in sorted(by_split.items()):
        for s in range(0, len(js), a.per_shard):
            tasks.append((cid, split, js[s: s + a.per_shard]))
            cid += 1
    print(f"{len(jobs)} unique images -> {len(tasks)} shards, {a.workers} workers", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    index = []
    with mp.Pool(a.workers, initializer=init, initargs=(a,)) as pool:
        for n, r in enumerate(pool.imap_unordered(work, tasks), 1):
            index.extend(r["index"])
            drops.update(r["drops"])
            if n % 10 == 0 or n == len(tasks):
                print(f"{n}/{len(tasks)} shards", flush=True)
    with open(a.out / "index.jsonl", "w", encoding="utf-8") as f:
        for row in sorted(index, key=lambda r: (r["split"], r["shard"], r["src_stem"], r["kind"])):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (a.out / "drops.json").write_text(json.dumps(dict(drops), indent=2), encoding="utf-8")
    (a.out / "run.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items()}, indent=2), encoding="utf-8")
    print(f"samples {len(index)}; drops {dict(drops)}", flush=True)

    if a.s3:
        if not a.s3.startswith("s3://"):
            raise SystemExit("--s3 must be s3://bucket/prefix")
        subprocess.run(["aws", "s3", "sync", str(a.out), a.s3, "--region", "ap-northeast-2",
                        "--exclude", "*.part", "--only-show-errors"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
