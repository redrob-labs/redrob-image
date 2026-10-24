#!/usr/bin/env python3
"""Download the Korean font set for synthetic data into data/fonts/ and write manifest.json.

Every family here is SIL Open Font License 1.1 (commercial use, embedding and redistribution
allowed; selling the font files alone is not). Google Fonts files are pinned to one commit of
github.com/google/fonts; Pretendard is pinned to its v1.3.9 release.

For each file the manifest records the source URL, sha256, and the measured glyph coverage of the
2,350 KS X 1001 syllables and of all 11,172 precomposed Hangul syllables (fontTools cmap).
A re-run verifies the sha256 of files already on disk against the existing manifest.

Usage:
    .venv/bin/python synth/fonts/fetch_fonts.py            # download + verify + write manifest
    .venv/bin/python synth/fonts/fetch_fonts.py --offline  # only re-measure files already on disk
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[2]
FONT_DIR = ROOT / "data" / "fonts"
LIC_DIR = Path(__file__).resolve().parent / "LICENSES"
MANIFEST = Path(__file__).resolve().parent / "manifest.json"

GF_COMMIT = "47d54ca1ebb14fd864695f1d39dc734f39017c33"
GF_RAW = f"https://raw.githubusercontent.com/google/fonts/{GF_COMMIT}/ofl"
PRETENDARD_TAG = "v1.3.9"
PRETENDARD_ZIP = f"https://github.com/orioncactus/pretendard/releases/download/{PRETENDARD_TAG}/Pretendard-1.3.9.zip"
PRETENDARD_LICENSE = f"https://raw.githubusercontent.com/orioncactus/pretendard/{PRETENDARD_TAG}/LICENSE"

# (family, google/fonts dir, style, [files])
GOOGLE = [
    ("Noto Sans KR", "notosanskr", "gothic", ["NotoSansKR[wght].ttf"]),
    ("Noto Serif KR", "notoserifkr", "myeongjo", ["NotoSerifKR[wght].ttf"]),
    ("Nanum Gothic", "nanumgothic", "gothic", ["NanumGothic-Regular.ttf", "NanumGothic-ExtraBold.ttf"]),
    ("Nanum Myeongjo", "nanummyeongjo", "myeongjo", ["NanumMyeongjo-Regular.ttf", "NanumMyeongjo-ExtraBold.ttf"]),
    ("Nanum Gothic Coding", "nanumgothiccoding", "gothic", ["NanumGothicCoding-Regular.ttf"]),
    ("Nanum Pen Script", "nanumpenscript", "handwriting", ["NanumPenScript-Regular.ttf"]),
    ("Nanum Brush Script", "nanumbrushscript", "brush", ["NanumBrushScript-Regular.ttf"]),
    ("Black Han Sans", "blackhansans", "display", ["BlackHanSans-Regular.ttf"]),
    ("Do Hyeon", "dohyeon", "display", ["DoHyeon-Regular.ttf"]),
    ("Jua", "jua", "rounded", ["Jua-Regular.ttf"]),
    ("Gaegu", "gaegu", "handwriting", ["Gaegu-Regular.ttf", "Gaegu-Bold.ttf"]),
    ("Gamja Flower", "gamjaflower", "handwriting", ["GamjaFlower-Regular.ttf"]),
    ("Gothic A1", "gothica1", "gothic", ["GothicA1-Light.ttf", "GothicA1-Regular.ttf", "GothicA1-Bold.ttf", "GothicA1-Black.ttf"]),
    ("Gowun Batang", "gowunbatang", "myeongjo", ["GowunBatang-Regular.ttf", "GowunBatang-Bold.ttf"]),
    ("Gowun Dodum", "gowundodum", "rounded", ["GowunDodum-Regular.ttf"]),
    ("Hi Melody", "himelody", "handwriting", ["HiMelody-Regular.ttf"]),
    ("IBM Plex Sans KR", "ibmplexsanskr", "gothic", ["IBMPlexSansKR-Light.ttf", "IBMPlexSansKR-Regular.ttf", "IBMPlexSansKR-Bold.ttf"]),
    ("Song Myung", "songmyung", "myeongjo", ["SongMyung-Regular.ttf"]),
    ("Sunflower", "sunflower", "gothic", ["Sunflower-Light.ttf", "Sunflower-Medium.ttf", "Sunflower-Bold.ttf"]),
    ("Yeon Sung", "yeonsung", "display", ["YeonSung-Regular.ttf"]),
    ("Dongle", "dongle", "rounded", ["Dongle-Regular.ttf", "Dongle-Bold.ttf"]),
    ("East Sea Dokdo", "eastseadokdo", "brush", ["EastSeaDokdo-Regular.ttf"]),
    ("Dokdo", "dokdo", "brush", ["Dokdo-Regular.ttf"]),
    ("Gugi", "gugi", "display", ["Gugi-Regular.ttf"]),
    ("Hahmlet", "hahmlet", "myeongjo", ["Hahmlet[wght].ttf"]),
    ("Orbit", "orbit", "display", ["Orbit-Regular.ttf"]),
    ("Bagel Fat One", "bagelfatone", "display", ["BagelFatOne-Regular.ttf"]),
    ("Kirang Haerang", "kiranghaerang", "brush", ["KirangHaerang-Regular.ttf"]),
    ("Poor Story", "poorstory", "handwriting", ["PoorStory-Regular.ttf"]),
    ("Single Day", "singleday", "handwriting", ["SingleDay-Regular.ttf"]),
    ("Stylish", "stylish", "rounded", ["Stylish-Regular.ttf"]),
    ("Cute Font", "cutefont", "handwriting", ["CuteFont-Regular.ttf"]),
    ("Black And White Picture", "blackandwhitepicture", "display", ["BlackAndWhitePicture-Regular.ttf"]),
    ("Diphylleia", "diphylleia", "myeongjo", ["Diphylleia-Regular.ttf"]),
    ("Gasoek One", "gasoekone", "display", ["GasoekOne-Regular.ttf"]),
    ("Moirai One", "moiraione", "display", ["MoiraiOne-Regular.ttf"]),
    ("Grandiflora One", "grandifloraone", "myeongjo", ["GrandifloraOne-Regular.ttf"]),
]
PRETENDARD_FILES = ["Pretendard-Light.otf", "Pretendard-Regular.otf", "Pretendard-SemiBold.otf", "Pretendard-Black.otf"]


def ksx1001_syllables() -> list[str]:
    """The 2,350 Hangul syllables of KS X 1001 (EUC-KR rows 0xB0-0xC8, cells 0xA1-0xFE)."""
    out = []
    for hi in range(0xB0, 0xC9):
        for lo in range(0xA1, 0xFF):
            out.append(bytes([hi, lo]).decode("euc_kr"))
    assert len(out) == 2350 and len(set(out)) == 2350
    return out


ALL_SYLLABLES = [chr(c) for c in range(0xAC00, 0xD7A4)]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "redrob-image-synth/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def coverage(path: Path) -> dict:
    font = TTFont(path, lazy=True)
    cmap = font.getBestCmap() or {}
    ks = ksx1001_syllables()
    return {
        "ksx1001": sum(1 for s in ks if ord(s) in cmap),
        "hangul_11172": sum(1 for s in ALL_SYLLABLES if ord(s) in cmap),
        "ascii_printable": sum(1 for c in range(0x20, 0x7F) if c in cmap),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    LIC_DIR.mkdir(parents=True, exist_ok=True)
    old = {}
    if MANIFEST.exists():
        for fam in json.loads(MANIFEST.read_text())["families"]:
            for f in fam["files"]:
                old[f["file"]] = f["sha256"]

    families = []
    pzip = None
    jobs = [(n, d, s, fs, "google") for n, d, s, fs in GOOGLE] + [("Pretendard", "pretendard", "gothic", PRETENDARD_FILES, "pretendard")]
    for name, d, style, files, src in jobs:
        lic_path = LIC_DIR / f"{name.replace(' ', '')}.txt"
        if src == "google":
            lic_url = f"{GF_RAW}/{d}/OFL.txt"
        else:
            lic_url = PRETENDARD_LICENSE
        if not args.offline and not lic_path.exists():
            lic_path.write_bytes(fetch(lic_url))
        lic_text = lic_path.read_text(encoding="utf-8", errors="replace")
        assert "SIL OPEN FONT LICENSE" in lic_text.upper(), f"{name}: license is not OFL"
        recs = []
        for fn in files:
            dst = FONT_DIR / fn
            if src == "google":
                url = f"{GF_RAW}/{d}/{fn}"
            else:
                url = f"{PRETENDARD_ZIP}#public/static/{fn}"
            if not dst.exists() and not args.offline:
                if src == "google":
                    data = fetch(url)
                else:
                    if pzip is None:
                        pzip = zipfile.ZipFile(io.BytesIO(fetch(PRETENDARD_ZIP)))
                    member = next(m for m in pzip.namelist() if m.endswith("/static/" + fn) or m == "public/static/" + fn)
                    data = pzip.read(member)
                dst.write_bytes(data)
            sha = hashlib.sha256(dst.read_bytes()).hexdigest()
            if fn in old and old[fn] != sha:
                print(f"sha256 mismatch for {fn}: {sha} != {old[fn]}", file=sys.stderr)
                return 1
            cov = coverage(dst)
            recs.append({"file": fn, "url": url, "sha256": sha, "variable": "[" in fn, "coverage": cov})
            print(f"{name:26s} {fn:36s} ks={cov['ksx1001']:4d} all={cov['hangul_11172']:5d}")
        families.append({
            "family": name, "style": style, "license": "OFL-1.1", "license_url": lic_url,
            "license_file": f"synth/fonts/LICENSES/{lic_path.name}", "files": recs,
        })
    manifest = {
        "note": "Font binaries live in data/fonts/ (gitignored). All families are SIL OFL 1.1.",
        "google_fonts_commit": GF_COMMIT, "pretendard_release": PRETENDARD_TAG,
        "ksx1001_total": 2350, "hangul_total": 11172,
        "families": families,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(families)} families, {sum(len(f['files']) for f in families)} files -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
