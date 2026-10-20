#!/usr/bin/env python3
"""Loop over an AI Hub dataset dir: rebuild zips the aihubshell merge left empty, extract every
complete zip (cp949 names) into <dir>/extracted/, then delete the zip to free disk.

  python3 infra/aihub_extract.py ~/aihub/105            # loop forever (60 s)
  python3 infra/aihub_extract.py ~/aihub/105 --once

A zip still being written has no central directory yet -> BadZipFile -> retried next pass.
Parts saved by infra/aihub_part_guard.sh live in <dir>/.parts-safe/<same relative path>.partN.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path


def name_of(info: zipfile.ZipInfo) -> str:
    if info.flag_bits & 0x800:
        return info.filename
    try:
        raw = info.filename.encode("cp437")
    except UnicodeEncodeError:  # already decoded properly (e.g. Unicode path extra field)
        return info.filename
    for enc in ("utf-8", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return info.filename


def rebuild(zp: Path, root: Path) -> bool:
    """Concatenate saved parts into a 0-byte zip. True if rebuilt."""
    safe = root / ".parts-safe" / zp.relative_to(root)
    parts = sorted(safe.parent.glob(safe.name + ".part*"), key=lambda p: int(re.sub(r"\D", "", p.suffix) or 0))
    if not parts:
        return False
    tmp = zp.with_suffix(".zip.rebuild")
    with tmp.open("wb") as out:
        for p in parts:
            with p.open("rb") as f:
                shutil.copyfileobj(f, out, 16 << 20)
    try:
        zipfile.ZipFile(tmp).close()
    except zipfile.BadZipFile:
        tmp.unlink()
        return False  # parts incomplete (guard still copying) -> next pass
    tmp.replace(zp)
    for p in parts:
        p.unlink()
    print(f"rebuilt {zp.relative_to(root)} from {len(parts)} parts", flush=True)
    return True


def extract(zp: Path, root: Path) -> bool:
    try:
        z = zipfile.ZipFile(zp)
    except zipfile.BadZipFile:
        return False  # still being written
    rel = zp.relative_to(root)
    dest = root / "extracted" / rel.parent / zp.stem
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with z:
        for info in z.infolist():
            name = name_of(info).lstrip("/\\")  # some AI Hub zips (71553) store absolute member names
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"unsafe path in {rel}: {name}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("wb") as out:  # CRC checked on read
                shutil.copyfileobj(src, out, 16 << 20)
            n += 1
    zp.unlink()
    (dest / ".EXTRACTED").write_text(f"{n}\n")
    print(f"extracted {rel} -> {n} files, zip removed", flush=True)
    return True


def one_pass(root: Path) -> None:
    for zp in sorted(root.rglob("*.zip")):
        if "extracted" in zp.relative_to(root).parts or ".parts-safe" in zp.relative_to(root).parts:
            continue
        try:
            if zp.stat().st_size == 0 and not rebuild(zp, root):
                continue
            extract(zp, root)
        except Exception as e:  # keep the loop alive; zip stays for the next pass
            print(f"error {zp.relative_to(root)}: {e!r}", file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=int, default=60)
    a = ap.parse_args()
    root = Path(a.root).expanduser().resolve()
    while True:
        one_pass(root)
        if a.once:
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
