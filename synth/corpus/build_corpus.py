#!/usr/bin/env python3
"""Build the Korean text corpus for synthetic images from a Korean Wikipedia dump.

Source: Korean Wikipedia, dump 20261001, file pages-articles1 (pages 1-82407).
License of the text: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), see
https://ko.wikipedia.org/wiki/위키백과:저작권 . We render short fragments (words, phrases, single
sentences) as glyph-learning material; attribution is recorded in docs/synth.md.

Output (gitignored, data/corpus/):
    sentences.txt   one cleaned sentence per line (8-48 chars, Hangul/ASCII/basic punctuation only)
    words.txt       word<TAB>count, Hangul-only eojeol stems (trailing particle stripped), 1-6 syllables
Unseen words/syllables are filtered here already, and again at generation time.

Usage:
    .venv/bin/python synth/corpus/build_corpus.py [--max-pages 40000]
"""
from __future__ import annotations

import argparse
import bz2
import collections
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unseen import UnseenFilter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CDIR = ROOT / "data" / "corpus"
DUMP_NAME = "kowiki-20261001-pages-articles1.xml-p1p82407.bz2"
DUMP_URL = f"https://dumps.wikimedia.org/kowiki/20261001/{DUMP_NAME}"
DUMP_SHA1 = "ce520af301952ce2ae4f1b0d31d1d2883e248390"

ALLOWED = re.compile(r"^[가-힣0-9A-Za-z ,.!?·()%\-~'\"]+$")
PARTICLES = sorted(["은", "는", "이", "가", "을", "를", "의", "에", "에서", "으로", "로", "와", "과", "도", "만",
                    "에게", "까지", "부터", "이다", "였다", "이며", "하고", "처럼"], key=len, reverse=True)


def strip_nested(text: str, op: str, cl: str) -> str:
    out, depth, i = [], 0, 0
    n = len(op)
    while i < len(text):
        if text.startswith(op, i):
            depth += 1
            i += n
        elif depth and text.startswith(cl, i):
            depth -= 1
            i += len(cl)
        else:
            if not depth:
                out.append(text[i])
            i += 1
    return "".join(out)


def clean_wikitext(t: str) -> str:
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>/]*/>", "", t)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<(math|code|pre|gallery|timeline|syntaxhighlight|score)[^>]*>.*?</\1>", "", t, flags=re.S)
    t = re.sub(r"<[^>]+>", "", t)
    t = strip_nested(t, "{{", "}}")
    t = strip_nested(t, "{|", "|}")
    t = re.sub(r"\[\[(?:파일|File|그림|Image|분류|Category):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", "", t)
    t = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", t)
    t = re.sub(r"'{2,}", "", t)
    lines = []
    for ln in t.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith(("=", "|", "!", "{", "}", "__")):
            continue
        ln = ln.lstrip("*#:; ")
        lines.append(ln)
    return "\n".join(lines)


def sentences(text: str):
    for ln in text.splitlines():
        for s in re.split(r"(?<=[.!?])\s+", ln):
            s = re.sub(r"\(\s*\)", "", s)
            s = re.sub(r"\s+", " ", s).strip()
            if 8 <= len(s) <= 48 and ALLOWED.match(s):
                hang = sum("가" <= c <= "힣" for c in s)
                if hang >= 0.6 * len(s.replace(" ", "")):
                    yield s


def stem(word: str) -> str:
    w = word.strip(",.!?·()\"'~-")
    for p in PARTICLES:
        if w.endswith(p) and len(w) - len(p) >= 2:
            return w[: -len(p)]
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=40000)
    a = ap.parse_args()
    CDIR.mkdir(parents=True, exist_ok=True)
    dump = CDIR / DUMP_NAME
    if not dump.exists():
        urllib.request.urlretrieve(DUMP_URL, dump)
    import hashlib
    h = hashlib.sha1(dump.read_bytes()).hexdigest()
    if h != DUMP_SHA1:
        print(f"dump sha1 {h} != {DUMP_SHA1}", file=sys.stderr)
        return 1
    flt = UnseenFilter.load()
    seen_s: set[str] = set()
    sents: list[str] = []
    words = collections.Counter()
    pages = rejected = 0
    with bz2.open(dump, "rb") as f:
        ns_t = None
        for ev, el in ET.iterparse(f, events=("end",)):
            tag = el.tag.rsplit("}", 1)[-1]
            if tag != "page":
                continue
            ns = el.find("{*}ns")
            rev = el.find("{*}revision/{*}text")
            if ns is not None and ns.text == "0" and el.find("{*}redirect") is None and rev is not None and rev.text:
                pages += 1
                for s in sentences(clean_wikitext(rev.text)):
                    if not flt.ok(s):
                        rejected += 1
                        continue
                    if s not in seen_s:
                        seen_s.add(s)
                        sents.append(s)
                    for w in s.split():
                        st = stem(w)
                        if 1 <= len(st) <= 6 and all("가" <= c <= "힣" for c in st) and flt.ok(st):
                            words[st] += 1
            el.clear()
            if pages >= a.max_pages:
                break
    (CDIR / "sentences.txt").write_text("\n".join(sents) + "\n", encoding="utf-8")
    (CDIR / "words.txt").write_text("".join(f"{w}\t{c}\n" for w, c in words.most_common()), encoding="utf-8")
    print(f"pages={pages} sentences={len(sents)} words={len(words)} unseen_rejected={rejected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
