#!/usr/bin/env python3
"""Write eval/unseen_syllables.txt: Hangul syllables held out of all training data.

The set is sampled (fixed seed) from the 8,822 modern syllables that are NOT in
KS X 1001 (the 2,350 common ones), so stage-1 coverage of KS X 1001 is untouched.
Any training sample whose text contains one of these syllables is dropped.
"""
from __future__ import annotations

import random
from pathlib import Path

SEED = 20261019
COUNT = 120


def ksx1001(ch: str) -> bool:
    # Python's euc-kr codec also encodes non-KS X 1001 syllables as 8-byte
    # jamo sequences (KS X 1001 annex 3); only 2-byte results are in the 2,350 set.
    try:
        return len(ch.encode("euc-kr")) == 2
    except UnicodeEncodeError:
        return False


def main() -> None:
    all_syl = [chr(c) for c in range(0xAC00, 0xD7A4)]
    rare = [s for s in all_syl if not ksx1001(s)]
    assert len(all_syl) == 11172 and len(rare) == 11172 - 2350, (len(all_syl), len(rare))
    picked = sorted(random.Random(SEED).sample(rare, COUNT))
    out = Path(__file__).with_name("unseen_syllables.txt")
    out.write_text("\n".join(picked) + "\n", encoding="utf-8")
    print(f"wrote {len(picked)} syllables to {out}")


if __name__ == "__main__":
    main()
