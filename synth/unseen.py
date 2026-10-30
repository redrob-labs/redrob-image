"""Held-out filter: reject any text that contains a frozen unseen word or syllable.

Rules (docs/plan.md, eval/): a text is rejected if
  - it contains ANY syllable listed in eval/unseen_syllables.txt, or
  - after removing all whitespace, it contains any entry of eval/unseen_words.txt
    (also with whitespace removed) as a substring.
The eval files are FROZEN and only read here.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_WS = re.compile(r"\s+")


def _lines(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


class UnseenFilter:
    def __init__(self, words: list[str], syllables: list[str]):
        self.words = sorted({_WS.sub("", w) for w in words if _WS.sub("", w)}, key=len)
        self.syllables = frozenset(s for line in syllables for s in line if not s.isspace())
        # Words longer than one syllable are checked by substring; group by first char for speed.
        self._by_first: dict[str, list[str]] = {}
        for w in self.words:
            self._by_first.setdefault(w[0], []).append(w)

    @classmethod
    def load(cls, root: Path = ROOT) -> "UnseenFilter":
        return cls(_lines(root / "eval" / "unseen_words.txt"), _lines(root / "eval" / "unseen_syllables.txt"))

    def bad_syllable(self, ch: str) -> bool:
        return ch in self.syllables

    def reason(self, text: str) -> str | None:
        """None if clean, else a short reason string."""
        for ch in text:
            if ch in self.syllables:
                return f"syllable:{ch}"
        t = _WS.sub("", text)
        for i, ch in enumerate(t):
            for w in self._by_first.get(ch, ()):
                if t.startswith(w, i):
                    return f"word:{w}"
        return None

    def ok(self, text: str) -> bool:
        return self.reason(text) is None
