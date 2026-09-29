"""Parse the model's terse output into structured sentences/tokens and validate it against the input.

Terse format (both languages): one sentence per line; tokens separated by spaces; every token is
WORD/POS; punctuation tokens are MARK/p; a run of tokens that renders one Sanskrit word or compound is
enclosed in [ ... ], optionally followed by =TYPE (Chinese compound type) or =lemma.
Tibetan words join syllables with "_" and write an attached affix as its own token with leading "+".
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional

_TOK = re.compile(r"^(\[?)(\+?)(.+?)/([A-Za-z])(\](?:=([^\s\]]+))?)?$")


@dataclass
class Token:
    surface: str            # as in the input (Tibetan: syllables separated by spaces; affix without "+")
    pos: str                # one-letter tag
    is_punct: bool = False
    is_affix: bool = False  # Tibetan: affix written attached to the previous syllable ('i, s, r, 'o ...)
    unit: Optional[int] = None  # index into Sentence.units, if inside a [ ] bracket
    start: int = 0          # char offsets into the raw (letters-only for Tibetan / kanji-only for Chinese) text
    end: int = 0


@dataclass
class Unit:
    tokens: List[int]       # token indices
    type: str = ""          # Chinese compound type letter (T K D B A O) or "" ; Tibetan: Sanskrit lemma if given
    text: str = ""


@dataclass
class Sentence:
    tokens: List[Token] = field(default_factory=list)
    units: List[Unit] = field(default_factory=list)
    raw: str = ""           # the terse line

    def words(self):
        return [t for t in self.tokens if not t.is_punct]


def _letters_bo(s: str) -> str:
    return re.sub(r"[^a-zA-Z'+_ ]", "", s).replace("_", "").replace("+", "").replace(" ", "")


def _letters_zh(s: str) -> str:
    return "".join(c for c in s if _is_cjk(c))


def _is_cjk(c: str) -> bool:
    o = ord(c)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x20000 <= o <= 0x2EBEF or c == "〇"


def parse_terse(text: str, lang: str) -> List[Sentence]:
    """Parse terse output. Raises ValueError on a malformed token."""
    sents: List[Sentence] = []
    pos_cursor = 0
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        sent = Sentence(raw=line)
        open_unit: Optional[int] = None
        for tok in line.split():
            m = _TOK.match(tok)
            if not m:
                raise ValueError(f"malformed token: {tok!r}")
            opens, plus, word, pos, closes, meta = m.groups()
            if opens:
                sent.units.append(Unit(tokens=[]))
                open_unit = len(sent.units) - 1
            is_punct = pos == "p"
            surface = word.replace("_", " ") if lang == "bo" else word
            n = len(_letters_bo(word)) if lang == "bo" else len(_letters_zh(word))
            t = Token(surface=surface, pos=pos, is_punct=is_punct, is_affix=bool(plus),
                      unit=open_unit, start=pos_cursor, end=pos_cursor + n)
            if not is_punct:
                pos_cursor += n
            sent.tokens.append(t)
            if open_unit is not None and not is_punct:
                sent.units[open_unit].tokens.append(len(sent.tokens) - 1)
            if closes:
                if open_unit is not None:
                    u = sent.units[open_unit]
                    u.type = meta or ""
                    u.text = " ".join(sent.tokens[i].surface for i in u.tokens)
                open_unit = None
        sents.append(sent)
    return sents


def reconstruct(sents: List[Sentence], lang: str) -> str:
    """Letters-only (Tibetan) / kanji-only (Chinese) string implied by the annotation."""
    out = []
    for s in sents:
        for t in s.tokens:
            if t.is_punct:
                continue
            out.append(_letters_bo(t.surface.replace(" ", "_")) if lang == "bo" else _letters_zh(t.surface))
    return "".join(out)


def normalize_input(text: str, lang: str) -> str:
    """The comparable form of an input string: Tibetan letters only / Chinese kanji only."""
    return _letters_bo(text) if lang == "bo" else _letters_zh(text)


def is_faithful(sents: List[Sentence], text: str, lang: str) -> bool:
    return reconstruct(sents, lang) == normalize_input(text, lang)


def to_dict(sents: List[Sentence]) -> list:
    return [{"tokens": [asdict(t) for t in s.tokens], "units": [asdict(u) for u in s.units], "raw": s.raw} for s in sents]
