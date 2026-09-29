"""Tibetan function labels and dharmamitra grammar-explained conventions on top of the tagger.

This module is the deployment layer that the dharmamitra main backend runs over the Tibetan tagger
output in its grammar-explained mode (backend: services/tibetan_grammar.py, branch feat/tibetan-tagger).
It is ported here verbatim where possible so that this package produces the same function labels:

* `tibetan_rules.py`   — the original rule layer (24-25 September 2026): case particle vs converb after
  a noun vs after a verb (Faggionato, Meelen & Hill 2023, §3.3/§3.5; the principle from Garrett & Hill
  2017), clitics (§3.4), determiners (§3.6), relator nouns (§3.8), negation (§3.10), fused
  demonstrative+case forms, and verb-stem tense notes from `tibetan_verbs.csv`.
* `tibetan_rules_ext.py` — the six rule families added in the 25 September ablation, on by default
  (env TIB_RULE_STEPS): special verbs (§3.14.2), pronouns (§3.12), adverbs (§3.2), nominalisers (§3.9),
  list gaps (§3.6/§3.8), and Garrett & Hill regex-tagger disambiguations (sandhi of final/question
  particles, semi-final de only after -d, su only after -s, gyis + shig, tense from context, small
  lexical rules).
* `rule_function()` below — the backend's dispatcher over both files, unchanged.

Backend display/link conventions (services/knn_translate_gemini.py) are also mirrored: case-ending
stripping for dictionary lookup, the trailing-tsheg citation form, Steinert links with the
single-syllable / real-headword rule, and the WordEvent record shape (typing_models/grammar_events.py).
"""
from __future__ import annotations
import json
import re
from typing import Callable, Dict, List, Optional
from urllib.parse import quote

from . import tibetan_rules as R
from . import tibetan_rules_ext as X

try:
    import pyewts
    _EWTS = pyewts.pyewts()
except Exception:  # pragma: no cover
    _EWTS = None

# ---------------------------------------------------------------- tag set (terse scheme; manual tags collapsed)
POS_NAMES: Dict[str, str] = {
    "N": "noun", "P": "proper noun", "V": "verb", "G": "verbal noun", "J": "adjective",
    "A": "adverb", "M": "numeral", "R": "pronoun", "D": "determiner", "X": "negation",
    "C": "case particle", "K": "clause particle", "L": "clitic or final particle",
    "I": "interjection", "p": "punctuation",
}

# affix tokens (+'i, +s, +r, +'o, +'am, +'ang, +'u): backend services/tibetan_grammar.py AFFIX_INFO
AFFIX_INFO = {
    "'i": ("genitive particle", "of; links the preceding word to the following noun"),
    "s": ("agentive/instrumental particle", "by; marks the agent or the instrument"),
    "r": ("terminative particle", "to, as, in; marks the goal, purpose or manner"),
    "'o": ("sentence-final particle", "closes the statement"),
    "'am": ("question / alternative particle", "or; whether"),
    "'ang": ("focus particle", "also, even"),
    "'u": ("diminutive suffix", "little"),
}


def _nw(s: str) -> str:
    return " ".join((s or "").replace("_", " ").split()).strip()


# ---------------------------------------------------------------- rule dispatcher (backend rule_function, verbatim)
def rule_function(w: dict, prev: dict = None, nxt: dict = None) -> str:
    """Authoritative function text for a token from the manual's rules (particles,
    determiners, negation, relator nouns, verb stems); "" when no rule applies.
    w / prev / nxt: {"wylie", "pos", "affix", "first", "last", "next"} as built by `flat_tokens`."""
    wy = w.get("wylie", "")
    pos = w.get("pos", "")
    first = bool(w.get("first")) or prev is None
    last = bool(w.get("last")) or nxt is None
    if X.on("regex") and first:
        prev = None  # the regex tagger reads sentence punctuation: no context across a shad
    prev_pos = (prev or {}).get("pos")
    prev_wy = (prev or {}).get("wylie")
    next_wy = (nxt or {}).get("wylie")
    rx = X.regex_particle(wy, prev_wy, prev_pos, next_wy, last)
    if rx == "skip":
        return ""
    if rx:
        return f"{rx[0]}: {rx[1]}"
    next2_wy = ((nxt or {}).get("next") or {}).get("wylie")
    for lab in (X.list_label(wy, pos, prev_wy), X.adverb_label(wy, pos, prev_wy, first), X.pronoun_label(wy, pos, prev_wy, next_wy, next2_wy)):
        if lab:
            return f"{lab[0]}: {lab[1]}" if lab[1] else lab[0]
    nzc = X.split_nominaliser_case(wy) if pos in ("G", "N", "V", "K", "C", "A") else None
    if nzc:
        stem, nom, case = nzc
        sv = X.special_verb(stem, prev_wy, prev_pos, next_wy)
        vi = sv[0] if sv else R.verb_info(stem, prev_wy, next_wy)
        if vi:
            return f"verbal noun ({vi} + nominaliser {nom}) + {X.NOMZ_CASE[case]}"
    if pos in ("C", "K", "L", "D", "X") or (pos in ("A", "M", "J", "R", "I") and R.is_closed_class(wy)):
        lab = R.particle_label(wy, prev_pos, prev_wy)
        if lab:
            return f"{lab[0]}: {lab[1]}"
        return ""
    nz = X.split_nominaliser(wy, pos) if pos in ("G", "N", "V", "A") else None
    if nz:
        stem, nom = nz
        sv = X.special_verb(stem, prev_wy, prev_pos, next_wy)
        vi = R.verb_info(stem, prev_wy, next_wy)
        gloss = X.NOMINALISERS.get(nom, "")
        if sv:
            return f"verbal noun: {sv[0]} + nominaliser {nom} {gloss}".rstrip()
        if vi:
            return f"verbal noun: {vi} + nominaliser {nom} {gloss}".rstrip()
    if pos in ("N", "A"):
        lab = R.relator_label(wy, prev_wy, prev_pos, next_wy)
        if lab:
            return f"{lab[0]}: {lab[1]}"
        return ""
    if pos in ("V", "G"):
        nominalised = bool(re.search(r"\s(pa|ba|pa'i|ba'i|par|bar|pas|bas)$", wy.strip()))
        stem = re.sub(r"\s+(pa|ba|par|bar|pas|bas|pa'i|ba'i)$", "", _nw(wy))
        sv = X.special_verb(stem, prev_wy, prev_pos, next_wy)
        if sv:
            if pos == "G" or nominalised:
                return f"verbal noun: {sv[0]} ({sv[1]}) + nominaliser"
            return f"{sv[0]}: {sv[1]}"
        vi = R.verb_info(wy, prev_wy, next_wy)
        if pos == "G" or nominalised:
            return f"verbal noun: {vi} + nominaliser" if vi else ("verbal noun" if pos == "V" else "")
        return f"verb ({vi})" if vi else ""
    return ""


def flat_tokens(sentences) -> List[dict]:
    """Content tokens of parsed sentences (mitra_tagger.parse.Sentence) in order, each with
    prev/next pointers, in the dict shape rule_function expects (backend flat_tokens)."""
    toks: List[dict] = []
    for s in sentences:
        start = len(toks)
        for t in s.tokens:
            if t.is_punct:
                continue
            toks.append({"wylie": _nw(t.surface), "unicode": wylie_to_unicode(t.surface).strip("་"),
                         "pos": t.pos, "affix": t.is_affix, "first": len(toks) == start, "last": False, "_tok": t})
        if len(toks) > start:
            toks[-1]["last"] = True
    for i, d in enumerate(toks):
        d["prev"] = toks[i - 1] if i else None
        d["next"] = toks[i + 1] if i + 1 < len(toks) else None
    return toks


def function_for(tokdict: dict) -> str:
    """Spelled-out function for one flat token: the rule layer's label, else the affix table,
    else the plain POS name (what the backend shows when no rule applies)."""
    if tokdict.get("affix"):
        info = AFFIX_INFO.get(tokdict["wylie"])
        return f"{info[0]}: {info[1]}" if info else f"attached affix {tokdict['wylie']}"
    func = rule_function(tokdict, tokdict.get("prev"), tokdict.get("next"))
    return func or POS_NAMES.get(tokdict.get("pos", ""), tokdict.get("pos", ""))


# ---------------------------------------------------------------- backend display / link conventions (verbatim)
# Same list and order as dharmamitra-main-backend `_remove_tibetan_case_endings`.
_CASE_ENDINGS_FOR_LOOKUP = ["kyis", "gis", "las", "nas", "kyi", "gyi", "'i", "su", "ru", "tu", "du", "la", "na", "dang"]


def remove_case_endings(wylie_term: str) -> str:
    """Strip one trailing case ending from a Wylie term for dictionary lookup (backend-identical)."""
    term = wylie_term.strip()
    for ending in _CASE_ENDINGS_FOR_LOOKUP:
        if term.endswith(ending):
            if len(term) == len(ending):
                break
            term = term[: -len(ending)]
            break
    return term


def steinert_url(wylie_term: str) -> str:
    """Christian Steinert dictionary deep link (backend-identical payload)."""
    params = {"activeTerm": wylie_term, "lang": "tib", "inputLang": "tib", "currentListTerm": wylie_term,
              "forceLeftSideVisible": False, "offset": 0}
    return "https://dictionary.christian-steinert.de/#" + quote(json.dumps(params, separators=(",", ":")), safe="")


_TSHEG = "་"
_TERMINATORS = ("་", "༌", "།", "༎")


def with_trailing_tsheg(tib: str) -> str:
    """Display a Tibetan-script term the way dictionaries cite it: ending in exactly one tsheg."""
    if not tib:
        return tib
    tib = re.sub("་{2,}", _TSHEG, tib)
    return tib if tib[-1] in _TERMINATORS else tib + _TSHEG


def wylie_to_unicode(wylie: str) -> str:
    return _EWTS.toUnicode(wylie) if _EWTS else wylie


def unicode_to_wylie(tib: str) -> str:
    return _EWTS.toWylie(tib) if _EWTS else tib


def is_linkable(lemma_wylie: str, headword_check: Optional[Callable[[str], bool]] = None) -> bool:
    """Backend rule (_tib_lemma_is_linkable): single-syllable lemmas always link; multi-syllable
    lemmas only if a real headword (raw or after case-ending stripping). Without a dictionary
    callback, multi-syllable lemmas are not linked."""
    t = (lemma_wylie or "").strip()
    if not t:
        return False
    if " " not in t:
        return True
    if headword_check is None:
        return False
    if headword_check(t):
        return True
    stripped = remove_case_endings(t).strip()
    return bool(stripped) and stripped != t and headword_check(stripped)


def to_word_events(sentence, headword_check: Optional[Callable[[str], bool]] = None,
                   prev_sentence=None) -> List[dict]:
    """Backend-compatible WordEvent dicts (typing_models/grammar_events.py) for one parsed sentence,
    with `function` from the rule layer. `meaning` is left empty: it comes from a dictionary or an
    LLM, not from the tagger."""
    events = []
    for d in flat_tokens([sentence]):
        t = d["_tok"]
        lemma = d["wylie"]
        link = (not d["affix"]) and is_linkable(lemma, headword_check) and t.pos in ("N", "P", "V", "G", "J", "A", "M", "R")
        ev = {
            "type": "word",
            "surface": with_trailing_tsheg(wylie_to_unicode(t.surface)),
            "lemma": lemma,
            "transliteration": t.surface,
            "function": function_for(d),
            "meaning": "",
            "external_source": "steinert" if link else None,
            "external_url": steinert_url(remove_case_endings(lemma)) if link else None,
            "mitra": None,
        }
        if t.unit is not None:
            u = sentence.units[t.unit]
            ev["sanskrit_unit"] = {"index": t.unit, "text": u.text, "lemma": u.type or None}
        events.append(ev)
    return events
