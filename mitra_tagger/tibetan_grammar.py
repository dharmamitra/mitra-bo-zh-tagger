"""Tibetan tag inventory, particle functions and the dharmamitra grammar-explained conventions.

The tables here mirror what the dharmamitra main backend hard-codes for its Tibetan
grammar-explained mode (services/knn_translate_gemini.py), so that tagger output can be turned
into the same WordEvent records the frontend renders:
  * segmentation at tsheg boundaries; multi-syllable words only for genuine lexemes; a stem and
    its case particle, an adverb and its verb, a verb and its auxiliary are always separate words;
  * lemmas in Wylie; case endings stripped for dictionary lookup (same list, same order);
  * surface forms shown in Tibetan script ending in exactly one tsheg (dictionary citation form);
  * a multi-syllable lemma only gets a dictionary link when it is a real headword; single
    syllables always link (Christian Steinert dictionary);
  * grammatical functions spelled out, never abbreviated tags.
"""
from __future__ import annotations
import json
import re
from typing import Callable, Dict, List, Optional
from urllib.parse import quote

try:
    import pyewts
    _EWTS = pyewts.pyewts()
except Exception:  # pragma: no cover
    _EWTS = None

# ---------------------------------------------------------------- tag set
POS_NAMES: Dict[str, str] = {
    "N": "noun", "P": "proper noun", "V": "verb", "G": "verbal noun", "J": "adjective",
    "A": "adverb", "M": "numeral", "R": "pronoun", "D": "determiner", "X": "negation",
    "C": "case particle", "K": "clause particle", "L": "clitic or final particle",
    "I": "interjection", "p": "punctuation",
}

# ---------------------------------------------------------------- particles
# Case particles (tag C) with the function the grammar-explained mode spells out.
CASE_PARTICLES: Dict[str, str] = {
    **{p: "agentive (instrumental) case particle" for p in ("kyis", "gyis", "gis", "yis", "'is", "s")},
    **{p: "genitive case particle" for p in ("kyi", "gyi", "gi", "yi", "'i")},
    "la": "dative-locative particle (la don)",
    "na": "locative particle (la don)",
    **{p: "terminative particle (la don)" for p in ("tu", "du", "ru", "su", "r")},
    "nas": "ablative particle (from, out of)",
    "las": "ablative particle (from; comparative than)",
    "dang": "associative particle (and, with)",
    "bas": "comparative particle (than)",
}

# Clause particles / converbs after a verb (tag K).
CLAUSE_PARTICLES: Dict[str, str] = {
    "nas": "clause connective (after doing ..., then)",
    **{p: "gerundive connective (and, having ...)" for p in ("te", "ste", "de")},
    **{p: "coordinating connective (and, while)" for p in ("zhing", "cing", "shing")},
    "na": "conditional / temporal connective (if, when)",
    **{p: "concessive connective (although, even)" for p in ("kyang", "yang", "'ang")},
    **{p: "continuative (while ...ing)" for p in ("gin", "kyin", "gyin", "yin")},
    "bas": "causal connective (because, since)",
    "pas": "causal connective (because, since)",
    "phyir": "purposive / causal (in order to, because)",
}

# Clitics and final particles (tag L).
FINAL_PARTICLES: Dict[str, str] = {
    **{p: "sentence-final particle" for p in ("'o", "so", "to", "do", "no", "bo", "mo", "ro", "lo", "ngo", "go", "'o")},
    **{p: "interrogative particle" for p in ("'am", "sam", "tam", "dam", "nam", "bam", "mam", "ram", "lam", "ngam", "gam")},
    "ni": "topic particle",
    **{p: "quotative particle (thus, saying)" for p in ("ces", "zhes", "shes")},
    **{p: "emphatic / concessive clitic (also, even)" for p in ("kyang", "yang", "'ang")},
    "'ung": "emphatic clitic",
}

# Affixes the tagger writes as "+affix" (written attached to the preceding syllable in the input).
AFFIXES: Dict[str, str] = {
    "'i": "genitive case (attached)", "s": "agentive case (attached)", "r": "terminative case (attached)",
    "'o": "sentence-final particle (attached)", "'am": "interrogative particle (attached)",
    "'ang": "concessive clitic (attached)", "'u": "diminutive (attached)", "'is": "agentive case (attached)",
}

# Negation (tag X).
NEGATION: Dict[str, str] = {"ma": "negation (prohibitive / past)", "mi": "negation (present / future)",
                            "med": "negative existential (there is not)", "min": "negative copula (is not)"}

# Determiners, demonstratives and plural markers (tag D).
DETERMINERS: Dict[str, str] = {"de": "demonstrative (that)", "'di": "demonstrative (this)", "rnams": "plural marker",
                               "dag": "plural / dual marker", "zhig": "indefinite article (a, some)",
                               "cig": "indefinite article (a, some)", "shig": "indefinite article (a, some)",
                               "gang": "relative / interrogative (which, what)", "su": "interrogative (who)"}


# ---------------------------------------------------------------- backend conventions (verbatim)
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
    """Backend rule: single-syllable lemmas always link; multi-syllable lemmas only if they are a real
    headword (checked raw and after case-ending stripping). Without a dictionary callback, multi-syllable
    lemmas are NOT linked (no invented links)."""
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


# ---------------------------------------------------------------- functions
def function_for(token, prev_pos: Optional[str] = None) -> str:
    """Spelled-out grammatical function for a tagged token (Token from mitra_tagger.parse)."""
    w = token.surface.strip()
    if token.is_affix:
        return AFFIXES.get(w, f"attached affix {w}")
    p = token.pos
    if p == "C":
        return CASE_PARTICLES.get(w, "case particle")
    if p == "K":
        return CLAUSE_PARTICLES.get(w, "clause particle / converb")
    if p == "L":
        return FINAL_PARTICLES.get(w, "clitic or final particle")
    if p == "X":
        return NEGATION.get(w, "negation")
    if p == "D":
        return DETERMINERS.get(w, "determiner")
    if p == "G":
        return "verbal noun (nominalised verb)"
    return POS_NAMES.get(p, p)


def lemma_for(token) -> str:
    """Wylie lemma: the word itself (affixes are separate tokens, so no stripping is needed)."""
    return token.surface.strip()


def to_word_events(sentence, headword_check: Optional[Callable[[str], bool]] = None) -> List[dict]:
    """Backend-compatible WordEvent dicts (typing_models/grammar_events.py) for one tagged sentence.
    `meaning` is left empty: it comes from a dictionary or an LLM, not from the tagger."""
    events = []
    prev = None
    for i, t in enumerate(sentence.tokens):
        if t.is_punct:
            continue
        lemma = lemma_for(t)
        surface_tib = with_trailing_tsheg(wylie_to_unicode(t.surface))
        link = is_linkable(lemma, headword_check) and t.pos in ("N", "P", "V", "G", "J", "A", "M", "R")
        ev = {
            "type": "word",
            "surface": surface_tib,
            "lemma": lemma,
            "transliteration": t.surface,
            "function": function_for(t, prev),
            "meaning": "",
            "external_source": "steinert" if link else None,
            "external_url": steinert_url(remove_case_endings(lemma)) if link else None,
            "mitra": None,
        }
        if t.unit is not None:
            u = sentence.units[t.unit]
            ev["sanskrit_unit"] = {"index": t.unit, "text": u.text, "lemma": u.type or None}
        events.append(ev)
        prev = t.pos
    return events
