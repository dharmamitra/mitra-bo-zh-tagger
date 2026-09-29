# Attribution: the analyses in this module follow Faggionato, Meelen & Hill (2023), Classical Tibetan
# Annotation Manual, Part II (Zenodo, doi:10.5281/zenodo.7880130, CC BY 4.0) and Garrett & Hill (2017),
# A rule based Tibetan POS tagger (SOAS, Zenodo, doi:10.5281/zenodo.574882, CC BY 4.0); closed-class lists
# were checked against Meelen's ACTib gold lexicon (github.com/mariekemeelen/actib, MIT). See CITATION.md.
"""Rule-based function labels for Classical Tibetan particles, determiners, negation,
relator nouns and verb stems, following the Classical Tibetan Annotation Manual
Part II (Faggionato, Meelen & Hill 2023; case markers §3.3, clitics §3.4, converbs
§3.5, determiners §3.6, nouns/relator nouns §3.8, negation §3.10) and the SOAS
rule-based tagger (Garrett & Hill 2017) for the case-vs-converb decision: the SAME
morpheme is a case marker after a noun / verbal noun and a converb after a verb.

Used by services/tibetan_grammar.py to (a) put authoritative function labels into
the grammar block of the prompt and (b) overwrite the model's function field on
particle bullets in the streamed output.
"""
import csv
import os
import re
from typing import Dict, List, Optional, Tuple

import pyewts

from . import tibetan_rules_ext as X

_conv = pyewts.pyewts()

# (function label, short gloss) after a NOUN / verbal noun / other  |  after a VERB
# form(s) -> (case_label, case_gloss, converb_label, converb_gloss)
_CASE_CV = {
    ("kyi", "gyi", "gi", "yi", "'i"): (
        "genitive particle", "of; links the preceding noun to the following one (after a verbal noun it forms a relative clause)",
        "genitive converb", "adversative or concessive link between two clauses: 'but, although, whereas'"),
    ("kyis", "gyis", "gis", "yis", "s"): (
        "agentive particle", "marks the agent of a transitive verb, or the instrument or cause ('by, with')",
        "agentive converb", "causal link between two clauses: 'because, since'"),
    ("la",): (
        "allative particle", "to, at, in; marks the goal, the location or an oblique argument",
        "allative converb", "coordinates two clauses: 'and, while, after …-ing'"),
    ("na",): (
        "locative particle", "in, at (a place or a time)",
        "locative converb", "conditional or temporal clause: 'if, when'"),
    ("nas",): (
        "elative particle", "from (the place or time where an action starts)",
        "elative converb", "sequence of actions: 'after …-ing, having …, and then'"),
    ("las",): (
        "ablative particle", "from; than (in a comparison)",
        "ablative converb", "'from, after' the preceding action (rare)"),
    ("du", "tu", "su", "ru", "r"): (
        "terminative particle", "to, into, as; marks a goal, a purpose, or turns the preceding word into an adverb",
        "terminative converb", "purpose clause: 'in order to …'"),
    ("dang",): (
        "associative particle", "and, with; joins nouns or marks an oblique argument",
        "associative converb", "after an imperative: 'do …, and (then)'"),
    ("te", "ste", "de"): (
        "semi-final particle", "'that is, namely' (after a noun phrase)",
        "semi-final converb", "joins the clause to what follows: 'and, having …' (same subject continues)"),
    ("zhing", "cing", "shing"): (
        "imperfective particle", "coordinates: 'and'",
        "imperfective converb", "coordinates two predicates: 'and, while …-ing'"),
    ("rung",): (
        "concessive particle", "'even, although'",
        "concessive converb", "'although, even if'"),
    ("kyin", "gyin", "gin"): (
        "continuative particle", "'while'",
        "continuative converb", "progressive aspect: 'while …-ing'"),
    ("pas", "bas"): (
        "comparative particle", "'than' (comparison)",
        "nominaliser + agentive", "'because …, by …-ing' (verbal noun pa + agentive s)"),
}
_QUES = ("'am", "sam", "nam", "dam", "gam", "ngam", "tam", "ram", "lam", "bam", "mam")
_FIN = ("'o", "so", "to", "do", "no", "ngo", "ro", "lo", "go", "bo", "mo")
_IPV = ("shig", "cig", "zhig")
_FOCUS = ("kyang", "yang", "'ang", "cang")
_QUOT = ("ces", "zhes", "shes", "ce", "zhe", "she", "ces pa", "zhes pa")
_DEM = ("de", "'di")
_PLURAL = ("rnams", "dag", "tsho")
_EMPH = ("nyid", "kho na")
_QUANT = ("kun", "thams cad", "gzhan", "re", "re re", "sha stag", "kha", "mang po", "spyi", "cung zad", "du ma", "shas", "mtha' dag", "sna tshogs", "'ga'", "la la", "so so")
_TSAM = ("tsam", "snyed", "tsam pa")
_NEG = ("ma", "mi", "med", "min")
_NARE = ("na re",)
_ARE = ("ta re", "'a re")

# relator nouns (manual §3.8 n.rel): a noun that takes a genitive before it and a
# spatial case after it, working like a postposition. gloss when used that way.
RELATOR_NOUNS = {
    "phyir": "for the sake of; because of (the preceding)", "phyi": "after; outside", "bzhin": "in accordance with, like; while",
    "tshe": "at the time when", "dus": "at the time of", "nang": "inside, within", "rjes": "after", "drung": "in the presence of, before",
    "'og": "under, below", "steng": "on, above", "ched": "for the sake of", "skabs": "on the occasion of, when", "mtha'": "the end, limit of",
    "sgo": "by way of, by means of", "don": "for the purpose of", "ngang": "in the state of, while", "bar": "between; until",
    "lta": "like, as (lta + r = ltar)", "lta bu": "like, such as", "skad": "the words of (introduces speech)", "'dra": "like",
    "slad": "for the sake of; after", "phyogs": "the side, direction of", "sgang": "on top of", "thog": "on, at the point of",
    "gong": "above, before", "mdun": "in front of", "rgyab": "behind", "khrod": "among, amidst", "dkyil": "in the middle of",
    "dbus": "in the centre of", "mjug": "at the end of", "thad": "concerning, opposite", "gan": "near", "ltag": "behind, on top of",
    "sngon": "before", "'phral": "immediately after", "mgo": "at the head of", "rtsa": "at the foot of, near", "logs": "at the side of",
}
_VERB_POS = {"V"}
_NOMINAL_POS = {"N", "P", "R", "M", "D", "J", "A", "G"}


def _nw(s: str) -> str:
    return " ".join((s or "").replace("_", " ").split()).strip()


def _find(form: str, table: dict):
    for keys, val in table.items():
        if form in keys:
            return val
    return None


def particle_label(form: str, prev_pos: Optional[str], prev_form: Optional[str] = None,
                   next_pos: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """(function label, gloss) for a particle-like token given the previous word's
    coarse POS from the tagger (V verb, G verbal noun, N/P/R/... nominal). None if
    the form is not a particle we have a rule for."""
    if _nw(form) in _DEM_FUSED:
        return _DEM_FUSED[_nw(form)]
    f = _nw(form)
    after_verb = prev_pos in _VERB_POS
    # de / 'di: the semi-final allomorph de exists only after a verb; elsewhere (sentence
    # start, after a noun) de is the demonstrative 'that'
    if f in _DEM and not after_verb:
        return ("demonstrative", "'this' ('di) or 'that' (de); as a pronoun or determiner")
    if f in ("phyir", "bzhin", "ched", "slad") and not after_verb:
        return ("relator noun", RELATOR_NOUNS[f])
    if f in ("deng", "da", "da lta", "da ltar", "sngon", "phyis", "da dung", "ding sang", "de ring", "nam"):
        return ("temporal adverb", "'today, now, before, later …' (a time word)")
    v = _find(f, _CASE_CV)
    if v:
        return (v[2], v[3]) if after_verb else (v[0], v[1])
    if f in _QUES:
        return ("question particle", "marks a polar or alternative question: 'or …?'") if after_verb else \
               ("question particle", "'or'; alternative question after a noun phrase")
    if f in _FIN:
        return ("sentence-final particle", "closes the statement; marks the finite verb") if after_verb else \
               ("sentence-final particle", "closes the statement after a noun phrase or verbal noun")
    if f in _IPV:
        return ("imperative particle", "marks an imperative or prohibitive") if after_verb else \
               ("indefinite particle", "'a, some, a certain'")
    if f in _FOCUS:
        return ("concessive particle", "'even though, although, even if' (after a verb)") if after_verb else \
               ("focus particle", "'also, even, too'")
    if f == "ni":
        return ("topic particle", "marks the topic: 'as for …'")
    if f in _QUOT:
        return ("quotative particle", "closes quoted speech or thought: 'thus, saying …'")
    if f in _NARE:
        return ("speech-introducing particle", "'… said:' (introduces direct speech)")
    if f in _ARE:
        return ("converb 'lest'", "'lest, so that … not'")
    if f in _DEM:
        return ("demonstrative", "'this' (‘di) or 'that' (de); also used as a pronoun")
    if f in _PLURAL:
        return ("plural marker", "marks the plural of the preceding noun phrase")
    if f in _EMPH:
        return ("emphasising determiner", "'that very, itself' (nyid) / 'the very same' (kho na)")
    if f in _QUANT:
        return ("quantifier", "'all, other, each, some, many' etc.")
    if f in _TSAM:
        return ("limiting determiner", "'merely, just, about, as much as'")
    if f in ("ma", "mi"):
        return ("negation", "'not' (ma with past and imperative stems, mi with present and future)")
    return None


# demonstrative + fused case affix that the tagger sometimes leaves as one token
_DEM_FUSED = {
    "'dir": ("demonstrative + terminative particle", "'di + r: 'in this, here'"),
    "der": ("demonstrative + terminative particle", "de + r: 'there, to that'"),
    "'dis": ("demonstrative + agentive particle", "'di + s: 'by this'"),
    "des": ("demonstrative + agentive particle", "de + s: 'by that, thereby'"),
    "'di'i": ("demonstrative + genitive particle", "'di + 'i: 'of this'"),
    "de'i": ("demonstrative + genitive particle", "de + 'i: 'of that, its'"),
    "gang gis": ("interrogative/relative pronoun + agentive particle", "gang + gis: 'by whom, by which'"),
    "gang du": ("interrogative/relative pronoun + terminative particle", "gang + du: 'where, in which'"),
    "gang la": ("interrogative/relative pronoun + allative particle", "gang + la: 'to whom, in which'"),
    "gang gi": ("interrogative/relative pronoun + genitive particle", "gang + gi: 'whose, of which'"),
}


def is_closed_class(form: str) -> bool:
    """True for forms in the particle / determiner / negation tables (any POS the tagger gave)."""
    f = _nw(form)
    return f in _DEM_FUSED or bool(_find(f, _CASE_CV)) or f in _QUES or f in _FIN or f in _IPV or f in _FOCUS or f == "ni" or f in _QUOT \
        or f in _DEM or f in _PLURAL or f in _EMPH or f in _QUANT or f in _TSAM or f in ("ma", "mi") or f in _NARE or f in _ARE


def relator_label(form: str, prev_form: Optional[str], prev_pos: Optional[str], next_form: Optional[str]) -> Optional[Tuple[str, str]]:
    """Relator-noun reading when preceded by a genitive or followed by a spatial case."""
    f = _nw(form)
    table = RELATOR_NOUNS
    extra = X.on("lists") and f in X.RELATOR_EXTRA and f not in RELATOR_NOUNS
    if extra and X.on("fix") and f in ("snying kha", "kha", "tsa", "drud", "bseb", "dbung", "rgyud", "zla", "sked", "phu", "byang"):
        return None
    if extra:
        table = X.RELATOR_EXTRA
    if f not in table:
        return None
    prev = _nw(prev_form or "")
    nxt = _nw(next_form or "")
    gen = prev in ("kyi", "gyi", "gi", "yi", "'i")
    spatial = nxt in ("la", "na", "nas", "du", "tu", "su", "ru", "r", "las")
    # a following spatial case alone is weak evidence for nouns that are common as plain nouns
    weak = extra or f in ("phyogs", "don", "tshe", "dus", "ngo", "khong", "gzhi", "steng", "'og", "nang", "mdun", "rgyab", "sngon", "rjes", "mtha'", "gseb", "dbus", "ngos", "rtsa")
    if gen or (spatial and not weak) or f in ("phyir", "bzhin", "ched", "slad"):
        return ("relator noun", table[f])
    return None


# ---------------------------------------------------------------- verb stems
_VERBS: Dict[str, List[Tuple[str, str]]] = {}
_TENSE = ("present", "past", "future", "imperative")


def _load_verbs():
    if _VERBS:
        return
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tibetan_verbs.csv")
    try:
        with open(path, encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if not row or row[0].strip() == "ད་ལྟ" or len(row) < 4:
                    continue
                pres = re.sub(r"[༼(][^༽)]*[༽)]", "", row[0].strip())
                if not pres:
                    continue
                for i, lab in enumerate(_TENSE):
                    form = re.sub(r"[༼(][^༽)]*[༽)]", "", row[i].strip())
                    if form:
                        entry = (pres, lab)
                        if entry not in _VERBS.setdefault(form, []):
                            _VERBS[form].append(entry)
    except OSError:
        pass


def verb_info(wylie: str, prev_form: Optional[str] = None, next_form: Optional[str] = None) -> Optional[str]:
    """'past stem of X' style note for a verb stem (Wylie); handles a verbal noun by
    stripping the nominaliser. None if the verbs database does not know the form.
    With the regex step on, prev_form / next_form narrow the tense reading."""
    _load_verbs()
    w = _nw(wylie)
    stem = re.sub(r"\s+(pa|ba|par|bar|pas|bas|pa'i|ba'i|mkhan|tshul)$", "", w)
    nominalised = stem != w
    try:
        uni = _conv.toUnicode(stem).strip("་")
    except Exception:
        return None
    entries = _VERBS.get(uni)
    if not entries:
        return None
    by_verb: Dict[str, List[str]] = {}
    for pres, lab in entries:
        by_verb.setdefault(pres, []).append(lab)
    # several verbs share the form: prefer the one where it is the present stem itself,
    # then a past stem (the common narrative reading), then the rest
    def rank(item):
        pres, labs = item
        return (0 if pres == uni else 1 if "past" in labs else 2, len(pres))
    pres, labs = sorted(by_verb.items(), key=rank)[0]
    pres_wy = _conv.toWylie(pres).strip()
    labs = list(dict.fromkeys(labs))
    invariant = pres == uni and len(labs) >= 3 or len([l for l in labs if l != "imperative"]) >= 3
    labs, why = X.tense_filter(labs, prev_form, next_form, stem) if not (invariant and X.on("fix")) else (labs, "")
    suffix = f", {why}" if why else ""
    if pres == uni:
        if why:
            return f"{'/'.join(labs)} stem{suffix}"
        return "present stem" if labs == ["present"] or "present" in labs and len(labs) < 3 else "invariant stem"
    core = [l for l in labs if l != "imperative"] if not why else labs
    if not core:
        # only an imperative reading: inside a nominalised form that reading is implausible
        return f"imperative stem of {pres_wy}" if not nominalised else f"form of {pres_wy}"
    if len(core) >= 3:
        return f"invariant stem of {pres_wy}"
    return f"{'/'.join(core)} stem of {pres_wy}{suffix}"


def final_particle_matches(prev_form: str, form: str) -> bool:
    """Sandhi check for the 'o family: go after -g, ngo after -ng, do after -d, no after -n,
    bo after -b, mo after -m, ro after -r, lo after -l, so after -s, 'o after a vowel."""
    p = _nw(prev_form).replace("'", "")
    f = _nw(form)
    if not p:
        return True
    last = p[-1]
    expect = {"g": "go", "d": "do", "n": "no", "b": "bo", "m": "mo", "r": "ro", "l": "lo", "s": "so"}
    if p.endswith("ng"):
        return f == "ngo"
    if last in expect:
        return f == expect[last]
    return f == "'o"
