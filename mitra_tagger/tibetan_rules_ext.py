# Attribution: the analyses in this module follow Faggionato, Meelen & Hill (2023), Classical Tibetan
# Annotation Manual, Part II (Zenodo, doi:10.5281/zenodo.7880130, CC BY 4.0) and Garrett & Hill (2017),
# A rule based Tibetan POS tagger (SOAS, Zenodo, doi:10.5281/zenodo.574882, CC BY 4.0); closed-class lists
# were checked against Meelen's ACTib gold lexicon (github.com/mariekemeelen/actib, MIT). See CITATION.md.
"""Candidate rule families on top of tibetan_rules.py, each switchable so
that an ablation can measure them one by one (env TIB_RULE_STEPS, comma separated):

  special   §3.14.2 special verbs: copulas, existentials, modal / light auxiliaries
  pronoun   §3.12 personal, reflexive, interrogative and indefinite pronouns
  adverb    §3.2 intensifying, directional, proclausal, temporal adverbs and fused
            terminative adverbials (nges par, de ltar, mthar …)
  nomz      §3.9 nominalisers beyond pa/ba: mkhan, sa, rgyu, thabs, lugs, tshul …
  lists     §3.6 / §3.8 missing demonstratives, quantifiers, plural markers, relator nouns
  regex     Garrett & Hill (2017) regex-tagger disambiguations: final / question particle
            sandhi, semi-final de only after -d, su only after -s, gyis as imperative of
            bgyid, tense from context (ma / mi / cig / nas), yongs su, rjes su, kho, dag
"""
import os
import re
from typing import Optional, Tuple

# Default: all six families with the trimmed labels (config "s7" of the 2026-09-25 ablation,
# +342 Elo / +0.25 rating over the previous rule set). TIB_RULE_STEPS="" disables them.
STEPS = set(s.strip() for s in os.getenv("TIB_RULE_STEPS", "all,fix").split(",") if s.strip())
ALL_STEPS = ("special", "pronoun", "adverb", "nomz", "lists", "regex")


def on(step: str) -> bool:
    return step in STEPS or "all" in STEPS


def _nw(s):
    return " ".join((s or "").replace("_", " ").split()).strip()


# ------------------------------------------------------------------ 1 special verbs
# surface stem -> (label, gloss). label is what goes into the function field.
SPECIAL_VERBS = {
    "yin": ("copula verb yin", "'is, am, are' (equational 'to be'; present/future stem, yind past)"),
    "yod": ("existential verb yod", "'there is, exists; to have' (invariant)"),
    "'dug": ("existential verb 'dug", "'is, there is, exists' (invariant)"),
    "med": ("negative existential verb med", "'there is not, does not have, is without' (invariant; negates yod)"),
    "min": ("negative copula min", "'is not' (negates yin)"),
    "lags": ("honorific copula lags", "'is' (polite equivalent of yin)"),
    "mchis": ("humble existential verb mchis", "'is, exists, there is' (humble yod); also past stem of mchi 'to go'"),
    "gda'": ("existential verb gda'", "'to be, to be there' (invariant)"),
    "red": ("copula verb red", "'is' (later language)"),
    "dgos": ("modal auxiliary dgos", "'must, need to' after a verb; as main verb 'to need' (invariant)"),
    "srid": ("modal auxiliary srid", "'to be possible, may' (invariant)"),
    "chog": ("modal auxiliary chog", "'may, be permitted; be enough' (invariant)"),
    "thub": ("modal verb thub", "'to be able, can' (invariant)"),
    "btub": ("modal verb thub", "'to be able' (invariant stem btub)"),
    "thang": ("modal verb thang", "'to be able' (invariant)"),
    "dka'": ("verb dka'", "'to be difficult'; after a verbal noun: 'hard to …' (invariant)"),
    "ran": ("verb ran", "'to be time, be proper, be due' (invariant)"),
    "grags": ("verb grags", "'to be known as, be famed' (invariant)"),
    "mdzad": ("honorific verb mdzad", "'to do, perform' (honorific of byed; light verb after a noun; invariant)"),
    "'dod": ("verb 'dod", "'to wish, want; to assert' (after a verbal noun: 'wants to'; invariant)"),
    "'dra": ("verb 'dra", "'to be similar, alike'; after a verbal noun: 'seems to' (invariant)"),
    "myong": ("verb myong", "'to experience, taste'; after a verb: 'have experienced …-ing' (present stem)"),
    "myangs": ("verb myong", "'to experience, taste' (past stem myangs)"),
    "myang": ("verb myong", "'to experience, taste' (future stem myang)"),
    "'tshal": ("verb 'tshal", "'to pay respect' (phyag 'tshal 'prostrate'); also 'to wish, beg, ask' (present/future stem)"),
    "btsal": ("verb 'tshal", "'to pay respect; to seek' (past/future stem btsal)"),
    "byed": ("verb byed 'to do, make'", "present stem (past byas, future bya, imperative byos); light verb in noun + byed"),
    "byas": ("verb byed 'to do, make'", "past stem byas (present byed, future bya)"),
    "bya": ("verb byed 'to do, make'", "future stem bya (present byed, past byas); 'to be done'"),
    "byos": ("verb byed 'to do, make'", "imperative stem byos"),
    "bgyid": ("humble verb bgyid 'to do'", "present stem (past bgyis, future bgyi, imperative gyis)"),
    "bgyis": ("humble verb bgyid 'to do'", "past stem bgyis"),
    "bgyi": ("humble verb bgyid 'to do'", "future stem bgyi"),
    "'gro": ("verb 'gro 'to go'", "present/future stem (past song or phyin, suppletive)"),
    "song": ("verb 'gro 'to go'", "past stem song (suppletive; also auxiliary 'went, became')"),
    "phyin": ("verb 'gro 'to go'", "past stem phyin (suppletive)"),
    "'gyur": ("verb 'gyur 'to become'", "present/future stem; after a terminative: 'will become, will be' (resultative of sgyur 'to change')"),
    "gyur": ("verb 'gyur 'to become'", "past stem gyur; after a terminative: 'became, was'"),
    "bsgyur": ("verb sgyur 'to change, transform'", "past/future stem bsgyur"),
    "sgyur": ("verb sgyur 'to change, transform'", "present stem"),
    "'byung": ("verb 'byung 'to arise, occur, come forth'", "present/future stem (past byung)"),
    "byung": ("verb 'byung 'to arise, occur'", "past stem byung; also auxiliary 'happened, came about'"),
    "'ong": ("verb 'ong 'to come'", "present/future stem (past 'ongs)"),
    "'ongs": ("verb 'ong 'to come'", "past stem 'ongs"),
    "yong": ("verb yong 'to come; will'", "invariant; as auxiliary marks the future"),
    "bzhag": ("verb 'jog 'to place, set, establish'", "past stem bzhag"),
    "phod": ("verb 'phod 'to dare, be able to bear'", "past stem phod"),
    "'phod": ("verb 'phod 'to dare, be able to bear'", "present/future stem"),
    "drag": ("verb drag 'to recover; be strong'", "invariant"),
}


def special_verb(stem: str, prev_form: Optional[str], prev_pos: Optional[str], next_form: Optional[str]) -> Optional[Tuple[str, str]]:
    """(label, gloss) for a special verb stem (Wylie, nominaliser already stripped); None otherwise."""
    if not on("special"):
        return None
    s = _nw(stem)
    p = _nw(prev_form or "")
    after_verb = prev_pos in ("V", "G") or p in ("ma", "mi")
    if on("fix") and s in ("nus", "shes", "mod", "thag", "rgyu", "shog"):
        r = _special_ctx(s, p, after_verb, prev_pos, next_form)
        return (r[0], r[1].split(";")[0].split(" (")[0].replace(", yind past", "")) if r else None
    return _special_ctx(s, p, after_verb, prev_pos, next_form) if s in ("nus", "shes", "mod", "thag", "rgyu", "shog") else _special_plain(s)


def _special_plain(s):
    lab = SPECIAL_VERBS.get(s)
    if lab and on("fix"):
        return (lab[0], lab[1].split(";")[0].split(" (")[0])
    return lab


def _special_ctx(s, p, after_verb, prev_pos, next_form):
    prev_pos_verb = prev_pos in ("V", "G")
    if s == "nus":
        return ("modal auxiliary nus", "'to be able, can' after a verb (invariant)") if after_verb else \
               ("verb nu", "'to suck' (past stem nus); after a verb nus would be the modal 'can'")
    if s == "shes":
        return ("modal auxiliary shes", "'to know how to, can' after a verb (invariant)") if after_verb else \
               ("verb shes", "'to know, understand' (invariant)")
    if s == "mod":
        return ("concessive auxiliary mod", "'indeed …, although, though' after a verb (invariant)") if after_verb else None
    if s == "thag":
        return ("auxiliary thag", "'as soon as' in verb + ma thag (tu)") if p == "ma" else None
    if s == "rgyu":
        return ("auxiliary rgyu", "'is to be done, to be …-ed' after a verb (invariant; future obligation)") if prev_pos_verb else None
    if s == "shog":
        if _nw(next_form or "") in ("cig", "shig", "zhig"):
            return ("imperative verb shog", "'come!' (imperative of 'ong, with cig)")
        return ("optative auxiliary shog", "'may it be …, may … come about' (invariant)")
    return None


# ------------------------------------------------------------------ 2 pronouns
PRONOUNS = {
    "nga": ("personal pronoun", "'I, me'"), "nged": ("personal pronoun", "'we, us'"),
    "khyod": ("personal pronoun", "'you' (singular)"), "khyed": ("personal pronoun", "'you' (honorific / plural)"),
    "bdag": ("personal pronoun (humble)", "'I, me' (as a noun: 'self, lord')"),
    "bdag cag": ("personal pronoun", "'we, us'"), "khyed cag": ("personal pronoun", "'you' (plural)"),
    "'o skol": ("personal pronoun", "'we'"), "'u cag": ("personal pronoun", "'we'"), "'o cag": ("personal pronoun", "'we'"),
    "rang re": ("personal pronoun", "'we, ourselves'"), "kho bo": ("personal pronoun", "'I' (male speaker)"),
    "kho mo": ("personal pronoun", "'I' (female speaker); 'she'"), "kho": ("personal pronoun", "'he'"),
    "khong": ("personal pronoun (honorific)", "'he, she'"), "mo": ("personal pronoun", "'she'"),
    "rang": ("reflexive pronoun", "'self, own, oneself'; after nga / khyed: 'I myself, you yourself'"),
    "su": ("interrogative pronoun", "'who'"), "gang": ("interrogative / relative pronoun", "'which, what; who(ever), what(ever)'"),
    "ci": ("interrogative pronoun", "'what'"), "ji": ("interrogative pronoun", "'what, how' (in ji ltar, ji snyed)"),
    "nam": ("interrogative pronoun", "'when'"), "ga": ("interrogative pronoun", "'which'"),
    "ji srid": ("relative conjunction (ji + srid)", "'as long as, how long'"), "ci srid": ("interrogative / relative pronoun", "'as long as'"),
    "ga re": ("interrogative pronoun", "'what'"), "ji snyed": ("interrogative / relative pronoun", "'how many, as many as'"),
    "la la": ("indefinite pronoun", "'some'"), "so so": ("indefinite pronoun", "'each, respective'"),
    "re re": ("indefinite pronoun", "'each one'"), "gnyi ga": ("indefinite pronoun", "'both'"),
}
def is_particle_like(w: Optional[str]) -> bool:
    w = _nw(w or "")
    return w in ("kyi", "gyi", "gi", "yi", "'i", "kyis", "gyis", "gis", "yis", "s", "la", "na", "nas", "las", "du", "tu", "su", "ru", "r",
                 "dang", "te", "ste", "de", "zhing", "cing", "shing", "kyang", "yang", "'ang", "ni", "ces", "zhes", "shes", "'am", "sam", "nam", "dam", "gam")


# forms that are pronouns whatever the tagger says
_PRON_STRICT = {"nga", "nged", "khyod", "khyed", "bdag cag", "khyed cag", "'o skol", "'u cag", "'o cag", "rang re", "kho bo", "kho mo", "khong",
                "ji srid", "ci srid", "ga re", "gnyi ga"}


# bdag: 'I' (humble) only as a free pronoun; before these it is the noun 'self' (ātman)
_BDAG_SELF_NEXT = {"gzhan", "med", "med pa", "'dzin", "'dzin pa", "nyid", "lta", "lta ba", "lta bar", "du 'dzin", "tu 'dzin", "dang gzhan"}
_BDAG_SELF_VIA_CASE = {"lta ba", "lta", "'dzin", "'dzin pa", "zhen pa", "chags pa", "rlom pa", "lta bar", "'dzin par"}
_BDAG_LORD_NEXT = {"po", "mo", "nyid can", "chen po"}


def pronoun_label(form: str, pos: str, prev_form: Optional[str], next_form: Optional[str] = None,
                  next2_form: Optional[str] = None) -> Optional[Tuple[str, str]]:
    if not on("pronoun"):
        return None
    f = _nw(form)
    if f == "bdag":
        n, n2 = _nw(next_form or ""), _nw(next2_form or "")
        if n in _BDAG_SELF_NEXT or (n in ("la", "tu", "du", "gi", "gis") and n2 in _BDAG_SELF_VIA_CASE):
            return ("noun", "'self' (ātman): bdag gzhan 'self and others', bdag med 'no-self', bdag 'dzin / bdag tu lta ba 'clinging to / viewing a self'")
        if n in _BDAG_LORD_NEXT or (pos in ("N", "J") and _nw(prev_form or "") and not is_particle_like(prev_form)):
            return ("noun", "'lord, owner, master' (bdag after a noun: khyim bdag 'householder', kun bdag 'lord of all')")
        if pos != "R":
            return None
    if pos == "R" or f in _PRON_STRICT or (f in ("gang", "ci", "ji") and pos in ("N", "A", "J", "D")):
        lab = PRONOUNS.get(f)
        if lab:
            if f == "rang" and _nw(prev_form or "") in ("nga", "khyed", "khyod", "kho", "khong", "nged", "bdag"):
                return ("reflexive pronoun", "'-self' (emphatic: nga rang 'I myself', khyed rang 'you yourself')")
            return lab
        if pos == "R":
            return ("pronoun", "")
    return None


# ------------------------------------------------------------------ 3 adverbs
INTENSE = {"rab": "'very, extremely' (rab tu)", "shin": "'very, extremely' (shin tu)", "ches": "'very, most'", "ha cang": "'very, too much'",
           "ye": "'at all, entirely' (with negation)", "cung zad": "'a little, somewhat'", "cung": "'a little'", "rab tu": "'very, extremely, thoroughly'",
           "shin tu": "'very, extremely'", "cung zad tsam": "'just a little'"}
DIRECTIONAL = {"yan chad": "'above, upwards of, and above'", "man chad": "'below, down to, and below'", "phyin chad": "'from now on, henceforth'",
               "tshun chad": "'up to, within, this side of'", "tshun cad": "'up to, within'", "sngon chad": "'formerly, before'",
               "slan chad": "'hereafter, from now on'", "phan chad": "'onwards, beyond'", "phan cad": "'onwards'", "phan tshun": "'mutually, one another'",
               "yan": "'above, upwards'", "tshun": "'this side, within'", "slar": "'again, back'"}
PROCLAUSAL = {"'o na": "'well then, in that case'", "gal te": "'if' (opens a conditional clause, closed by na)", "'on kyang": "'nevertheless, but'",
              "'on te": "'or, but if'", "de nas": "'then, after that'", "des na": "'therefore, so'", "de bas na": "'therefore'",
              "de lta bas na": "'therefore'", "de'i phyir": "'therefore, for that reason'", "de bzhin du": "'likewise, in the same way'",
              "de ltar": "'thus, in that way, so'", "'di ltar": "'thus, as follows, like this'", "ji ltar": "'how; just as, in whatever way'",
              "ci ltar": "'how'", "de lta bu": "'such, like that'", "'di lta bu": "'such, like this'", "de lta na": "'in that case'",
              "mdor na": "'in short'", "de yang": "'moreover, and that'", "'on": "'but'", "yang na": "'or else, otherwise'"}
TEMPORAL = {"da lta": "'now'", "da ltar": "'now, at present'", "deng sang": "'nowadays'", "ding sang": "'nowadays'", "gdod": "'at first; only then'",
            "de ring": "'today'", "mtshan mo": "'at night'", "do nub": "'tonight'", "tho rangs": "'at dawn'", "gzod": "'only then, at first'",
            "yun ring": "'for a long time'", "kha sang": "'yesterday'", "sang": "'tomorrow'", "mdang": "'last night'", "cig car": "'at once, suddenly'",
            "gnangs": "'the day after tomorrow'", "da dung": "'still, yet, moreover'", "da rung": "'still, again'", "phyis": "'later, afterwards'",
            "sngon": "'formerly, before'", "deng": "'today, now'", "da": "'now'", "nam": "'when; ever'", "rtag tu": "'always'", "gtan du": "'permanently, for ever'",
            "myur du": "'quickly, soon'", "'phral du": "'immediately'", "res 'ga'": "'sometimes'", "lan cig": "'once'",
            "yang yang": "'again and again'", "thog mar": "'at first'", "mthar": "'finally, in the end' (mtha' + terminative r)"}
TERM_ADV = {"nges par": "'certainly, definitely' (nges pa + r)", "mngon par": "'manifestly, clearly' (mngon pa + r)", "rgyas par": "'extensively, in detail' (rgyas pa + r)",
            "lhag par": "'especially, particularly' (lhag pa + r)", "bye brag tu": "'in particular, specifically'", "yongs su": "'completely, entirely'",
            "so sor": "'separately, individually' (so so + r)", "gcig tu": "'solely; as one'", "gsal bar": "'clearly' (gsal ba + r)", "legs par": "'well, properly' (legs pa + r)",
            "ji bzhin": "'just as, exactly as'", "rim par": "'in order, gradually' (rim pa + r)", "khyad par du": "'especially'", "mtshungs par": "'equally' (mtshungs pa + r)",
            "rnam par": "'completely, thoroughly' (rnam pa + r; Sanskrit vi-)", "mngon sum du": "'directly, perceptibly'", "rang bzhin du": "'naturally'",
            "shin tu": "'very'", "rab tu": "'very'", "dngos su": "'actually, directly'", "gtan nas": "'at all' (with negation)", "phyir": "'again, back; out'",
            "phal cher": "'mostly, for the most part' (phal che + r)", "dus gcig tu": "'at one time, simultaneously'", "gzhan du": "'otherwise; elsewhere'",
            "sa ler": "'vividly, clearly' (sa le + r)", "lhan cig tu": "'together'", "mnyam du": "'together, equally'", "thog mar": "'at first'",
            "ci rigs par": "'as appropriate'", "ji ltar rigs par": "'as is fitting, as appropriate'", "rim gyis": "'gradually'", "mngon du": "'manifestly, directly'"}
_TERM_END = re.compile(r"^(.*?[a-zA-Z'])\s?(r|du|tu|su|ru)$")


def adverb_label(form: str, pos: str, prev_form: Optional[str], first: bool) -> Optional[Tuple[str, str]]:
    if not on("adverb"):
        return None
    f = _nw(form)
    if f in INTENSE:
        return ("intensifying adverb", INTENSE[f] + (" (uninflected stem; the following tu is a terminative)" if f in ("rab", "shin") else ""))
    if f in DIRECTIONAL:
        return ("directional adverb", DIRECTIONAL[f])
    if f in ("du", "tu") and _nw(prev_form or "") in ("rab", "shin", "ches", "cung zad", "cung", "ha cang"):
        return ("terminative particle", f"forms the adverb {_nw(prev_form)} {f} 'very'")
    if f in PROCLAUSAL and (pos in ("A", "D", "L", "K", "C", "J") or first):
        return ("clause-linking adverb" if on("fix") else "proclausal adverb", PROCLAUSAL[f])
    if f in TEMPORAL and pos not in ("V", "G"):
        return ("temporal adverb", TEMPORAL[f])
    if pos == "A":
        if f in TERM_ADV:
            return ("adverbial (with terminative)", TERM_ADV[f])
        if f.endswith(" par") or f.endswith(" bar"):
            stem = f[:-1]
            return ("adverbial (with terminative)", f"verbal noun {stem} + terminative r: 'so as to be …, in the manner of …-ing' (adverbial or complement use)")
        m = _TERM_END.match(f)
        if m and " " in f:
            return ("adverbial (with terminative)", f"{m.group(1)} + terminative {m.group(2)}: 'to, at, in, as {m.group(1)}' (noun or adjective used adverbially)")
        if f.endswith("ltar"):
            return ("adverbial", f"{f[:-4].strip() or f} + lta + r: 'in the way of …, thus / how'")
    return None


# ------------------------------------------------------------------ 4 nominalisers
NOMINALISERS = {
    "pa": "", "ba": "",
    "mkhan": "'the one who …s' (agent nominaliser)", "sa": "'the place or object of …-ing'", "rgyu": "'that which is to be …-ed' (future obligation)",
    "thabs": "'the means, way of …-ing'", "lugs": "'the way, manner of …-ing'", "tshul": "'the manner of …-ing'", "mi": "'the person who …s'",
    "grabs": "'being about to …, preparation for …-ing'", "tshad": "'as much as …, all that …'", "thengs": "'an occasion of …-ing'",
    "stabs": "'the circumstance of …-ing'", "bya": "'that which is to be …-ed'", "po": "'the one who …s'",
}
_NOMZ_RE = re.compile(r"^(.+?)\s+(mkhan|sa|rgyu|thabs|lugs|tshul|mi|grabs|tshad|thengs|stabs|bya)$")
# on a noun-tagged token only the unambiguous agent / manner nominalisers count
_NOMZ_ON_NOUN = ("mkhan", "thabs", "lugs", "tshul", "grabs")
# nominaliser fused with a case particle (the tagger keeps 'yin pas', 'byed par' whole)
_NOMZ_CASE_RE = re.compile(r"^(.+?)\s+(pa|ba)(s|r|'i)$")
NOMZ_CASE = {"s": "agentive particle s: 'because …, by …-ing'", "r": "terminative particle r: after a verb 'in order to …, so that …', after an adjective-like stem '…-ly, in a … manner'",
             "'i": "genitive particle 'i: relative clause 'which …, of …-ing'"}


def split_nominaliser(wylie: str, pos: str = "G") -> Optional[Tuple[str, str]]:
    """(stem, nominaliser) for a form ending in one of the extended nominalisers; None otherwise."""
    if not on("nomz"):
        return None
    m = _NOMZ_RE.match(_nw(wylie))
    if not m:
        return None
    if pos != "G" and m.group(2) not in _NOMZ_ON_NOUN:
        return None
    return (m.group(1), m.group(2))


def split_nominaliser_case(wylie: str) -> Optional[Tuple[str, str, str]]:
    """(stem, nominaliser, case) for 'yin pas', 'byed par', 'bstan pa'i'; None otherwise."""
    if not on("nomz"):
        return None
    m = _NOMZ_CASE_RE.match(_nw(wylie))
    return (m.group(1), m.group(2), m.group(3)) if m else None


# ------------------------------------------------------------------ 5 lists
DEM_EXTRA = {"de nyid": "'that very (thing), itself' (de + emphatic nyid)", "'di nyid": "'this very (thing), itself'", "ya": "'that up there'", "tshu": "'this side, here'", "pha gi": "'that over there'", "ma gi": "'that down there'", "ya gi": "'that up there'",
             "ma ki": "'that down there'", "pha": "'yonder, over there'"}
QUANT_EXTRA = {"ka": "'all' (after a numeral: gnyis ka 'both')", "ga": "'all' (after a numeral)", "'ga'": "'some, a few'", "sna dgu": "'all kinds of'",
               "'ba'": "'only, solely' ('ba' zhig)", "shas": "'some, a few'", "nyung shas": "'a few'", "bag tsam": "'a little'", "yo": "'all'",
               "shas dag": "'some'", "ya re": "'each one of two'", "sna re": "'a few'", "gzhin": "'?' (quantifier)", "phal cher": "'most, mostly'"}
ETC = {"sogs": "'and so on, et cetera' (sogs / la sogs pa: closes a list)", "la sogs pa": "'and so on, et cetera'", "la sogs": "'and so on'",
       "sogs pa": "'and so on, et cetera'"}
PLURAL_EXTRA = {"yongs": "'all, entire, whole'", "sna tshogs": "'various, of all kinds' (marks a plural of kinds)", "cog": "'all' (thams cog)"}
RELATOR_EXTRA = {
    "skor": "concerning, about; around", "khongs": "within, among", "rten": "on the basis of, relying on", "smad": "the lower part of",
    "thog ma": "the beginning of", "'gram": "at the bank, edge of", "mthil": "at the bottom of", "gseb": "among, amidst", "shul": "in the wake of, after",
    "tha ma": "the end of", "ring": "during (the time of)", "g.yas g.yon": "to the right and left of", "phu": "at the upper end of", "'khris": "beside",
    "zhor": "incidentally, while (doing)", "sked": "at the middle of", "stod": "the upper part of", "stengs": "on top of", "snying kha": "at the heart of",
    "phyi bzhin": "following after", "kha": "at the surface, mouth, edge of", "mtha' logs": "at the edge of", "tshun": "up to, within",
    "byang": "north of", "dbung": "in the middle of", "drud": "near", "bseb": "among", "tsa": "near, beside", "rting": "after, behind",
    "ngos": "on the surface, side of", "sne": "at the end of", "rgyud": "along; in the continuum of", "'og ma": "below", "gong ma": "above, the one above",
    "nang du": "inside", "mtshams": "at the border, junction of", "bar du": "until, between", "zla": "the companion of",
}


def list_label(form: str, pos: str, prev_form: Optional[str]) -> Optional[Tuple[str, str]]:
    if not on("lists"):
        return None
    f = _nw(form)
    if f in ETC:
        return ("et cetera marker", ETC[f])
    if f in DEM_EXTRA and pos in ("D", "R", "A", "N"):
        return ("demonstrative", DEM_EXTRA[f])
    if f in QUANT_EXTRA and pos in ("D", "M", "A", "J", "N"):
        return ("quantifier", QUANT_EXTRA[f])
    if f in PLURAL_EXTRA and pos in ("D", "A", "J", "N"):
        return ("plural marker", PLURAL_EXTRA[f])
    return None


# ------------------------------------------------------------------ 6 regex-tagger rules
_FIN = ("'o", "so", "to", "do", "no", "ngo", "ro", "lo", "go", "bo", "mo")
_QUES = ("'am", "sam", "nam", "dam", "gam", "ngam", "tam", "ram", "lam", "bam", "mam")
_IPV = ("shig", "cig", "zhig")


def sandhi_matches(prev_form: str, form: str) -> bool:
    """The 'o / 'am families copy the final consonant of the preceding syllable (go after -g,
    ngo after -ng, … 'o / 'am after a vowel)."""
    p = _nw(prev_form).replace("'", "").rstrip()
    f = _nw(form)
    if not p:
        return True
    if p.endswith("ng"):
        return f[:2] == "ng"
    last = p[-1]
    if last in "gdnbmrls":
        return f[0] == last
    return f[0] == "'"


def regex_particle(form: str, prev_form: Optional[str], prev_pos: Optional[str], next_form: Optional[str], last: bool):
    """Overrides from the regex tagger for particle-like forms. Returns (label, gloss),
    the string "skip" (leave the model's label alone) or None (no opinion)."""
    if not on("regex"):
        return None
    f = _nw(form)
    p = _nw(prev_form or "")
    n = _nw(next_form or "")
    if f in _FIN and p:
        if f == "lo" and prev_pos == "M":
            return ("noun", "'year' (lo after a numeral)")
        if not sandhi_matches(p, f):
            return "skip"
    if f in _QUES and p and not sandhi_matches(p, f):
        return "skip"
    if f == "de" and prev_pos == "V" and not p.endswith("d"):
        return ("demonstrative", "'that' (the semi-final converb de occurs only after a verb ending in -d)")
    if f == "su" and p and not p.replace("'", "").endswith("s"):
        return ("interrogative pronoun", "'who' (the terminative allomorph su follows only a syllable ending in -s)")
    if f == "gyis" and n in _IPV:
        return ("imperative verb", "'do!' (gyis, imperative stem of bgyid 'to do', with shig)")
    if f == "yongs" and n == "su":
        return ("totality marker", "'completely, entirely' (yongs su + verb)")
    if f == "rjes" and n == "su":
        return ("relator noun", "'after, following; in accordance with' (rjes su)")
    if f == "khong" and n == "du":
        return ("relator noun", "'within, inside' (khong du)")
    if f == "kho":
        return ("personal pronoun", "'he'") if not p else ("emphasising determiner", "'the very, same' (variant of kho na)")
    if f == "dag" and p in ("de", "'di"):
        return ("plural marker", "'those, these' (de dag / 'di dag)")
    return None


def tense_filter(labs, prev_form: Optional[str], next_form: Optional[str], stem_wylie: str):
    """Narrow a verb form's tense readings by context (regex tagger): ma → past / imperative,
    mi → present / future, a following cig → imperative, a following nas excludes the future,
    and n/r/l-final stems before kyang / cing / to / tu / tam are not future."""
    if not on("regex") or not labs:
        return labs, ""
    p = _nw(prev_form or "")
    n = _nw(next_form or "")
    order = {"present": 0, "past": 1, "future": 2, "imperative": 3}
    labs = sorted(labs, key=lambda l: order.get(l, 9))
    keep, why = list(labs), ""
    if p == "ma":
        k = [l for l in labs if l in ("past", "imperative")]
        if k:
            keep, why = k, "after ma: past / prohibitive reading"
    elif p == "mi":
        k = [l for l in labs if l in ("present", "future")]
        if k:
            keep, why = k, "after mi: present / future reading"
    if n in _IPV:
        k = [l for l in keep if l == "imperative"]
        if k:
            keep, why = k, "before cig: imperative"
    if n == "nas":
        k = [l for l in keep if l != "future"]
        if k:
            keep = k
    if n in ("kyang", "cing", "to", "tu", "tam") and stem_wylie and stem_wylie[-1] in "nrl":
        k = [l for l in keep if l != "future"]
        if k:
            keep = k
    return keep, why
