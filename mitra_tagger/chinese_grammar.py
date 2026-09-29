"""Chinese tag inventory, compound types and dharmamitra grammar-explained conventions."""
from __future__ import annotations
from typing import List
from urllib.parse import quote

POS_NAMES = {
    "N": "noun", "P": "proper noun", "V": "verb", "J": "adjective", "A": "adverb", "R": "pronoun",
    "M": "numeral", "X": "auxiliary", "C": "adposition", "K": "coordinating conjunction",
    "S": "subordinating conjunction", "L": "particle", "I": "interjection", "p": "punctuation",
}
COMPOUND_TYPES = {"T": "tatpuruṣa (dependent determinative)", "K": "karmadhāraya (descriptive)",
                  "D": "dvandva (coordinative)", "B": "bahuvrīhi (possessive / exocentric)",
                  "A": "avyayībhāva (adverbial)", "O": "other"}


def ddb_url(term: str) -> str:
    """Digital Dictionary of Buddhism lookup (backend-identical)."""
    return "http://buddhism-dict.net/cgi-bin/xpr-ddb.pl?q=" + quote(term, safe="")


def function_for(token) -> str:
    return POS_NAMES.get(token.pos, token.pos)


def to_word_events(sentence) -> List[dict]:
    """Backend-compatible WordEvent dicts for one tagged Chinese sentence (pinyin left empty)."""
    events = []
    for t in sentence.tokens:
        if t.is_punct:
            continue
        link = t.pos in ("N", "P", "V", "J", "A", "M")
        ev = {"type": "word", "surface": t.surface, "lemma": t.surface, "transliteration": "",
              "function": function_for(t), "meaning": "",
              "external_source": "ddb" if link else None, "external_url": ddb_url(t.surface) if link else None,
              "mitra": None}
        if t.unit is not None:
            u = sentence.units[t.unit]
            ev["compound"] = {"index": t.unit, "members": [sentence.tokens[i].surface for i in u.tokens],
                              "type": u.type, "type_name": COMPOUND_TYPES.get(u.type, "")}
        events.append(ev)
    return events


def punctuated(sentence) -> str:
    """The sentence with the predicted punctuation reinserted."""
    return "".join(t.surface for t in sentence.tokens)
