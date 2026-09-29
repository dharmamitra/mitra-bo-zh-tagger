"""Exact instruction strings the model was trained with (do not edit: the model expects them verbatim)."""

BO_INSTRUCTION = ("Segment the Tibetan into sentences (one per line) and dictionary words (syllables joined by _, "
                  "attached affixes as +affix), tag each word with its part of speech (/N noun, /P proper noun, "
                  "/V verb, /G verbal noun, /J adjective, /A adverb, /M numeral, /R pronoun, /D determiner, "
                  "/X negation, /C case particle, /K clause particle, /L clitic or final particle, /I interjection, "
                  "/p punctuation) and enclose in [ ] the words that together render one Sanskrit word or compound.")

ZH_INSTRUCTION = ("Punctuate the Chinese, segment it into sentences (one per line) and words, tag each word with its "
                  "part of speech (/N noun, /P proper noun, /V verb, /J adjective, /A adverb, /R pronoun, /M numeral, "
                  "/X auxiliary, /C adposition, /K coordinating conjunction, /S subordinating conjunction, /L particle, "
                  "/I interjection, /p punctuation) and enclose in [ ] the words that together render one Sanskrit "
                  "compound, followed by its type (=T tatpurusa, =K karmadharaya, =D dvandva, =B bahuvrihi, "
                  "=A avyayibhava, =O other).")


def build_prompt(text: str, lang: str) -> str:
    if lang == "bo":
        return f"{BO_INSTRUCTION}\nTIBETAN: {text}\nOUTPUT:\n"
    if lang == "zh":
        return f"{ZH_INSTRUCTION}\nCHINESE: {text}\nOUTPUT:\n"
    raise ValueError(f"lang must be 'bo' or 'zh', got {lang!r}")
