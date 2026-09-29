# mitra-bo-zh-tagger

Sentence segmentation, word segmentation, part-of-speech tagging and Sanskrit-unit annotation for
**classical Tibetan** (Wylie or Unicode) and **Buddhist Chinese** (Taishō-style, unpunctuated), using
the [`buddhist-nlp/mitra-bo-zh-tagger`](https://huggingface.co/buddhist-nlp/mitra-bo-zh-tagger) model
(a 9B Qwen3.5-based model finetuned by the MITRA project). For Chinese the model also restores punctuation.

```
in : de nas tshe dang ldan pa kun dga' bos bcom ldan 'das la 'di skad ces gsol to/ /
out: [de_nas/A] [tshe_dang_ldan_pa/J] [kun_dga'_bo/P] +s/C [bcom_ldan_'das/N] la/C 'di_skad/A ces/L [gsol/V] to/L ||/p

in : 爾時世尊告諸比丘汝等當知一切諸法皆悉無常苦空無我
out: 爾/R 時/N 世尊/N 告/V 諸/L 比丘/N ：/p 「/p 汝/R 等/L 當/X 知/V ，/p [一切/N 諸/L 法/N]=K 皆/A 悉/A 無常/J 、/p 苦/J 、/p 空/J 、/p [無/V 我/R]=B 。/p
```

## Install

```bash
pip install git+https://github.com/dharmamitra/mitra-bo-zh-tagger
# or, for development
git clone https://github.com/dharmamitra/mitra-bo-zh-tagger && cd mitra-bo-zh-tagger && pip install -e .
```

Requirements: Python 3.10+, PyTorch 2.2+, transformers 4.57+ (Qwen3.5 architecture). The model weights
(18 GB, bfloat16) are downloaded from the Hugging Face Hub on first use.

**Mac (Apple Silicon):** works out of the box on the MPS backend in float16; you need a machine with at
least 24 GB of unified memory (32 GB recommended). Install PyTorch with `pip install torch` (the default
wheel includes MPS). If an operator is missing on MPS, run with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
**Linux/Windows with an NVIDIA GPU:** bfloat16 on CUDA, ~19 GB of VRAM.
**CPU only:** works but slow (float32, ~36 GB RAM); expect a minute or more per 200 characters.

## Command line

```bash
mitra-tag --lang bo input_wylie_or_unicode.txt            # terse annotation, one sentence per line
mitra-tag --lang zh --format json input.txt                # parsed tokens, offsets, compounds
mitra-tag --lang bo --format events input.txt              # dharmamitra grammar-explained WordEvent records
echo "爾時世尊告諸比丘" | mitra-tag --lang zh
```

Options: `--model` (Hub id or local path), `--device cuda|mps|cpu` (auto by default), `--max-new-tokens`.

## Python

```python
from mitra_tagger import Tagger, tibetan_grammar, chinese_grammar

tagger = Tagger()                       # device and dtype chosen automatically
res = tagger.tag("བཅོམ་ལྡན་འདས་ཀྱིས་བཀའ་སྩལ་པ།", lang="bo")
for chunk in res:
    print(chunk["output"], chunk["valid"])
    for sent in chunk["sentences"]:
        for tok in sent.tokens:
            print(tok.surface, tok.pos, tibetan_grammar.function_for(tok))
        print(tibetan_grammar.to_word_events(sent))      # backend-compatible records

res = tagger.tag("若能如是觀者則得解脫", lang="zh")
print(chinese_grammar.punctuated(res[0]["sentences"][0]))   # 若能如是觀者，則得解脫。
```

`tag()` splits long input into chunks the model was trained on (Tibetan: sentence groups of up to ~500
characters of Wylie, split at shad; Chinese: windows of up to 200 characters, split at existing sentence
marks if any), generates greedily, parses the output and **validates that the annotation reproduces the
input exactly** (letters for Tibetan, CJK characters for Chinese). An invalid chunk is split in half and
retried once; `valid` reports the result. Chinese punctuation in the input is discarded before tagging
and re-predicted.

## Output format

One sentence per line. Tokens are `WORD/POS`; punctuation is `MARK/p`; a run of words that translates one
Sanskrit word or compound is bracketed `[ ... ]`. Tibetan words join syllables with `_`; an affix written
attached to the preceding syllable in the input (`'i`, `s`, `r`, `'o`, `'am`, `'ang`) is its own token
`+affix`; shad runs are `|/p` (single) and `||/p` (double). Chinese brackets carry the compound type:
`=T` tatpuruṣa, `=K` karmadhāraya, `=D` dvandva, `=B` bahuvrīhi, `=A` avyayībhāva, `=O` other.

| Tibetan tag | | Chinese tag | |
|---|---|---|---|
| N noun | P proper noun | N noun | P proper noun |
| V verb (finite / predicate stem) | G verbal noun | V verb | J adjective |
| J adjective | A adverb | A adverb | R pronoun |
| M numeral | R pronoun | M numeral | X auxiliary |
| D determiner / demonstrative / plural | X negation | C adposition | K coordinating conjunction |
| C case particle | K clause particle | S subordinating conjunction | L particle |
| L clitic or final particle | I interjection | I interjection | p punctuation |
| p punctuation | | | |

Segmentation is at the dictionary-lexeme level: a stem and its case particle, an adverb and its verb, a
verb and its auxiliary are always separate words; genuine multi-syllable lexemes (`byang_chub_sems_dpa'`)
stay whole. Chinese words are Sanskrit-lemma sized (compound members are separate words, transliterated
names are one word).

## Tibetan grammar layer (dharmamitra conventions)

`mitra_tagger.tibetan_grammar` carries the tables and rules that the dharmamitra main backend uses in
its Tibetan grammar-explained mode, so tagger output can feed the same UI:

* spelled-out functions for every tag and for each particle (`kyis` → *agentive (instrumental) case
  particle*, `la` → *dative-locative particle (la don)*, `nas` as a case particle vs as a clause
  connective, final and quotative particles, negation, determiners, attached affixes);
* `remove_case_endings()` — the backend's case-ending stripping for dictionary lookup (same list, same order);
* `with_trailing_tsheg()` — display convention: Tibetan-script surface forms end in exactly one tsheg;
* `steinert_url()` and `is_linkable()` — Christian Steinert dictionary links; single-syllable lemmas
  always link, multi-syllable lemmas only when a `headword_check(wylie)` callback confirms a real headword;
* `to_word_events(sentence, headword_check=None)` — `WordEvent` dicts as in the backend's
  `typing_models/grammar_events.py` (`surface`, `lemma`, `transliteration`, `function`, `meaning`,
  `external_source`, `external_url`, `mitra`), plus `sanskrit_unit` for bracketed words. `meaning`
  is left empty for a dictionary or LLM to fill.

`mitra_tagger.chinese_grammar` does the same for Chinese (DDB links, compound types, `punctuated()`).

## Quality

On held-out test data (100 items per language, scored against Gemini-corrected gold annotations):
word boundary F1 0.94 (zh) / 0.95 (bo), POS accuracy 0.94 for both, sentence boundary F1 0.81 (zh) /
0.87 (bo), Chinese punctuation position F1 0.79. Sanskrit-unit brackets are the weakest layer
(F1 0.52 zh / 0.65 bo, mostly under-marking). Outputs that fail the round-trip check are rare for
Tibetan (1%) and occasional for 200-character Chinese windows (14%, single-character slips); the
`valid` flag and the split-and-retry handle these.

## How the model was trained

Targets were generated from cross-lingual evidence and validated mechanically: Sanskrit–Tibetan and
Sanskrit–Chinese word alignments with Sanskrit morphology, the Japanese kundoku readings of the
Kokuyaku Issaikyō for Chinese, and Gemini 3.8 Flash producing (Tibetan) or correcting (Chinese) the
annotation under the constraint that it reproduce the input exactly. The two languages were trained
jointly on the `buddhist-nlp` 9B stage-2 base model. See the model card for details.

## License

MIT for this code. The model weights follow the license on the model card.
