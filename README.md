# mitra-bo-zh-tagger

Sentence segmentation, word segmentation, part-of-speech tagging and Sanskrit-unit annotation for
classical Tibetan (Wylie or Unicode) and Buddhist Chinese, using
the [`buddhist-nlp/mitra-bo-zh-tagger`](https://huggingface.co/buddhist-nlp/mitra-bo-zh-tagger) model
(a 9B Qwen3.5-based model). For Chinese the model also restores punctuation.

```
in : de nas tshe dang ldan pa kun dga' bos bcom ldan 'das la 'di skad ces gsol to/ /
out: [de_nas/A] [tshe_dang_ldan_pa/J] [kun_dga'_bo/P] +s/C [bcom_ldan_'das/N] la/C 'di_skad/A ces/L [gsol/V] to/L ||/p

in : 爾時世尊告諸比丘汝等當知一切諸法皆悉無常苦空無我
out: 爾/R 時/N 世尊/N 告/V 諸/L 比丘/N ：/p 「/p 汝/R 等/L 當/X 知/V ，/p [一切/N 諸/L 法/N]=K 皆/A 悉/A 無常/J 、/p 苦/J 、/p 空/J 、/p [無/V 我/R]=B 。/p
```

## Credits

The Tibetan grammar layer in this package is built on the work of Nathan W. Hill, Marieke Meelen,
Christian Faggionato and Edward Garrett, whose annotation manual, rule-based tagger and ACTib
lexicon define the tag set, the particle functions and the disambiguation rules used here:

* Faggionato, C., Meelen, M. & Hill, N. W. (2023). *Classical Tibetan Annotation Manual, Part II:
  Segmentation & POS tagging* (version 1.0). Zenodo. https://doi.org/10.5281/zenodo.7880130 (CC BY 4.0)
* Garrett, E. & Hill, N. W. (2017). *A rule based Tibetan part-of-speech (POS) tagger for the creation of
  gold standard training data*. SOAS University of London, Zenodo. https://doi.org/10.5281/zenodo.574882
  (CC BY 4.0)
* Meelen, M. et al. *ACTib: Annotated Corpus of Classical Tibetan*, gold lexicon.
  https://github.com/mariekemeelen/actib (MIT)

See [Sources of the rules](#sources-of-the-rules) for exactly which sections and rules were used, and
`CITATION.md` for citation forms. If you use the Tibetan function labels in published work, please
cite these sources alongside this package.

## Install

```bash
pip install git+https://github.com/dharmamitra/mitra-bo-zh-tagger
# or, for development
git clone https://github.com/dharmamitra/mitra-bo-zh-tagger && cd mitra-bo-zh-tagger && pip install -e .
```

Requirements: Python 3.10+, PyTorch 2.2+, transformers 4.57+ (Qwen3.5 architecture). The model weights
(18 GB, bfloat16) are downloaded from the Hugging Face Hub on first use.

**Mac (Apple Silicon):** works out of the box on the MPS backend in float16. You need a machine with at
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
marks if any), generates greedily, parses the output and validates that the annotation reproduces the
input exactly (letters for Tibetan, CJK characters for Chinese). An invalid chunk is split in half and
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
stay whole. 

### Sentence segmentation

The model decides sentence boundaries itself; each output line is one sentence.

* **Tibetan:** boundaries are not taken from the shad. A sentence is a main clause ending in a finite
  verb plus its final particle, or a complete utterance; subordinate clauses stay on the same line, and
  direct speech together with its speech verb counts as one sentence. So a double shad inside a verse or
  list does not force a break, and a break can fall after a single shad. Shad runs are kept as `|/p`
  and `||/p` tokens wherever they occur in the input.

  ```
  in : de nas tshe dang ldan pa kun dga' bos bcom ldan 'das la 'di skad ces gsol to/ /bcom ldan 'das chos kyi rgyal po ni gang lags/
  out: [de_nas/A] [tshe_dang_ldan_pa/J] [kun_dga'_bo/P] +s/C [bcom_ldan_'das/N] la/C 'di_skad/A ces/L [gsol/V] to/L ||/p
       [bcom_ldan_'das/N] [chos/N kyi/C rgyal_po/N] ni/L [gang/R] lags/V |/p
  ```

* **Chinese:** the input carries no punctuation, so the model places the sentence-final marks
  (。？！, with closing quotation marks) and the clause marks (，、：；) and thereby fixes the sentence
  division. The standard follows the Japanese kundoku readings the model was trained on, which divide
  sentences more finely than CBETA punctuation does (a CBETA ， between two finite clauses often becomes
  a 。). `chinese_grammar.punctuated()` returns a sentence with the marks reinserted.

  ```
  in : 爾時世尊告諸比丘汝等當知一切諸法皆悉無常苦空無我若能如是觀者則得解脫
  out: 爾/R 時/N 世尊/N 告/V 諸/L 比丘/N ：/p 「/p 汝/R 等/L 當/X 知/V ，/p [一切/N 諸/L 法/N]=K 皆/A 悉/A 無常/J 、/p 苦/J 、/p 空/J 、/p [無/V 我/R]=B 。/p
       若/S 能/X 如/V 是/R 觀/V 者/L ，/p 則/K 得/X 解脫/V 。/p 」/p
  ```

## Tibetan grammar layer

The tagger gives segmentation and a coarse tag. For deployment, dharamitra's grammar-explained mode
runs a **rule-based layer on top of the tagger output** that turns tags into spelled-out grammatical
functions and tense notes.

* `mitra_tagger/tibetan_rules.py`: The case-particle
  versus converb decision (the same morpheme is a case particle after a noun or verbal noun and a
  converb after a verb, for the 13 morpheme groups kyi, kyis, la, na, nas, las, du, dang, te, zhing,
  rung, kyin, pas), clitics, determiners, relator nouns (a noun that takes a genitive before it or a
  spatial case after it), negation, fused demonstrative + case forms (der, des, 'dir, gang gis …), and
  verb-stem notes ("past stem of 'jog") from `tibetan_verbs.csv`.
* `mitra_tagger/tibetan_rules_ext.py` on by default; `TIB_RULE_STEPS="" ` disables them,
  `TIB_RULE_STEPS=special,regex` picks families: special verbs (copulas, existentials, modals nus /
  dgos / srid / shes, the byed / 'gro / 'gyur paradigms), pronouns, adverbs (intensifiers such as rab tu,
  the -chad directionals, proclausal and temporal adverbs, terminative adverbials), nominalisers beyond
  pa/ba (mkhan, tshul, sa, rgyu, thabs, lugs …), list gaps (demonstratives, quantifiers, plural markers,
  relator nouns), and the regex-tagger disambiguations.
* `mitra_tagger/tibetan_grammar.py` — the backend's dispatcher `rule_function()` over both files
  (verbatim), `flat_tokens()` to give each word its neighbours, `function_for()`, and the display and
  link conventions of the backend: case-ending stripping for dictionary lookup, the trailing-tsheg
  citation form, Steinert links (single-syllable lemmas always; multi-syllable only when a
  `headword_check(wylie)` callback confirms a real headword), and `to_word_events()` producing the
  backend's `WordEvent` records (`surface`, `lemma`, `transliteration`, `function`, `meaning`,
  `external_source`, `external_url`, `mitra`, plus `sanskrit_unit`). `meaning` is left empty.

When no rule applies, the label falls back to the plain POS name. The
regex family deliberately declines in some cases (e.g. a final particle that fails the sandhi check).

### Sources of the rules

1. **Faggionato, Meelen & Hill (2023), *Classical Tibetan Annotation Manual, Part II: Segmentation
   & POS tagging*, Zenodo, doi:10.5281/zenodo.7880130 (CC BY 4.0).** The main source. Sections used:
   §3.3 and §3.5 (case particles and converbs, the 13 morpheme groups), §3.4 (clitics: topic ni, focus
   kyang/yang, quotatives, question and final particles, imperative cig), §3.6 (determiners:
   demonstratives, plural markers, nyid and kho na, quantifiers, tsam), §3.8 (relator nouns and their
   genitive-before / spatial-case-after rule), §3.9 (nominalisers incl. mkhan, tshul), §3.10 (negation),
   §3.12 (pronouns), §3.2 (adverbs: intensifiers, -chad directionals), §3.14.2 (special verbs: copulas,
   existentials, modals, byed / 'gro / 'gyur).
2. **Garrett & Hill (2017), *A rule based Tibetan part-of-speech (POS) tagger for the creation of gold
   standard training data*, SOAS, Zenodo, doi:10.5281/zenodo.574882 (CC BY 4.0).** 307 regex
   disambiguation rules for the older tagset. The base layer took its case-versus-converb principle;
   the regex family adds the sandhi checks for final and question particles, de as semi-final only
   after a d-final syllable, su as terminative only after an s-final syllable (otherwise "who"),
   gyis before shig as the imperative of bgyid, tense from context (ma / mi before a verb, a following
   cig or nas), and the lexical rules for yongs su, rjes su, khong du, kho, de dag.
3. **A Tibetan verbs database** (`tibetan_verbs.csv`, 2,492 rows: present, past, future, imperative
   stems), from the dharmamitra backend; it supplies the "past stem of …" notes. Its original
   provenance is not recorded; it predates this work.
4. **Marieke Meelen's ACTib gold lexicon** (github.com/mariekemeelen/actib, MIT). Its per-tag word
   lists were used as a reference when filling the closed-class lists (pronouns, adverbs, quantifiers,
   relator nouns). It is not loaded at runtime; its verb list (verblex.txt) is not used yet.

`mitra_tagger.chinese_grammar` is a thin counterpart for Chinese (POS names, compound types, DDB
links, `punctuated()`).

## Quality

On held-out test data (100 items per language, scored against Gemini-corrected gold annotations):
word boundary F1 0.94 (zh) / 0.95 (bo), POS accuracy 0.94 for both, sentence boundary F1 0.81 (zh) /
0.87 (bo), Chinese punctuation position F1 0.79. Sanskrit-unit brackets are the weakest layer
(F1 0.52 zh / 0.65 bo, mostly under-marking). Outputs that fail the round-trip check are rare for
Tibetan (1%) and occasional for 200-character Chinese windows (14%, single-character slips); the
`valid` flag and the split-and-retry handle these.

## License
GPL on the code -- attribution is welcome, if you find the model useful in your work, please give a reference to the Dharmamitra project! 
