# Citing

## This package and model

Nehrdich, S. and the MITRA project (2026). *mitra-bo-zh-tagger: sentence/word segmentation, POS tagging
and Sanskrit-unit annotation for classical Tibetan and Buddhist Chinese.*
https://github.com/dharmamitra/mitra-bo-zh-tagger, model https://huggingface.co/buddhist-nlp/mitra-bo-zh-tagger

## Work the Tibetan grammar layer is built on

The rule layer (`mitra_tagger/tibetan_rules.py`, `mitra_tagger/tibetan_rules_ext.py`) implements the
analyses of the Classical Tibetan Annotation Manual and the disambiguation rules of the SOAS rule-based
tagger; its closed-class lists were checked against the ACTib gold lexicon. Please cite:

```
@misc{faggionato2023manual,
  author    = {Faggionato, Christian and Meelen, Marieke and Hill, Nathan W.},
  title     = {Classical Tibetan Annotation Manual, Part II: Segmentation \& POS tagging (version 1.0)},
  year      = {2023},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.7880130},
  note      = {CC BY 4.0}
}

@misc{garrett2017tagger,
  author    = {Garrett, Edward and Hill, Nathan W.},
  title     = {A rule based Tibetan part-of-speech (POS) tagger for the creation of gold standard training data},
  year      = {2017},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.574882},
  note      = {SOAS University of London; CC BY 4.0}
}

@misc{meelen_actib,
  author    = {Meelen, Marieke and Hill, Nathan W. and Faggionato, Christian},
  title     = {ACTib: Annotated Corpus of Classical Tibetan (gold lexicon)},
  howpublished = {\url{https://github.com/mariekemeelen/actib}},
  note      = {MIT}
}
```

The Tibetan verbs database (`tibetan_verbs.csv`, 2,492 verbs with present, past, future and imperative
stems) comes from the dharmamitra backend; its original provenance is not recorded.
