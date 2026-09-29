"""mitra-bo-zh-tagger: sentence/word segmentation, POS and Sanskrit-unit annotation for
classical Tibetan (Wylie or Unicode) and Buddhist Chinese, with the mitra-bo-zh-tagger model."""
from .model import Tagger
from .parse import parse_terse, Sentence, Token, Unit
from . import tibetan_grammar, chinese_grammar

__all__ = ["Tagger", "parse_terse", "Sentence", "Token", "Unit", "tibetan_grammar", "chinese_grammar"]
__version__ = "0.1.0"
