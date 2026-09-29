from mitra_tagger.parse import parse_terse, is_faithful
from mitra_tagger import tibetan_grammar, chinese_grammar


def test_bo_roundtrip_and_events():
    inp = "de nas tshe dang ldan pa kun dga' bos bcom ldan 'das la 'di skad ces gsol to/ /"
    out = "de_nas/A [tshe_dang_ldan_pa/J] [kun_dga'_bo/P] +s/C [bcom_ldan_'das/N] la/C 'di_skad/A ces/L [gsol/V] to/L ||/p"
    s = parse_terse(out, "bo")
    assert is_faithful(s, inp, "bo")
    ev = tibetan_grammar.to_word_events(s[0])
    assert ev[3]["function"].startswith("agentive") and ev[3]["lemma"] == "s"
    assert ev[5]["function"].startswith("dative-locative")
    assert ev[0]["surface"].endswith("་")


def test_zh_roundtrip_and_punct():
    inp = "謂摶食觸食思食識食"
    out = "謂/V [摶/N 食/N]=T 、/p [觸/N 食/N]=T 、/p [思/N 食/N]=T 、/p [識/N 食/N]=T 。/p"
    s = parse_terse(out, "zh")
    assert is_faithful(s, inp, "zh")
    assert chinese_grammar.punctuated(s[0]) == "謂摶食、觸食、思食、識食。"
    assert s[0].units[0].type == "T" and len(s[0].units) == 4
