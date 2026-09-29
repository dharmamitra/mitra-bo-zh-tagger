from mitra_tagger.parse import parse_terse, is_faithful
from mitra_tagger import tibetan_grammar, chinese_grammar

TSHEG = "་"


def test_bo_roundtrip_and_events():
    inp = "de nas tshe dang ldan pa kun dga' bos bcom ldan 'das la 'di skad ces gsol to/ /"
    out = "de_nas/A [tshe_dang_ldan_pa/J] [kun_dga'_bo/P] +s/C [bcom_ldan_'das/N] la/C 'di_skad/A ces/L [gsol/V] to/L ||/p"
    s = parse_terse(out, "bo")
    assert is_faithful(s, inp, "bo")
    ev = tibetan_grammar.to_word_events(s[0])
    assert ev[3]["function"].startswith("agentive/instrumental particle") and ev[3]["lemma"] == "s"
    assert ev[5]["function"].startswith("allative particle")          # la after a noun: case, not converb
    assert ev[0]["surface"].endswith(TSHEG)
    # regex family: 'to' after l-final gsol fails the sandhi check, so the layer declines (plain POS name)
    assert ev[-1]["function"] == "clitic or final particle"
    s2 = parse_terse("[stong_pa_nyid/N] do/L |/p", "bo")
    assert tibetan_grammar.to_word_events(s2[0])[1]["function"].startswith("sentence-final particle")


def test_case_vs_converb_and_verbs():
    # nas after a noun is a case particle, after a verb a converb; verb stems get tense notes
    s = parse_terse("khyim/N nas/C byung/V nas/K |/p", "bo")
    ev = tibetan_grammar.to_word_events(s[0])
    assert ev[1]["function"].startswith("elative particle")
    assert ev[3]["function"].startswith("elative converb")
    assert "stem" in ev[2]["function"]


def test_zh_roundtrip_and_punct():
    inp = "謂摶食觸食思食識食"
    out = "謂/V [摶/N 食/N]=T 、/p [觸/N 食/N]=T 、/p [思/N 食/N]=T 、/p [識/N 食/N]=T 。/p"
    s = parse_terse(out, "zh")
    assert is_faithful(s, inp, "zh")
    assert chinese_grammar.punctuated(s[0]) == "謂摶食、觸食、思食、識食。"
    assert s[0].units[0].type == "T" and len(s[0].units) == 4
