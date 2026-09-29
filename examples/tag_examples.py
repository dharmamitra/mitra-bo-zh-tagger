"""Tag one Tibetan and one Chinese passage and print terse output, punctuated Chinese and word events."""
import json
from mitra_tagger import Tagger, tibetan_grammar, chinese_grammar

tagger = Tagger()   # pass model="/path/to/local/checkpoint" or device="cpu" if needed
bo = "de nas tshe dang ldan pa kun dga' bos bcom ldan 'das la 'di skad ces gsol to/ /bcom ldan 'das chos kyi rgyal po ni gang lags/"
zh = "爾時世尊告諸比丘汝等當知一切諸法皆悉無常苦空無我若能如是觀者則得解脫"
for chunk in tagger.tag(bo, "bo"):
    print(chunk["output"])
    print(json.dumps(tibetan_grammar.to_word_events(chunk["sentences"][0])[:4], ensure_ascii=False, indent=1))
for chunk in tagger.tag(zh, "zh"):
    print(chunk["output"])
    print(" ".join(chinese_grammar.punctuated(s) for s in chunk["sentences"]))
