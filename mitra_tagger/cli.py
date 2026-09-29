"""Command line: mitra-tag --lang bo|zh [--format terse|json|events] [file]  (reads stdin if no file)."""
import argparse, json, sys

from .model import Tagger, DEFAULT_MODEL
from .parse import to_dict
from . import tibetan_grammar, chinese_grammar


def main():
    ap = argparse.ArgumentParser(description="Sentence/word segmentation, POS and Sanskrit-unit tagging for Tibetan and Buddhist Chinese")
    ap.add_argument("file", nargs="?", help="input text file (default: stdin)")
    ap.add_argument("--lang", required=True, choices=["bo", "zh"])
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local path")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--format", default="terse", choices=["terse", "json", "events"],
                    help="terse = model output; json = parsed tokens; events = dharmamitra WordEvent records")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    a = ap.parse_args()
    text = open(a.file, encoding="utf-8").read() if a.file else sys.stdin.read()
    tagger = Tagger(a.model, device=a.device, max_new_tokens=a.max_new_tokens)
    results = tagger.tag(text, a.lang)
    if a.format == "terse":
        for r in results:
            print(r["output"] + ("" if r["valid"] else "\n# WARNING: output does not reproduce the input"))
    elif a.format == "json":
        print(json.dumps([{"input": r["input"], "valid": r["valid"], "sentences": to_dict(r["sentences"])} for r in results],
                         ensure_ascii=False, indent=1))
    else:
        grammar = tibetan_grammar if a.lang == "bo" else chinese_grammar
        out = []
        for r in results:
            for s in r["sentences"]:
                out.append({"type": "source_text", "text": s.raw})
                out.extend(grammar.to_word_events(s))
        print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
