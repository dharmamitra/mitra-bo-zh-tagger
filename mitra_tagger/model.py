"""Load the tagger model and run it on Tibetan (Wylie or Unicode) or Chinese text.

Works on CUDA, Apple Silicon (MPS) and CPU. The model is a 9B-parameter causal LM
(Qwen3.5 architecture); it needs ~19 GB of memory in 16-bit precision.
"""
from __future__ import annotations
import re
from typing import List, Optional

import torch

from .prompts import build_prompt
from .parse import Sentence, parse_terse, is_faithful, normalize_input
from . import tibetan_grammar

DEFAULT_MODEL = "buddhist-nlp/mitra-bo-zh-tagger"
_CJK_PUNCT_SENT = "。？！"


def _pick_device(device: Optional[str]) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(device: str, dtype):
    if dtype is not None:
        return dtype
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _disable_cuda_kernels(device: str) -> None:
    """transformers' Qwen3.5 uses the causal-conv1d / flash-linear-attention CUDA kernels whenever they
    are importable, even on CPU or MPS, and then crashes. Force the pure-PyTorch path off CUDA."""
    if device.startswith("cuda"):
        return
    try:
        import transformers.models.qwen3_5.modeling_qwen3_5 as m
    except Exception:
        return
    for name in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
        if hasattr(m, name):
            setattr(m, name, None)
    if hasattr(m, "is_fast_path_available"):
        m.is_fast_path_available = False


class Tagger:
    def __init__(self, model: str = DEFAULT_MODEL, device: Optional[str] = None, dtype=None,
                 max_new_tokens: int = 1024, bo_max_chars: int = 500, zh_max_chars: int = 200):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = _pick_device(device)
        self.dtype = _pick_dtype(self.device, dtype)
        _disable_cuda_kernels(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=self.dtype).to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        self.bo_max_chars = bo_max_chars
        self.zh_max_chars = zh_max_chars

    # ------------------------------------------------------------ generation
    @torch.no_grad()
    def _generate(self, prompts: List[str]) -> List[str]:
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
                                  pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return [t.strip() for t in self.tokenizer.batch_decode(gen, skip_special_tokens=True)]

    # ------------------------------------------------------------ chunking
    @staticmethod
    def _chunks_bo(wylie: str, max_chars: int) -> List[str]:
        """Split Wylie at shad runs into pieces of <= max_chars (sentences kept whole)."""
        parts = re.split(r"(/\s*/|/)", wylie)
        sents, cur = [], ""
        for i in range(0, len(parts), 2):
            piece = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
            cur += piece
            if i + 1 < len(parts) and parts[i + 1].replace(" ", "") == "//" or len(cur) > max_chars:
                sents.append(cur.strip()); cur = ""
        if cur.strip():
            sents.append(cur.strip())
        chunks, cur = [], ""
        for s in sents:
            if cur and len(cur) + len(s) + 1 > max_chars:
                chunks.append(cur); cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur:
            chunks.append(cur)
        return chunks or [wylie.strip()]

    @staticmethod
    def _chunks_zh(text: str, max_chars: int) -> List[str]:
        """Keep only CJK characters (the model expects unpunctuated input); cut at existing sentence
        marks where present, otherwise hard-cut at max_chars."""
        pieces = re.split(f"[{_CJK_PUNCT_SENT}]", text)
        sents = [normalize_input(p, "zh") for p in pieces]
        sents = [s for s in sents if s]
        chunks, cur = [], ""
        for s in sents:
            while len(s) > max_chars:
                if cur:
                    chunks.append(cur); cur = ""
                chunks.append(s[:max_chars]); s = s[max_chars:]
            if cur and len(cur) + len(s) > max_chars:
                chunks.append(cur); cur = s
            else:
                cur += s
        if cur:
            chunks.append(cur)
        return chunks

    # ------------------------------------------------------------ public API
    def tag_chunk(self, text: str, lang: str, retries: int = 1) -> dict:
        """Tag one chunk. Returns {"input", "output", "sentences", "valid"}. If the output does not
        reproduce the input, the chunk is split at its midpoint sentence boundary and retried once."""
        out = self._generate([build_prompt(text, lang)])[0]
        try:
            sents = parse_terse(out, lang)
            valid = is_faithful(sents, text, lang)
        except ValueError:
            sents, valid = [], False
        if not valid and retries > 0:
            halves = self._split_half(text, lang)
            if len(halves) == 2:
                a = self.tag_chunk(halves[0], lang, retries - 1)
                b = self.tag_chunk(halves[1], lang, retries - 1)
                return {"input": text, "output": a["output"] + "\n" + b["output"],
                        "sentences": a["sentences"] + b["sentences"], "valid": a["valid"] and b["valid"]}
        return {"input": text, "output": out, "sentences": sents, "valid": valid}

    def _split_half(self, text: str, lang: str) -> List[str]:
        if lang == "bo":
            parts = self._chunks_bo(text, max(60, len(text) // 2))
        else:
            half = max(30, len(text) // 2)
            parts = [text[:half], text[half:]]
        return parts if len(parts) >= 2 else [text]

    def tag(self, text: str, lang: str, batch_size: int = 4) -> List[dict]:
        """Tag a whole text. Tibetan may be Wylie or Unicode (converted to Wylie). Returns one dict
        per chunk with the parsed sentences; use `mitra_tagger.parse.to_dict` to serialise."""
        if lang == "bo":
            if re.search(r"[ༀ-࿿]", text):
                text = tibetan_grammar.unicode_to_wylie(text)
            chunks = self._chunks_bo(text, self.bo_max_chars)
        elif lang == "zh":
            chunks = self._chunks_zh(text, self.zh_max_chars)
        else:
            raise ValueError("lang must be 'bo' or 'zh'")
        results = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            outs = self._generate([build_prompt(c, lang) for c in batch])
            for c, o in zip(batch, outs):
                try:
                    sents = parse_terse(o, lang); valid = is_faithful(sents, c, lang)
                except ValueError:
                    sents, valid = [], False
                if not valid:
                    results.append(self.tag_chunk(c, lang))
                else:
                    results.append({"input": c, "output": o, "sentences": sents, "valid": True})
        return results
