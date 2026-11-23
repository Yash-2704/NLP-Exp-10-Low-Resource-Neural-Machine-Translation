"""Shared pieces: language codes, data loading, tokenizers, batched translation."""
import json
from pathlib import Path

import torch

from meitei import to_beng, to_mtei

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NLLB = "facebook/nllb-200-distilled-600M"
EN = "eng_Latn"

CODES = {  # our lang -> NLLB-style code
    "as": "asm_Beng", "njz": "njz_Latn", "lus": "lus_Latn", "kha": "kha_Latn", "mni": "mni_Beng",
    "mtei": "mni_Mtei", "brx": "brx_Deva", "tgj": "tgj_Latn", "trp": "trp_Latn", "nag": "nag_Latn",
    "mjw": "mjw_Latn",
}
NLLB_KNOWN = {"as", "lus", "mni"}
# new code token -> existing NLLB code whose embedding initializes it (related language / same script)
INIT_FROM = {
    "njz_Latn": "lus_Latn", "kha_Latn": "lus_Latn", "tgj_Latn": "lus_Latn", "trp_Latn": "lus_Latn",
    "mjw_Latn": "lus_Latn", "nag_Latn": "asm_Beng", "brx_Deva": "hin_Deva", "mni_Mtei": "mni_Beng",
}


def load_split(split, langs=None):
    rows = [json.loads(l) for l in open(DATA / f"{split}.jsonl")]
    rows = [r for r in rows if langs is None or r["lang"] in langs]
    for r in rows:  # model always sees Meitei Mayek as Bengali script (see meitei.py)
        if r["lang"] == "mtei":
            r["xx_model"] = to_beng(r["xx"])
        else:
            r["xx_model"] = r["xx"]
    return rows


def directions(r):
    """Both translation directions for one pair: (src_text, src_code, tgt_text, tgt_code)."""
    c = CODES[r["lang"]]
    return [(r["en"], EN, r["xx_model"], c), (r["xx_model"], c, r["en"], EN)]


def postprocess(text, lang, tgt_code):
    return to_mtei(text) if lang == "mtei" and tgt_code != EN else text


class NllbTok:
    def __init__(self, path=NLLB):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        new = [c for c in CODES.values() if c not in self.tok.get_vocab()]
        if new:
            self.tok.add_tokens(new, special_tokens=True)
        self.pad_id, self.eos_id = self.tok.pad_token_id, self.tok.eos_token_id
        self.code_ids = {c: self.tok.convert_tokens_to_ids(c) for c in list(CODES.values()) + [EN]}

    def __len__(self):
        return len(self.tok)

    def encode(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def save(self, d):
        self.tok.save_pretrained(d)


class SpTok:
    """SentencePiece tokenizer for the from-scratch model; language codes are user symbols."""
    def __init__(self, model_file):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=str(model_file))
        self.model_file = Path(model_file)
        self.pad_id, self.eos_id = self.sp.pad_id(), self.sp.eos_id()
        self.code_ids = {c: self.sp.piece_to_id(c) for c in list(CODES.values()) + [EN]}
        self._special = set(self.code_ids.values()) | {self.pad_id, self.eos_id}

    @staticmethod
    def train(texts, out_dir, vocab=32000):
        import sentencepiece as spm
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(texts), model_prefix=str(out_dir / "spm"), vocab_size=vocab,
            model_type="unigram", character_coverage=0.9999, pad_id=0, unk_id=1, bos_id=-1, eos_id=2,
            user_defined_symbols=list(CODES.values()) + [EN], input_sentence_size=1_000_000,
            shuffle_input_sentence=True, num_threads=16, hard_vocab_limit=False)
        return SpTok(out_dir / "spm.model")

    def __len__(self):
        return self.sp.get_piece_size()

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, ids):
        return self.sp.decode([i for i in ids if i not in self._special])

    def save(self, d):
        import shutil
        shutil.copy(self.model_file, Path(d) / "spm.model")


def load_tok(model_dir):
    model_dir = Path(model_dir)
    return SpTok(model_dir / "spm.model") if (model_dir / "spm.model").exists() else NllbTok(model_dir)


def encode_example(tok, src, src_code, tgt, tgt_code, max_len):
    """NLLB layout: source = [src_code] text </s>; target = [tgt_code] text </s>."""
    s = [tok.code_ids[src_code]] + tok.encode(src)[: max_len - 2] + [tok.eos_id]
    t = [tok.code_ids[tgt_code]] + tok.encode(tgt)[: max_len - 2] + [tok.eos_id]
    return {"input_ids": s, "labels": t}


@torch.no_grad()
def translate(model, tok, srcs, src_code, tgt_code, beams=4, bsz=32, max_len=128, no_repeat=0):
    """Translate a list of sentences that share one direction. Sorted by length for speed."""
    model.eval()
    order = sorted(range(len(srcs)), key=lambda i: len(srcs[i]))
    out = [None] * len(srcs)
    dev = next(model.parameters()).device
    for b in range(0, len(order), bsz):
        idx = order[b:b + bsz]
        enc = [[tok.code_ids[src_code]] + tok.encode(srcs[i])[: max_len - 2] + [tok.eos_id] for i in idx]
        L = max(map(len, enc))
        ids = torch.tensor([e + [tok.pad_id] * (L - len(e)) for e in enc], device=dev)
        gen = model.generate(input_ids=ids, attention_mask=ids.ne(tok.pad_id), num_beams=beams,
                             forced_bos_token_id=tok.code_ids[tgt_code], decoder_start_token_id=tok.eos_id,
                             max_new_tokens=int(max_len * 1.5), no_repeat_ngram_size=no_repeat)
        for i, g in zip(idx, gen.tolist()):
            out[i] = tok.decode(g)
    return out
