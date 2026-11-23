"""Translate the test set (both directions) with a trained or the base NLLB model.

  python src/translate.py --model_dir checkpoints/m3_nllb_multi/best --name m3
  python src/translate.py --model_dir facebook/nllb-200-distilled-600M --name m2 --langs as lus mni   # zero-shot
Writes results/predictions/<name>.jsonl  (lang, direction, src, ref, hyp)
"""
import argparse, json
from collections import defaultdict

import torch
from transformers import AutoModelForSeq2SeqLM

from common import CODES, EN, ROOT, NllbTok, directions, load_split, load_tok, postprocess, translate

p = argparse.ArgumentParser()
p.add_argument("--model_dir", required=True)
p.add_argument("--name", required=True)
p.add_argument("--langs", nargs="*", default=list(CODES))
p.add_argument("--split", default="test")
p.add_argument("--beams", type=int, default=4)
p.add_argument("--bsz", type=int, default=32)
p.add_argument("--no_repeat", type=int, default=0, help="block repeated n-grams of this size (0 = off); chosen on dev")
a = p.parse_args()

tok = NllbTok(a.model_dir) if a.model_dir.startswith("facebook/") else load_tok(a.model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(a.model_dir, torch_dtype=torch.bfloat16).cuda()  # loads Marian or NLLB

groups = defaultdict(list)
for r in load_split(a.split, a.langs):
    for s, sc, t, tc in directions(r):
        groups[(r["lang"], sc, tc)].append({"src": r["en"] if sc == EN else r["xx"], "src_model": s,
                                            "ref": r["en"] if tc == EN else r["xx"]})

outp = ROOT / "results" / "predictions"
outp.mkdir(parents=True, exist_ok=True)
with open(outp / f"{a.name}.jsonl", "w") as f:
    for (lang, sc, tc), items in groups.items():
        hyps = translate(model, tok, [i["src_model"] for i in items], sc, tc, beams=a.beams, bsz=a.bsz, no_repeat=a.no_repeat)
        d = "en-xx" if sc == EN else "xx-en"
        for i, h in zip(items, hyps):
            f.write(json.dumps({"lang": lang, "direction": d, "src": i["src"], "ref": i["ref"],
                                "hyp": postprocess(h, lang, tc)}, ensure_ascii=False) + "\n")
        print(lang, d, len(items), "done", flush=True)
