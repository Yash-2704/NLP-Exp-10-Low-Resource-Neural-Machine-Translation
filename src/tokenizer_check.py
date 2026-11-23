"""Measure NLLB tokenizer UNK rate and fertility (subword tokens per word) for each language.

High UNK => model cannot even represent the script (decides Meitei Mayek handling).
High fertility => text is split into many small pieces (harder to learn, slower).
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
unk = tok.unk_token_id

rows = defaultdict(list)
for line in open(ROOT / "data" / "train.jsonl"):
    r = json.loads(line)
    if len(rows[r["lang"]]) < 3000:
        rows[r["lang"]].append(r)

out = []
for lang, rs in rows.items():
    stat = {"lang": lang}
    for side in ("en", "xx"):
        ids = [tok(r[side], add_special_tokens=False).input_ids for r in rs]
        words = sum(len(r[side].split()) for r in rs)
        n = sum(len(i) for i in ids)
        stat[f"{side}_fertility"] = round(n / words, 2)
        stat[f"{side}_unk_pct"] = round(100 * sum(t == unk for i in ids for t in i) / n, 2)
    out.append(stat)

df = pd.DataFrame(out).set_index("lang")
print(df.to_string())
(ROOT / "results").mkdir(exist_ok=True)
df.to_csv(ROOT / "results" / "tokenizer_stats.csv")
