"""Score prediction files: chrF++ (primary), spBLEU, TER, COMET (xx->en only), with bootstrap 95% CIs
and paired bootstrap significance between systems.

  python src/evaluate.py --systems m1 m2 m3 m4 --compare m3:m1 m3:m2 m3:m4 [--comet]
Reads results/predictions/<system>.jsonl, writes results/metrics.csv and results/significance.csv.

Notes
- spBLEU = BLEU with the FLORES-200 SentencePiece tokenizer (as in the NLLB paper); plain 13a BLEU
  tokenizes Indic scripts badly. FLORES-200 SPM has no Meitei Mayek, so for mtei we compute spBLEU on the
  Bengali-script transliteration of hypothesis and reference; chrF++ is always on the original script.
- COMET (wmt22-comet-da) is computed only into English: its XLM-R encoder never saw most of our languages.
"""
import argparse, json
from collections import defaultdict

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF, TER

from common import ROOT
from meitei import to_beng

p = argparse.ArgumentParser()
p.add_argument("--systems", nargs="+", required=True)
p.add_argument("--compare", nargs="*", default=[], help="pairs a:b, tests whether a > b")
p.add_argument("--comet", action="store_true")
p.add_argument("--n_boot", type=int, default=1000)
p.add_argument("--out", default="metrics.csv", help="file name under results/")
a = p.parse_args()

PRED = ROOT / "results" / "predictions"
chrf, bleu, ter = CHRF(word_order=2), BLEU(tokenize="flores200"), TER()
rng = np.random.default_rng(42)


def load(sys):
    g = defaultdict(list)
    for l in open(PRED / f"{sys}.jsonl"):
        r = json.loads(l)
        g[(r["lang"], r["direction"])].append(r)
    return g


def stats(metric, hyp, ref):
    return np.array(metric._extract_corpus_statistics(hyp, [ref]), dtype=float)


def score(metric, st):
    return metric._compute_score_from_stats(list(st.sum(axis=0))).score


preds = {s: load(s) for s in a.systems}
boot_idx = {}  # same resamples for every system -> paired test
rows, chrf_stats = [], {}

comet_model = None
if a.comet:
    from comet import download_model, load_from_checkpoint
    comet_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

for sys, groups in preds.items():
    for (lang, d), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["src"])  # stable order so systems align for paired tests
        hyp, ref = [r["hyp"] for r in rs], [r["ref"] for r in rs]
        n = len(rs)
        if (lang, d) not in boot_idx:
            boot_idx[(lang, d)] = rng.integers(0, n, size=(a.n_boot, n))
        cs = stats(chrf, hyp, ref)
        chrf_stats[(sys, lang, d)] = (cs, [r["src"] for r in rs])
        boots = [score(chrf, cs[i]) for i in boot_idx[(lang, d)]]
        bh, br = (([to_beng(h) for h in hyp], [to_beng(r) for r in ref]) if lang == "mtei" and d == "en-xx" else (hyp, ref))
        row = {"system": sys, "lang": lang, "direction": d, "n": n, "chrF++": round(score(chrf, cs), 2),
               "chrF++_lo": round(np.percentile(boots, 2.5), 2), "chrF++_hi": round(np.percentile(boots, 97.5), 2),
               "spBLEU": round(bleu.corpus_score(bh, [br]).score, 2), "TER": round(ter.corpus_score(hyp, [ref]).score, 2)}
        if comet_model and d == "xx-en":
            data = [{"src": r["src"], "mt": r["hyp"], "ref": r["ref"]} for r in rs]
            row["COMET"] = round(100 * comet_model.predict(data, batch_size=64, gpus=1, progress_bar=False).system_score, 2)
        rows.append(row)
        print(row, flush=True)

df = pd.DataFrame(rows)
out = ROOT / "results"
df.to_csv(out / a.out, index=False)

sig = []
for pair in a.compare:
    s1, s2 = pair.split(":")
    for (lang, d), idx in boot_idx.items():
        if (s1, lang, d) not in chrf_stats or (s2, lang, d) not in chrf_stats:
            continue
        (st1, src1), (st2, src2) = chrf_stats[(s1, lang, d)], chrf_stats[(s2, lang, d)]
        assert src1 == src2, f"{pair} {lang} {d}: systems translated different sentences"
        diffs = np.array([score(chrf, st1[i]) - score(chrf, st2[i]) for i in idx])
        sig.append({"a": s1, "b": s2, "lang": lang, "direction": d,
                    "delta_chrF++": round(score(chrf, st1) - score(chrf, st2), 2),
                    "ci_lo": round(np.percentile(diffs, 2.5), 2), "ci_hi": round(np.percentile(diffs, 97.5), 2),
                    "p_value": round(float((diffs <= 0).mean()), 4)})
if sig:
    pd.DataFrame(sig).to_csv(out / "significance.csv", index=False)
    print(pd.DataFrame(sig).to_string())
