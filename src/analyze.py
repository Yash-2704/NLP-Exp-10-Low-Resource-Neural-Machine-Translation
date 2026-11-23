"""Turn results/metrics.csv (+ data/tokenizer stats) into slide charts and markdown tables.

  python src/analyze.py      -> results/figures/*.png, results/summary.md
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT

R = ROOT / "results"
FIG = R / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# validated categorical slots 1-3 (dataviz reference palette, light mode) + recessive ink
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NAMES = {"as": "Assamese", "njz": "Nyishi", "lus": "Mizo", "kha": "Khasi", "mni": "Manipuri",
         "mtei": "Meitei", "brx": "Bodo", "tgj": "Tagin", "trp": "Kokborok", "nag": "Nagamese", "mjw": "Karbi"}
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})

m = pd.read_csv(R / "metrics.csv")
stats = pd.read_csv(R / "data_stats.csv", index_col=0)
order = stats.sort_values("train", ascending=False).index.tolist()  # largest -> smallest training set
systems = set(m.system)


def save(fig, name, top_legend=False):
    if top_legend:  # one shared legend above the panels, clear of the bars
        h, l = fig.axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


def bars(ax, langs, series, d, title):
    """Grouped bars (2px gap between fills) of chrF++ per language for one direction."""
    w = 0.8 / len(series)
    for k, (sys, label, color) in enumerate(series):
        sub = m[(m.system == sys) & (m.direction == d)].set_index("lang")
        xs = [i + (k - (len(series) - 1) / 2) * w for i in range(len(langs))]
        ys = [sub["chrF++"].get(l, float("nan")) for l in langs]
        ax.bar(xs, ys, w * 0.92, color=color, label=label, edgecolor=SURF, linewidth=1)
    ax.set_xticks(range(len(langs)), [NAMES[l] for l in langs], rotation=35, ha="right")
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.set_ylabel("chrF++")
    ax.grid(axis="x", visible=False)


# 1. headline: quality vs training-data size (M3)
if "m3" in systems:
    fig, ax = plt.subplots(figsize=(8, 5))
    for d, color, label in (("en-xx", C1, "English → X"), ("xx-en", C2, "X → English")):
        sub = m[(m.system == "m3") & (m.direction == d)].set_index("lang")
        x = stats.loc[sub.index, "train"]
        ax.scatter(x, sub["chrF++"], s=60, color=color, label=label, edgecolor=SURF, linewidth=2, zorder=3)
        if d == "en-xx":
            for l in sub.index:
                ax.annotate(NAMES[l], (x[l], sub.loc[l, "chrF++"]), xytext=(5, -12), textcoords="offset points",
                            fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("training sentence pairs (log scale)")
    ax.set_ylabel("chrF++ (test)")
    ax.set_title("Fine-tuned NLLB (M3): quality vs. training data", color=INK, loc="left")
    ax.legend()
    save(fig, "1_chrf_vs_data_size")

# 2. pretraining effect: scratch (M1) vs fine-tuned NLLB (M3)
if {"m1", "m3"} <= systems:
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, d, t in zip(axs, ("en-xx", "xx-en"), ("English → X", "X → English")):
        bars(ax, order, [("m1", "M1 scratch Transformer", C2), ("m3", "M3 fine-tuned NLLB", C1)], d, t)
    save(fig, "2_scratch_vs_nllb", top_legend=True)

# 3. zero-shot vs fine-tuned on languages NLLB already knows
if {"m2", "m3"} <= systems:
    langs = [l for l in order if l in set(m[m.system == "m2"].lang)]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, d, t in zip(axs, ("en-xx", "xx-en"), ("English → X", "X → English")):
        bars(ax, langs, [("m2", "M2 NLLB zero-shot", C3), ("m3", "M3 fine-tuned NLLB", C1)], d, t)
    save(fig, "3_zeroshot_vs_finetuned", top_legend=True)

# 4. multilingual vs bilingual fine-tuning
if {"m3", "m4"} <= systems:
    langs = [l for l in order if l in set(m[m.system == "m4"].lang)]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, d, t in zip(axs, ("en-xx", "xx-en"), ("English → X", "X → English")):
        bars(ax, langs, [("m4", "M4 one model per language", C3), ("m3", "M3 one multilingual model", C1)], d, t)
    save(fig, "4_bilingual_vs_multilingual", top_legend=True)

# 5. tokenizer fertility vs quality (into the language)
tokp = R / "tokenizer_stats.csv"
if tokp.exists() and "m3" in systems:
    tk = pd.read_csv(tokp, index_col=0)
    sub = m[(m.system == "m3") & (m.direction == "en-xx")].set_index("lang")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(tk.loc[sub.index, "xx_fertility"], sub["chrF++"], s=60, color=C1, edgecolor=SURF, linewidth=2, zorder=3)
    for l in sub.index:
        ax.annotate(NAMES[l], (tk.loc[l, "xx_fertility"], sub.loc[l, "chrF++"]), xytext=(5, -14) if l == "lus" else (5, 4),
                    textcoords="offset points", fontsize=9, color=INK2)
    ax.set_xlabel("NLLB subword tokens per word (target language)")
    ax.set_ylabel("chrF++ English → X (M3)")
    ax.set_title("Tokenizer fertility vs. translation quality", color=INK, loc="left")
    save(fig, "5_fertility_vs_chrf")

# markdown tables for README / slides
lines = ["# Results\n"]
for metric in ("chrF++", "spBLEU", "TER", "COMET"):
    if metric not in m:
        continue
    t = m.pivot_table(index=["lang", "direction"], columns="system", values=metric).reset_index()
    t["lang"] = pd.Categorical(t["lang"], order, ordered=True)
    t = t.sort_values(["direction", "lang"]).dropna(how="all", subset=[c for c in t.columns if c.startswith("m")])
    t["lang"] = t["lang"].map(NAMES)
    lines += [f"\n## {metric}\n", t.round(2).to_markdown(index=False)]
sig = R / "significance.csv"
if sig.exists():
    s = pd.read_csv(sig)
    # 95% CI of the difference decides: entirely above 0 -> a wins, entirely below 0 -> b wins, else tie
    s["winner"] = [r.a if r.ci_lo > 0 else r.b if r.ci_hi < 0 else "tie" for r in s.itertuples()]
    s["lang"] = s["lang"].map(NAMES)
    lines += ["\n## Paired bootstrap significance (chrF++, 1000 resamples)\n", s.to_markdown(index=False)]
(R / "summary.md").write_text("\n".join(lines) + "\n")
print("wrote", R / "summary.md", "and", sorted(p.name for p in FIG.glob("*.png")))
