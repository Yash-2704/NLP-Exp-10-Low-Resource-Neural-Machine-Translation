"""Phase 1: load, clean, filter and split the WMT26 Indic MT training data.

Usage: python src/prepare_data.py [--raw DIR] [--out DIR]
Writes {train,dev,test}.jsonl (fields: lang, en, xx) and results/data_stats.csv.
"""
import argparse, hashlib, json, re, unicodedata
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# code: (file, has_header, en_col, xx_col, target script, NLLB code)
LANGS = {
    "as":   ("Category I/English-Assamese Training Data 2026.csv", True, 0, 1, "BENGALI", "asm_Beng"),
    "njz":  ("Category I/English-Nyishi Training Data 2026.xlsx", True, 0, 1, "LATIN", "njz_Latn"),
    "lus":  ("Category I/English-Mizo Traning Data 2026.xlsx", False, 0, 1, "LATIN", "lus_Latn"),
    "kha":  ("Category I/English-Khasi Training Data 2026.xlsx", True, 0, 1, "LATIN", "kha_Latn"),
    "mni":  ("Category I/English-Manipuri Training Data 2026.xlsx", True, 0, 1, "BENGALI", "mni_Beng"),
    "mtei": ("Category I/English-Meitei-Mayek Transining Data 2026.xlsx", True, 0, 1, "MEETEI", "mni_Mtei"),
    "brx":  ("Category II/English-Bodo Training Data 2026.xlsx", True, 0, 1, "DEVANAGARI", "brx_Deva"),
    "tgj":  ("Category II/English-Tagin Training Data 2026.xlsx", True, 1, 0, "LATIN", "tgj_Latn"),
    "trp":  ("Category II/English-Kokborok Training Data 2026.xlsx", True, 0, 1, "LATIN", "trp_Latn"),
    "nag":  ("Category II/English-Nagamese  Training Data 2026.xlsx", False, 0, 1, "LATIN", "nag_Latn"),
    "mjw":  ("Category II/English - Karbi Training Data 2026.xlsx", True, 0, 1, "LATIN", "mjw_Latn"),
}
CATEGORY = {c: (1 if v[0].startswith("Category I/") else 2) for c, v in LANGS.items()}

MAX_WORDS, MAX_RATIO, SHORT = 100, 3.0, 3
TEST_CAP, TEST_FRAC = 1000, 0.10

PUNCT_BEFORE = re.compile(r"\s+([,.;:?!%)\]}।॥꯫’”])")
PUNCT_AFTER = re.compile(r"([(\[{‘“])\s+")


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[\s ​﻿]+", " ", s).strip()
    s = PUNCT_BEFORE.sub(r"\1", s)   # undo pre-tokenized spacing: "word ," -> "word,"
    s = PUNCT_AFTER.sub(r"\1", s)
    return s


def scripts(s: str) -> set:
    out = set()
    for ch in s:
        if ch.isalpha():
            try:
                out.add(unicodedata.name(ch).split()[0])
            except ValueError:
                out.add("?")
    return out


def script_ok(s: str, allowed: set) -> bool:
    return scripts(s) <= allowed | {"MODIFIER"}


def key(en: str) -> str:
    """Grouping key: same English sentence across languages -> same split."""
    return re.sub(r"[^\w]+", " ", en.lower()).strip()


def rank(k: str) -> str:
    return hashlib.md5(k.encode()).hexdigest()  # deterministic shuffle


def load(code: str, raw: Path) -> pd.DataFrame:
    f, hdr, e, x = LANGS[code][:4]
    p = raw / f
    df = pd.read_csv(p, header=0 if hdr else None) if p.suffix == ".csv" else pd.read_excel(p, header=0 if hdr else None)
    return pd.DataFrame({"en": df.iloc[:, e], "xx": df.iloc[:, x]})


def clean(code: str, df: pd.DataFrame, log: dict) -> pd.DataFrame:
    tscript = LANGS[code][4]
    step = lambda name, d: log.__setitem__(name, len(d)) or d
    step("raw_rows", df)
    df = step("non_empty", df.dropna().astype(str))
    df = df.assign(en=df.en.map(normalize), xx=df.xx.map(normalize))
    df = step("non_empty_after_norm", df[(df.en != "") & (df.xx != "")])
    df = step("no_tag_rows", df[~df.xx.str.match(r"^\d+ from ")])
    df = step("dedup_pairs", df.drop_duplicates(["en", "xx"]))
    df = step("src_ne_tgt", df[df.en.str.lower() != df.xx.str.lower()])
    ew, xw = df.en.str.split().str.len(), df.xx.str.split().str.len()
    df = step("max_len", df[(ew <= MAX_WORDS) & (xw <= MAX_WORDS)])
    ew, xw = df.en.str.split().str.len(), df.xx.str.split().str.len()
    ratio = pd.concat([ew / xw, xw / ew], axis=1).max(axis=1)
    df = step("length_ratio", df[(ratio <= MAX_RATIO) | (pd.concat([ew, xw], axis=1).max(axis=1) <= 5)])
    allowed_x = {tscript} if tscript == "LATIN" else {tscript, "LATIN"}
    ok = df.en.map(lambda s: script_ok(s, {"LATIN"})) & df.xx.map(lambda s: script_ok(s, allowed_x))
    df = step("script_ok", df[ok])
    return df.assign(lang=code, key=df.en.map(key), short=df.en.str.split().str.len() <= SHORT)


def split(all_df: pd.DataFrame) -> pd.DataFrame:
    """Grouped split: each English key lives in exactly one split across ALL languages."""
    assign = {}  # key -> "dev"/"test"
    sizes = all_df.groupby("lang").size().sort_values()  # smallest languages pick first
    for lang in sizes.index:
        d = all_df[(all_df.lang == lang) & ~all_df.short]
        n = min(TEST_CAP, round(TEST_FRAC * sizes[lang]))
        have = {"dev": d.key.map(assign.get).eq("dev").sum(), "test": d.key.map(assign.get).eq("test").sum()}
        free = sorted(set(d.key) - assign.keys(), key=rank)
        for s in ("test", "dev"):
            need = max(0, n - have[s])
            for k in free[:need]:
                assign[k] = s
            free = free[need:]
    all_df = all_df.assign(split=all_df.key.map(assign).fillna("train"))
    # one reference per source sentence in dev/test
    ev = all_df[all_df.split != "train"].drop_duplicates(["lang", "key"])
    return pd.concat([all_df[all_df.split == "train"], ev])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    logs, parts = {}, []
    for code in LANGS:
        logs[code] = {}
        parts.append(clean(code, load(code, a.raw), logs[code]))
        print(code, logs[code])
    df = split(pd.concat(parts, ignore_index=True))

    for s in ("train", "dev", "test"):
        sub = df[df.split == s].sample(frac=1, random_state=42) if s == "train" else df[df.split == s].sort_values(["lang", "key"])
        with open(a.out / f"{s}.jsonl", "w") as f:
            for r in sub.itertuples():
                f.write(json.dumps({"lang": r.lang, "en": r.en, "xx": r.xx}, ensure_ascii=False) + "\n")

    stats = pd.DataFrame(logs).T
    counts = df.groupby(["lang", "split"]).size().unstack(fill_value=0)
    stats = stats.join(counts)
    stats.insert(0, "category", stats.index.map(CATEGORY))
    stats.insert(1, "nllb_code", stats.index.map(lambda c: LANGS[c][5]))
    stats["short_in_train"] = df[df.split == "train"].groupby("lang").short.sum()
    stats["en_words_median"] = df.groupby("lang").en.apply(lambda s: s.str.split().str.len().median())
    stats["xx_words_median"] = df.groupby("lang").xx.apply(lambda s: s.str.split().str.len().median())
    (ROOT / "results").mkdir(exist_ok=True)
    stats.to_csv(ROOT / "results" / "data_stats.csv")
    print(stats.to_string())

    # leakage self-check: no English key in more than one split
    per_key = df.groupby("key").split.nunique()
    assert (per_key == 1).all(), f"leak: {(per_key > 1).sum()} keys in multiple splits"
    print("OK: no English sentence shared across splits")


if __name__ == "__main__":
    main()
