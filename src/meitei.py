"""Meitei Mayek <-> Bengali-script transliteration for Manipuri.

Why: the NLLB tokenizer has no Meitei Mayek characters (~49% UNK), but NLLB already
knows Manipuri written in Bengali script (mni_Beng). We train/decode in Bengali script
and convert back to Meitei Mayek before scoring against the original references.

The only ambiguity: lonsum (final) k/l/m/p/n and consonant + apun iyek (conjunct) both
become "X্". Reverse rule (fitted on the training data): word-initial -> apun (a word can't
start with a final consonant); k before র/ৱ/য় and p before র/ল/ৱ/য় -> apun; otherwise lonsum.
Round-trip accuracy on the real data is printed by `python src/meitei.py`.
"""
import unicodedata

M2B = {
    "ꯀ": "ক", "ꯁ": "স", "ꯂ": "ল", "ꯃ": "ম", "ꯄ": "প", "ꯅ": "ন", "ꯆ": "চ", "ꯇ": "ত",
    "ꯈ": "খ", "ꯉ": "ঙ", "ꯊ": "থ", "ꯋ": "ৱ", "ꯌ": "য়", "ꯍ": "হ", "ꯎ": "উ", "ꯏ": "ই",
    "ꯐ": "ফ", "ꯑ": "অ", "ꯒ": "গ", "ꯓ": "ঝ", "ꯔ": "র", "ꯕ": "ব", "ꯖ": "জ", "ꯗ": "দ",
    "ꯘ": "ঘ", "ꯙ": "ধ", "ꯚ": "ভ",
    # lonsum (syllable-final) letters
    "ꯛ": "ক্", "ꯜ": "ল্", "ꯝ": "ম্", "ꯞ": "প্", "ꯟ": "ন্", "ꯠ": "ৎ", "ꯡ": "ং", "ꯢ": "ঈ",
    # vowel signs and marks
    "ꯣ": "ো", "ꯤ": "ি", "ꯥ": "া", "ꯦ": "ে", "ꯧ": "ৌ", "ꯨ": "ু", "ꯩ": "ৈ", "ꯪ": "ঁ",
    "꯫": "।", "꯭": "্",
    **{chr(0xABF0 + i): chr(0x09E6 + i) for i in range(10)},  # digits
}
M2B = {k: unicodedata.normalize("NFC", v) for k, v in M2B.items()}

_LONSUM = {"ক": "ꯛ", "ল": "ꯜ", "ম": "ꯝ", "প": "ꯞ", "ন": "ꯟ"}
_CONJ_NEXT = {"ক": ("র", "ৱ", "য়"), "প": ("র", "ল", "ৱ", "য়")}
_B2M = {v: k for k, v in M2B.items() if k not in "ꯛꯜꯝꯞꯟ"}
_KEYS = sorted(_B2M, key=len, reverse=True)


def to_beng(s: str) -> str:
    return unicodedata.normalize("NFC", "".join(M2B.get(c, c) for c in s))


def to_mtei(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c in _LONSUM and s[i + 1:i + 2] == "্":
            word_start = i == 0 or not "\u0980" <= s[i - 1] <= "\u09ff"
            conj = any(s.startswith(k, i + 2) for k in _CONJ_NEXT.get(c, ()))
            out.append(_B2M[c] + "꯭" if word_start or conj else _LONSUM[c])
            i += 2
            continue
        k = next((k for k in _KEYS if s.startswith(k, i)), None)
        out.append(_B2M[k] if k else c)
        i += len(k) if k else 1
    return "".join(out)


if __name__ == "__main__":
    import json, sys
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data" / "train.jsonl"
    refs = [json.loads(l)["xx"] for l in open(path) if '"lang": "mtei"' in l]
    same = sum(to_mtei(to_beng(r)) == r for r in refs)
    chars = sum(len(r) for r in refs)
    bad = sum(sum(a != b for a, b in zip(to_mtei(to_beng(r)), r)) + abs(len(to_mtei(to_beng(r))) - len(r)) for r in refs)
    print(f"round-trip exact: {same}/{len(refs)} = {same/len(refs):.2%}; char mismatch ≈ {bad/chars:.3%}")
    assert same / len(refs) > 0.9, "transliteration round-trip too lossy"
