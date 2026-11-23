# Exp 10 — Low-Resource Neural Machine Translation: Strategy

**Task:** English ⇔ 11 low-resource Indic / North-East Indian languages, using the WMT26 Indic MT 2026 training data.
**Status:** plan only. No code written yet.

---

## 0. The story we are telling (one line)

> *How far can one multilingual model take 11 languages whose training data ranges from 54k pairs down to 1k, and what actually helps: pretraining, multilingual sharing, or more data?*

Every step below exists to answer one of four research questions (RQs). If a step doesn't serve an RQ, it's out.

| RQ | Question | Answered by |
|---|---|---|
| RQ1 | Does a pretrained model beat training from scratch in low-resource MT? | M1 vs M3 |
| RQ2 | Does fine-tuning on WMT26 data help, even for languages the pretrained model already knows? | M2 vs M3 (as, mni, lus) |
| RQ3 | Does training all languages together help the smallest ones, compared with one model per language? | M3 vs M4 |
| RQ4 | How does quality depend on data size, domain and script? | Analysis of M3 (no extra training) |

---

## 1. Data facts that drive every decision (from the profiling)

| Lang | Code | Script | Usable pairs | Domain | NLLB knows it? |
|---|---|---|---|---|---|
| Assamese | as | Bengali | ~53.9k | mixed (Bible 26%, agri, news) | yes (`asm_Beng`) |
| Nyishi | njz | Latin | ~50.1k | Tatoeba-style + dictionary | no |
| Mizo | lus | Latin | ~49.8k | Bible, stories, chat | yes (`lus_Latn`) |
| Khasi | kha | Latin | ~26.0k | Bible (57%) | no |
| Manipuri | mni | Bengali | ~23.1k | government press | yes (`mni_Beng`) |
| Meitei | mtei | Meitei Mayek | ~16.8k | news, health | no (script likely unseen) |
| Bodo | brx | Devanagari | ~12.8k | history, sports, news | no |
| Tagin | tgj | Latin | ~5.2k | short phrases, culture | no |
| Kokborok | trp | Latin | ~2.3k | Bible | no |
| Nagamese | nag | Latin | ~2.0k | history, sports, news | no |
| Karbi | mjw | Latin | ~1.0k | news headlines (noisy) | no |

Key consequences:
- **Huge imbalance** (54×), so the training sampling strategy matters (§4.3).
- **Different domains per language**, so scores are not comparable across languages without noting the domain. Report a domain column.
- **The same English sentences appear across languages** (as–kha 7.4k, brx–nag 2k, njz–tgj 1k), so a naive split leaks test data (§3.5).
- The official 2026 test set is password-protected and its references were never published, so **our own held-out split is the test set**.

---

## 2. Compute

| Server | GPU | VRAM free now | Role |
|---|---|---|---|
| t7 | RTX 4090 | ~1 GB of 24 GB (another job is holding it, 0% util) | Primary if it gets freed |
| t8 | RTX 4070 | ~10.5 GB of 12 GB | **Default primary** |
| t3 | RTX 3080 Ti | ~5 GB of 12 GB | Scratch baseline / evaluation runs |

Feasibility check for the biggest job (full fine-tuning of NLLB-600M) on 12 GB:
- With AdamW in fp32 it needs ~10 GB of weights + gradients + optimizer state before activations, which **doesn't fit**.
- With **bf16 + Adafactor (factored optimizer state) + gradient checkpointing + max length 128**, it needs about 7–9 GB, which **fits on t8**.
- If it's still tight, fall back to 8-bit AdamW (bitsandbytes). We don't use LoRA (see §4.2 for why).

Rough time: ~240k pairs × 2 directions ≈ 480k examples per epoch.
- About 1.5–2.5 h per epoch on the 4070, or about half that on the 4090.
- 3 epochs plus evaluation is well under a day.
- Scratch models are about 1–2 h each.

Ops: `uv` venv on the server, data copied with `rsync` (never committed), runs inside `tmux`, logs and checkpoints in `~/nlp-lab10/` on the server. Disk: t7 is 95% full, so keep checkpoints on t8 (236 GB free) and save only the best one.

---

## 3. Phase 1 — Data preparation (`prepare_data.py`)

All steps are deterministic, rule-based and logged with counts for each language, so the README and slides can show a "rows removed at each step" table.

| Step | What | Why (defense) |
|---|---|---|
| 3.1 Load | Per-file rules: Tagin columns swapped; Mizo and Nagamese have no header; select columns by position; drop the 4 tag rows ("500 from History") in Bodo/Nagamese and the 5.2k blank rows in Nagamese | Every file has a different layout; silent mis-loading would corrupt a whole language |
| 3.2 Normalize | Unicode NFC, NBSP → space, collapse repeated spaces, strip; **undo pre-tokenized spacing** (`word ,` → `word,`) on both sides | Files mix two spacing styles. Without this the model learns two ways to write a comma, and BLEU penalizes outputs arbitrarily |
| 3.3 Filter | Drop empty pairs, exact duplicate pairs, pairs where source = target, pairs >100 words, pairs with a length ratio >3 (only when the longer side has >5 words, so short dictionary pairs are kept), and pairs containing a wrong script (e.g. Bengali inside Nagamese, Kannada inside Meitei) | Profiling found ~800 misaligned pairs (e.g. a full sentence ↔ "Question 3"). Standard filters from the WMT/NLLB data pipelines; cheap and explainable |
| 3.4 Keep but tag | Dictionary-style entries (≤3 English words): **kept for training, excluded from dev/test** | They're valid word translations (useful vocabulary for tiny languages), but a test set of single words would distort sentence-level metrics |
| 3.5 Split | **Grouped split by normalized English sentence, shared across all languages**: a given English sentence goes to train, dev or test globally. Per language: test = dev = min(1000, 10%), rest = train. Fixed seed | Prevents cross-lingual leakage: in a multilingual model, a test sentence's English side seen in training for another language inflates scores |
| 3.6 Report | Save `stats.csv`: counts at each step, length distributions, domain mix | Gives the "Dataset Description" slide and README section |

Expected result: ~235k training pairs; about 1k dev + 1k test for most languages, and about 100 dev/test for Karbi. We'll state that Karbi results have wide confidence intervals.

**Alternative considered:** the official WMT26 test set. It's password-protected and its references are unpublished, so it's unusable. The WMT25 gold set was ruled out by the faculty brief. A held-out split is standard practice when no official reference exists.

---

## 4. Phase 2 — Models

### 4.1 The four models

| ID | Model | Trained on | Purpose |
|---|---|---|---|
| **M1** | Transformer from scratch (~60M params, 6+6 layers, d=512), shared SentencePiece 32k vocab, target-language tags | all 11 languages, both directions | Baseline: what the data alone can do (RQ1) |
| **M2** | NLLB-200-distilled-600M, **zero-shot**, no training | evaluated on as, mni, lus only | What pretraining alone gives (RQ2) |
| **M3** | NLLB-200-distilled-600M, **fine-tuned multilingually** | all 11 languages, both directions | Main system (RQ1, RQ2, RQ4) |
| **M4** | NLLB-200-distilled-600M fine-tuned **bilingually** (one model per language) | 4 representative languages: as, lus (Cat 1), brx, trp (Cat 2) | Does sharing help small languages? (RQ3) |

M4 covers only 4 languages because 22 directions × separate models adds a lot of compute for little extra evidence. Two large and two small languages are enough to show the trend.

### 4.2 Why NLLB-200-distilled-600M? (alternatives considered)

| Option | Verdict | Reason |
|---|---|---|
| **NLLB-200 distilled 600M** | ✅ chosen | One model covers both directions and all scripts we need except Meitei Mayek. Already knows 3 of our languages plus close relatives (Bengali, Hindi). Built for low-resource MT. Fits 12 GB with Adafactor |
| NLLB-200 1.3B / 3.3B | ❌ | Doesn't fit full fine-tuning on 12 GB; the 4090 isn't reliably free |
| IndicTrans2 | ⚠️ future scope | Stronger for as/brx/mni and has native Meitei Mayek, but covers only 4 of 11 languages and needs **two** separate models (en→indic, indic→en), which doubles the work. Good "future work" slide |
| mBART-50 | ❌ | Lacks Assamese and every NE language; weaker low-resource coverage |
| mT5 / ByT5 | ❌ | Not MT-pretrained, so it needs more data than we have |
| LoRA / adapters instead of full fine-tuning | ❌ | New languages need new tokens and full embedding updates; LoRA on embeddings is awkward and underperforms for new-language adaptation. Full FT fits anyway |
| LLM prompting (GPT, Claude, Llama) | ⚠️ future scope | Interesting comparison, but not "training a model", which is what the guidelines require; costs API money; unreproducible |

### 4.3 Key training decisions

| Decision | Choice | Defense |
|---|---|---|
| New languages in NLLB | Add 8 new language-code tokens (`kha_Latn`, `brx_Deva`, `trp_Latn`, `njz_Latn`, `tgj_Latn`, `mjw_Latn`, `nag_Latn`, `mni_Mtei`), each initialized from a related existing code (brx ← `hin_Deva`, nag ← `asm_Beng`, Latin-script languages ← `lus_Latn`, mni_Mtei ← `mni_Beng`) | Standard way to extend NLLB; a related-language initialization starts training from a sensible point instead of random noise |
| Meitei Mayek script | **Phase 1 check:** measure the NLLB tokenizer's UNK rate on Meitei Mayek. If it's high, transliterate Meitei Mayek → Bengali script for the model and back after decoding (a near one-to-one character mapping) | NLLB never saw this script; with UNKs the model literally cannot output the characters. Measured, then decided, so the choice is evidence-based |
| Same check for all languages | Report tokenizer **fertility** (tokens per word) per language | Explains part of the score differences (RQ4); a nice slide |
| Imbalance | **Temperature sampling, T = 5** (sample language i with probability ∝ (nᵢ/N)^(1/5)) | Used by M2M-100, NLLB and Arivazhagan et al. 2019. Without it, Karbi is 0.4% of batches and gets ignored |
| Both directions in one model | Yes, using the target-language code as the forced BOS token | Halves the number of models; NLLB was trained this way |
| Optimizer | Adafactor, lr 1e-4, warmup 500 steps, bf16, grad checkpointing, effective batch ~64 sentences | Fits 12 GB; the standard recipe for fine-tuning NLLB/T5-family models |
| Stopping | Evaluate dev chrF++ every ~2k steps; early stopping with patience 3; max 5 epochs | Prevents overfitting the small languages; picks the checkpoint by the metric we report |
| M1 config | Transformer-base-small, dropout 0.3, label smoothing 0.1, same tags and sampling | Standard low-resource settings (Sennrich & Zhang 2019, "Revisiting Low-Resource NMT") |
| Framework | Hugging Face `transformers` for all models (M1 uses a MarianMT config with random init) | One codebase and one evaluation path. fairseq is unmaintained and breaks on current Python |
| Decoding | Beam size 4, max 256 tokens | Standard; identical across all models |
| Seeds | Fixed (42) everywhere | Reproducibility |

---

## 5. Phase 3 — Evaluation (`evaluate.py`)

| Metric | Where | Why |
|---|---|---|
| **chrF++** (primary) | all 22 directions | Character-level, so it's robust for agglutinative languages and small test sets; recommended over BLEU for low-resource MT (Popović 2017; WMT findings) |
| BLEU (sacreBLEU) | all 22 | Required by convention and listed by the task; signature reported for reproducibility |
| TER | all 22 | Listed by the WMT26 task page |
| COMET (wmt22-comet-da) | **xx→en only** | COMET's encoder (XLM-R) doesn't know most target languages, so scores into them would be meaningless. Into English it's reliable. Saying this explicitly is itself a strong point in the defense |

Rigor:
- **Paired bootstrap resampling** (sacreBLEU, 1000 samples) for every model comparison, reported with 95% confidence intervals and significance. Without it, "M3 beats M1 by 0.8" could be noise, especially for Karbi's 100 sentences.
- Same test set, same detokenized references, same metric settings for every model.
- All predictions saved as files, so every number in the slides can be regenerated.

---

## 6. Phase 4 — Analysis (answers RQ4, makes the slides)

1. **Score vs. training-data size** scatter (log x-axis), one point per language with Cat 1/Cat 2 colors. The headline chart.
2. **Gain from pretraining** (M3 − M1) per language: where does NLLB help most? (Hypothesis: seen languages and same-script relatives.)
3. **Multilingual vs bilingual** (M3 vs M4) bar chart for the 4 languages (RQ3). Hypothesis: sharing helps Cat 2, with a slight drop or no change for Cat 1.
4. **en→xx vs xx→en** gap. Expect xx→en to be easier, since generating English benefits from pretraining.
5. **Tokenizer fertility vs score**: explains the script effect (Meitei Mayek).
6. **Domain note**: Bible-heavy languages (kha, trp) vs news (mjw, brx). Explains why the scores aren't directly comparable.
7. **Qualitative examples**: 3–4 good and bad translations per category, with typical errors (names, numbers, hallucination on Karbi).

---

## 7. Phase 5 — Deliverables

```
lab 10/
  README.md            dataset, preprocessing, models, how to run, results (required by guidelines)
  STRATEGY.md          this file
  requirements.txt
  src/prepare_data.py  Phase 1
  src/train.py         M1/M3/M4 (one script, config flags)
  src/evaluate.py      metrics + bootstrap + COMET
  src/analyze.py       tables + charts for slides
  results/             metrics CSVs, charts, sample predictions (committed)
  data/, checkpoints/  .gitignored (dataset license unclear; models too large)
```

Slides map one-to-one: Introduction → §0; Dataset → §1 + the §3.6 table; Methodology → §3–4; Results → §5–6 (model comparison table + charts); Conclusion/Future → §8.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| t7/t8 busy when we need them | Code runs on any single 12 GB GPU; batch size/accumulation configurable |
| NLLB OOM on 12 GB | Adafactor → 8-bit AdamW → max length 96, in that order |
| Meitei Mayek UNKs | Transliteration fallback (§4.3), decided by measurement |
| Cat 2 scores near zero | Expected; it's a finding, not a failure. chrF++ still shows differences where BLEU ≈ 0 |
| Karbi test set tiny (~100) | Report confidence intervals; don't over-claim |
| Time | M4 is the first thing to cut if time runs short; M1–M3 alone answer RQ1, RQ2 and RQ4 |

## 9. Future scope (for the last slide)
IndicTrans2 for its 4 covered languages; back-translation using monolingual data; LLM few-shot prompting comparison; human evaluation by native speakers; larger NLLB on the 4090.

---

## 10. Likely questions in the viva and our answers

- **Why not the official test set?** Its references are unpublished and the files are password-protected. We use a leakage-safe grouped held-out split, which is standard practice.
- **Why chrF++ over BLEU?** These languages are agglutinative and the test sets are small, so BLEU is near zero and noisy. chrF++ correlates better with human judgment there. We still report BLEU.
- **Why no COMET into Indic languages?** COMET's encoder never saw them, so the score would be meaningless.
- **Why one multilingual model?** Small languages borrow from large ones (RQ3 tests this directly), and it's one model instead of 22.
- **Why 600M and not bigger?** Full fine-tuning has to fit on a 12 GB GPU; we say so honestly and list bigger models as future scope.
- **Is the comparison fair?** Same splits, same preprocessing, same decoding and metrics for all models, with significance testing.
- **Why are Khasi and Karbi scores not comparable?** Different domains (Bible vs news) and 26× different data sizes; shown in the analysis.

---

## 11. Order of work

1. `prepare_data.py` + tokenizer fertility/UNK check → **checkpoint: review data stats together**
2. M2 zero-shot evaluation (fast, also tests the evaluation pipeline)
3. M1 scratch training (t3 or t8)
4. M3 fine-tuning (t8, or t7 if freed)
5. M4 bilingual runs (if time allows)
6. Evaluation + bootstrap + COMET
7. Analysis charts → README → slides
