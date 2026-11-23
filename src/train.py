"""Train M1 (scratch Transformer) or M3/M4 (fine-tuned NLLB-200-distilled-600M).

  M1: python src/train.py --model scratch --out checkpoints/m1_scratch
  M3: python src/train.py --model nllb --out checkpoints/m3_nllb_multi
  M4: python src/train.py --model nllb --langs brx --out checkpoints/m4_nllb_brx
"""
import argparse, json, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import sacrebleu
import torch
from transformers import (EarlyStoppingCallback, MarianConfig, MarianMTModel, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM)

from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right

from common import CODES, EN, INIT_FROM, NLLB, NllbTok, SpTok, directions, encode_example, load_split, \
    postprocess, translate

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["scratch", "nllb"], required=True)
p.add_argument("--langs", nargs="*", default=list(CODES))
p.add_argument("--out", required=True)
p.add_argument("--epochs", type=float, default=None)
p.add_argument("--lr", type=float, default=None)
p.add_argument("--bsz", type=int, default=16)
p.add_argument("--accum", type=int, default=4)
p.add_argument("--max_len", type=int, default=128)
p.add_argument("--temp", type=float, default=5.0, help="temperature for language sampling (1 = proportional)")
p.add_argument("--eval_steps", type=int, default=2000)
p.add_argument("--patience", type=int, default=3)
p.add_argument("--dev_per_dir", type=int, default=100, help="dev sentences per direction used for checkpoint selection")
p.add_argument("--optim", default="adafactor")
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

SCRATCH = a.model == "scratch"
epochs = a.epochs or (40 if SCRATCH else 3)
lr = a.lr or (5e-4 if SCRATCH else 1e-4)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

train_rows = load_split("train", a.langs)
dev_rows = load_split("dev", a.langs)

# ---------- tokenizer + model ----------
if SCRATCH:
    texts = [t for r in train_rows for t in (r["en"], r["xx_model"])]
    tok = SpTok.train(texts, out / "spm")
    cfg = MarianConfig(
        vocab_size=len(tok), decoder_vocab_size=len(tok), d_model=512, encoder_layers=6, decoder_layers=6,
        encoder_attention_heads=8, decoder_attention_heads=8, encoder_ffn_dim=2048, decoder_ffn_dim=2048,
        dropout=0.3, attention_dropout=0.1, activation_dropout=0.1, activation_function="relu",
        max_position_embeddings=256, pad_token_id=tok.pad_id, eos_token_id=tok.eos_id,
        decoder_start_token_id=tok.eos_id, forced_eos_token_id=tok.eos_id, scale_embedding=True,
        share_encoder_decoder_embeddings=True, tie_word_embeddings=True)
    model = MarianMTModel(cfg)
else:
    tok = NllbTok()
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB)
    model.resize_token_embeddings(len(tok))
    emb = model.get_input_embeddings().weight
    with torch.no_grad():  # new language codes start from a related language's embedding
        for new, src in INIT_FROM.items():
            emb[tok.code_ids[new]] = emb[tok.tok.convert_tokens_to_ids(src)]
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M, vocab {len(tok)}")

# ---------- data: both directions, temperature sampling per language, resampled every epoch ----------
pool, by_lang = [], defaultdict(list)  # pool = every encoded example once; by_lang = indices into it
for r in train_rows:
    for d in directions(r):
        by_lang[r["lang"]].append(len(pool))
        pool.append(encode_example(tok, *d, a.max_len))
sizes = {l: len(v) for l, v in by_lang.items()}
total = sum(sizes.values())
w = {l: n ** (1 / a.temp) for l, n in sizes.items()}
quota = {l: round(total * w[l] / sum(w.values())) for l in sizes}
print("examples/epoch per lang (real -> sampled):", {l: (sizes[l], quota[l]) for l in sizes})
json.dump({"sizes": sizes, "quota": quota, "temp": a.temp}, open(out / "sampling.json", "w"), indent=1)

rng = np.random.default_rng(a.seed)
epoch_idx = []  # fresh temperature-sampled draw every epoch, stored as compact int arrays (not copies)
for e in range(int(np.ceil(epochs))):
    ep = []
    for l, ix in by_lang.items():
        ix, q = np.asarray(ix, dtype=np.int32), quota[l]
        ep.append(rng.choice(ix, q, replace=False) if q <= len(ix) else
                  np.concatenate([np.tile(ix, q // len(ix)), rng.choice(ix, q % len(ix), replace=False)]))
    ep = np.concatenate(ep)
    rng.shuffle(ep)
    epoch_idx.append(ep)
epoch_idx = np.concatenate(epoch_idx)[: int(epochs * total)]


class IndexDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(epoch_idx)

    def __getitem__(self, i):
        return pool[epoch_idx[i]]


train_ds = IndexDataset()

dev = defaultdict(list)  # (lang, src_code, tgt_code) -> [(src, ref)]
for r in dev_rows:
    for s, sc, t, tc in directions(r):
        if len(dev[(r["lang"], sc, tc)]) < a.dev_per_dir:
            dev[(r["lang"], sc, tc)].append((s, r["en"] if tc == EN else r["xx"]))


def collate(batch):
    def pad(seqs, v):
        L = max(map(len, seqs))
        return torch.tensor([s + [v] * (L - len(s)) for s in seqs])
    ids = pad([b["input_ids"] for b in batch], tok.pad_id)
    labels = pad([b["labels"] for b in batch], -100)
    # label smoothing pops `labels` before forward, so decoder inputs must be built here
    dec = shift_tokens_right(labels, tok.pad_id, tok.eos_id)  # decoder starts with </s>, as in NLLB
    return {"input_ids": ids, "attention_mask": ids.ne(tok.pad_id).long(), "labels": labels, "decoder_input_ids": dec}


class MTTrainer(Seq2SeqTrainer):
    """Checkpoint selection by mean dev chrF++ over all directions (greedy decoding for speed)."""
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        scores = {}
        for (lang, sc, tc), pairs in dev.items():
            hyp = translate(self.model, tok, [s for s, _ in pairs], sc, tc, beams=1, bsz=64, max_len=a.max_len)
            hyp = [postprocess(h, lang, tc) for h in hyp]
            scores[f"{lang}:{'en-xx' if sc == EN else 'xx-en'}"] = sacrebleu.corpus_chrf(hyp, [[r for _, r in pairs]], word_order=2).score
        metrics = {f"{metric_key_prefix}_chrf": float(np.mean(list(scores.values())))}
        metrics.update({f"{metric_key_prefix}_chrf_{k}": round(v, 2) for k, v in scores.items()})
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self.model.train()
        return metrics


args = Seq2SeqTrainingArguments(
    output_dir=str(out), per_device_train_batch_size=a.bsz, gradient_accumulation_steps=a.accum,
    learning_rate=lr, num_train_epochs=1, warmup_steps=4000 if SCRATCH else 500,
    lr_scheduler_type="inverse_sqrt" if SCRATCH else "linear", weight_decay=0.0 if SCRATCH else 0.01,
    label_smoothing_factor=0.1, optim=a.optim, bf16=True, gradient_checkpointing=not SCRATCH,
    eval_strategy="steps", eval_steps=a.eval_steps, save_strategy="steps", save_steps=a.eval_steps,
    save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="chrf", greater_is_better=True,
    logging_steps=100, report_to="none", group_by_length=True, remove_unused_columns=False,
    dataloader_num_workers=2, dataloader_persistent_workers=True, dataloader_pin_memory=True, seed=a.seed, save_only_model=True, disable_tqdm=True)

trainer = MTTrainer(model=model, args=args, train_dataset=train_ds, data_collator=collate,
                    eval_dataset=[0],  # placeholder: evaluate() is overridden to decode the dev set
                    callbacks=[EarlyStoppingCallback(early_stopping_patience=a.patience)])
trainer.train()

best = out / "best"
trainer.save_model(best)
tok.save(best)
json.dump(trainer.state.log_history, open(out / "log_history.json", "w"), indent=1)
print("best checkpoint:", trainer.state.best_model_checkpoint, "dev chrF++:", trainer.state.best_metric)
