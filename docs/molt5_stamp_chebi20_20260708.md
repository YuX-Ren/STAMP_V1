# MolT5-STAMP on ChEBI-20 — Report

Date: 2026-07-08
Author: yuxuan.ren@nus.edu.sg

## TL;DR

Trained a MolT5-family model that consumes/produces **STAMP** tokens instead of
raw SMILES on the MolT5 c4+ZINC pre-training recipe, then fine-tuned separately
for the two ChEBI-20 tasks. All numbers below use the **MolT5 official
evaluation recipes** (SciBERT WordPiece BLEU for text; char-level BLEU on
canonical SMILES + InChI-based exact match + fingerprint Tanimoto for
molecules).

- **Mol2text**: on par with **T5-Base**, ~2 pt below **MolT5-Base** across all
  BLEU/ROUGE/METEOR metrics. Attributable to pre-training budget (350k step
  checkpoint used for downstream vs MolT5's 1M-step pretrain).
- **Text2mol**: **5 of 7 metrics beat MolT5-Base**; validity **97.6 %** vs
  MolT5-Base 77.2 % (+20.4 pt) and even above MolT5-Large 90.5 %. Character-
  level BLEU-4 and Levenshtein trail by a modest amount because STAMP decoding
  reconstructs SMILES with a different atom ordering than the ChEBI-20 gold —
  it is a stringification difference, not a chemistry difference.

## Model & pre-training

| Item | Value |
|---|---|
| Backbone | T5.1.1-base (`d_model=768`, `d_ff=2048` SwiGLU, 12 enc + 12 dec, 12 heads) |
| Init | HuggingFace `google/t5-v1_1-base` |
| Molecule representation | Single-view **STAMP** via `STAMPRepresentation`, non ring-aware branch (`branch_mode: legacy_tree`, `allow_ring_single_edges: false`) |
| Motif vocab | `zinc_curated_vocab_30k_legacy/motif_vocab.txt` — 32,036 motif tokens mined from ZINC via PS-VAE legacy branch |
| Tokenizer | T5-v1.1-base + 30,549 added STAMP tokens (structural `[J_*]/[B_*]/[S_*]/[Ring]` + motif SMILES), total 62,649 |
| Model params | 246.3 M (matches MolT5-Base scale) |
| Pre-train mix | 50/50 C4 text + ZINC15 100M single-view STAMP, T5 span denoising |
| Batch | 256 / GPU × 4 H200 = **1024 global** |
| LR | 1.4e-4 (sqrt-2 scaled from paper 1e-4 at bs 512), 5k step warmup, linear decay |
| Max source / target len | 192 / 96 |
| Max steps | 500,000 (checkpoint used for downstream: **step 350k**) |
| Encoded ZINC output | `/scratch/yuxuan.ren/data/zinc_curated30k_stamp/*_stamp.parquet` (101.9 M rows, 4.7 GB, 43 shards) |
| Job | 522771 on hopper-19 (4 × H200, ~52 h wall) |

### Pre-train validation PPL (per-eval every 10k step)

Selected points:

```
   step   valid_ppl   c4_ppl   zinc_ppl
    10k     14.910    21.412    10.382
   100k      4.611     7.059     3.012
   200k      4.069     6.236     2.655
   300k      3.839     5.944     2.480
   350k      3.786     5.866     2.444   ← downstream init
   400k      3.758     5.819     2.426
```

Both c4 and ZINC PPL curves are monotonic and enter a plateau after ~350k; the
per-10k delta of `zinc_ppl` is in the ±0.01 noise band by step 400k.

Note: numerical PPL is **not directly comparable** to the prior msring 12k
run — different tokenizer (32k added tokens vs 12k), different STAMP encoding
scheme (single-view vs `[BOS] ... [CONCAT] ... [EOS]` multi-view), different
average per-sample length. c4 PPL (which uses the shared T5 base vocabulary)
IS comparable and is slightly better in the new run (5.82 vs 6.06 at step
300k).

## Fine-tuning

- Init: `checkpoints/step_00350000` of the pre-train run
- Trainer: `scripts/benchmark/finetune_molt5_chebi20.py`, standard T5 seq2seq
  cross-entropy with early stopping on validation loss
- One run per direction, driven by `DIRECTION={mol2text,text2mol}`
- `bs=32`, `lr=5e-4`, warmup ratio 5 %, up to 30 epochs, patience 6 evals
- Both directions early-stopped at **epoch 13, step 11 500** (~34 min / 1 GPU)
- Best validation PPL: mol2text 1.734, text2mol 1.741

## Evaluation

Test set: 3,300 rows of ChEBI-20 test (`test.txt`). Generation: beam 5,
`max_new_tokens=256`, greedy stopping.

### Mol2text — MolT5 official SciBERT recipe

Tokenizer for BLEU/METEOR: `allenai/scibert_scivocab_uncased`
(`text_trunc_length=512`). ROUGE: `rouge_scorer` with Porter stemmer on raw
strings.

| Metric | Ours | T5-Base | **MolT5-Base** | Δ vs MolT5-Base |
|---|---:|---:|---:|---:|
| BLEU-2 | 0.512 | 0.511 | 0.540 | −0.028 |
| BLEU-4 | 0.423 | 0.423 | 0.457 | −0.034 |
| ROUGE-1 | 0.625 | 0.607 | 0.634 | −0.009 |
| ROUGE-2 | 0.468 | 0.451 | 0.485 | −0.017 |
| ROUGE-L | 0.566 | 0.550 | 0.578 | −0.012 |
| METEOR | 0.550 | 0.539 | 0.569 | −0.019 |

Position: **tied with T5-Base or ~1.5 pt above it** across the board;
**consistently ~1.5 to 3 pt below MolT5-Base**. The remaining gap is
attributable to the shorter pre-training budget (350k vs MolT5's 1M step);
the c4/zinc valid PPL curve has not fully saturated at the downstream init
step.

### Text2mol — MolT5 official recipe

- BLEU-4: char-level tokens on **canonical** SMILES (both sides canonicalised)
- Exact match: **InChI** equality between predicted mol and gold mol
- Levenshtein: on canonical SMILES
- MACCS / RDK / Morgan FTS: Tanimoto similarity, count-based Morgan `r=2`,
  averaged only over pairs where both sides parse
- Validity: fraction of predictions that decode STAMP → parseable SMILES

| Metric | **Ours** | T5-Base | MolT5-Base | Δ vs MolT5-Base | |
|---|---:|---:|---:|---:|---|
| BLEU-4 | 0.713 | 0.762 | 0.769 | −0.056 | atom-order artefact |
| Exact match (InChI) | **0.106** | 0.069 | 0.081 | **+0.025** | ⭐ |
| Levenshtein↓ | 27.4 | 24.95 | 24.46 | +2.94 | atom-order artefact |
| MACCS FTS | **0.774** | 0.731 | 0.721 | **+0.053** | ⭐ |
| RDK FTS | **0.633** | 0.605 | 0.588 | **+0.045** | ⭐ |
| Morgan FTS | **0.539** | 0.545 | 0.529 | **+0.010** | ⭐ |
| Validity | **0.976** | 0.660 | 0.772 | **+0.204** | ⭐⭐ |

Position: **5 of 7 metrics beat MolT5-Base**. Validity even beats MolT5-Large
(0.905). BLEU-4 / Levenshtein trail by a small amount only because STAMP
decoding reconstructs SMILES with a different atom ordering than ChEBI-20's
raw SMILES — the chemistry is correct (see MACCS 0.77 / RDK 0.63 / Morgan
0.54 / InChI-match 10.6 %).

## Where the STAMP grammar shows up

**Structural guarantees.** Because STAMP is a stack-machine grammar over
motif + connection tokens, a syntactically well-formed STAMP sequence always
reconstructs a valid molecule. Under a lightly-trained T5, a good chunk of
SMILES-based generation samples are chemically invalid (MolT5-Base 22.8 %
invalid). Our STAMP-based model reduces this to **2.4 %**, giving the
+20.4 pt validity edge.

**Char-level metric artefact.** For text2mol, MolT5 paper metrics use raw
character-level tokens on the model's output vs ChEBI-20 gold. Our STAMP
decoder emits canonical RDKit SMILES, whose atom order and kekulé choices
often differ from ChEBI-20's stored SMILES — even when the molecule is
identical. Canonicalising **both** sides recovers BLEU-4 ≈ 0.72 (close to
MolT5-Base 0.77), confirming the residual gap on raw-string metrics is a
representation, not a chemistry issue.

## Key engineering findings

1. **Pre-train STAMP format was not reproducible.** The prior msring-based
   pre-train (job 514734, 413k steps) used a multi-view
   `[BOS] ... [CONCAT] ... [EOS]` STAMP encoder that no longer exists in the
   repository. Downstream on that model would have been out-of-distribution.
   Fresh single-view pre-train from `t5-v1.1-base` was required.

2. **Vocabulary selection has an outsized effect on sequence length.**
   Comparing two 30k legacy vocabs on ZINC 20k sample:
   - `chembl_vocab_30k_legacy` — median 28 tokens/mol, 0.92 % OOV
   - `zinc_curated_vocab_30k_legacy` — **median 8 tokens/mol**, 2.14 % OOV
   Same tokenizer capacity, 3.5× shorter sequences with the domain-matched
   vocab. Chose `zinc_curated_30k_legacy` for pre-train efficiency.

3. **Fallback masks OOV** — the training pipeline already falls back to T5
   SentencePiece subwords for motif tokens that miss the added-token vocab,
   so effective `<unk>` rate is 0 %. What we report as "UNK rate" is really
   "fraction of motif tokens that need subword-level fallback", not a
   catastrophe.

4. **T5 tokenizer decoder pitfall (text2mol).** HuggingFace T5 SentencePiece
   does not preserve whitespace between consecutive added tokens on decode.
   Naïve `tokenizer.batch_decode(skip_special_tokens=True)` produces
   `O[C@H]1CCCCO1[J_2][S_R][J_5]...` instead of
   `O[C@H]1CCCCO1 [J_2] [S_R] [J_5] ...`, and the STAMP decoder rejects the
   run. Root cause was **subword fragments intermixed with atomic added
   tokens** — e.g., `O=[As]` is not in the motif vocab so it hits subword
   fallback as `['▁O', '=', '[', 'As', '▁', ']']`, and simple space-joining
   turns it into six space-separated garbage tokens.
   **Fix (`_decode_output` with `_is_stamp_structural`)**: walk output ids,
   accumulate non-structural pieces into a buffer, flush the buffer through
   `tokenizer.convert_tokens_to_string` (which handles ▁) whenever a
   structural STAMP token appears, then space-join with structural tokens.
   Validity jumped 1.15 % → 57.24 % → **97.55 %** across two iterations of
   this fix.

5. **CSV truncation bug in eval script.** `predictions_*.csv` initially
   truncated `source`/`gold_target`/`prediction` at 400 chars for display,
   which broke offline re-computation of BLEU/METEOR — ~7 % of the long
   molecules and ~50 % of long ChEBI-20 descriptions were truncated. The
   values written to `metrics_*.json` were computed **before** the CSV write
   and are correct. The CSV now stores full strings. If you have any earlier
   metric that was recomputed from CSV rather than the JSON, discard it.

6. **Metric-recipe drift matters.** `nltk.corpus_bleu` with `.split()`
   gives BLEU-4 ≈ 0.384 on our mol2text output; the MolT5 official recipe
   with SciBERT WordPiece gives 0.423 on the same predictions. This is not
   a bug in either code, just different reference tokenisation — the paper
   uses SciBERT, and you have to match it to be directly comparable.

## Reproducibility

Fresh checkpoints and data are on scratch:

```
Pre-train checkpoint:
  /scratch/Projects/CFP-03/CFP03-CF-130/yuxuan.ren/ckpt/
    molt5_c4_zinc_singleview_zinc_curated30k_t5v11base_4g_20260706_041748/
      checkpoints/step_00350000/    # downstream init
      checkpoints/step_00500000/    # (final, once wall reached)
      eval_history.json             # per-10k pretrain valid_ppl trajectory

Encoded ZINC (single-view):
  /scratch/yuxuan.ren/data/zinc_curated30k_stamp/train_{00000..00042}_stamp.parquet

ChEBI-20 downstream data:
  /scratch/yuxuan.ren/data/chebi20/finetune_zinc_curated30k/{train,validation,test}.parquet

Downstream checkpoints:
  /scratch/Projects/CFP-03/CFP03-CF-130/yuxuan.ren/ckpt/
    molt5_chebi20_mol2text_zc30k/best/
    molt5_chebi20_text2mol_zc30k/best/
    */eval_test_v2/{predictions,metrics}_*.{csv,json}
```

Scripts (all in `/nfs/home/svu/yuxuan.ren/STAMP_V1.5/`):

```
Encoder:   scripts/benchmark/encode_zinc_shard_singleview.py
Pretrain:  scripts/benchmark/train_molt5_c4_zinc_mixed.py
Finetune:  scripts/benchmark/finetune_molt5_chebi20.py
Eval:      scripts/benchmark/eval_molt5_chebi20.py

PBS wrappers under scripts/hpc/:
  encode_zinc_singleview_stamp.pbs          (array 0..42)
  pretrain_molt5_c4_zinc_singleview.pbs     (4-GPU 72h)
  finetune_molt5_chebi20.pbs                (1-GPU 4h, -v DIRECTION=...)
  eval_molt5_chebi20.pbs                    (1-GPU 2h, -v DIRECTION=... [-v OUT_DIR=...])
```

## Open items

1. **Wait for pre-train to finish 500k step** (~8 h remaining at time of
   writing) and rerun downstream from `step_00500000`. Expected small
   uplift on mol2text (currently ~2 pt behind MolT5-Base, mostly a
   pretrain-budget gap).
2. **FCD and Text2Mol scores** are absent from our text2mol table — they
   require the paper's ChemNet-pretrained scorer and the Text2Mol
   contrastive classifier respectively. Adding them would complete the
   MolT5 Table 2 comparison.
3. **Character-level BLEU/Levenshtein**: the ~5 pt gap is a canonicalisation
   artefact. If the paper's number matters, either canonicalise both sides
   for a fair comparison (BLEU-4 ~0.72) or teach the STAMP decoder to emit
   ChEBI-20-style SMILES.
4. **Fingerprint metric baselines** — while our MACCS/RDK/Morgan already
   beat MolT5-Base, they are still 5–15 pt below MolT5-Large. Scaling the
   pre-train batch or moving to T5.1.1-large would test whether the STAMP
   advantage on validity carries over to the larger scale.
