#!/bin/bash
# Phase 2 P2.α: Partial-QED guided AR decoding. STAMP-exclusive.
# At each AR step, evaluate top-K candidates via partial-molecule decode → QED/SA,
# select by combined (model_log_prob + α * partial_QED - β * partial_SA).
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/phase2_partial}"
: "${N:=1024}"
: "${TOPK_CAND:=5}"

echo "[remote] host=$(hostname) date=$(date)"
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

python - <<PY
import sys, json, time, random
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import RDLogger, Chem, DataStructs
from rdkit.Chem import AllChem, QED, RDConfig
sys.path.append(str(Path(RDConfig.RDContribDir) / 'SA_Score'))
import sascorer
RDLogger.DisableLog('rdApp.*')

from stamp.benchmark.lm import (
    LMConfig, TinyDecoderLM, _normalize_model_type, _trim_special_tail,
    sample_model_sequences,
)
from stamp.benchmark.representations import build_representation
from scripts.benchmark.train_ao_gpt_parquet import _load_scan_cache

ckpt_dir = Path('${CKPT_DIR}'); out_dir = Path('${OUT_DIR}')
_, vocab = _load_scan_cache(ckpt_dir)
ckpt = torch.load(ckpt_dir / 'model.pt', map_location='cpu', weights_only=False)
cfg = LMConfig(**ckpt['cfg'])
cfg.model_type = _normalize_model_type(str(ckpt.get('model_type', cfg.model_type)))
dev = torch.device('cuda:0')
model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg).to(dev)
state = ckpt['model_state']
if any(k.startswith('_orig_mod.') for k in state):
    state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
model.load_state_dict(state); model.eval()
rep = build_representation('stamp')
stamp_codec = rep.codec  # direct access
print(f'[p2.α] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

N = ${N}; K_CAND = ${TOPK_CAND}

# ---- Helper: partial-decode a token-id list to molecule ----
def partial_mol_and_score(ids_list):
    """Given token ids (excluding BOS), append EOS and decode via STAMP codec.
    Returns (mol, qed, sa) or (None, 0.0, 10.0) on fail. STAMP R4 fallback ensures
    any input decodes to SOME molecule."""
    try:
        tokens = vocab.decode_ids(ids_list)  # may include specials
        # Remove any EOS/PAD/BOS in middle (safety)
        tokens_clean = [t for t in tokens if t not in {vocab.PAD, vocab.BOS, vocab.EOS, vocab.UNK, vocab.MASK}]
        stamp_text = rep.detokenize(tokens_clean)
        if not stamp_text.strip():
            return None, 0.0, 10.0
        mol = stamp_codec.decode_stamp_to_mol(stamp_text)
        if mol is None:
            return None, 0.0, 10.0
        # take largest fragment in case of disconnect
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags:
            return None, 0.0, 10.0
        main = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        qed = QED.qed(main)
        sa  = sascorer.calculateScore(main)
        return main, float(qed), float(sa)
    except Exception:
        return None, 0.0, 10.0


def partial_qed_guided_ar(
    model, vocab, rep, n_samples, max_new_tokens=64,
    base_temp=0.6, base_topk=10, cand_K=5,
    qed_weight=3.0, sa_weight=0.5, dev=dev,
    guidance_start_step=2,
):
    """AR decoding with per-step partial-QED guidance.

    At each step:
      1. Forward → logits → take top-`base_topk` tokens as candidates
      2. Keep top-`cand_K` of those for partial eval (cand_K <= base_topk)
      3. For each candidate: build partial ids (current + cand), STAMP-decode to mol,
         compute QED/SA.
      4. Score = log_prob(cand)/T + qed_weight*QED - sa_weight*SA
      5. Pick argmax (or sample softmax of scores with Gumbel for diversity).
    """
    # Start from [BOS]; generate per-sample variable length until EOS
    ids_list = [[vocab.bos_id] for _ in range(n_samples)]
    finished = [False] * n_samples
    n_active = n_samples

    for step in range(max_new_tokens):
        if all(finished): break
        active_idx = [i for i in range(n_samples) if not finished[i]]
        # Pad active contexts
        max_ctx = max(len(ids_list[i]) for i in active_idx)
        ctx = torch.full((len(active_idx), max_ctx), vocab.pad_id, dtype=torch.long, device=dev)
        for j, i in enumerate(active_idx):
            ctx[j, :len(ids_list[i])] = torch.tensor(ids_list[i], dtype=torch.long, device=dev)
        # Truncate if > max_seq_len
        if max_ctx > cfg.max_seq_len:
            ctx = ctx[:, -cfg.max_seq_len:]

        with torch.no_grad():
            # AO-GPT expects identity order for AR. Use default.
            if cfg.model_type in ('ao_gpt', 'ao_gpt_pp'):
                ao_orders = torch.arange(ctx.size(1), device=dev).unsqueeze(0).expand(ctx.size(0), -1)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(ctx, orders=ao_orders).float()[:, -1, :]
            else:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(ctx).float()[:, -1, :]
        # Suppress specials for middle sampling (keep EOS as a choice)
        for sup in (vocab.pad_id, vocab.bos_id, vocab.mask_id, vocab.unk_id):
            if 0 <= int(sup) < logits.size(-1):
                logits[:, int(sup)] = float('-inf')

        # top-K candidates by log-prob (for evaluation)
        scaled = logits / base_temp
        topk_lp, topk_ids = torch.topk(scaled, k=min(base_topk, scaled.size(-1)), dim=-1)
        # Keep only top cand_K for partial eval (cheapest, still uses all in fallback)
        cand_lp = topk_lp[:, :cand_K]      # (B, cand_K)
        cand_id = topk_ids[:, :cand_K]     # (B, cand_K)

        # Apply guidance only after step >= guidance_start_step
        use_guidance = step >= guidance_start_step
        for b, i in enumerate(active_idx):
            if not use_guidance:
                # Standard sampling from top-k at base_temp
                probs = torch.softmax(scaled[b] - scaled[b].max(), dim=-1)
                # top-k mask
                top_vals, top_inds = torch.topk(probs, k=min(base_topk, probs.size(-1)))
                top_vals = top_vals / top_vals.sum()
                pick = top_inds[torch.multinomial(top_vals, num_samples=1).item()]
                tok = int(pick.item())
            else:
                # Evaluate each candidate's partial-QED, SAMPLE from combined score
                # distribution (argmax would mode-collapse across samples).
                scores = torch.empty(cand_K, device=dev)
                for k in range(cand_K):
                    cand_tok = int(cand_id[b, k].item())
                    cand_log_p = float(cand_lp[b, k].item())
                    probe_ids = ids_list[i][1:] + [cand_tok]
                    _, qed_v, sa_v = partial_mol_and_score(probe_ids)
                    scores[k] = cand_log_p + qed_weight * qed_v - sa_weight * sa_v
                # Normalize & sample (softmax over combined scores)
                scores = scores - scores.max()
                probs = torch.softmax(scores, dim=0)
                pick_idx = int(torch.multinomial(probs, num_samples=1).item())
                tok = int(cand_id[b, pick_idx].item())

            ids_list[i].append(tok)
            if tok == vocab.eos_id:
                finished[i] = True
    # Done: convert to token strings
    out = []
    for i in range(n_samples):
        row = ids_list[i][1:]  # strip BOS
        row = _trim_special_tail(row, vocab)
        out.append(vocab.decode_ids(row))
    return out


def score_subset(token_seqs):
    smiles = []
    for toks in token_seqs:
        try: smiles.append(rep.decode(rep.detokenize(toks)))
        except Exception: smiles.append(None)
    canon = []
    for smi in smiles:
        if not smi: canon.append(None); continue
        try:
            m = Chem.MolFromSmiles(smi)
            canon.append(Chem.MolToSmiles(m) if m is not None else None)
        except Exception:
            canon.append(None)
    valid = [c for c in canon if c]
    if not valid: return dict(validity=0.,unique=0.,quality=0.,qed=0.,sa=0.,div=0.,genmol=0.)
    mols = [Chem.MolFromSmiles(c) for c in valid]
    qeds = np.array([QED.qed(m) for m in mols])
    sas  = np.array([sascorer.calculateScore(m) for m in mols])
    qmask = (qeds > 0.6) & (sas < 4.0)
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols]
    rng = random.Random(0); sims=[]; n=len(fps); pairs=min(5000, n*(n-1)//2)
    for _ in range(pairs):
        i, j = rng.sample(range(n), 2); sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    total = len(canon); validity = len(valid)/total
    unique_rate = len(set(valid))/len(valid); quality = float(qmask.mean())
    return dict(validity=validity*100, unique=unique_rate*100, quality=quality*100,
                qed=float(qeds.mean()), sa=float(sas.mean()),
                div=(1.0 - float(np.mean(sims))) if sims else 0.0,
                genmol=validity*unique_rate*quality*100)

def fmt(name, s):
    return (f'[{name:<52s}] q={s["quality"]:>6.2f}%  u={s["unique"]:>6.2f}%  '
            f'GenMol={s["genmol"]:>6.2f}%  d={s["div"]:.3f}  QED={s["qed"]:.3f}  SA={s["sa"]:.2f}')

results = []

# Baseline: AR (no guidance)
print('\n==== Baseline: plain AR ====', flush=True)
t0 = time.time()
with torch.no_grad():
    seqs = sample_model_sequences(
        model, vocab, n_samples=N, max_new_tokens=64,
        temperature=0.6, top_k=10, device=dev, ao_decode_mode='ar',
    )
print(f'  sampled in {time.time()-t0:.1f}s', flush=True)
s = score_subset(seqs); results.append(dict(name='AR baseline (T=0.6, K=10)', **s))
print(fmt('AR baseline (T=0.6, K=10)', s), flush=True)

# Partial-QED guided AR, various (qed_w, sa_w) combos
print('\n==== Partial-QED guided AR ====', flush=True)
for qw, sw, tag in [
    (3.0, 0.5, 'guide qw=3.0 sw=0.5'),
    (5.0, 0.5, 'guide qw=5.0 sw=0.5'),
    (5.0, 1.0, 'guide qw=5.0 sw=1.0'),
    (10.0, 1.0, 'guide qw=10 sw=1.0'),
    (10.0, 2.0, 'guide qw=10 sw=2.0'),
]:
    t0 = time.time()
    seqs = partial_qed_guided_ar(
        model, vocab, rep, n_samples=N, max_new_tokens=64,
        base_temp=0.6, base_topk=10, cand_K=5,
        qed_weight=qw, sa_weight=sw, dev=dev,
        guidance_start_step=2,
    )
    dt = time.time() - t0
    s = score_subset(seqs); results.append(dict(name=tag + f' ({dt:.0f}s)', **s))
    print(fmt(tag, s), flush=True); print(f'    wall: {dt:.1f}s', flush=True)

(out_dir / 'p2_partial_results.json').write_text(json.dumps(results, indent=2))
print('\n---- TOP-3 by GenMol ----', flush=True)
for r in sorted(results, key=lambda r: -r['genmol'])[:3]:
    print(fmt(r['name'], r), flush=True)
print(f'\n[p2.α] wrote {out_dir}/p2_partial_results.json', flush=True)
PY
echo "[remote] end=$(date)"
