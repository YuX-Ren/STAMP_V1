#!/bin/bash
# Quick QED/SA threshold breakdown on ep9 best non-guided config.
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
CKPT_DIR=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009

echo "[remote] host=$(hostname) date=$(date)"
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

python - <<PY
import sys, json, time, random
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import RDLogger, Chem
from rdkit.Chem import QED, RDConfig
sys.path.append(str(Path(RDConfig.RDContribDir) / 'SA_Score'))
import sascorer
RDLogger.DisableLog('rdApp.*')

from stamp.benchmark.lm import (
    LMConfig, TinyDecoderLM, sample_model_sequences, _normalize_model_type, _sample_ao_orders,
)
from stamp.benchmark.representations import build_representation
from scripts.benchmark.train_ao_gpt_parquet import _load_scan_cache

ckpt_dir = Path('${CKPT_DIR}')
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

N = 1024; POOL = 4096; M = 16
with torch.no_grad():
    token_seqs = sample_model_sequences(
        model, vocab, n_samples=POOL, max_new_tokens=64,
        temperature=0.6, top_k=10, device=dev, ao_decode_mode='ar',
    )

# canonicalize + QED/SA per sample
qed_all = np.full(POOL, -1.0); sa_all = np.full(POOL, 10.0); canon_all = [None]*POOL
for i, toks in enumerate(token_seqs):
    try:
        smi = rep.decode(rep.detokenize(toks))
        if smi:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                canon_all[i] = Chem.MolToSmiles(m)
                qed_all[i] = QED.qed(m); sa_all[i] = sascorer.calculateScore(m)
    except Exception:
        pass

# AO NLL
bos, eos, pad = vocab.bos_id, vocab.eos_id, vocab.pad_id
id_seqs = [[bos] + vocab.encode_tokens(toks) + [eos] for toks in token_seqs]
max_len = min(max(len(s) for s in id_seqs), cfg.max_seq_len)
lengths = torch.tensor([min(len(s), max_len) for s in id_seqs], dtype=torch.long)
padded = torch.full((POOL, max_len), pad, dtype=torch.long)
for i, s in enumerate(id_seqs):
    L = min(len(s), max_len); padded[i, :L] = torch.tensor(s[:L], dtype=torch.long)
padded = padded.to(dev)
x = padded[:, :-1].contiguous(); y = padded[:, 1:].contiguous()
seq_mask_y = torch.arange(max_len - 1, device=dev).unsqueeze(0) < (lengths.to(dev).unsqueeze(1) - 1)
labels = torch.where(seq_mask_y, y, torch.full_like(y, -100))
BSZ = 512
all_nlls = torch.zeros(M, POOL, device=dev)
for m_idx in range(M):
    with torch.no_grad():
        for s in range(0, POOL, BSZ):
            e = min(s + BSZ, POOL)
            orders = _sample_ao_orders(labels[s:e], random_ratio=1.0)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x[s:e], orders=orders).float()
            bi = torch.arange(e - s, device=dev).unsqueeze(1).expand_as(orders)
            targets = y[s:e][bi, orders]; lab_sh = labels[s:e][bi, orders]
            log_probs = F.log_softmax(logits, dim=-1)
            nll_per = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            valid = (lab_sh != -100).float()
            all_nlls[m_idx, s:e] = (nll_per * valid).sum(-1) / valid.sum(-1).clamp(min=1.0)
nll_min = all_nlls.min(dim=0).values
nll_std = all_nlls.std(dim=0)
max_8 = all_nlls[:8].max(dim=0).values

# Best non-guided config: (min_16 - 0.5*std) top-1.3N → max_8 top-N
outer = nll_min - 0.5 * nll_std
idx_in = torch.argsort(outer, descending=True)[:int(1.3 * N)].cpu().tolist()
inner_set = torch.tensor(idx_in, device=dev)
rel = torch.argsort(max_8[inner_set], descending=True)[:N].cpu().tolist()
sel = [idx_in[i] for i in rel]

# Extract QED/SA for selected
sel_qed = qed_all[sel]
sel_sa  = sa_all[sel]
valid_mask = sel_qed >= 0  # -1 = invalid decode
sq = sel_qed[valid_mask]; ss = sel_sa[valid_mask]
N_eval = len(sq)

print(f'Config: AR + (min_16 - 0.5*std) top-1.3N → max_8')
print(f'N_selected = {len(sel)}  N_valid_eval = {N_eval}')
print()
print(f'QED: mean={sq.mean():.3f}  std={sq.std():.3f}  median={np.median(sq):.3f}')
print(f'SA : mean={ss.mean():.3f}   std={ss.std():.3f}   median={np.median(ss):.3f}')
print()
pass_qed  = (sq > 0.6).mean()
pass_sa   = (ss < 4.0).mean()
pass_both = ((sq > 0.6) & (ss < 4.0)).mean()
fail_qed_only = ((sq <= 0.6) & (ss < 4.0)).mean()
fail_sa_only  = ((sq > 0.6) & (ss >= 4.0)).mean()
fail_both     = ((sq <= 0.6) & (ss >= 4.0)).mean()
print('Marginal pass rates:')
print(f'  P(QED > 0.6) = {pass_qed*100:.2f}%')
print(f'  P(SA  < 4.0) = {pass_sa*100:.2f}%')
print()
print('Joint breakdown:')
print(f'  Both pass (Quality)   = {pass_both*100:.2f}%')
print(f'  Fail QED only         = {fail_qed_only*100:.2f}%')
print(f'  Fail SA only          = {fail_sa_only*100:.2f}%')
print(f'  Fail both             = {fail_both*100:.2f}%')
print()
print('Lifting thresholds:')
print(f'  Remove QED: +{(pass_sa - pass_both)*100:.2f}pp  →  {pass_sa*100:.2f}%')
print(f'  Remove SA : +{(pass_qed - pass_both)*100:.2f}pp  →  {pass_qed*100:.2f}%')
PY
echo "[remote] end=$(date)"
