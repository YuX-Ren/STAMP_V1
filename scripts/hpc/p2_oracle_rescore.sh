#!/bin/bash
# Phase 2 step 1: oracle (QED/SA) + AO log-prob combined rescore.
# Generates a pool, computes BOTH oracle and AO signals per sample, tests
# many weighted combinations.
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/phase2_oracle}"
: "${N:=1024}"
: "${POOL:=4096}"
: "${M_PERMS:=16}"

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
    LMConfig, TinyDecoderLM, sample_model_sequences, _normalize_model_type, _sample_ao_orders,
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
print(f'[p2] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

N = ${N}; POOL = ${POOL}; M = ${M_PERMS}

# ---- Draft pool ----
t0 = time.time()
with torch.no_grad():
    token_seqs = sample_model_sequences(
        model, vocab, n_samples=POOL, max_new_tokens=64,
        temperature=0.6, top_k=10, device=dev, ao_decode_mode='ar',
    )
print(f'[p2] generated {POOL} drafts in {time.time()-t0:.1f}s', flush=True)

# ---- Pre-compute canon + QED + SA for ALL drafts ----
print('[p2] computing QED / SA for all drafts...', flush=True)
t0 = time.time()
qed_all = np.full(POOL, -1.0)  # -1 if failed
sa_all  = np.full(POOL, 10.0)
canon_all = [None] * POOL
mols_all = [None] * POOL
for i, toks in enumerate(token_seqs):
    try:
        smi = rep.decode(rep.detokenize(toks))
        if smi:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                canon_all[i] = Chem.MolToSmiles(m)
                mols_all[i] = m
                qed_all[i] = QED.qed(m)
                sa_all[i]  = sascorer.calculateScore(m)
    except Exception:
        pass
print(f'[p2]   done in {time.time()-t0:.1f}s', flush=True)
qed_t = torch.tensor(qed_all, dtype=torch.float32, device=dev)
sa_t  = torch.tensor(sa_all,  dtype=torch.float32, device=dev)

# ---- Pad and compute AO NLL ----
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
t0 = time.time()
for m_idx in range(M):
    with torch.no_grad():
        for s in range(0, POOL, BSZ):
            e = min(s + BSZ, POOL)
            x_c = x[s:e]; y_c = y[s:e]; lab_c = labels[s:e]
            orders = _sample_ao_orders(lab_c, random_ratio=1.0)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x_c, orders=orders).float()
            batch_idx = torch.arange(e - s, device=dev).unsqueeze(1).expand_as(orders)
            targets = y_c[batch_idx, orders]; lab_sh = lab_c[batch_idx, orders]
            log_probs = F.log_softmax(logits, dim=-1)
            nll_per = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            valid = (lab_sh != -100).float()
            all_nlls[m_idx, s:e] = (nll_per * valid).sum(-1) / valid.sum(-1).clamp(min=1.0)
nll_min = all_nlls.min(dim=0).values
nll_std = all_nlls.std(dim=0)
max_8   = all_nlls[:8].max(dim=0).values
print(f'[p2] M={M} perms done in {time.time()-t0:.1f}s', flush=True)

# ---- Scoring function (same as before) ----
def score_subset(idx_list):
    canon = [canon_all[i] for i in idx_list]
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
    return (f'[{name:<48s}] q={s["quality"]:>6.2f}%  u={s["unique"]:>6.2f}%  '
            f'GenMol={s["genmol"]:>6.2f}%  d={s["div"]:.3f}  SA={s["sa"]:.2f}')

# For rescore signals: we want HIGH score = select (top-N by signal)
def top(signal, k=N):
    return torch.argsort(signal, descending=True)[:k].cpu().tolist()

# QED higher = better; SA lower = better; NLL higher = better (drug-like mode outlier)
sa_rew = 4.0 - sa_t                                # in [0..10], higher = easier to synth
qed_norm = qed_t.clamp(min=0.0)                    # in [0..1]
sa_norm = (sa_rew / 4.0).clamp(min=0.0)            # in [0..1] mostly

results = []

# ---- Pure AO references (our best so far) ----
print('\n---- AO-only references ----', flush=True)
for name, sig in [
    ('min_16 - 0.5*std → max_8 top-1.3N (best)', None),
    ('min_16 - 0.2*std', nll_min - 0.2 * nll_std),
    ('max_8', max_8),
]:
    if sig is None:
        outer = nll_min - 0.5 * nll_std
        idx_inner = top(outer, k=int(1.3 * N))
        inner_max = max_8[torch.tensor(idx_inner, device=dev)]
        rel = torch.argsort(inner_max, descending=True)[:N].cpu().tolist()
        sel = [idx_inner[i] for i in rel]
    else:
        sel = top(sig)
    s = score_subset(sel); results.append(dict(name=name, **s)); print(fmt(name, s), flush=True)

# ---- Pure oracle filters ----
print('\n---- Pure oracle filters ----', flush=True)
for name, sig in [
    ('QED only', qed_norm),
    ('SA only (low)', sa_norm),
    ('QED + (4-SA)', qed_norm + sa_norm),
    ('QED * (SA<4)', qed_norm * (sa_t < 4.0).float()),
    ('pass quality ind (binary)', ((qed_t > 0.6) & (sa_t < 4.0)).float() + torch.rand(POOL, device=dev) * 0.01),
]:
    sel = top(sig)
    s = score_subset(sel); results.append(dict(name=name, **s)); print(fmt(name, s), flush=True)

# ---- Oracle + AO combined linear ----
print('\n---- Oracle + AO combined (higher=select) ----', flush=True)
# Use nll_min (drug-like outlier signal, higher=better) + QED + SA reward
for qw, sw, llw in [
    (1.0, 0.0, 0.0),         # QED only (duplicate)
    (0.0, 1.0, 0.0),         # SA only (duplicate)
    (1.0, 1.0, 0.0),         # pure oracle (duplicate)
    (1.0, 1.0, 0.5),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 2.0),
    (2.0, 2.0, 1.0),
    (1.0, 0.5, 1.0),
    (0.5, 1.0, 1.0),
    (1.0, 1.0, 0.3),
]:
    signal = qw * qed_norm + sw * sa_norm + llw * nll_min
    sel = top(signal)
    s = score_subset(sel); tag = f'{qw}*QED + {sw}*SA_rew + {llw}*min_NLL'
    results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)

# ---- 2-stage: AO rescore then oracle ----
print('\n---- 2-stage: AO pre-filter → oracle ----', flush=True)
for m_mult in (1.5, 2.0, 3.0):
    M_inner = min(int(m_mult * N), POOL)
    outer = nll_min - 0.5 * nll_std
    idx_inner = top(outer, k=M_inner)
    inner_set = torch.tensor(idx_inner, device=dev)
    # Stage 2: oracle score within
    or_sig = qed_norm[inner_set] + sa_norm[inner_set]
    rel = torch.argsort(or_sig, descending=True)[:N].cpu().tolist()
    sel = [idx_inner[i] for i in rel]
    s = score_subset(sel); tag = f'(min-0.5std) top{m_mult}N → oracle top-N'
    results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)

# ---- 2-stage reverse: oracle → AO ----
print('\n---- 2-stage reverse: oracle pre-filter → AO ----', flush=True)
for m_mult in (1.5, 2.0, 3.0):
    M_inner = min(int(m_mult * N), POOL)
    or_sig = qed_norm + sa_norm
    idx_inner = top(or_sig, k=M_inner)
    inner_set = torch.tensor(idx_inner, device=dev)
    ao_sig = (nll_min - 0.5 * nll_std)[inner_set]
    rel = torch.argsort(ao_sig, descending=True)[:N].cpu().tolist()
    sel = [idx_inner[i] for i in rel]
    s = score_subset(sel); tag = f'oracle top{m_mult}N → (min-0.5std) top-N'
    results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)

(out_dir / 'p2_results.json').write_text(json.dumps(results, indent=2))
print('\n---- TOP-5 by GenMol ----', flush=True)
for r in sorted(results, key=lambda r: -r['genmol'])[:5]:
    print(fmt(r['name'], r), flush=True)
print('\n---- TOP-5 by Quality ----', flush=True)
for r in sorted(results, key=lambda r: -r['quality'])[:5]:
    print(fmt(r['name'], r), flush=True)
print(f'\n[p2] wrote {out_dir}/p2_results.json', flush=True)
PY
echo "[remote] end=$(date)"
