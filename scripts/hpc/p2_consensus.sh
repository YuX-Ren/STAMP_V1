#!/bin/bash
# P2 #5: Consensus decoding. Generate M drafts per "slot", select medoid
# (highest avg fingerprint similarity to other group members).
# AO-GPT concept: M drafts are different factorization samples of same mode.
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/phase2_consensus}"
: "${N:=1024}"
: "${M_GROUP:=4}"        # drafts per consensus group

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
from rdkit import RDLogger, Chem, DataStructs
from rdkit.Chem import AllChem, QED, RDConfig
sys.path.append(str(Path(RDConfig.RDContribDir) / 'SA_Score'))
import sascorer
RDLogger.DisableLog('rdApp.*')

from stamp.benchmark.lm import (
    LMConfig, TinyDecoderLM, sample_model_sequences, _normalize_model_type,
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
print(f'[consensus] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

N = ${N}; M = ${M_GROUP}
POOL = N * M

# ---- Draft M*N drafts via AR (same T/K; different random seeds from multinomial) ----
t0 = time.time()
with torch.no_grad():
    token_seqs = sample_model_sequences(
        model, vocab, n_samples=POOL, max_new_tokens=64,
        temperature=0.6, top_k=10, device=dev, ao_decode_mode='ar',
    )
print(f'[consensus] generated {POOL} drafts ({M}×{N}) in {time.time()-t0:.1f}s', flush=True)

# ---- Decode to canonical + compute fingerprints + QED/SA ----
t0 = time.time()
canon = [None] * POOL; fps = [None] * POOL
qed_all = np.full(POOL, -1.0); sa_all = np.full(POOL, 10.0)
for i, toks in enumerate(token_seqs):
    try:
        smi = rep.decode(rep.detokenize(toks))
        if smi:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                canon[i] = Chem.MolToSmiles(m)
                fps[i] = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024)
                qed_all[i] = QED.qed(m)
                sa_all[i]  = sascorer.calculateScore(m)
    except Exception:
        pass
print(f'[consensus] canonicalize + fp + QED/SA in {time.time()-t0:.1f}s', flush=True)

# ---- Assign drafts to groups; each slot i has M consecutive drafts ----
# For each slot, compute medoid (highest avg Tanimoto to peers)
t0 = time.time()
medoid_idx = []
for slot in range(N):
    group_idx = list(range(slot * M, slot * M + M))
    group_fps = [fps[g] for g in group_idx]
    # If any fps missing, fall back to first valid
    if any(f is None for f in group_fps):
        valid_in_grp = [g for g in group_idx if fps[g] is not None]
        if valid_in_grp:
            medoid_idx.append(valid_in_grp[0])
        else:
            medoid_idx.append(group_idx[0])
        continue
    # Compute avg Tanimoto to other M-1 members
    best_g = group_idx[0]; best_sim = -1.0
    for i_idx, g in enumerate(group_idx):
        peers = [group_fps[j] for j in range(M) if j != i_idx]
        sims = [DataStructs.TanimotoSimilarity(group_fps[i_idx], p) for p in peers]
        avg = sum(sims) / len(sims) if sims else 0.0
        if avg > best_sim:
            best_sim = avg; best_g = g
    medoid_idx.append(best_g)
print(f'[consensus] medoid selection in {time.time()-t0:.1f}s', flush=True)

# For comparison: plain "first of each group" (= random selection)
first_idx = [slot * M for slot in range(N)]

# Anti-consensus (pick most-different within group)
t0 = time.time()
anti_idx = []
for slot in range(N):
    group_idx = list(range(slot * M, slot * M + M))
    group_fps = [fps[g] for g in group_idx]
    if any(f is None for f in group_fps):
        valid_in_grp = [g for g in group_idx if fps[g] is not None]
        anti_idx.append(valid_in_grp[0] if valid_in_grp else group_idx[0]); continue
    worst_g = group_idx[0]; worst_sim = 2.0
    for i_idx, g in enumerate(group_idx):
        peers = [group_fps[j] for j in range(M) if j != i_idx]
        sims = [DataStructs.TanimotoSimilarity(group_fps[i_idx], p) for p in peers]
        avg = sum(sims) / len(sims) if sims else 0.0
        if avg < worst_sim:
            worst_sim = avg; worst_g = g
    anti_idx.append(worst_g)
print(f'[consensus] anti-consensus selection in {time.time()-t0:.1f}s', flush=True)

def score_subset(idx_list):
    sub = [token_seqs[i] for i in idx_list]
    smiles_list = [canon[i] if canon[i] else None for i in idx_list]
    valid = [c for c in smiles_list if c]
    if not valid: return dict(validity=0.,unique=0.,quality=0.,qed=0.,sa=0.,div=0.,genmol=0.)
    mols = [Chem.MolFromSmiles(c) for c in valid]
    qeds = np.array([QED.qed(m) for m in mols]); sas = np.array([sascorer.calculateScore(m) for m in mols])
    qmask = (qeds > 0.6) & (sas < 4.0)
    sub_fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols]
    rng = random.Random(0); sims=[]; n=len(sub_fps); pairs=min(5000, n*(n-1)//2)
    for _ in range(pairs):
        i, j = rng.sample(range(n), 2); sims.append(DataStructs.TanimotoSimilarity(sub_fps[i], sub_fps[j]))
    total = len(smiles_list); validity = len(valid)/total
    unique_rate = len(set(valid))/len(valid); quality = float(qmask.mean())
    return dict(validity=validity*100, unique=unique_rate*100, quality=quality*100,
                qed=float(qeds.mean()), sa=float(sas.mean()),
                div=(1.0 - float(np.mean(sims))) if sims else 0.0,
                genmol=validity*unique_rate*quality*100)

def fmt(name, s):
    return (f'[{name:<40s}] q={s["quality"]:>6.2f}%  u={s["unique"]:>6.2f}%  '
            f'GenMol={s["genmol"]:>6.2f}%  d={s["div"]:.3f}  QED={s["qed"]:.3f}  SA={s["sa"]:.2f}')

results = []
# Plain baseline: random / first in each group
s = score_subset(first_idx); results.append(dict(name='random (first-of-group)', **s)); print(fmt('random (first-of-group)', s))
# Medoid
s = score_subset(medoid_idx); results.append(dict(name='medoid (consensus)', **s)); print(fmt('medoid (consensus)', s))
# Anti-consensus
s = score_subset(anti_idx); results.append(dict(name='anti-medoid (outlier)', **s)); print(fmt('anti-medoid (outlier)', s))

# Try with M=2 (smaller groups)
print('\n---- With M=2 groups (cheaper consensus) ----', flush=True)
for M_alt in (2, 3):
    slots_alt = POOL // M_alt
    # Only use enough for N slots
    if slots_alt < N: continue
    med_alt = []
    for slot in range(N):
        group_idx = list(range(slot * M_alt, slot * M_alt + M_alt))
        group_fps = [fps[g] for g in group_idx]
        if any(f is None for f in group_fps):
            valid_in_grp = [g for g in group_idx if fps[g] is not None]
            med_alt.append(valid_in_grp[0] if valid_in_grp else group_idx[0]); continue
        best_g = group_idx[0]; best_sim = -1.0
        for i_idx, g in enumerate(group_idx):
            peers = [group_fps[j] for j in range(M_alt) if j != i_idx]
            sims = [DataStructs.TanimotoSimilarity(group_fps[i_idx], p) for p in peers]
            avg = sum(sims) / len(sims) if sims else 0.0
            if avg > best_sim:
                best_sim = avg; best_g = g
        med_alt.append(best_g)
    s = score_subset(med_alt)
    results.append(dict(name=f'medoid M={M_alt}', **s)); print(fmt(f'medoid M={M_alt}', s))

(out_dir / 'consensus_results.json').write_text(json.dumps(results, indent=2))
print('\n---- TOP by GenMol ----', flush=True)
for r in sorted(results, key=lambda r: -r['genmol']):
    print(fmt(r['name'], r), flush=True)
print(f'\n[consensus] wrote {out_dir}/consensus_results.json', flush=True)
PY
echo "[remote] end=$(date)"
