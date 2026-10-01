#!/bin/bash
# Phase 2 push: larger pool + explicit dedup + diversity-aware filtering
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/phase2_push}"
: "${N:=1024}"
: "${POOL:=16384}"             # 16x oversample
: "${M_PERMS:=8}"              # cheaper on bigger pool
: "${SAMPLE_CHUNK:=2048}"

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
print(f'[p2-push] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

N = ${N}; POOL = ${POOL}; M = ${M_PERMS}; CHUNK = ${SAMPLE_CHUNK}

# ---- Draft pool (chunked to fit memory) ----
t0 = time.time()
token_seqs = []
for cs in range(0, POOL, CHUNK):
    ce = min(cs + CHUNK, POOL)
    with torch.no_grad():
        seqs = sample_model_sequences(
            model, vocab, n_samples=ce-cs, max_new_tokens=64,
            temperature=0.6, top_k=10, device=dev, ao_decode_mode='ar',
        )
    token_seqs.extend(seqs)
    torch.cuda.empty_cache()
print(f'[p2-push] generated {POOL} drafts in {time.time()-t0:.1f}s', flush=True)

# QED + SA + canonicalize
t0 = time.time()
qed_all = np.full(POOL, -1.0)
sa_all  = np.full(POOL, 10.0)
canon_all = [None] * POOL
for i, toks in enumerate(token_seqs):
    try:
        smi = rep.decode(rep.detokenize(toks))
        if smi:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                canon_all[i] = Chem.MolToSmiles(m)
                qed_all[i] = QED.qed(m)
                sa_all[i]  = sascorer.calculateScore(m)
    except Exception:
        pass
print(f'[p2-push] QED/SA in {time.time()-t0:.1f}s', flush=True)

# Count pass
qed_arr = qed_all; sa_arr = sa_all
pass_mask = (qed_arr > 0.6) & (sa_arr < 4.0) & np.array([c is not None for c in canon_all])
print(f'[p2-push] pool stats: quality={pass_mask.sum()}/{POOL} = {pass_mask.mean()*100:.2f}%', flush=True)

# ---- AO NLL for select configs ----
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
print(f'[p2-push] {M} perms in {time.time()-t0:.1f}s', flush=True)

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
    return (f'[{name:<52s}] q={s["quality"]:>6.2f}%  u={s["unique"]:>6.2f}%  '
            f'GenMol={s["genmol"]:>6.2f}%  d={s["div"]:.3f}  SA={s["sa"]:.2f}')

results = []

# ---- A. Binary filter + first N ----
pass_idx = np.where(pass_mask)[0]
print(f'\n[p2-push] pass count: {len(pass_idx)}', flush=True)

if len(pass_idx) >= N:
    # Simple: first N passing
    sel = pass_idx[:N].tolist()
    s = score_subset(sel); tag = f'binary pass, first {N}'
    results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)

# ---- B. Binary filter + DEDUP + first N unique ----
if len(pass_idx) >= N:
    seen = set(); dedup_idx = []
    for i in pass_idx:
        c = canon_all[i]
        if c and c not in seen:
            seen.add(c); dedup_idx.append(int(i))
            if len(dedup_idx) >= N: break
    if len(dedup_idx) >= N:
        s = score_subset(dedup_idx); tag = f'binary pass + dedup, first {N} unique'
        results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)
    else:
        print(f'[p2-push] NOT ENOUGH unique passing: {len(dedup_idx)} < {N}', flush=True)

# ---- C. Binary filter + DEDUP + top-N by QED ----
if len(pass_idx) >= N:
    seen = set(); uniq_pass = []
    for i in pass_idx:
        c = canon_all[i]
        if c and c not in seen:
            seen.add(c); uniq_pass.append(int(i))
    if len(uniq_pass) >= N:
        # Sort by QED descending
        uniq_pass.sort(key=lambda i: -qed_all[i])
        sel = uniq_pass[:N]
        s = score_subset(sel); tag = f'binary pass + dedup + top-{N} by QED'
        results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)
        # By oracle combined
        uniq_pass2 = sorted(uniq_pass, key=lambda i: -(qed_all[i] + (4 - sa_all[i])/4))
        sel2 = uniq_pass2[:N]
        s = score_subset(sel2); tag = f'binary pass + dedup + top-{N} by (QED+SA_rew)'
        results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)
        # By AO min_16 (drug-like outlier within passing+unique)
        uniq_pass3 = sorted(uniq_pass, key=lambda i: -nll_min[i].item())
        sel3 = uniq_pass3[:N]
        s = score_subset(sel3); tag = f'binary pass + dedup + top-{N} by min_NLL'
        results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)
        # Combined score
        combined = lambda i: -(qed_all[i] + (4 - sa_all[i])/4) - 1.0 * nll_min[i].item()
        uniq_pass4 = sorted(uniq_pass, key=combined)
        sel4 = uniq_pass4[:N]
        s = score_subset(sel4); tag = f'binary+dedup + top-{N} by (QED+SA+NLL)'
        results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)

# ---- D. Diversity-aware: greedy farthest-point on fingerprints among passing+unique ----
if len(pass_idx) >= N:
    seen = set(); uniq_pass = []
    for i in pass_idx:
        c = canon_all[i]
        if c and c not in seen:
            seen.add(c); uniq_pass.append(int(i))
    if len(uniq_pass) >= N:
        print(f'\n[p2-push] computing fingerprints for {len(uniq_pass)} passing+unique...', flush=True)
        mols_subset = [Chem.MolFromSmiles(canon_all[i]) for i in uniq_pass]
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols_subset]
        # Greedy farthest-first traversal
        # Start with highest QED, then iteratively add the one maximizing min-distance to selected
        start = max(range(len(uniq_pass)), key=lambda i: qed_all[uniq_pass[i]])
        selected_local = [start]
        min_sims = [DataStructs.TanimotoSimilarity(fps[start], fps[i]) for i in range(len(uniq_pass))]
        for _ in range(N - 1):
            # Pick the one with smallest min-sim (maximally distant)
            nxt = min([i for i in range(len(uniq_pass)) if i not in set(selected_local)],
                      key=lambda i: min_sims[i])
            selected_local.append(nxt)
            # Update min_sims
            nxt_sims = [DataStructs.TanimotoSimilarity(fps[nxt], fps[i]) for i in range(len(uniq_pass))]
            for i in range(len(uniq_pass)):
                if nxt_sims[i] > min_sims[i]: pass
                min_sims[i] = max(min_sims[i], nxt_sims[i])
        sel = [uniq_pass[i] for i in selected_local]
        s = score_subset(sel); tag = f'binary+dedup + greedy farthest-point (N={N})'
        results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)

(out_dir / 'p2_push_results.json').write_text(json.dumps(results, indent=2))
print('\n---- TOP-3 by GenMol ----', flush=True)
for r in sorted(results, key=lambda r: -r['genmol'])[:3]:
    print(fmt(r['name'], r), flush=True)
print(f'\n[p2-push] wrote {out_dir}/p2_push_results.json', flush=True)
PY
echo "[remote] end=$(date)"
