#!/bin/bash
# Combined min_16 rescore with second criteria + double-filter
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/rescore_combined}"
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
print(f'[combined] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

N = ${N}; POOL = ${POOL}; M = ${M_PERMS}

# Draft pool
t0 = time.time()
with torch.no_grad():
    token_seqs = sample_model_sequences(
        model, vocab, n_samples=POOL, max_new_tokens=64,
        temperature=0.6, top_k=10, device=dev, ao_decode_mode='ar',
    )
print(f'[combined] generated {POOL} drafts in {time.time()-t0:.1f}s', flush=True)

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

# Compute per-perm NLL for M=16
BSZ = 512
all_nlls = torch.zeros(M, POOL, device=dev)
t_rescore = time.time()
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
print(f'[combined] computed {M} perms in {time.time()-t_rescore:.1f}s', flush=True)

nll_min = all_nlls.min(dim=0).values
nll_max = all_nlls.max(dim=0).values
nll_mean = all_nlls.mean(dim=0)
nll_std = all_nlls.std(dim=0)
nll_range = nll_max - nll_min                      # factorization sensitivity

def score_subset(idx_list):
    seqs = [token_seqs[i] for i in idx_list]
    smiles = []
    for toks in seqs:
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
    if not valid: return dict(validity=0.,unique=0.,quality=0.,qed=0.,sa=0.,div=0., genmol=0.)
    mols = [Chem.MolFromSmiles(c) for c in valid]
    qeds = np.array([QED.qed(m) for m in mols]); sas = np.array([sascorer.calculateScore(m) for m in mols])
    qmask = (qeds > 0.6) & (sas < 4.0)
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols]
    rng = random.Random(0); sims=[]; n=len(fps); pairs=min(5000, n*(n-1)//2)
    for _ in range(pairs):
        i, j = rng.sample(range(n), 2); sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    total = len(canon); validity = len(valid)/total
    unique_rate = len(set(valid))/len(valid); quality = float(qmask.mean())
    return dict(
        validity=validity*100, unique=unique_rate*100, quality=quality*100,
        qed=float(qeds.mean()), sa=float(sas.mean()),
        div=(1.0 - float(np.mean(sims))) if sims else 0.0,
        genmol=validity*unique_rate*quality*100,
    )

def fmt(name, s):
    return (f'[{name:<38s}] q={s["quality"]:>6.2f}%  u={s["unique"]:>6.2f}%  '
            f'GenMol={s["genmol"]:>6.2f}%  d={s["div"]:.3f}  QED={s["qed"]:.3f}  SA={s["sa"]:.2f}')

def bot(signal, k=N):
    return torch.argsort(signal, descending=True)[:k].cpu().tolist()

def top(signal, k=N):
    return torch.argsort(signal, descending=False)[:k].cpu().tolist()

results = []

print('\n---- References ----', flush=True)
for name, signal in [('bot-N min_16 (prev best)', nll_min),
                     ('bot-N max_8 (prev best Q)', all_nlls[:8].max(dim=0).values),
                     ('bot-N mean_8 (old prod)', all_nlls[:8].mean(dim=0))]:
    s = score_subset(bot(signal)); results.append(dict(name=name, **s)); print(fmt(name, s), flush=True)

print('\n---- min_16 - α*std ----', flush=True)
for alpha in (0.5, 1.0, 2.0, -0.5, -1.0):
    signal = nll_min - alpha * nll_std
    s = score_subset(bot(signal)); tag = f'min_16 - {alpha}*std'
    results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)

print('\n---- min_16 + β*max_8  (weighted blend) ----', flush=True)
max_8 = all_nlls[:8].max(dim=0).values
for beta in (0.25, 0.5, 0.75, 1.0, 1.5):
    signal = nll_min + beta * max_8
    s = score_subset(bot(signal)); tag = f'min_16 + {beta}*max_8'
    results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)

print('\n---- min_16 - α*range (max-min) ----', flush=True)
for alpha in (0.25, 0.5, 1.0):
    signal = nll_min - alpha * nll_range
    s = score_subset(bot(signal)); tag = f'min_16 - {alpha}*range'
    results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)

print('\n---- Double-filter: min_16 top-M, then max_8 top-N ----', flush=True)
for m_mult in (1.5, 2.0, 3.0):
    M_inner = int(m_mult * N)
    if M_inner > POOL: continue
    # Step 1: pick top M by min_16 (high min-NLL)
    inner_idx = bot(nll_min, k=M_inner)
    inner_set = torch.tensor(inner_idx, device=dev)
    # Step 2: rank by max_8 within these
    inner_max8 = max_8[inner_set]
    second_local = torch.argsort(inner_max8, descending=True)[:N].cpu().tolist()
    final_idx = [inner_idx[i] for i in second_local]
    s = score_subset(final_idx); tag = f'min_16-top{m_mult}N then max_8-top-N'
    results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)

print('\n---- Double-filter reverse: max_8 top-M, then min_16 top-N ----', flush=True)
for m_mult in (1.5, 2.0, 3.0):
    M_inner = int(m_mult * N)
    if M_inner > POOL: continue
    inner_idx = bot(max_8, k=M_inner)
    inner_set = torch.tensor(inner_idx, device=dev)
    inner_min16 = nll_min[inner_set]
    second_local = torch.argsort(inner_min16, descending=True)[:N].cpu().tolist()
    final_idx = [inner_idx[i] for i in second_local]
    s = score_subset(final_idx); tag = f'max_8-top{m_mult}N then min_16-top-N'
    results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)

(out_dir / 'combined_results.json').write_text(json.dumps(results, indent=2))
print(f'\n[combined] wrote {out_dir}/combined_results.json', flush=True)
PY
echo "[remote] end=$(date)"
