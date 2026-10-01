#!/bin/bash
# #2 + #7: max-over-orders rescore + larger M
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/rescore_maxperm}"
: "${N:=1024}"
: "${POOL:=4096}"        # OS=4x (known knee)
: "${M_PERMS:=16}"       # double the default
: "${TEMP:=0.6}"
: "${TOPK:=10}"

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
print(f'[maxperm] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

N = ${N}; POOL = ${POOL}; M = ${M_PERMS}
TEMP = ${TEMP}; TOPK = ${TOPK}

# ---- Draft pool ----
t0 = time.time()
with torch.no_grad():
    token_seqs = sample_model_sequences(
        model, vocab, n_samples=POOL, max_new_tokens=64,
        temperature=TEMP, top_k=TOPK, device=dev, ao_decode_mode='ar',
    )
print(f'[maxperm] generated {POOL} drafts in {time.time()-t0:.1f}s', flush=True)

# Pad
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

# ---- Compute per-perm NLL for M=16 perms ----
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
print(f'[maxperm] computed NLL for {M} perms x {POOL} samples in {time.time()-t_rescore:.1f}s', flush=True)

# Aggregations (all shape (POOL,))
# "NLL" here is per-token; larger = less likely = drug-like outlier (our observation)
nll_mean_16 = all_nlls.mean(dim=0)
nll_mean_8  = all_nlls[:8].mean(dim=0)
nll_max_16  = all_nlls.max(dim=0).values
nll_max_8   = all_nlls[:8].max(dim=0).values
nll_min_16  = all_nlls.min(dim=0).values
nll_median_16 = all_nlls.median(dim=0).values
nll_p90_16   = all_nlls.quantile(0.90, dim=0)
nll_p75_16   = all_nlls.quantile(0.75, dim=0)

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
    return dict(
        validity=len(canon) and len(valid)/len(canon)*100 or 0,
        unique=len(set(valid))/len(valid)*100,
        quality=float(qmask.mean())*100,
        qed=float(qeds.mean()), sa=float(sas.mean()),
        div=(1.0 - float(np.mean(sims))) if sims else 0.0,
        genmol=(len(valid)/len(canon))*(len(set(valid))/len(valid))*float(qmask.mean())*100,
    )

def fmt(name, s):
    return (f'[{name:<30s}] q={s["quality"]:>6.2f}%  u={s["unique"]:>6.2f}%  '
            f'GenMol={s["genmol"]:>6.2f}%  d={s["div"]:.3f}  QED={s["qed"]:.3f}  SA={s["sa"]:.2f}')

results = []

print('\n---- Aggregation comparison (all bot-N @ OS=4x, highest NLL) ----', flush=True)

def bot_n(signal):
    return torch.argsort(signal, descending=True)[:N].cpu().tolist()

# Baselines
s = score_subset(list(range(N)))
results.append(dict(name='baseline first-N', **s)); print(fmt('baseline first-N', s), flush=True)
s = score_subset(random.Random(0).sample(range(POOL), N))
results.append(dict(name='baseline random-N', **s)); print(fmt('baseline random-N', s), flush=True)

# Reference: M=8 mean (the original winner)
s = score_subset(bot_n(nll_mean_8))
results.append(dict(name='bot-N mean_8 (ref)', **s)); print(fmt('bot-N mean_8 (ref)', s), flush=True)

# M=16 mean
s = score_subset(bot_n(nll_mean_16))
results.append(dict(name='bot-N mean_16 (#7)', **s)); print(fmt('bot-N mean_16 (#7)', s), flush=True)

# M=8 max
s = score_subset(bot_n(nll_max_8))
results.append(dict(name='bot-N max_8 (#2)', **s)); print(fmt('bot-N max_8 (#2)', s), flush=True)

# M=16 max
s = score_subset(bot_n(nll_max_16))
results.append(dict(name='bot-N max_16 (#2+#7)', **s)); print(fmt('bot-N max_16 (#2+#7)', s), flush=True)

# M=16 min (pessimistic for best-case LL)
s = score_subset(bot_n(nll_min_16))
results.append(dict(name='bot-N min_16', **s)); print(fmt('bot-N min_16', s), flush=True)

# M=16 median (robust aggregation)
s = score_subset(bot_n(nll_median_16))
results.append(dict(name='bot-N median_16', **s)); print(fmt('bot-N median_16', s), flush=True)

# M=16 p75 (between median and max)
s = score_subset(bot_n(nll_p75_16))
results.append(dict(name='bot-N p75_16', **s)); print(fmt('bot-N p75_16', s), flush=True)

# M=16 p90 (between median and max, closer to max)
s = score_subset(bot_n(nll_p90_16))
results.append(dict(name='bot-N p90_16', **s)); print(fmt('bot-N p90_16', s), flush=True)

(out_dir / 'maxperm_results.json').write_text(json.dumps(results, indent=2))
print(f'\n[maxperm] wrote {out_dir}/maxperm_results.json', flush=True)
PY
echo "[remote] end=$(date)"
