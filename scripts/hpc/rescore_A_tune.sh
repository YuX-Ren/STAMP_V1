#!/bin/bash
# A-tune: single draft pool, compare 7 rerank strategies on same samples
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/rescore_tune}"
: "${N:=1024}"
: "${POOL:=4096}"       # draft pool size
: "${M_PERMS:=8}"
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
    LMConfig, TinyDecoderLM, sample_model_sequences, _normalize_model_type,
    _sample_ao_orders,
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
print(f'[A-tune] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

N = ${N}; POOL = ${POOL}; M = ${M_PERMS}
TEMP = ${TEMP}; TOPK = ${TOPK}

# ---- Generate draft pool ----
t0 = time.time()
with torch.no_grad():
    token_seqs = sample_model_sequences(
        model, vocab, n_samples=POOL, max_new_tokens=64,
        temperature=TEMP, top_k=TOPK, device=dev, ao_decode_mode='ar',
    )
print(f'[A-tune] generated {POOL} drafts in {time.time()-t0:.1f}s', flush=True)

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

# ---- Compute per-sample NLL under M random AO permutations ----
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
print(f'[A-tune] computed NLL for {M} perms x {POOL} samples in {time.time()-t_rescore:.1f}s', flush=True)

mean_nll = all_nlls.mean(dim=0)                    # (POOL,)
std_nll  = all_nlls.std(dim=0)                     # (POOL,) robustness signal
lengths_t = lengths.to(dev).float()
length_bin = lengths_t - lengths_t.mean()          # length-normalized NLL
nll_len_adj = mean_nll - 0.02 * length_bin         # mild length correction (heuristic)

print(f'[A-tune] mean_nll stats: min={mean_nll.min().item():.3f} median={mean_nll.median().item():.3f} max={mean_nll.max().item():.3f}', flush=True)
print(f'[A-tune] std_nll  stats: min={std_nll.min().item():.3f}  median={std_nll.median().item():.3f}  max={std_nll.max().item():.3f}', flush=True)

# ---- Scoring function ----
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
    if not valid: return dict(validity=0.,unique=0.,quality=0.,qed=0.,sa=0.,div=0.)
    mols = [Chem.MolFromSmiles(c) for c in valid]
    qeds = np.array([QED.qed(m) for m in mols]); sas = np.array([sascorer.calculateScore(m) for m in mols])
    qmask = (qeds > 0.6) & (sas < 4.0)
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols]
    rng = random.Random(0); sims=[]; n=len(fps); pairs=min(5000, n*(n-1)//2)
    for _ in range(pairs):
        i, j = rng.sample(range(n), 2); sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return dict(
        validity=len(valid)/len(canon)*100, unique=len(set(valid))/len(valid)*100,
        quality=float(qmask.mean())*100, qed=float(qeds.mean()), sa=float(sas.mean()),
        div=(1.0 - float(np.mean(sims))) if sims else 0.0,
    )

def fmt(name, s):
    return f'[{name:<28s}] q={s["quality"]:>6.2f}%  u={s["unique"]:>6.2f}%  d={s["div"]:.3f}  QED={s["qed"]:.3f}  SA={s["sa"]:.2f}'

strategies = []
# Baselines (from draft pool, no rerank)
strategies.append(('baseline first-N', list(range(N))))
strategies.append(('baseline random-N', random.Random(0).sample(range(POOL), N)))

# Oversample sweep on bot-N mean-NLL (the winning inverse direction)
for os_mult in (1.5, 2.0, 2.5, 3.0, 4.0):
    pool_size = int(N * os_mult)
    if pool_size > POOL: continue
    # Take pool_size samples, rank by mean NLL ascending, keep bottom-N (highest NLL = "unconfident", our winner)
    sub_nll = mean_nll[:pool_size]
    order = torch.argsort(sub_nll, descending=True)   # highest NLL first (inverted)
    idx = order[:N].cpu().tolist()
    strategies.append((f'bot-N (mean_nll, OS={os_mult})', idx))

# For 2x oversample specifically, also try:
#   - top-N by mean NLL (confirmation of inverse direction)
#   - low-variance filter (robust to ordering = canonical)
#   - high-variance filter (fragile = noisy)
#   - length-adjusted mean NLL
pool_2x = 2 * N
sub_nll = mean_nll[:pool_2x]; sub_std = std_nll[:pool_2x]; sub_nlj = nll_len_adj[:pool_2x]
# top-N mean NLL (lowest NLL = most "confident" = higher LL, previously shown to be worse)
strategies.append(('top-N (mean_nll, OS=2x)', torch.argsort(sub_nll)[:N].cpu().tolist()))
# low-var-N (robust across factorizations)
strategies.append(('low-var-N (OS=2x)',        torch.argsort(sub_std)[:N].cpu().tolist()))
# high-var-N (fragile)
strategies.append(('high-var-N (OS=2x)',       torch.argsort(sub_std, descending=True)[:N].cpu().tolist()))
# length-adj mean NLL, bottom
strategies.append(('bot-N (len_adj_nll, OS=2x)', torch.argsort(sub_nlj, descending=True)[:N].cpu().tolist()))
# combined: mean_nll + lambda*std  (low value = low NLL & low var = "too canonical"? or high NLL & high var?)
# We want drug-like mode -> high NLL (surprising to LM) AND low var (robust across orders)
combined = sub_nll - 1.0 * sub_std     # higher = good: high NLL + low std
strategies.append(('bot-N (nll-1.0*std, OS=2x)', torch.argsort(combined, descending=True)[:N].cpu().tolist()))

# Middle percentile from 2x pool: take 25-75% of mean NLL
sub_order = torch.argsort(sub_nll)     # ascending
mid_start = pool_2x // 4; mid_end = mid_start + N
strategies.append(('middle-N (mean_nll, OS=2x)', sub_order[mid_start:mid_end].cpu().tolist()))

print('\n---- Strategy comparison ----', flush=True)
results = []
for name, idx in strategies:
    s = score_subset(idx)
    results.append(dict(name=name, **s))
    print(fmt(name, s), flush=True)

(out_dir / 'tune_results.json').write_text(json.dumps(results, indent=2))
print(f'\n[A-tune] wrote {out_dir}/tune_results.json', flush=True)
PY
echo "[remote] end=$(date)"
