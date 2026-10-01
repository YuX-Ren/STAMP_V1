#!/bin/bash
# A: Permutation-averaged rescoring. Generate 2N draft samples, rerank by
# mean log-prob across M random AO orders, keep top N, evaluate.
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/rescore}"
: "${N:=1024}"
: "${OVERSAMPLE:=2}"   # generate 2N then filter to N
: "${M_PERMS:=8}"      # number of random AO orders for averaging
: "${TEMP:=0.6}"
: "${TOPK:=10}"

echo "[remote] host=$(hostname) date=$(date) CKPT_DIR=$CKPT_DIR"
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
print(f'[rescore] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

N = ${N}; M = ${M_PERMS}; OS = ${OVERSAMPLE}
TEMP = ${TEMP}; TOPK = ${TOPK}
N_DRAFT = N * OS

# ---- Step 1: Generate draft samples ----
t0 = time.time()
with torch.no_grad():
    token_seqs = sample_model_sequences(
        model, vocab, n_samples=N_DRAFT, max_new_tokens=64,
        temperature=TEMP, top_k=TOPK, device=dev, ao_decode_mode='ar',
    )
print(f'[rescore] generated {N_DRAFT} drafts in {time.time()-t0:.1f}s '
      f'(T={TEMP} K={TOPK})', flush=True)

# ---- Step 2: Convert tokens back to IDs and pad ----
# token_seqs is List[List[str]]. Reconstruct BOS + ids + EOS (the sampler trims EOS tail).
bos, eos, pad = vocab.bos_id, vocab.eos_id, vocab.pad_id
id_seqs = []
for toks in token_seqs:
    ids = [bos] + vocab.encode_tokens(toks) + [eos]
    id_seqs.append(ids)

max_len = max(len(s) for s in id_seqs)
max_len = min(max_len, cfg.max_seq_len)
lengths = torch.tensor([min(len(s), max_len) for s in id_seqs], dtype=torch.long)
padded = torch.full((N_DRAFT, max_len), pad, dtype=torch.long)
for i, s in enumerate(id_seqs):
    L = min(len(s), max_len)
    padded[i, :L] = torch.tensor(s[:L], dtype=torch.long)
padded = padded.to(dev)

# Build x, y, labels with ignore_index=-100 outside real tokens
x = padded[:, :-1].contiguous()                       # (N, max_len-1)
y = padded[:, 1:].contiguous()                        # (N, max_len-1)
labels = y.clone()
seq_mask_y = torch.arange(max_len - 1, device=dev).unsqueeze(0) < (lengths.to(dev).unsqueeze(1) - 1)
labels = torch.where(seq_mask_y, labels, torch.full_like(labels, -100))

# ---- Step 3: Compute mean log-prob across M random AO orders ----
# Each permutation: orders = _sample_ao_orders(labels, random_ratio=1.0)
# logits[i, t, :] predicts y[i, orders[i, t]]. NLL = -log_softmax(logits).gather(target).
BSZ = 512  # mini-batch over drafts to fit memory
mean_nll = torch.zeros(N_DRAFT, device=dev)
total_valid = torch.zeros(N_DRAFT, device=dev)
t_score_start = time.time()
for pass_idx in range(M):
    # Fresh random order for this pass
    with torch.no_grad():
        # Process in chunks
        nll_sum = torch.zeros(N_DRAFT, device=dev)
        valid_cnt = torch.zeros(N_DRAFT, device=dev)
        for s in range(0, N_DRAFT, BSZ):
            e = min(s + BSZ, N_DRAFT)
            x_c = x[s:e]; y_c = y[s:e]; lab_c = labels[s:e]
            orders = _sample_ao_orders(lab_c, random_ratio=1.0)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x_c, orders=orders).float()  # (B, t, V)
            batch_idx = torch.arange(e - s, device=dev).unsqueeze(1).expand_as(orders)
            targets = y_c[batch_idx, orders]               # (B, t)
            lab_sh = lab_c[batch_idx, orders]
            # Cross-entropy per position
            log_probs = F.log_softmax(logits, dim=-1)
            nll_per = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, t)
            valid = (lab_sh != -100).float()
            nll_sum[s:e] = (nll_per * valid).sum(dim=-1)
            valid_cnt[s:e] = valid.sum(dim=-1)
        per_token_nll = nll_sum / valid_cnt.clamp(min=1.0)
        mean_nll += per_token_nll
mean_nll = mean_nll / M
print(f'[rescore] computed mean NLL over {M} orders in {time.time()-t_score_start:.1f}s', flush=True)

# ---- Step 4: Rank ----
order = torch.argsort(mean_nll)                              # ascending NLL
keep = order[:N].cpu().tolist()
drop = order[N:].cpu().tolist()
keep_seqs = [token_seqs[i] for i in keep]
drop_seqs = [token_seqs[i] for i in drop]
print(f'[rescore] NLL: kept mean={mean_nll[order[:N]].mean().item():.4f}  '
      f'dropped mean={mean_nll[order[N:]].mean().item():.4f}', flush=True)

# ---- Step 5: Score both halves for comparison ----
def score(seqs):
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
    if not valid:
        return dict(validity=0., unique=0., quality=0., qed=0., sa=0., div=0.)
    mols = [Chem.MolFromSmiles(c) for c in valid]
    qeds = np.array([QED.qed(m) for m in mols])
    sas  = np.array([sascorer.calculateScore(m) for m in mols])
    qmask = (qeds > 0.6) & (sas < 4.0)
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols]
    rng = random.Random(0); sims = []; n = len(fps); pairs = min(5000, n*(n-1)//2)
    for _ in range(pairs):
        i, j = rng.sample(range(n), 2); sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return dict(
        validity=len(valid)/len(canon)*100, unique=len(set(valid))/len(valid)*100,
        quality=float(qmask.mean())*100, qed=float(qeds.mean()), sa=float(sas.mean()),
        div=(1.0 - float(np.mean(sims))) if sims else 0.0,
    )

print('[rescore] scoring ...', flush=True)
s_base = score(token_seqs[:N])         # first N as unranked baseline (same draft distribution)
s_keep = score(keep_seqs)               # top-N by perm-avg log-prob
s_drop = score(drop_seqs)               # bottom-N by perm-avg log-prob
s_full = score(token_seqs)              # full 2N baseline
def fmt(s):
    return f'q={s["quality"]:.2f}%  u={s["unique"]:.2f}%  d={s["div"]:.3f}  QED={s["qed"]:.3f}  SA={s["sa"]:.2f}'
print(f'[baseline first-N = {N}]       {fmt(s_base)}', flush=True)
print(f'[baseline full 2N = {N_DRAFT}]  {fmt(s_full)}', flush=True)
print(f'[top-N by perm-avg LL]          {fmt(s_keep)}   <-- RESCORED', flush=True)
print(f'[bot-N by perm-avg LL]          {fmt(s_drop)}', flush=True)

(out_dir / 'rescore.json').write_text(json.dumps(dict(
    config=dict(T=TEMP, K=TOPK, N=N, OS=OS, M=M),
    baseline_firstN=s_base, baseline_all=s_full,
    rescored_top=s_keep, rescored_bot=s_drop,
), indent=2))
print(f'[rescore] wrote {out_dir}/rescore.json', flush=True)
PY
echo "[remote] end=$(date)"
