#!/bin/bash
# Phase 2 bidirectional consistency: forward vs reverse-middle NLL asymmetry.
# AO-GPT exclusive signal (SAFE/GenMol can't compute this).
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/phase2_bidir}"
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
print(f'[bidir] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

N = ${N}; POOL = ${POOL}; M = ${M_PERMS}

# ---- Draft pool ----
t0 = time.time()
with torch.no_grad():
    token_seqs = sample_model_sequences(
        model, vocab, n_samples=POOL, max_new_tokens=64,
        temperature=0.6, top_k=10, device=dev, ao_decode_mode='ar',
    )
print(f'[bidir] generated {POOL} drafts in {time.time()-t0:.1f}s', flush=True)

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
L_y = max_len - 1
last_valid_y = (lengths.to(dev) - 2).clamp(min=0)  # y-index of EOS

def compute_nll(orders):
    BSZ = 512
    out = torch.zeros(POOL, device=dev)
    with torch.no_grad():
        for s in range(0, POOL, BSZ):
            e = min(s + BSZ, POOL)
            x_c = x[s:e]; y_c = y[s:e]; lab_c = labels[s:e]; ord_c = orders[s:e]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x_c, orders=ord_c).float()
            bi = torch.arange(e - s, device=dev).unsqueeze(1).expand_as(ord_c)
            targets = y_c[bi, ord_c]; lab_sh = lab_c[bi, ord_c]
            log_probs = F.log_softmax(logits, dim=-1)
            nll_per = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            valid = (lab_sh != -100).float()
            out[s:e] = (nll_per * valid).sum(-1) / valid.sum(-1).clamp(min=1.0)
    return out

# ---- Forward order (identity with first/last preserved) ----
forward_orders = torch.zeros(POOL, L_y, dtype=torch.long, device=dev)
for b in range(POOL):
    lv = int(last_valid_y[b].item())
    if lv <= 0:
        forward_orders[b] = torch.arange(L_y, device=dev)
        continue
    row = [0] + list(range(1, lv)) + [lv] + list(range(lv+1, L_y))
    forward_orders[b] = torch.tensor(row, device=dev)

# ---- Reverse order (first/last preserved, middle REVERSED) ----
reverse_orders = torch.zeros(POOL, L_y, dtype=torch.long, device=dev)
for b in range(POOL):
    lv = int(last_valid_y[b].item())
    if lv <= 0:
        reverse_orders[b] = torch.arange(L_y, device=dev)
        continue
    row = [0] + list(range(lv-1, 0, -1)) + [lv] + list(range(lv+1, L_y))
    reverse_orders[b] = torch.tensor(row, device=dev)

t0 = time.time()
nll_fwd = compute_nll(forward_orders)
nll_rev = compute_nll(reverse_orders)
print(f'[bidir] forward+reverse NLL in {time.time()-t0:.1f}s', flush=True)
asymm = (nll_fwd - nll_rev).abs()

# ---- Also compute M=16 random permutations NLL for comparison ----
all_nlls = torch.zeros(M, POOL, device=dev)
t0 = time.time()
for m_idx in range(M):
    orders = torch.zeros(POOL, L_y, dtype=torch.long, device=dev)
    for b in range(POOL):
        lv = int(last_valid_y[b].item())
        if lv <= 1:
            orders[b] = torch.arange(L_y, device=dev); continue
        mid = list(range(1, lv)); random.shuffle(mid)
        row = [0] + mid + [lv] + list(range(lv+1, L_y))
        orders[b] = torch.tensor(row, device=dev)
    all_nlls[m_idx] = compute_nll(orders)
print(f'[bidir] {M} random perms NLL in {time.time()-t0:.1f}s', flush=True)

nll_min = all_nlls.min(dim=0).values
nll_std = all_nlls.std(dim=0)
max_8 = all_nlls[:8].max(dim=0).values

print(f'[bidir] stats: asymm mean={asymm.mean():.3f} median={asymm.median():.3f} max={asymm.max():.3f}', flush=True)
print(f'[bidir] stats: nll_fwd mean={nll_fwd.mean():.3f}  nll_rev mean={nll_rev.mean():.3f}', flush=True)

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

def top(signal, k=N, descending=True):
    return torch.argsort(signal, descending=descending)[:k].cpu().tolist()

results = []
print('\n---- References ----', flush=True)
s = score_subset(list(range(N))); results.append(dict(name='baseline first-N', **s)); print(fmt('baseline first-N', s))
# Pure min_16
s = score_subset(top(nll_min)); results.append(dict(name='bot-N min_16', **s)); print(fmt('bot-N min_16', s))
# Existing winner
outer = nll_min - 0.5 * nll_std
idx_in = top(outer, k=int(1.3 * N))
inner_max = max_8[torch.tensor(idx_in, device=dev)]
rel = torch.argsort(inner_max, descending=True)[:N].cpu().tolist()
sel = [idx_in[i] for i in rel]
s = score_subset(sel); results.append(dict(name='(min-0.5std) top1.3N → max_8', **s)); print(fmt('(min-0.5std) top1.3N → max_8', s))

print('\n---- Bidirectional filters (AO-exclusive) ----', flush=True)
# Small asymmetry alone
s = score_subset(top(-asymm)); results.append(dict(name='bot-N -asymm (small first)', **s)); print(fmt('bot-N -asymm (small first)', s))
s = score_subset(top(asymm)); results.append(dict(name='bot-N asymm (large first)', **s)); print(fmt('bot-N asymm (large first)', s))

# min_nll - α*asymm (add penalty for asymmetry)
print('\n---- min_16 - α*asymm (low asymmetry bias) ----', flush=True)
for alpha in (0.1, 0.3, 0.5, 1.0, 2.0):
    signal = nll_min - alpha * asymm
    s = score_subset(top(signal)); tag = f'min_16 - {alpha}*asymm'
    results.append(dict(name=tag, **s)); print(fmt(tag, s))

# min_16 - α*std - β*asymm (combined)
print('\n---- min_16 - 0.2*std - β*asymm ----', flush=True)
for beta in (0.1, 0.3, 0.5, 1.0):
    signal = nll_min - 0.2 * nll_std - beta * asymm
    s = score_subset(top(signal)); tag = f'min_16 - 0.2*std - {beta}*asymm'
    results.append(dict(name=tag, **s)); print(fmt(tag, s))

# 2-stage outer combined
print('\n---- 2-stage: (min-0.2*std-β*asymm) top-1.3N → max_8 ----', flush=True)
for beta in (0.2, 0.5, 1.0):
    outer = nll_min - 0.2 * nll_std - beta * asymm
    idx_in = top(outer, k=int(1.3 * N))
    inner_max = max_8[torch.tensor(idx_in, device=dev)]
    rel = torch.argsort(inner_max, descending=True)[:N].cpu().tolist()
    sel = [idx_in[i] for i in rel]
    s = score_subset(sel); tag = f'(min-0.2std-{beta}asymm) top1.3N→max_8'
    results.append(dict(name=tag, **s)); print(fmt(tag, s))

(out_dir / 'bidir_results.json').write_text(json.dumps(results, indent=2))
print('\n---- TOP-3 by GenMol ----', flush=True)
for r in sorted(results, key=lambda r: -r['genmol'])[:3]:
    print(fmt(r['name'], r), flush=True)
print('\n---- TOP-3 by Quality ----', flush=True)
for r in sorted(results, key=lambda r: -r['quality'])[:3]:
    print(fmt(r['name'], r), flush=True)
print(f'\n[bidir] wrote {out_dir}/bidir_results.json', flush=True)
PY
echo "[remote] end=$(date)"
