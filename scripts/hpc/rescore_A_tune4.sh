#!/bin/bash
# tune4: negative α, max_16 inner, median signals, draft T-mix
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/rescore_tune4}"
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
print(f'[tune4] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

N = ${N}; POOL = ${POOL}; M = ${M_PERMS}

# Draft T-mix: equal quarters from 4 configs, total POOL
print('[tune4] T-mix draft: 4 quarters, 1024 each', flush=True)
token_seqs = []
for T, K in [(0.55, 8), (0.6, 10), (0.7, 15), (0.55, 10)]:
    t0 = time.time()
    with torch.no_grad():
        seqs = sample_model_sequences(
            model, vocab, n_samples=POOL // 4, max_new_tokens=64,
            temperature=T, top_k=K, device=dev, ao_decode_mode='ar',
        )
    token_seqs.extend(seqs)
    print(f'  T={T} K={K} -> {len(seqs)} drafts in {time.time()-t0:.1f}s', flush=True)
random.seed(0); random.shuffle(token_seqs)

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
print(f'[tune4] computed {M} perms in {time.time()-t_rescore:.1f}s', flush=True)

nll_min = all_nlls.min(dim=0).values
nll_max = all_nlls.max(dim=0).values
nll_mean = all_nlls.mean(dim=0)
nll_std = all_nlls.std(dim=0)
nll_range = nll_max - nll_min
nll_median = all_nlls.median(dim=0).values
nll_mad = (all_nlls - nll_median.unsqueeze(0)).abs().median(dim=0).values  # MAD
max_8 = all_nlls[:8].max(dim=0).values
max_16 = nll_max

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
    return (f'[{name:<44s}] q={s["quality"]:>6.2f}%  u={s["unique"]:>6.2f}%  '
            f'GenMol={s["genmol"]:>6.2f}%  d={s["div"]:.3f}  SA={s["sa"]:.2f}')

def bot(signal, k=N):
    return torch.argsort(signal, descending=True)[:k].cpu().tolist()

results = []

# References (on T-mix pool)
print('\n---- References (T-mix pool) ----', flush=True)
for name, sig in [
    ('min_16', nll_min),
    ('max_8', max_8),
    ('mean_8', all_nlls[:8].mean(dim=0)),
    ('best prev: min_16-0.2*std', nll_min - 0.2 * nll_std),
    ('best prev: (min-0.5std) top1.3N → max_8', None),
]:
    if sig is None:
        outer = nll_min - 0.5 * nll_std
        inner_idx = bot(outer, k=int(1.3 * N))
        inner_set = torch.tensor(inner_idx, device=dev)
        inner_max8 = max_8[inner_set]
        second_local = torch.argsort(inner_max8, descending=True)[:N].cpu().tolist()
        final_idx = [inner_idx[i] for i in second_local]
        s = score_subset(final_idx)
    else:
        s = score_subset(bot(sig))
    results.append(dict(name=name, **s)); print(fmt(name, s), flush=True)

# #1: negative α (min_16 - α*std, α < 0 would be min + |α|*std → favor unstable)
print('\n---- Negative α: min_16 - α*std ----', flush=True)
for alpha in (-0.3, -0.2, -0.1, 0.0, 0.05, 0.15, 0.25):
    signal = nll_min - alpha * nll_std
    s = score_subset(bot(signal)); tag = f'min_16 - {alpha}*std'
    results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)

# #2: max_16 inner in 2-stage
print('\n---- 2-stage with max_16 inner ----', flush=True)
for alpha, m_mult in [(0.2, 1.3), (0.2, 1.5), (0.3, 1.3), (0.5, 1.3), (0.5, 1.5)]:
    outer = nll_min - alpha * nll_std
    M_inner = int(m_mult * N)
    inner_idx = bot(outer, k=M_inner)
    inner_set = torch.tensor(inner_idx, device=dev)
    inner_max = max_16[inner_set]
    second_local = torch.argsort(inner_max, descending=True)[:N].cpu().tolist()
    final_idx = [inner_idx[i] for i in second_local]
    s = score_subset(final_idx); tag = f'(min-{alpha}*std) top{m_mult}N → max_16'
    results.append(dict(name=tag, **s)); print(fmt(tag, s), flush=True)

# #3: median + MAD (robust alternatives)
print('\n---- Median-based signals ----', flush=True)
for name, sig in [
    ('median', nll_median),
    ('median - 0.2*MAD', nll_median - 0.2 * nll_mad),
    ('median - 0.5*MAD', nll_median - 0.5 * nll_mad),
    ('(median-0.2*MAD) top1.3N → max_8', None),
]:
    if sig is None:
        outer = nll_median - 0.2 * nll_mad
        inner_idx = bot(outer, k=int(1.3 * N))
        inner_set = torch.tensor(inner_idx, device=dev)
        inner_max = max_8[inner_set]
        second_local = torch.argsort(inner_max, descending=True)[:N].cpu().tolist()
        final_idx = [inner_idx[i] for i in second_local]
        s = score_subset(final_idx)
    else:
        s = score_subset(bot(sig))
    results.append(dict(name=name, **s)); print(fmt(name, s), flush=True)

(out_dir / 'tune4_results.json').write_text(json.dumps(results, indent=2))
print('\n---- TOP-5 by GenMol ----', flush=True)
for r in sorted(results, key=lambda r: -r['genmol'])[:5]:
    print(fmt(r['name'], r), flush=True)
print('\n---- TOP-5 by Quality ----', flush=True)
for r in sorted(results, key=lambda r: -r['quality'])[:5]:
    print(fmt(r['name'], r), flush=True)
print(f'\n[tune4] wrote {out_dir}/tune4_results.json', flush=True)
PY
echo "[remote] end=$(date)"
