#!/bin/bash
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d512_l8_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/grid_fine}"

echo "[remote] host=$(hostname) date=$(date)"
nvidia-smi -L || true
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

ckpt_dir = Path('${CKPT_DIR}')
out_dir  = Path('${OUT_DIR}')

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
print(f'[grid-fine] model loaded: epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

def score(token_seqs):
    smiles = []
    for toks in token_seqs:
        text = rep.detokenize(toks)
        try: smi = rep.decode(text)
        except Exception: smi = None
        smiles.append(smi)
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
    rng = random.Random(0)
    sims = []
    n = len(fps)
    pairs = min(5000, n*(n-1)//2)
    for _ in range(pairs):
        i, j = rng.sample(range(n), 2)
        sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return dict(
        validity=len(valid)/len(canon)*100,
        unique=len(set(valid))/len(valid)*100,
        quality=float(qmask.mean())*100,
        qed=float(qeds.mean()),
        sa=float(sas.mean()),
        div=(1.0 - float(np.mean(sims))) if sims else 0.0,
    )

def bench(name, mode, N, T, K):
    t0 = time.time()
    with torch.no_grad():
        seqs = sample_model_sequences(
            model, vocab, n_samples=N, max_new_tokens=64,
            temperature=T, top_k=K, device=dev, ao_decode_mode=mode,
        )
    s = score(seqs)
    dt = time.time() - t0
    print(f'[{name:>10s}] T={T:.2f} K={K:<3d} N={N:<4d} => '
          f'q={s["quality"]:>5.2f}%  u={s["unique"]:>5.2f}%  d={s["div"]:.3f}  QED={s["qed"]:.3f}  SA={s["sa"]:.2f}  ({dt:5.1f}s)', flush=True)
    return dict(mode=mode, temp=T, top_k=K, n=N, dt_s=dt, **s)

# ---- Part A: AR fine grid around T=0.7 K=10 ----
print('\n==== Part A: AR fine grid (1024 samples each) ====', flush=True)
TEMPS_A = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
KS_A    = [5, 8, 10, 15, 20, 30]
rows_a = []
for T in TEMPS_A:
    for K in KS_A:
        rows_a.append(bench('AR_fine', 'ar', 1024, T, K))
(out_dir / 'grid_fine_ar.json').write_text(json.dumps(rows_a, indent=2))

best = max(rows_a, key=lambda r: r['quality'])
best_95 = max([r for r in rows_a if r['unique'] >= 95.0], key=lambda r: r['quality'], default=None)
best_98 = max([r for r in rows_a if r['unique'] >= 98.0], key=lambda r: r['quality'], default=None)
print(f'\n[AR-fine] absolute max quality : T={best["temp"]} K={best["top_k"]} => q={best["quality"]:.2f}%  u={best["unique"]:.2f}%', flush=True)
if best_95: print(f'[AR-fine] max q with unique>=95: T={best_95["temp"]} K={best_95["top_k"]} => q={best_95["quality"]:.2f}%  u={best_95["unique"]:.2f}%', flush=True)
if best_98: print(f'[AR-fine] max q with unique>=98: T={best_98["temp"]} K={best_98["top_k"]} => q={best_98["quality"]:.2f}%  u={best_98["unique"]:.2f}%', flush=True)

# ---- Part B: Confidence-guided (MaskGIT-style) decoding ----
# Much slower (O(n^2) fwd), so smaller sample + sparser grid
print('\n==== Part B: ao_canvas_confidence decoding (512 samples each, slower) ====', flush=True)
TEMPS_B = [0.6, 0.7, 0.8]
KS_B    = [10, 20, 50]
rows_b = []
for T in TEMPS_B:
    for K in KS_B:
        rows_b.append(bench('conf', 'ao_canvas_confidence', 512, T, K))
(out_dir / 'grid_conf.json').write_text(json.dumps(rows_b, indent=2))

best_b = max(rows_b, key=lambda r: r['quality'])
best_b_95 = max([r for r in rows_b if r['unique'] >= 95.0], key=lambda r: r['quality'], default=None)
print(f'\n[conf] absolute max quality : T={best_b["temp"]} K={best_b["top_k"]} => q={best_b["quality"]:.2f}%  u={best_b["unique"]:.2f}%', flush=True)
if best_b_95: print(f'[conf] max q with unique>=95: T={best_b_95["temp"]} K={best_b_95["top_k"]} => q={best_b_95["quality"]:.2f}%  u={best_b_95["unique"]:.2f}%', flush=True)

print('[grid-fine] done', flush=True)
PY
echo "[remote] end=$(date)"
