#!/bin/bash
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d512_l8_e10/eval_epoch_0005}"
: "${OUT_DIR:=${CKPT_DIR}/grid}"

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
out_dir = Path('${OUT_DIR}')

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

print(f'[grid] model loaded: epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

N = 1024
# Grid
TEMPS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
TOP_KS = [10, 20, 50, 100, 200]

def score_samples(token_seqs):
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
    valid_canon = [c for c in canon if c]
    if not valid_canon:
        return dict(validity=0.0, unique=0.0, quality=0.0, qed=0.0, sa=0.0, div=0.0)
    mols = [Chem.MolFromSmiles(c) for c in valid_canon]
    qeds = np.array([QED.qed(m) for m in mols])
    sas  = np.array([sascorer.calculateScore(m) for m in mols])
    quality_mask = (qeds > 0.6) & (sas < 4.0)
    # Internal diversity via ECFP4, 5k random pairs
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols]
    rng = random.Random(0); pairs = 5000
    sims = []
    n = len(fps)
    for _ in range(min(pairs, n*(n-1)//2)):
        i, j = rng.sample(range(n), 2)
        sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    div = 1.0 - float(np.mean(sims)) if sims else 0.0
    return dict(
        validity=len(valid_canon)/len(canon)*100,
        unique=len(set(valid_canon))/len(valid_canon)*100,
        quality=float(quality_mask.mean())*100,
        qed=float(qeds.mean()),
        sa=float(sas.mean()),
        div=div,
    )

rows = []
print(f'{"T":<6}{"K":<6}{"valid%":>8}{"uniq%":>8}{"qual%":>8}{"QED":>8}{"SA":>7}{"div":>8}{"t(s)":>8}', flush=True)
for T in TEMPS:
    for K in TOP_KS:
        t0 = time.time()
        with torch.no_grad():
            seqs = sample_model_sequences(
                model, vocab, n_samples=N, max_new_tokens=64,
                temperature=T, top_k=K, device=dev, ao_decode_mode='ar',
            )
        s = score_samples(seqs)
        dt = time.time() - t0
        row = dict(temp=T, top_k=K, dt_s=dt, **s)
        rows.append(row)
        print(f'{T:<6.2f}{K:<6d}{s["validity"]:>8.2f}{s["unique"]:>8.2f}{s["quality"]:>8.2f}'
              f'{s["qed"]:>8.3f}{s["sa"]:>7.2f}{s["div"]:>8.3f}{dt:>8.1f}', flush=True)

(out_dir / 'grid_results.json').write_text(json.dumps(rows, indent=2))
print(f'[grid] wrote {out_dir/"grid_results.json"}', flush=True)

# Identify winners
best_qual = max(rows, key=lambda r: r['quality'])
best_qd = max(rows, key=lambda r: r['quality'] * (r['unique']/100) * r['div'])  # balance
print(f'[grid] Max quality: T={best_qual["temp"]}, K={best_qual["top_k"]} -> {best_qual["quality"]:.2f}%  (QED={best_qual["qed"]:.3f} SA={best_qual["sa"]:.2f} div={best_qual["div"]:.3f})', flush=True)
print(f'[grid] Balanced (quality*unique*div): T={best_qd["temp"]}, K={best_qd["top_k"]} -> {best_qd["quality"]:.2f}% quality  {best_qd["unique"]:.2f}% unique  div={best_qd["div"]:.3f}', flush=True)
PY

echo "[remote] end=$(date)"
