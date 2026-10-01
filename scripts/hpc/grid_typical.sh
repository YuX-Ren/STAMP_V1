#!/bin/bash
# Typical sampling grid: T ∈ {0.6, 0.7, 0.8, 0.9, 1.0} × typical_p ∈ {0.2, 0.5, 0.7, 0.85, 0.9, 0.95}
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:?must set CKPT_DIR}"
: "${OUT_DIR:=${CKPT_DIR}/grid_typical}"

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
print(f'[typical] model: epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

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
    rng = random.Random(0); sims = []
    n = len(fps); pairs = min(5000, n*(n-1)//2)
    for _ in range(pairs):
        i, j = rng.sample(range(n), 2); sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return dict(
        validity=len(valid)/len(canon)*100, unique=len(set(valid))/len(valid)*100,
        quality=float(qmask.mean())*100, qed=float(qeds.mean()), sa=float(sas.mean()),
        div=(1.0 - float(np.mean(sims))) if sims else 0.0,
    )

TEMPS  = [0.6, 0.7, 0.8, 0.9, 1.0]
TYPS   = [0.2, 0.5, 0.7, 0.85, 0.9, 0.95]
rows = []
print(f'{"T":<5}{"typ_p":<7}{"q%":>7}{"u%":>7}{"div":>7}{"QED":>7}{"SA":>6}{"t(s)":>7}', flush=True)
for T in TEMPS:
    for P in TYPS:
        t0 = time.time()
        with torch.no_grad():
            seqs = sample_model_sequences(
                model, vocab, n_samples=1024, max_new_tokens=64,
                temperature=T, top_k=0, top_p=0.0, typical_p=P,
                device=dev, ao_decode_mode='ar',
            )
        s = score(seqs); dt = time.time() - t0
        rows.append(dict(temp=T, typical_p=P, top_k=0, top_p=0, dt_s=dt, **s))
        print(f'{T:<5.2f}{P:<7.2f}{s["quality"]:>7.2f}{s["unique"]:>7.2f}{s["div"]:>7.3f}{s["qed"]:>7.3f}{s["sa"]:>6.2f}{dt:>7.1f}', flush=True)

(out_dir / 'grid_typical.json').write_text(json.dumps(rows, indent=2))
best = max(rows, key=lambda r: r['quality'])
b95 = max([r for r in rows if r['unique']>=95.0], key=lambda r:r['quality'], default=None)
b98 = max([r for r in rows if r['unique']>=98.0], key=lambda r:r['quality'], default=None)
print(f'\n[typical] max-q : T={best["temp"]} p={best["typical_p"]} => q={best["quality"]:.2f}% u={best["unique"]:.2f}% d={best["div"]:.3f}', flush=True)
if b95: print(f'[typical] u>=95: T={b95["temp"]} p={b95["typical_p"]} => q={b95["quality"]:.2f}% u={b95["unique"]:.2f}% d={b95["div"]:.3f}', flush=True)
if b98: print(f'[typical] u>=98: T={b98["temp"]} p={b98["typical_p"]} => q={b98["quality"]:.2f}% u={b98["unique"]:.2f}% d={b98["div"]:.3f}', flush=True)
PY
echo "[remote] end=$(date)"
