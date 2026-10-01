#!/bin/bash
# Eval Eso-LM (alpha_0=1) ep8 checkpoint. Sweep L ∈ {14,16,18,20}.
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
RUN_DIR=/scratch/yuxuan.ren/data/ckpt_stamp/stamp_eso_30m_d512_l8_e10
CKPT_PATH=${RUN_DIR}/checkpoints/epoch_0008/model.pt

echo "[remote] host=$(hostname) date=$(date)"
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

python - <<'PY'
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
    LMConfig, TinyDecoderLM, TokenVocab,
    _sample_model_sequences_diffusion,
)
from stamp.benchmark.representations import build_representation
from scripts.benchmark.train_ao_gpt_parquet import _load_scan_cache

run_dir = Path('/scratch/yuxuan.ren/data/ckpt_stamp/stamp_eso_30m_d512_l8_e10')
ckpt_path = run_dir / 'checkpoints/epoch_0008/model.pt'
_, vocab = _load_scan_cache(run_dir)
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
cfg = LMConfig(**ckpt['cfg'])
# Eso-LM: causal + AdaLN target-pos conditioning.
cfg.use_adaln = True
dev = torch.device('cuda:0')
model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg, bidirectional=False).to(dev)
state = ckpt['model_state']
if any(k.startswith('_orig_mod.') for k in state):
    state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
model.load_state_dict(state); model.eval()
rep = build_representation('stamp')
print(f'[eso-eval] ckpt={ckpt_path} epoch={ckpt.get("epoch","?")} vocab_size={len(vocab.itos)}', flush=True)

T_, TOPK, STEPS, N = 1.0, 50, 128, 512

def sample_at_L(L, n):
    t0 = time.time()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        seqs = _sample_model_sequences_diffusion(
            model=model, vocab=vocab, n_samples=n, max_new_tokens=L,
            temperature=T_, top_k=TOPK,
            diffusion_steps=STEPS, diffusion_block_size=L,
            diffusion_remasking='low_confidence', device=dev,
        )
    dt = time.time() - t0
    return seqs, dt

def decode(seqs):
    raw, can = [], []
    for toks in seqs:
        try:
            text = rep.detokenize(toks)
            mol = rep.codec.decode_stamp_to_mol(text)
            if mol is None or len(Chem.GetMolFrags(mol)) != 1:
                raw.append(None); can.append(None); continue
            r = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
            c = Chem.MolToSmiles(mol, canonical=True,  isomericSmiles=True)
            raw.append(r if r else None); can.append(c if c else None)
        except Exception:
            raw.append(None); can.append(None)
    return raw, can

def report(tag, raw_smi, canon_smi, N, lens):
    valid = [s for s in raw_smi if s]
    validity = len(valid) / N
    uniq_raw = len(set(valid)) / len(valid) if valid else 0.0
    seen=set(); keep=[]
    for i, s in enumerate(raw_smi):
        if s is None or s in seen: continue
        seen.add(s); keep.append(i)
    hits=0
    for i in keep:
        m = Chem.MolFromSmiles(raw_smi[i])
        if m is None: continue
        if QED.qed(m) >= 0.6 and sascorer.calculateScore(m) <= 4.0: hits += 1
    gm = hits / N * 100
    mols = [Chem.MolFromSmiles(c) for c in canon_smi if c]
    qeds = np.array([QED.qed(m) for m in mols]) if mols else np.array([0.0])
    sas  = np.array([sascorer.calculateScore(m) for m in mols]) if mols else np.array([10.0])
    qmask = (qeds >= 0.6) & (sas <= 4.0)
    qov = float(qmask.mean()) * 100 if mols else 0.0
    print(f'{tag:<16} V={validity*100:5.1f}%  U={uniq_raw*100:5.1f}%  Q/V={qov:5.1f}%  GenMol={gm:5.1f}%  QED={float(qeds.mean()):.3f}  SA={float(sas.mean()):.2f}  len p50={int(np.median(lens))}', flush=True)
    return dict(tag=tag, validity=validity*100, uniq_raw=uniq_raw*100, quality_over_valid=qov, genmol=gm,
                qed=float(qeds.mean()), sa=float(sas.mean()), len_p50=int(np.median(lens)))

results = {}
for L in (14, 16, 18, 20):
    seqs, dt = sample_at_L(L, N)
    lens = [len([t for t in row if t]) for row in seqs]
    raw, can = decode(seqs)
    results[f'L{L}'] = report(f'L={L} ({dt:.1f}s)', raw, can, N, lens)

(run_dir / 'eval_ep8_lensweep.json').write_text(json.dumps(results, indent=2))
print('\nfirst 3 canonical SMILES at best config:', flush=True)
best = max(results, key=lambda k: results[k]['genmol'])
print(f'  (best = {best})')
PY
echo "[remote] end=$(date)"
