#!/bin/bash
# GenMol-aligned eval of the old-vocab d=512/l=8 AO-GPT (ep9) for apples-to-apples.
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${RUN_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d512_l8_e10}"
: "${EPOCH:=0009}"
: "${CKPT_PATH:=${RUN_DIR}/checkpoints/epoch_${EPOCH}/model.pt}"
: "${N:=1024}"
: "${TEMPERATURE:=0.6}"
: "${TOP_K:=50}"

echo "[remote] host=$(hostname) date=$(date)"
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

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

from stamp.benchmark.lm import LMConfig, TinyDecoderLM, _sample_model_sequences_decoder
from stamp.benchmark.representations import build_representation
from scripts.benchmark.train_ao_gpt_parquet import _load_scan_cache

run_dir = Path('${RUN_DIR}')
ckpt_path = Path('${CKPT_PATH}')
_, vocab = _load_scan_cache(run_dir)
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
cfg = LMConfig(**ckpt['cfg']); cfg.use_adaln = True
cfg.model_type = 'ao_gpt'

dev = torch.device('cuda:0')
model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg, bidirectional=False).to(dev)
state = ckpt['model_state']
if any(k.startswith('_orig_mod.') for k in state):
    state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
model.load_state_dict(state); model.eval()
rep = build_representation('stamp')
print(f'[old-ao] ckpt={ckpt_path} vocab_size={len(vocab.itos)}', flush=True)

for T, K in [(0.6, 50), (0.7, 20), (0.8, 50), (0.9, 50), (1.0, 50)]:
    t0 = time.time()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        seqs = _sample_model_sequences_decoder(
            model=model, vocab=vocab, n_samples=${N}, max_new_tokens=64,
            temperature=T, top_k=K, device=dev,
        )
    dt = time.time() - t0
    canon_smi, raw_smi = [], []
    for toks in seqs:
        try:
            text = rep.detokenize(toks)
            mol = rep.codec.decode_stamp_to_mol(text)
            if mol is None or len(Chem.GetMolFrags(mol)) != 1:
                raw_smi.append(None); canon_smi.append(None); continue
            r = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
            c = Chem.MolToSmiles(mol, canonical=True,  isomericSmiles=True)
            raw_smi.append(r if r else None); canon_smi.append(c if c else None)
        except Exception:
            raw_smi.append(None); canon_smi.append(None)
    total = ${N}
    valid_raw = [s for s in raw_smi if s]
    valid_canon = [s for s in canon_smi if s]
    validity = len(valid_raw) / total
    uniq_raw = len(set(valid_raw)) / len(valid_raw) if valid_raw else 0.0
    seen=set(); keep=[]
    for i, s in enumerate(raw_smi):
        if s is None or s in seen: continue
        seen.add(s); keep.append(i)
    gm_hits = 0
    for i in keep:
        m = Chem.MolFromSmiles(raw_smi[i])
        if m is None: continue
        if QED.qed(m) >= 0.6 and sascorer.calculateScore(m) <= 4.0: gm_hits += 1
    genmol = gm_hits / total * 100
    mols = [Chem.MolFromSmiles(c) for c in valid_canon if c]
    qeds = np.array([QED.qed(m) for m in mols]) if mols else np.array([0.0])
    sas  = np.array([sascorer.calculateScore(m) for m in mols]) if mols else np.array([10.0])
    qov = float(((qeds>=0.6)&(sas<=4.0)).mean())*100 if mols else 0.0
    print(f'[old-ao] T={T} K={K}: V={validity*100:.1f}%  U={uniq_raw*100:.1f}%  Q/V={qov:.1f}%  GenMol={genmol:.2f}%  QED={float(qeds.mean()):.3f}  SA={float(sas.mean()):.2f}  ({dt:.1f}s)', flush=True)
PY
echo "[remote] end=$(date)"
