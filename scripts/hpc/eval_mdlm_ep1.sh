#!/bin/bash
# Quick GenMol-aligned eval of a diffusion (MDLM) checkpoint.
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${RUN_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/stamp_mdlm_30m_d512_l8_e3}"
: "${CKPT_PATH:=${RUN_DIR}/checkpoints/epoch_0001/model.pt}"
: "${N:=1024}"
: "${MAX_NEW:=64}"
: "${STEPS:=128}"
: "${TEMPERATURE:=1.0}"
: "${TOP_K:=50}"
: "${TAG:=epoch_0001}"

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

from stamp.benchmark.lm import (
    LMConfig, TinyDecoderLM, TokenVocab,
    _sample_model_sequences_diffusion,
)
from stamp.benchmark.representations import build_representation
from scripts.benchmark.train_ao_gpt_parquet import _load_scan_cache

run_dir = Path('${RUN_DIR}')
ckpt_path = Path('${CKPT_PATH}')
_, vocab = _load_scan_cache(run_dir)
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
cfg = LMConfig(**ckpt['cfg'])
dev = torch.device('cuda:0')
# MDLM arch: bidirectional, no AdaLN.
cfg.use_adaln = False
model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg, bidirectional=True).to(dev)
state = ckpt['model_state']
if any(k.startswith('_orig_mod.') for k in state):
    state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
model.load_state_dict(state); model.eval()
rep = build_representation('stamp')
print(f'[mdlm-eval] ckpt={ckpt_path} epoch={ckpt.get("epoch","?")} vocab_size={len(vocab.itos)} mask_id={vocab.mask_id}', flush=True)

N = ${N}; MAX_NEW = ${MAX_NEW}; STEPS = ${STEPS}; T = ${TEMPERATURE}; TOPK = ${TOP_K}

t0 = time.time()
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    seqs = _sample_model_sequences_diffusion(
        model=model, vocab=vocab, n_samples=N, max_new_tokens=MAX_NEW,
        temperature=T, top_k=TOPK,
        diffusion_steps=STEPS, diffusion_block_size=MAX_NEW,
        diffusion_remasking='low_confidence', device=dev,
    )
dt = time.time() - t0
print(f'[mdlm-eval] sampled {N} in {dt:.1f}s  (avg {dt/N*1000:.1f} ms/sample)', flush=True)

# Decode to SMILES (STAMP rep already canonicalizes).
canon_smi, raw_smi = [], []
for toks in seqs:
    try:
        text = rep.detokenize(toks)
        mol = rep.codec.decode_stamp_to_mol(text)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            raw_smi.append(None); canon_smi.append(None); continue
        raw = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
        can = Chem.MolToSmiles(mol, canonical=True,  isomericSmiles=True)
        raw_smi.append(raw if raw else None)
        canon_smi.append(can if can else None)
    except Exception:
        raw_smi.append(None); canon_smi.append(None)

total = N
valid_raw   = [s for s in raw_smi   if s]
valid_canon = [s for s in canon_smi if s]
validity = len(valid_raw) / total
uniq_raw   = len(set(valid_raw))   / len(valid_raw)   if valid_raw   else 0.0
uniq_canon = len(set(valid_canon)) / len(valid_canon) if valid_canon else 0.0

# Sample-length stats
lens = [len([t for t in row if t]) for row in seqs]
print(f'[mdlm-eval] len stats: min={min(lens)} p50={int(np.median(lens))} p90={int(np.percentile(lens,90))} max={max(lens)}', flush=True)

# GenMol single-intersection Quality: dedup (raw) -> QED>=0.6 & SA<=4 -> /N
seen = set(); keep = []
for i, s in enumerate(raw_smi):
    if s is None or s in seen: continue
    seen.add(s); keep.append(i)
gm_hits = 0
qeds_all, sas_all = [], []
for i in keep:
    m = Chem.MolFromSmiles(raw_smi[i])
    if m is None: continue
    q = QED.qed(m); sa = sascorer.calculateScore(m)
    qeds_all.append(q); sas_all.append(sa)
    if q >= 0.6 and sa <= 4.0: gm_hits += 1
genmol_score = gm_hits / total * 100

# QED/SA over all valid mols (for reporting)
mols = [Chem.MolFromSmiles(c) for c in valid_canon]
qeds = np.array([QED.qed(m) for m in mols]) if mols else np.array([0.0])
sas  = np.array([sascorer.calculateScore(m) for m in mols]) if mols else np.array([10.0])
qmask = (qeds >= 0.6) & (sas <= 4.0)
quality_over_valid = float(qmask.mean()) * 100 if mols else 0.0

# Diversity (Tanimoto on subset)
fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols] if mols else []
rng = random.Random(0); sims = []
if fps and len(fps) > 1:
    pairs = min(5000, len(fps) * (len(fps)-1) // 2)
    for _ in range(pairs):
        i, j = rng.sample(range(len(fps)), 2)
        sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
div = (1.0 - float(np.mean(sims))) if sims else 0.0

print('')
print(f'[mdlm-eval] validity       = {validity*100:.2f}%')
print(f'[mdlm-eval] uniqueness raw = {uniq_raw*100:.2f}%   canon = {uniq_canon*100:.2f}%')
print(f'[mdlm-eval] quality (valid)= {quality_over_valid:.2f}%')
print(f'[mdlm-eval] GenMol score   = {genmol_score:.2f}%  (apples-to-apples vs AR 79.64%)')
print(f'[mdlm-eval] QED mean={float(qeds.mean()):.3f}  SA mean={float(sas.mean()):.3f}  diversity={div:.3f}')

# First few samples for sanity
print('')
print('[mdlm-eval] first 5 valid canonical SMILES:')
for s in valid_canon[:5]: print(f'  {s}')

out = dict(
    ckpt=str(ckpt_path), n=total, dt_s=dt,
    validity=validity*100, uniq_raw=uniq_raw*100, uniq_canon=uniq_canon*100,
    quality_over_valid=quality_over_valid, genmol_score=genmol_score,
    qed_mean=float(qeds.mean()), sa_mean=float(sas.mean()), diversity=div,
    len_p50=int(np.median(lens)), len_p90=int(np.percentile(lens,90)), len_max=int(max(lens)),
)
out_path = run_dir / f'eval_${TAG}.json'
out_path.write_text(json.dumps(out, indent=2))
print(f'\n[mdlm-eval] wrote {out_path}', flush=True)
PY
echo "[remote] end=$(date)"
