#!/bin/bash
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
CKPT_DIR=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d512_l8_e10/eval_epoch_0005
OUT_DIR=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d512_l8_e10/eval_epoch_0005/samples

echo "[remote] host=$(hostname) date=$(date)"
nvidia-smi -L || true
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

python - <<PY
import sys, json, time
sys.path.insert(0, '.')
from pathlib import Path
import torch

from stamp.benchmark.lm import (
    LMConfig, TinyDecoderLM, TokenVocab, sample_model_sequences,
    _normalize_model_type,
)
from scripts.benchmark.train_ao_gpt_parquet import _load_scan_cache
from stamp.benchmark.representations import build_representation

ckpt_dir = Path('${CKPT_DIR}')
out_dir = Path('${OUT_DIR}')

# ---- load vocab from scan cache (training not finished yet) ----
_, vocab = _load_scan_cache(ckpt_dir)
(ckpt_dir / 'vocab.json').write_text(json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2))
print(f'[sample] vocab={len(vocab.itos)}  pad={vocab.pad_id} bos={vocab.bos_id} eos={vocab.eos_id}', flush=True)

# ---- load model ----
ckpt = torch.load(ckpt_dir / 'model.pt', map_location='cpu', weights_only=False)
cfg = LMConfig(**ckpt['cfg'])
cfg.model_type = _normalize_model_type(str(ckpt.get('model_type', cfg.model_type)))
print(f'[sample] model_type={cfg.model_type}  epoch={ckpt.get("epoch","?")}  params={sum(ckpt["model_state"][k].numel() for k in ckpt["model_state"])/1e6:.2f}M', flush=True)

dev = torch.device('cuda:0')
model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg).to(dev)
state = ckpt['model_state']
if any(k.startswith('_orig_mod.') for k in state):
    state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
model.load_state_dict(state)
model.eval()

# ---- sample ----
N = 1024
print(f'[sample] generating {N} sequences (temp=1.0, top_k=50, max_new_tokens=64, AR decode)...', flush=True)
t0 = time.time()
with torch.no_grad():
    token_seqs = sample_model_sequences(
        model, vocab,
        n_samples=N,
        max_new_tokens=64,
        temperature=1.0,
        top_k=50,
        device=dev,
        ao_decode_mode='ar',
    )
dt = time.time() - t0
print(f'[sample] done in {dt:.1f}s  ({N/dt:.1f} samples/s)', flush=True)

# ---- decode STAMP tokens -> SMILES ----
rep = build_representation('stamp')
print(f'[sample] decoding STAMP -> SMILES ...', flush=True)
t0 = time.time()
smiles = []
for toks in token_seqs:
    # detokenize first (joins tokens with space), then decode back to SMILES
    text = rep.detokenize(toks)
    try:
        smi = rep.decode(text)
    except Exception:
        smi = None
    smiles.append(smi)
dt = time.time() - t0
print(f'[sample] decode done in {dt:.1f}s', flush=True)

# ---- canonicalize with RDKit, compute validity + uniqueness ----
from rdkit import Chem
canon = []
for smi in smiles:
    if not smi:
        canon.append(None); continue
    try:
        mol = Chem.MolFromSmiles(smi)
        canon.append(Chem.MolToSmiles(mol) if mol is not None else None)
    except Exception:
        canon.append(None)

n_valid = sum(1 for c in canon if c)
uniq = set(c for c in canon if c)
print(f'[sample] validity = {n_valid}/{N}  ({100.0*n_valid/N:.2f}%)', flush=True)
print(f'[sample] unique among valid = {len(uniq)}/{n_valid}  ({100.0*len(uniq)/max(1,n_valid):.2f}%)', flush=True)

# ---- show some examples ----
print('---- 15 example generations (canonicalized) ----', flush=True)
shown = 0
for c in canon:
    if c and shown < 15:
        print(f'  {c}')
        shown += 1

# ---- distribution-property quick check (length, % single component) ----
heavy = []
multi = 0
for c in canon:
    if not c: continue
    mol = Chem.MolFromSmiles(c)
    if mol is None: continue
    heavy.append(mol.GetNumHeavyAtoms())
    if '.' in c:
        multi += 1
if heavy:
    import statistics
    print(f'[sample] heavy-atom stats over {len(heavy)} valid: mean={statistics.mean(heavy):.1f}  median={statistics.median(heavy)}  min={min(heavy)}  max={max(heavy)}', flush=True)
    print(f'[sample] multi-component (contains "."): {multi}/{len(heavy)} ({100.0*multi/len(heavy):.2f}%)', flush=True)

# dump all samples to file for later inspection
with (out_dir / 'samples.tsv').open('w') as f:
    f.write('raw_stamp\traw_smiles\tcanonical\n')
    for toks, smi, c in zip(token_seqs, smiles, canon):
        f.write(f'{rep.detokenize(toks)}\t{smi or ""}\t{c or ""}\n')
print(f'[sample] wrote samples to {out_dir/"samples.tsv"}', flush=True)
PY

echo "[remote] end=$(date)"
