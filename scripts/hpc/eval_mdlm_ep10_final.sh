#!/bin/bash
# Final MDLM ep10 eval: (a) N=1024 canonical @ L=16, T=1.0, k=50,
# (b) N=1024 mixed-length over L ∈ {14,15,16,17,18} (≈ AR baseline dist).
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
RUN_DIR=/scratch/yuxuan.ren/data/ckpt_stamp/stamp_mdlm_30m_d512_l8_e3
CKPT_PATH=${RUN_DIR}/checkpoints/epoch_0010/model.pt

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

run_dir = Path('/scratch/yuxuan.ren/data/ckpt_stamp/stamp_mdlm_30m_d512_l8_e3')
ckpt_path = run_dir / 'checkpoints/epoch_0010/model.pt'
_, vocab = _load_scan_cache(run_dir)
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
cfg = LMConfig(**ckpt['cfg'])
cfg.use_adaln = False
dev = torch.device('cuda:0')
model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg, bidirectional=True).to(dev)
state = ckpt['model_state']
if any(k.startswith('_orig_mod.') for k in state):
    state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
model.load_state_dict(state); model.eval()
rep = build_representation('stamp')
print(f'[final-eval] ckpt={ckpt_path} epoch={ckpt.get("epoch","?")} vocab_size={len(vocab.itos)}', flush=True)

T, TOPK, STEPS = 1.0, 50, 128

def sample_at_L(L, n):
    t0 = time.time()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        seqs = _sample_model_sequences_diffusion(
            model=model, vocab=vocab, n_samples=n, max_new_tokens=L,
            temperature=T, top_k=TOPK,
            diffusion_steps=STEPS, diffusion_block_size=L,
            diffusion_remasking='low_confidence', device=dev,
        )
    print(f'  [sample] L={L} n={n} {time.time()-t0:.1f}s', flush=True)
    return seqs

def decode(seqs):
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
    return raw_smi, canon_smi

def report(tag, raw_smi, canon_smi, N, lens):
    valid_raw   = [s for s in raw_smi   if s]
    valid_canon = [s for s in canon_smi if s]
    validity = len(valid_raw) / N
    uniq_raw   = len(set(valid_raw))   / len(valid_raw)   if valid_raw   else 0.0
    uniq_canon = len(set(valid_canon)) / len(valid_canon) if valid_canon else 0.0
    seen=set(); keep=[]
    for i, s in enumerate(raw_smi):
        if s is None or s in seen: continue
        seen.add(s); keep.append(i)
    gm_hits=0; qeds_all=[]; sas_all=[]
    for i in keep:
        m = Chem.MolFromSmiles(raw_smi[i])
        if m is None: continue
        q = QED.qed(m); sa = sascorer.calculateScore(m)
        qeds_all.append(q); sas_all.append(sa)
        if q >= 0.6 and sa <= 4.0: gm_hits += 1
    genmol = gm_hits / N * 100
    mols = [Chem.MolFromSmiles(c) for c in valid_canon]
    qeds = np.array([QED.qed(m) for m in mols]) if mols else np.array([0.0])
    sas  = np.array([sascorer.calculateScore(m) for m in mols]) if mols else np.array([10.0])
    qmask = (qeds >= 0.6) & (sas <= 4.0)
    quality_over_valid = float(qmask.mean()) * 100 if mols else 0.0
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols] if mols else []
    rng = random.Random(0); sims=[]
    if fps and len(fps) > 1:
        pairs = min(5000, len(fps) * (len(fps)-1) // 2)
        for _ in range(pairs):
            i, j = rng.sample(range(len(fps)), 2)
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    div = (1.0 - float(np.mean(sims))) if sims else 0.0
    print(f'\n=== {tag} (N={N}) ===')
    print(f'  validity     = {validity*100:.2f}%')
    print(f'  uniq raw/can = {uniq_raw*100:.2f}% / {uniq_canon*100:.2f}%')
    print(f'  quality(V)   = {quality_over_valid:.2f}%')
    print(f'  GenMol       = {genmol:.2f}%  (AR=79.64)')
    print(f'  QED/SA/div   = {float(qeds.mean()):.3f} / {float(sas.mean()):.3f} / {div:.3f}')
    print(f'  len p50/p90  = {int(np.median(lens))} / {int(np.percentile(lens,90))}', flush=True)
    return dict(tag=tag, n=N, validity=validity*100, uniq_raw=uniq_raw*100, uniq_canon=uniq_canon*100,
                quality_over_valid=quality_over_valid, genmol=genmol,
                qed_mean=float(qeds.mean()), sa_mean=float(sas.mean()), diversity=div,
                len_p50=int(np.median(lens)), len_p90=int(np.percentile(lens,90)))

# (a) canonical: N=1024 at L=16
seqs_a = sample_at_L(16, 1024)
lens_a = [len([t for t in row if t]) for row in seqs_a]
raw_a, canon_a = decode(seqs_a)
res_a = report('canonical L=16 T=1.0 K=50 N=1024', raw_a, canon_a, 1024, lens_a)

# (b) mixed-length: L ∈ {14..18}, ≈205 each, total ≈1024
mix = [(14, 205), (15, 205), (16, 205), (17, 205), (18, 204)]
all_seqs = []
for L, n in mix:
    all_seqs.extend(sample_at_L(L, n))
lens_b = [len([t for t in row if t]) for row in all_seqs]
raw_b, canon_b = decode(all_seqs)
res_b = report('mixed L∈{14..18} T=1.0 K=50 N=1024', raw_b, canon_b, 1024, lens_b)

out_path = run_dir / 'eval_ep10_final.json'
out_path.write_text(json.dumps({'canonical_L16': res_a, 'mixed_L14_18': res_b}, indent=2))
print(f'\n[final-eval] wrote {out_path}', flush=True)
PY
echo "[remote] end=$(date)"
