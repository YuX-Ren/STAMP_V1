#!/bin/bash
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
SAMPLES_TSV=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d512_l8_e10/eval_epoch_0005/samples/samples.tsv
PARQUET=/scratch/yuxuan.ren/data/dataset_stamp.parquet
OUT_DIR=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d512_l8_e10/eval_epoch_0005/samples

source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

python - <<PY
import sys, json, time, random, math, statistics
sys.path.insert(0, '.')
from pathlib import Path
from collections import Counter
import numpy as np
from rdkit import RDLogger, Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, Crippen, Lipinski
RDLogger.DisableLog('rdApp.*')

tsv = Path('${SAMPLES_TSV}')
rows = [ln.rstrip('\n').split('\t') for ln in tsv.open() if ln.strip()]
hdr, *rows = rows
canon = [r[2] for r in rows if len(r) >= 3 and r[2]]
print(f'[qual] loaded {len(canon)} canonical samples', flush=True)

# ---- parse canon -> Mol once ----
mols = []
for c in canon:
    m = Chem.MolFromSmiles(c)
    if m is not None: mols.append((c, m))
print(f'[qual] rdkit-parsed: {len(mols)}/{len(canon)}', flush=True)

# ---- QED ----
qeds = [QED.qed(m) for _, m in mols]
print(f'[qual] QED         : mean={np.mean(qeds):.3f}  median={np.median(qeds):.3f}  p10={np.percentile(qeds,10):.3f}  p90={np.percentile(qeds,90):.3f}', flush=True)

# ---- Drug-likeness descriptors ----
mw = [Descriptors.MolWt(m) for _, m in mols]
logp = [Crippen.MolLogP(m) for _, m in mols]
hba = [Lipinski.NumHAcceptors(m) for _, m in mols]
hbd = [Lipinski.NumHDonors(m) for _, m in mols]
rotb = [Lipinski.NumRotatableBonds(m) for _, m in mols]
rings = [Lipinski.RingCount(m) for _, m in mols]
tpsa = [Descriptors.TPSA(m) for _, m in mols]

def stat(x):
    return f'mean={np.mean(x):.2f}  median={np.median(x):.1f}  p10={np.percentile(x,10):.1f}  p90={np.percentile(x,90):.1f}'

print(f'[qual] MW          : {stat(mw)}', flush=True)
print(f'[qual] LogP        : {stat(logp)}', flush=True)
print(f'[qual] HBA         : {stat(hba)}', flush=True)
print(f'[qual] HBD         : {stat(hbd)}', flush=True)
print(f'[qual] RotBonds    : {stat(rotb)}', flush=True)
print(f'[qual] RingCount   : {stat(rings)}', flush=True)
print(f'[qual] TPSA        : {stat(tpsa)}', flush=True)

# ---- Lipinski rule-of-5 pass rate ----
def lipinski_ok(mw_, logp_, hba_, hbd_):
    return (mw_ <= 500) + (logp_ <= 5) + (hba_ <= 10) + (hbd_ <= 5)
passes = [lipinski_ok(mw[i], logp[i], hba[i], hbd[i]) for i in range(len(mols))]
print(f'[qual] Lipinski-4/4 : {sum(1 for v in passes if v==4)/len(passes)*100:.1f}%  (avg {np.mean(passes):.2f}/4 rules satisfied)', flush=True)

# ---- SA score (synthetic accessibility) ----
try:
    from rdkit.Chem import RDConfig
    import sys as _sys
    _sys.path.append(str(Path(RDConfig.RDContribDir) / 'SA_Score'))
    import sascorer
    sa = [sascorer.calculateScore(m) for _, m in mols[:500]]
    print(f'[qual] SA score (n=500): mean={np.mean(sa):.2f}  median={np.median(sa):.2f}  p10={np.percentile(sa,10):.2f}  p90={np.percentile(sa,90):.2f}  (1=easy, 10=hard)', flush=True)
except Exception as e:
    print(f'[qual] SA score skipped: {e}', flush=True)

# ---- Internal diversity (ECFP4 Tanimoto) ----
print(f'[qual] computing ECFP4 fingerprints...', flush=True)
t0 = time.time()
fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for _, m in mols]
print(f'[qual]   fp done in {time.time()-t0:.1f}s', flush=True)

# Sample 500 pairs for diversity (O(n^2) too expensive on 1024)
import itertools
rng = random.Random(0)
N = len(fps)
pair_count = 20000
sims = []
for _ in range(pair_count):
    i, j = rng.sample(range(N), 2)
    sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
print(f'[qual] InternalDiversity (1-mean_pairwise_Tanimoto over 20k random pairs) = {1-np.mean(sims):.3f}', flush=True)
print(f'[qual] mean pairwise similarity = {np.mean(sims):.3f}  (lower = more diverse)', flush=True)

# ---- Novelty vs training set ----
# Stream through parquet, collect train canonical smiles into a hash set. To keep
# memory reasonable, sample every k-th row (we have 19M train rows, sampling ~2M).
print('[qual] loading training canonical smiles from parquet (sampling)...', flush=True)
t0 = time.time()
import pyarrow.parquet as pq
pf = pq.ParquetFile('${PARQUET}')
train_canon = set()
sample_every = 10  # ~2M from 19M train rows — plenty for novelty estimation
for rg in range(pf.num_row_groups):
    tab = pf.read_row_group(rg, columns=['std_smiles', 'split_random', 'all_pass'])
    sm = tab.column('std_smiles').to_pylist()
    sp = tab.column('split_random').to_pylist()
    ap = tab.column('all_pass').to_pylist()
    for i, (s, split, apass) in enumerate(zip(sm, sp, ap)):
        if split != 'train' or not apass or s is None: continue
        if i % sample_every: continue
        # fast canonicalization-free compare: std_smiles already canonical per pipeline
        train_canon.add(s.strip())
print(f'[qual] train canonical set size = {len(train_canon):,}  (sample_every={sample_every})  in {time.time()-t0:.1f}s', flush=True)

# Canonicalize generated samples using the SAME canonical form expected to match train
# (std_smiles column is canonicalized via RDKit Chem.MolToSmiles of Mol; so should match
#  our Chem.MolToSmiles output).
novel = sum(1 for c, _ in mols if c not in train_canon)
print(f'[qual] Novelty (vs {len(train_canon):,} sampled train): {novel}/{len(mols)} = {100.0*novel/len(mols):.2f}%', flush=True)
PY
