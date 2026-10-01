#!/bin/bash
# Eval Eso-LM (alpha_0=1) ep8 with AR-style sampling (sequential unmask
# under a random permutation per sample, matching training distribution).
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
import torch.nn.functional as F
from rdkit import RDLogger, Chem, DataStructs
from rdkit.Chem import AllChem, QED, RDConfig
sys.path.append(str(Path(RDConfig.RDContribDir) / 'SA_Score'))
import sascorer
RDLogger.DisableLog('rdApp.*')

from stamp.benchmark.lm import (
    LMConfig, TinyDecoderLM, TokenVocab,
)
from stamp.benchmark.representations import build_representation
from scripts.benchmark.train_ao_gpt_parquet import _load_scan_cache

run_dir = Path('/scratch/yuxuan.ren/data/ckpt_stamp/stamp_eso_30m_d512_l8_e10')
ckpt_path = run_dir / 'checkpoints/epoch_0008/model.pt'
_, vocab = _load_scan_cache(run_dir)
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
cfg = LMConfig(**ckpt['cfg'])
cfg.use_adaln = True
dev = torch.device('cuda:0')
model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg, bidirectional=False).to(dev)
state = ckpt['model_state']
if any(k.startswith('_orig_mod.') for k in state):
    state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
model.load_state_dict(state); model.eval()
rep = build_representation('stamp')
print(f'[eso-ar] ckpt={ckpt_path} epoch={ckpt.get("epoch","?")}', flush=True)

BOS, EOS, MASK, PAD = vocab.bos_id, vocab.eos_id, vocab.mask_id, vocab.pad_id

@torch.no_grad()
def sample_eso_ar(n, L, temperature=1.0, top_k=50, seed=0):
    """Random-order AR sampling matching training.

    Canvas = [BOS] + [MASK]*(L-1) + [EOS_slot]. BOS fixed at shuffled pos 0,
    EOS pinned at last shuffled pos; content positions 1..L-1 filled in a
    random order, one position per forward pass.
    """
    T = L + 1  # BOS + L content slots (last is EOS slot)
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.full((n, T), MASK, dtype=torch.long, device=dev)
    x[:, 0] = BOS
    x[:, -1] = EOS  # pin EOS at last shuffled pos — model learned this

    # Per-sample random permutation: [0, random(1..T-2), T-1]
    orders = torch.zeros((n, T), dtype=torch.long, device=dev)
    for b in range(n):
        middle = list(range(1, T-1))
        random.Random(seed * 97 + b).shuffle(middle)
        orders[b] = torch.tensor([0] + middle + [T-1], dtype=torch.long)

    # Fill from step k=1 to k=T-2 (BOS at k=0, EOS at k=T-1 both fixed).
    for k in range(1, T-1):
        logits = model(x, orders=orders)  # [B, T, V]
        step_logits = logits[:, k, :]
        # Suppress PAD/BOS/MASK/UNK/EOS (EOS pinned; don't let it appear mid-sequence).
        for sid in (PAD, BOS, MASK, vocab.unk_id, EOS):
            if 0 <= int(sid) < step_logits.size(-1):
                step_logits[:, int(sid)] = float('-inf')
        # top-k + temperature
        if top_k > 0:
            tk = min(top_k, step_logits.size(-1))
            top_vals, _ = torch.topk(step_logits, tk, dim=-1)
            thresh = top_vals[:, -1:].expand_as(step_logits)
            step_logits = torch.where(step_logits < thresh, torch.full_like(step_logits, float('-inf')), step_logits)
        probs = torch.softmax(step_logits / max(1e-4, temperature), dim=-1)
        sampled = torch.multinomial(probs, num_samples=1, generator=g).squeeze(-1)  # [B]
        # Fill at the original position orders[:, k]
        x[torch.arange(n, device=dev), orders[:, k]] = sampled
    return x

def decode_rows(x):
    ids = x[:, 1:].detach().cpu().tolist()
    out, lens = [], []
    for row in ids:
        cut = len(row)
        for tid in (EOS, PAD):
            if tid in row:
                cut = min(cut, row.index(tid))
        row = row[:cut]
        lens.append(len(row))
        out.append(vocab.decode_ids(row))
    return out, lens

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
    print(f'{tag:<20} V={validity*100:5.1f}%  U={uniq_raw*100:5.1f}%  Q/V={qov:5.1f}%  GenMol={gm:5.1f}%  QED={float(qeds.mean()):.3f}  SA={float(sas.mean()):.2f}  len p50={int(np.median(lens))}', flush=True)
    return dict(validity=validity*100, uniq_raw=uniq_raw*100, quality_over_valid=qov, genmol=gm,
                qed=float(qeds.mean()), sa=float(sas.mean()), len_p50=int(np.median(lens)))

N = 512
results = {}
for L in (14, 16, 18, 20):
    t0 = time.time()
    x = sample_eso_ar(N, L, temperature=1.0, top_k=50, seed=0)
    dt = time.time() - t0
    seqs, lens = decode_rows(x)
    raw, can = decode(seqs)
    results[f'L{L}'] = report(f'L={L} ({dt:.1f}s)', raw, can, N, lens)

(run_dir / 'eval_ep8_ar.json').write_text(json.dumps(results, indent=2))
print('first canonical SMILES:')
for toks in seqs[:3]:
    try:
        text = rep.detokenize(toks)
        mol = rep.codec.decode_stamp_to_mol(text)
        print(f'  {Chem.MolToSmiles(mol, canonical=True) if mol else None}')
    except Exception:
        print('  <err>')
PY
echo "[remote] end=$(date)"
