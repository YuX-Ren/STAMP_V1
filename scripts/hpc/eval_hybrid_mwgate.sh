#!/bin/bash
# MW-gated hybrid eval: cumulative token-MW tracking + per-step logit masking
# for tokens whose addition would exceed (TARGET_MW + TOLERANCE).
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${RUN_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/stamp_hybrid_30m_d512_l8_e10_cut5k}"
: "${EPOCH:=0009}"
: "${CKPT_PATH:=${RUN_DIR}/checkpoints/epoch_${EPOCH}/model.pt}"
: "${N:=1024}"
: "${MAX_NEW:=64}"
: "${TEMPERATURE:=0.95}"
: "${TOP_K:=0}"
: "${TOP_P:=0.90}"
: "${TARGET_MW:=368}"
: "${MW_TOLERANCE:=30}"
: "${TAG:=epoch_${EPOCH}_mw${TARGET_MW}}"

echo "[remote] host=$(hostname) date=$(date)"
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

python - <<PY
import sys, json, time, random, re
sys.path.insert(0, '.')
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import RDLogger, Chem, DataStructs
from rdkit.Chem import AllChem, QED, RDConfig, Descriptors
sys.path.append(str(Path(RDConfig.RDContribDir) / 'SA_Score'))
import sascorer
RDLogger.DisableLog('rdApp.*')

from stamp.benchmark.lm import LMConfig, TinyDecoderLM
from stamp.benchmark.hybrid_vocab import HybridVocab, is_stamp_structural
from stamp.benchmark.representations import build_representation

run_dir = Path('${RUN_DIR}')
ckpt_path = Path('${CKPT_PATH}')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
cfg = LMConfig(**ckpt['cfg']); cfg.use_adaln = True

vocab_json = run_dir / 'hybrid_vocab.json'
if vocab_json.exists():
    vocab = HybridVocab.load(vocab_json)
else:
    vocab = HybridVocab.build('/scratch/yuxuan.ren/data/v3_cm_union.txt', freq_cutoff=5000)

dev = torch.device('cuda:0')
model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg, bidirectional=False).to(dev)
state = ckpt['model_state']
if any(k.startswith('_orig_mod.') for k in state):
    state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
model.load_state_dict(state); model.eval()
rep = build_representation('stamp')

BOS, EOS, MASK, PAD, UNK = vocab.bos_id, vocab.eos_id, vocab.mask_id, vocab.pad_id, vocab.unk_id
atomic_set = set(vocab.atomic_motifs.values())
struct_set = {vocab.stoi[t] for t in vocab.itos if is_stamp_structural(t)}
special_set = {BOS, EOS, MASK, PAD, UNK}
suppress_ids = {PAD, BOS, MASK, UNK}

# Precompute per-token MW contribution.
ATOM_MW = {'C': 12.01, 'N': 14.01, 'O': 16.00, 'S': 32.07, 'F': 19.00,
           'Cl': 35.45, 'Br': 79.90, 'I': 126.90, 'P': 30.97, 'B': 10.81}
BRACKET_RE = re.compile(r'\[([A-Za-z]+)')

def char_token_mw(tok: str) -> float:
    if tok.startswith('['):
        m = BRACKET_RE.match(tok)
        if m:
            atom = m.group(1)
            for a, w in sorted(ATOM_MW.items(), key=lambda x: -len(x[0])):
                if atom.startswith(a) or atom.startswith(a.capitalize()):
                    return w
            return 0.0
        return 0.0
    if tok in ATOM_MW:
        return ATOM_MW[tok]
    return 0.0  # bonds, digits, parens

token_mw = torch.zeros(len(vocab.itos), dtype=torch.float32, device=dev)
for tok, tid in vocab.stoi.items():
    if tid in atomic_set:
        m = Chem.MolFromSmiles(tok)
        token_mw[tid] = float(Descriptors.ExactMolWt(m)) if m else 0.0
    elif tid in struct_set or tid in special_set:
        token_mw[tid] = 0.0
    else:
        token_mw[tid] = char_token_mw(tok)

TARGET_MW = float(${TARGET_MW})
MW_TOLERANCE = float(${MW_TOLERANCE})
print(f'[mw-gate] TARGET_MW={TARGET_MW} TOLERANCE={MW_TOLERANCE} | atomic MW range: min={float(token_mw[list(atomic_set)].min()):.1f} max={float(token_mw[list(atomic_set)].max()):.1f}', flush=True)

@torch.no_grad()
def ar_sample_mw(n_samples, max_new, temperature, top_k, top_p, device):
    x = torch.full((n_samples, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(n_samples, dtype=torch.bool, device=device)
    mw_so_far = torch.zeros(n_samples, dtype=torch.float32, device=device)
    for step in range(max_new):
        ctx = x[:, -cfg.max_seq_len:]
        orders = torch.arange(ctx.size(1), device=device).unsqueeze(0).expand(ctx.size(0), -1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(ctx, orders=orders)[:, -1, :].float()
        # Specials.
        for sid in suppress_ids:
            logits[:, sid] = float('-inf')
        # MW gate: mask tokens that would push over TARGET_MW + TOLERANCE.
        budget = (TARGET_MW + MW_TOLERANCE) - mw_so_far  # [B]
        would_exceed = token_mw.unsqueeze(0) > budget.unsqueeze(1)  # [B, V]
        logits = logits.masked_fill(would_exceed, float('-inf'))
        # top-k
        if top_k > 0:
            tk = min(top_k, logits.size(-1))
            top_vals, _ = torch.topk(logits, tk, dim=-1)
            thresh = top_vals[:, -1:].expand_as(logits)
            logits = torch.where(logits < thresh, torch.full_like(logits, float('-inf')), logits)
        # top-p
        if top_p > 0.0 and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits / max(1e-4, temperature), dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            remove = cum > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
            logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
        probs = torch.softmax(logits / max(1e-4, temperature), dim=-1)
        # Handle -inf rows (no valid token) -> force EOS.
        all_bad = ~torch.isfinite(probs).any(dim=-1)
        if bool(all_bad.any().item()):
            probs = torch.where(all_bad.unsqueeze(1).expand_as(probs), torch.zeros_like(probs), probs)
            probs[all_bad, EOS] = 1.0
        g = torch.Generator(device=device).manual_seed(42 + step)
        nxt = torch.multinomial(probs, 1, generator=g).squeeze(-1)
        nxt = torch.where(finished, torch.full_like(nxt, EOS), nxt)
        # Update cumulative MW.
        mw_so_far = mw_so_far + token_mw[nxt]
        x = torch.cat([x, nxt.unsqueeze(1)], dim=1)
        finished = finished | (nxt == EOS)
        if bool(finished.all().item()):
            break
    return x

def decode_row(ids):
    out, buf = [], []
    for i in ids:
        if i in special_set:
            continue
        tok = vocab.itos[i]
        if i in struct_set:
            if buf:
                out.append(''.join(buf)); buf = []
            out.append(tok)
        else:
            buf.append(tok)
    if buf:
        out.append(''.join(buf))
    return out

def trim(row):
    for t in (EOS, PAD):
        if t in row:
            return row[:row.index(t)]
    return row

N, MAX_NEW, T_, TOPK, TOPP = ${N}, ${MAX_NEW}, ${TEMPERATURE}, ${TOP_K}, ${TOP_P}
t0 = time.time()
x = ar_sample_mw(N, MAX_NEW, T_, TOPK, TOPP, dev)
dt = time.time() - t0
print(f'[mw-gate] sampled {N} in {dt:.1f}s', flush=True)

ids_list = x[:, 1:].cpu().tolist()
canon_smi, raw_smi, seq_lens = [], [], []
for row in ids_list:
    row = trim(row)
    seq_lens.append(len(row))
    toks = decode_row(row)
    try:
        text = rep.detokenize(toks)
        mol = rep.codec.decode_stamp_to_mol(text)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            raw_smi.append(None); canon_smi.append(None); continue
        raw = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
        can = Chem.MolToSmiles(mol, canonical=True,  isomericSmiles=True)
        raw_smi.append(raw if raw else None); canon_smi.append(can if can else None)
    except Exception:
        raw_smi.append(None); canon_smi.append(None)

total = N
valid_raw = [s for s in raw_smi if s]
valid_canon = [s for s in canon_smi if s]
validity = len(valid_raw) / total
uniq_raw = len(set(valid_raw)) / len(valid_raw) if valid_raw else 0.0
seen = set(); keep = []
for i, s in enumerate(raw_smi):
    if s is None or s in seen: continue
    seen.add(s); keep.append(i)
gm_hits = 0
for i in keep:
    m = Chem.MolFromSmiles(raw_smi[i])
    if m is None: continue
    if QED.qed(m) >= 0.6 and sascorer.calculateScore(m) <= 4.0: gm_hits += 1
genmol_score = gm_hits / total * 100
mols = [Chem.MolFromSmiles(c) for c in valid_canon if c]
qeds = np.array([QED.qed(m) for m in mols]) if mols else np.array([0.0])
sas = np.array([sascorer.calculateScore(m) for m in mols]) if mols else np.array([10.0])
mw_dist = np.array([Descriptors.ExactMolWt(m) for m in mols]) if mols else np.array([0.0])
qmask = (qeds >= 0.6) & (sas <= 4.0)
quality_over_valid = float(qmask.mean()) * 100 if mols else 0.0

print(f'[mw-gate] validity     = {validity*100:.2f}%')
print(f'[mw-gate] uniqueness   = {uniq_raw*100:.2f}%')
print(f'[mw-gate] quality/V    = {quality_over_valid:.2f}%')
print(f'[mw-gate] GenMol       = {genmol_score:.2f}%  (vs AR baseline 79.64%)')
print(f'[mw-gate] QED={float(qeds.mean()):.3f}  SA={float(sas.mean()):.3f}  MW={float(mw_dist.mean()):.1f} (p90={float(np.percentile(mw_dist,90)):.1f})')
print(f'[mw-gate] len p50={int(np.median(seq_lens))} p90={int(np.percentile(seq_lens,90))}')
PY
echo "[remote] end=$(date)"
