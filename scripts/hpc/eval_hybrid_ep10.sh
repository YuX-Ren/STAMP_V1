#!/bin/bash
# GenMol-aligned eval of hybrid-vocab AO-GPT checkpoint.
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${RUN_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/stamp_hybrid_30m_d512_l8_e10_cut5k}"
: "${EPOCH:=0009}"
: "${CKPT_PATH:=${RUN_DIR}/checkpoints/epoch_${EPOCH}/model.pt}"
: "${N:=1024}"
: "${MAX_NEW:=64}"
: "${TEMPERATURE:=1.0}"
: "${TOP_K:=50}"
: "${TOP_P:=0.0}"
: "${TAG:=epoch_${EPOCH}}"

echo "[remote] host=$(hostname) date=$(date)"
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

python - <<PY
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

from stamp.benchmark.lm import LMConfig, TinyDecoderLM
from stamp.benchmark.hybrid_vocab import HybridVocab, is_stamp_structural, SMILES_RE
from stamp.benchmark.representations import build_representation

run_dir = Path('${RUN_DIR}')
ckpt_path = Path('${CKPT_PATH}')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
cfg = LMConfig(**ckpt['cfg']); cfg.use_adaln = True

# Prefer vocab dump alongside the run; fall back to the fresh build.
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
print(f'[hybrid-eval] ckpt={ckpt_path} epoch={ckpt.get("epoch","?")} vocab_size={len(vocab.itos)} atomic={len(vocab.atomic_motifs)}', flush=True)

BOS, EOS, MASK, PAD, UNK = vocab.bos_id, vocab.eos_id, vocab.mask_id, vocab.pad_id, vocab.unk_id

atomic_set = set(vocab.atomic_motifs.values())
struct_set = {vocab.stoi[t] for t in vocab.itos if is_stamp_structural(t)}
special_set = {BOS, EOS, MASK, PAD, UNK}
suppress_ids = {PAD, BOS, MASK, UNK}

@torch.no_grad()
def ar_sample(n_samples, max_new, temperature, top_k, top_p, device):
    x = torch.full((n_samples, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(n_samples, dtype=torch.bool, device=device)
    for step in range(max_new):
        ctx = x[:, -cfg.max_seq_len:]
        orders = torch.arange(ctx.size(1), device=device).unsqueeze(0).expand(ctx.size(0), -1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(ctx, orders=orders)[:, -1, :]
        # Suppress specials (keep EOS).
        for sid in suppress_ids:
            logits[:, sid] = float('-inf')
        if top_k > 0:
            tk = min(top_k, logits.size(-1))
            top_vals, _ = torch.topk(logits, tk, dim=-1)
            thresh = top_vals[:, -1:].expand_as(logits)
            logits = torch.where(logits < thresh, torch.full_like(logits, float('-inf')), logits)
        # nucleus (top-p)
        if top_p > 0.0 and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits / max(1e-4, temperature), dim=-1)
            cum = torch.cumsum(sorted_probs, dim=-1)
            remove = cum > top_p
            # Keep the first token that crosses the threshold.
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(remove, float('-inf'))
            logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)
        probs = torch.softmax(logits / max(1e-4, temperature), dim=-1)
        g = torch.Generator(device=device).manual_seed(42 + step)
        nxt = torch.multinomial(probs, 1, generator=g).squeeze(-1)
        nxt = torch.where(finished, torch.full_like(nxt, EOS), nxt)
        x = torch.cat([x, nxt.unsqueeze(1)], dim=1)
        finished = finished | (nxt == EOS)
        if bool(finished.all().item()):
            break
    return x

def decode_row(ids):
    """Walk token ids. Only STAMP structural tokens ([J_*], [B_*], [S_*],
    [END], [BACK]) act as motif boundaries. Atomic motifs and SMILES chars
    are both appended to the current motif buffer — concatenation of any
    valid SMILES fragments stays a valid SMILES."""
    out = []
    buf = []
    for i in ids:
        if i in special_set:
            continue
        tok = vocab.itos[i]
        if i in struct_set:
            if buf:
                out.append(''.join(buf)); buf = []
            out.append(tok)
        else:
            # Atomic motif string OR SMILES char — both are SMILES fragments.
            buf.append(tok)
    if buf:
        out.append(''.join(buf))
    return out

N, MAX_NEW, T, TOPK, TOPP = ${N}, ${MAX_NEW}, ${TEMPERATURE}, ${TOP_K}, ${TOP_P}

t0 = time.time()
x = ar_sample(N, MAX_NEW, T, TOPK, TOPP, dev)
dt = time.time() - t0
print(f'[hybrid-eval] sampled {N} in {dt:.1f}s  ({dt/N*1000:.1f} ms/sample)', flush=True)

ids_list = x[:, 1:].cpu().tolist()  # strip BOS
# Cut at EOS / PAD
def trim(row):
    for tid in (EOS, PAD):
        if tid in row:
            row = row[:row.index(tid)]
            break
    return row

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
sas  = np.array([sascorer.calculateScore(m) for m in mols]) if mols else np.array([10.0])
qmask = (qeds >= 0.6) & (sas <= 4.0)
quality_over_valid = float(qmask.mean()) * 100 if mols else 0.0

fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024) for m in mols] if mols else []
rng = random.Random(0); sims = []
if fps and len(fps) > 1:
    pairs = min(5000, len(fps) * (len(fps)-1) // 2)
    for _ in range(pairs):
        i, j = rng.sample(range(len(fps)), 2)
        sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
div = (1.0 - float(np.mean(sims))) if sims else 0.0

print()
print(f'[hybrid-eval] seq_len p50={int(np.median(seq_lens))} p90={int(np.percentile(seq_lens,90))} max={int(max(seq_lens))}')
print(f'[hybrid-eval] validity       = {validity*100:.2f}%')
print(f'[hybrid-eval] uniqueness raw = {uniq_raw*100:.2f}%   canon = {uniq_canon*100:.2f}%')
print(f'[hybrid-eval] quality (valid)= {quality_over_valid:.2f}%')
print(f'[hybrid-eval] GenMol score   = {genmol_score:.2f}%  (vs AR baseline 79.64%)')
print(f'[hybrid-eval] QED mean={float(qeds.mean()):.3f}  SA mean={float(sas.mean()):.3f}  diversity={div:.3f}')

print()
print('[hybrid-eval] first 5 valid canonical SMILES:')
for s in valid_canon[:5]:
    if s: print(f'  {s}')

out = dict(
    ckpt=str(ckpt_path), n=total, dt_s=dt,
    validity=validity*100, uniq_raw=uniq_raw*100, uniq_canon=uniq_canon*100,
    quality_over_valid=quality_over_valid, genmol_score=genmol_score,
    qed_mean=float(qeds.mean()), sa_mean=float(sas.mean()), diversity=div,
    len_p50=int(np.median(seq_lens)), len_p90=int(np.percentile(seq_lens,90)), len_max=int(max(seq_lens)),
)
out_path = run_dir / f'eval_${TAG}.json'
out_path.write_text(json.dumps(out, indent=2))
print(f'\n[hybrid-eval] wrote {out_path}', flush=True)
PY
echo "[remote] end=$(date)"
