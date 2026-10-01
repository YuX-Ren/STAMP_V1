#!/bin/bash
# P1.B: Fixed-length random-order AR on AO-GPT.
# - Sample target length L_i from empirical distribution
# - For each sample, pick a random permutation of middle y-indices (y[0] stays first, EOS stays last)
# - Sequentially fill in that permutation order; each step's prefix is ONLY filled (real) tokens
# - No PAD pollution of context
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/random_order_ar}"
: "${N:=1024}"

echo "[remote] host=$(hostname) date=$(date)"
source /scratch/yuxuan.ren/miniconda3/bin/activate "${CONDA_ENV}"
cd "${REPO_ROOT}"
mkdir -p "${OUT_DIR}"

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

from stamp.benchmark.lm import (
    LMConfig, TinyDecoderLM, _normalize_model_type, _trim_special_tail,
    _apply_top_k_logits, sample_model_sequences, _sample_next_tokens,
)
from stamp.benchmark.representations import build_representation
from scripts.benchmark.train_ao_gpt_parquet import _load_scan_cache

ckpt_dir = Path('${CKPT_DIR}'); out_dir = Path('${OUT_DIR}')
scan_summary, vocab = _load_scan_cache(ckpt_dir)
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
print(f'[p1b] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

train_lens = np.array(scan_summary.train_lengths)
ids_lens_dist = train_lens + 2
print(f'[p1b] length dist (ids): mean={ids_lens_dist.mean():.1f}  median={np.median(ids_lens_dist)}', flush=True)


def random_order_ar(model, vocab, n_samples, lengths, *, max_len=64, temperature=1.0, top_k=50, dev=dev):
    """Fixed-length random-order AR for AO-GPT.

    Each sample gets a random permutation of middle y-indices. y[0] at perm[0],
    y[last_valid_y]=EOS is fixed and never in perm. Sequentially fill in perm
    order; each step's model forward uses only real-token context (no placeholder
    pollution), so it stays in the training distribution.
    """
    assert max_len >= int(lengths.max().item())
    L_y = max_len - 1
    rows = torch.arange(n_samples, device=dev)
    lengths = lengths.to(dev).long()
    eos_pos = lengths - 1                # ids-index of EOS
    last_valid_y = eos_pos - 1           # y-index of EOS

    # Init ids: [BOS, PAD..., EOS, PAD_tail]
    ids = torch.full((n_samples, max_len), vocab.pad_id, dtype=torch.long, device=dev)
    ids[:, 0] = vocab.bos_id
    ids[rows, eos_pos] = vocab.eos_id

    # Build per-sample permutation (y-indices to fill, in order)
    # perm[b, 0] = 0 (y[0]); perm[b, 1..mid_count] = shuffle of middle (1..last_valid_y[b] - 1)
    # After perm: last_valid_y[b] (EOS, already filled, we don't include it)
    # Beyond last_valid_y: tail pad, fill with any dummy y-index (we won't predict them)

    # Construct orders[b, :]:
    #   orders[b, 0..n_valid[b]-2] = fill order (perm)
    #   orders[b, n_valid[b]-1] = last_valid_y[b]
    #   orders[b, n_valid[b]..L_y-1] = beyond-EOS y-indices (y[last_valid_y+1..L_y-1]) in identity order
    n_valid = last_valid_y + 1  # number of valid y positions per sample

    orders = torch.zeros(n_samples, L_y, dtype=torch.long, device=dev)
    for b in range(n_samples):
        nv = int(n_valid[b].item())
        lv = int(last_valid_y[b].item())
        mid = list(range(1, lv))
        random.shuffle(mid)
        perm = [0] + mid  # length nv - 1
        beyond = list(range(lv + 1, L_y))
        row = perm + [lv] + beyond
        orders[b] = torch.tensor(row, dtype=torch.long, device=dev)

    t0 = time.time()
    # Sequentially fill perm[b, 0], perm[b, 1], ..., perm[b, nv-2]
    # Batched: at step k, each sample fills ITS perm[b, k]. Different samples may have
    # different n_valid. For samples with k >= nv[b]-1, skip (already done).
    max_steps = int((n_valid - 1).max().item())

    for k in range(max_steps):
        active = (torch.tensor(k, device=dev) < (n_valid - 1))  # (N,)
        if not bool(active.any().item()):
            break

        # target y-index per sample = orders[b, k]
        target_y = orders[rows, torch.full((n_samples,), k, device=dev, dtype=torch.long)]

        # Model forward with current ids and orders
        x_in = ids[:, :-1].contiguous()
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x_in, orders=orders).float()   # (N, L_y, V)

        # Extract logit at output position k (predicts y[orders[b,k]] = target_y)
        logit_k = logits[:, k, :].clone()                     # (N, V)

        # Suppress special tokens for middle token (never sample EOS/BOS/PAD/UNK/MASK)
        for sup in (vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.mask_id, vocab.unk_id):
            if 0 <= int(sup) < logit_k.size(-1):
                logit_k[:, int(sup)] = float('-inf')

        new_tok = _sample_next_tokens(logit_k, temperature=temperature, top_k=top_k)

        # Write back to ids[b, target_y + 1] for active samples only
        target_ids_pos = target_y + 1                          # ids-index
        current = ids[rows, target_ids_pos]
        ids[rows, target_ids_pos] = torch.where(active, new_tok, current)

    dt = time.time() - t0
    print(f'[p1b]   decode done in {dt:.1f}s ({max_steps} steps)', flush=True)

    out = []
    for row in ids[:, 1:].cpu().tolist():
        row = _trim_special_tail(row, vocab)
        out.append(vocab.decode_ids(row))
    return out


def score(seqs):
    smiles = []
    for toks in seqs:
        try: smiles.append(rep.decode(rep.detokenize(toks)))
        except Exception: smiles.append(None)
    canon = []
    for smi in smiles:
        if not smi: canon.append(None); continue
        try:
            m = Chem.MolFromSmiles(smi)
            canon.append(Chem.MolToSmiles(m) if m is not None else None)
        except Exception:
            canon.append(None)
    valid = [c for c in canon if c]
    if not valid: return dict(validity=0.,unique=0.,quality=0.,qed=0.,sa=0.,div=0., genmol=0.)
    mols = [Chem.MolFromSmiles(c) for c in valid]
    qeds = np.array([QED.qed(m) for m in mols]); sas = np.array([sascorer.calculateScore(m) for m in mols])
    qmask = (qeds > 0.6) & (sas < 4.0)
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols]
    rng = random.Random(0); sims=[]; n=len(fps); pairs=min(5000, n*(n-1)//2)
    for _ in range(pairs):
        i, j = rng.sample(range(n), 2); sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    N_total = len(canon)
    validity = len(valid) / N_total
    unique_rate = len(set(valid)) / len(valid)
    quality = float(qmask.mean())
    genmol = validity * unique_rate * quality
    return dict(
        validity=validity*100, unique=unique_rate*100,
        quality=quality*100, qed=float(qeds.mean()), sa=float(sas.mean()),
        div=(1.0 - float(np.mean(sims))) if sims else 0.0,
        genmol=genmol*100,
    )

def fmt(name, s):
    return (f'[{name:<32s}] q={s["quality"]:>6.2f}%  u={s["unique"]:>6.2f}%  v={s["validity"]:>6.2f}%  '
            f'GenMol={s["genmol"]:>6.2f}%  d={s["div"]:.3f}  QED={s["qed"]:.3f}  SA={s["sa"]:.2f}')

N = ${N}
rng = np.random.default_rng(42)
sampled_lengths_np = rng.choice(ids_lens_dist, size=N, replace=True)
sampled_lengths = torch.tensor(np.clip(sampled_lengths_np, 4, 63), dtype=torch.long, device=dev)
print(f'[p1b] sampled target lengths: mean={sampled_lengths.float().mean().item():.1f}', flush=True)

results = []

print('\n==== Ref: plain AR T=0.6 K=10 (variable length) ====', flush=True)
t0 = time.time()
with torch.no_grad():
    seqs_ar = sample_model_sequences(
        model, vocab, n_samples=N, max_new_tokens=64,
        temperature=0.6, top_k=10, device=dev, ao_decode_mode='ar',
    )
print(f'  sampled in {time.time()-t0:.1f}s', flush=True)
s = score(seqs_ar); results.append(dict(name='AR T=0.6 K=10', **s)); print(fmt('AR T=0.6 K=10', s), flush=True)

print('\n==== P1.B: Fixed-length random-order AR ====', flush=True)
for temp, topk in [(0.6, 10), (0.7, 10), (0.7, 15), (0.6, 15), (0.6, 5), (0.8, 10)]:
    random.seed(0)  # reset permutations for reproducibility
    print(f'\n-- T={temp}, K={topk} --', flush=True)
    t0 = time.time()
    seqs = random_order_ar(model, vocab, N, sampled_lengths, max_len=64, temperature=temp, top_k=topk, dev=dev)
    print(f'  total {time.time()-t0:.1f}s', flush=True)
    s = score(seqs); results.append(dict(name=f'rand-order-AR T={temp} K={topk}', **s))
    print(fmt(f'rand-order-AR T={temp} K={topk}', s), flush=True)

(out_dir / 'p1b_results.json').write_text(json.dumps(results, indent=2))
print(f'\n[p1b] wrote {out_dir}/p1b_results.json', flush=True)
PY
echo "[remote] end=$(date)"
