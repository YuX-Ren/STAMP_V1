#!/bin/bash
# P2 #3: Tree-level mask-and-regenerate. STAMP tree parser identifies subtrees;
# AO-GPT refills a leaf subtree with K candidates; keep best by QED.
# STAMP-exclusive (needs canonical group parse) + AO-exclusive (multi-token refill).
set -u
REPO_ROOT=/home/svu/yuxuan.ren/STAMP_V1.5
CONDA_ENV=stamp-train-rdkit-2025_09_4
: "${CKPT_DIR:=/scratch/yuxuan.ren/data/ckpt_stamp/ao_gpt_30m_d768_l12_e10/eval_epoch_0009}"
: "${OUT_DIR:=${CKPT_DIR}/phase2_treemask}"
: "${N:=1024}"
: "${K_CAND:=5}"
: "${R_ITER:=1}"

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
    sample_model_sequences, _sample_next_tokens,
)
from stamp.benchmark.representations import build_representation
from stamp.benchmark.mcts_unified.subtree import parse_terminal_sequence, splice_span
from scripts.benchmark.train_ao_gpt_parquet import _load_scan_cache

ckpt_dir = Path('${CKPT_DIR}'); out_dir = Path('${OUT_DIR}')
_, vocab = _load_scan_cache(ckpt_dir)
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
stamp_codec = rep.codec
print(f'[treemask] model epoch={ckpt.get("epoch","?")}  vocab={len(vocab.itos)}', flush=True)

N = ${N}; K_CAND = ${K_CAND}; R_ITER = ${R_ITER}

def smi_mol_from_tokens(toks):
    try:
        smi = rep.decode(rep.detokenize(toks))
        if smi:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                return Chem.MolToSmiles(m), m
    except Exception:
        pass
    return None, None

def score_qed_sa(m):
    if m is None: return 0.0, 10.0
    try: return float(QED.qed(m)), float(sascorer.calculateScore(m))
    except Exception: return 0.0, 10.0

# ---- Draft pool (same as our baseline) ----
t0 = time.time()
with torch.no_grad():
    token_seqs = sample_model_sequences(
        model, vocab, n_samples=N, max_new_tokens=64,
        temperature=0.6, top_k=10, device=dev, ao_decode_mode='ar',
    )
print(f'[treemask] generated {N} drafts in {time.time()-t0:.1f}s', flush=True)

# ---- Pre-compute baseline metrics ----
base_canon = [None]*N; base_qed = np.zeros(N); base_sa = np.full(N, 10.0)
for i, toks in enumerate(token_seqs):
    c, m = smi_mol_from_tokens(toks)
    if c:
        base_canon[i] = c
        q, s = score_qed_sa(m)
        base_qed[i] = q; base_sa[i] = s
print(f'[treemask] baseline: q={(((base_qed>0.6) & (base_sa<4)).sum()/N)*100:.2f}%  '
      f'QED={base_qed.mean():.3f}  SA={base_sa.mean():.3f}', flush=True)

# ---- Parse each draft's STAMP tree and pick worst subtree ----
# Heuristic: for each leaf subtree (not root, smallest span), compute QED
# contribution = full_QED - QED_without_subtree. Pick the subtree with MIN
# contribution (most harmful or least helpful) as mask target.
t0 = time.time()
parse_cache = []      # per-sample: (parsed, target_span) or None
n_parsed = 0; n_target = 0
for i, toks in enumerate(token_seqs):
    parsed = parse_terminal_sequence(toks, stamp_codec)
    if parsed is None:
        parse_cache.append(None); continue
    n_parsed += 1
    # Find leaf subtrees (group_count == 1), not root, span_len >= 2
    leaf_spans = [s for s in parsed.subtree_spans
                  if (not s.is_root) and s.span_len >= 2 and s.group_count == 1]
    if not leaf_spans:
        parse_cache.append(None); continue
    # Evaluate contribution: remove each leaf's tokens, re-decode, compute ΔQED
    best_span = None; worst_contrib = float('inf')
    for sp in leaf_spans:
        spliced = parsed.flat_tokens[:sp.token_start] + parsed.flat_tokens[sp.token_end:]
        if not spliced: continue
        _, m2 = smi_mol_from_tokens(spliced)
        if m2 is None:
            q2 = 0.0
        else:
            q2 = score_qed_sa(m2)[0]
        contrib = base_qed[i] - q2
        if contrib < worst_contrib:
            worst_contrib = contrib; best_span = sp
    if best_span is not None:
        parse_cache.append((parsed, best_span))
        n_target += 1
    else:
        parse_cache.append(None)
print(f'[treemask] parsed {n_parsed}/{N}  leaf-target found {n_target}/{N}  '
      f'in {time.time()-t0:.1f}s', flush=True)

# ---- For each draft with a target, generate K AO refills and pick best by QED ----
# Refill strategy: given observed tokens = flat_tokens[:start] + flat_tokens[end:],
# and mask = positions [start..end), build AO order placing masked positions at
# order's tail (after observed). Sequentially sample each masked position with
# temperature + softmax (no argmax).

def ao_refill_span(parsed, span, K=K_CAND, temperature=1.0, top_k=30):
    """Generate K candidate replacements for tokens[span.start:span.end].
    Returns list of K candidate full token lists (canon form)."""
    tokens = parsed.flat_tokens
    L = len(tokens)
    mask_len = span.token_end - span.token_start
    if mask_len < 1 or mask_len > 8:
        return []
    # Encode tokens to ids; prepend BOS (ids = [BOS, tokens, EOS])
    # We only need the x/y AR format for model feed. Construct fully:
    # ids = [BOS] + tokens + [EOS], length L+2
    ids = [vocab.bos_id] + vocab.encode_tokens(tokens) + [vocab.eos_id]
    ids_len = len(ids)
    # y-indices 0..L (y[0]=ids[1]=first token, y[L]=EOS)
    # Mask y-indices correspond to tokens span [span.token_start..span.token_end)
    # which maps to y[span.token_start..span.token_end)
    mask_ys = list(range(span.token_start, span.token_end))
    # Construct candidate set of K full sequences
    candidates = []
    for k_i in range(K):
        cur_ids = list(ids)
        # Zero out masked ids first (use PAD as placeholder; model won't attend because
        # AO order places these at tail — they're never in any masked position's prefix
        # when we sample them IN ORDER.)
        for m_y in mask_ys:
            cur_ids[m_y + 1] = vocab.pad_id   # ids[m_y+1] corresponds to y[m_y]
        # Fill masked positions one at a time (AR within the masked span)
        for mask_i, my in enumerate(mask_ys):
            # Build AO order: all other y-indices in natural order + masked positions at tail
            # Actually simpler: put observed in their natural order, mask positions at end,
            # but within mask we process sequentially: this mask at the position
            # right after all observed (and previously-filled masks).
            other = [y for y in range(ids_len - 1) if (y not in mask_ys) or (y in mask_ys[:mask_i])]
            order_yidx = other + [my] + mask_ys[mask_i+1:]
            # x = ids[:-1], we need orders over y-space (length ids_len-1)
            x = torch.tensor(cur_ids[:-1], dtype=torch.long, device=dev).unsqueeze(0)
            orders_t = torch.tensor(order_yidx, dtype=torch.long, device=dev).unsqueeze(0)
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(x, orders=orders_t).float()
            # target is at order position = len(other)
            target_pos = len(other)
            logit = logits[0, target_pos, :].clone()   # (V,)
            # Suppress specials
            for sup in (vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.mask_id, vocab.unk_id):
                if 0 <= int(sup) < logit.size(0):
                    logit[int(sup)] = float('-inf')
            new_tok = _sample_next_tokens(logit.unsqueeze(0), temperature=temperature,
                                          top_k=top_k).item()
            cur_ids[my + 1] = int(new_tok)
        # Reconstruct token string
        new_tokens = vocab.decode_ids(cur_ids[1:-1])  # strip BOS, EOS
        candidates.append(new_tokens)
    return candidates

# ---- Main refine loop ----
t0 = time.time()
refined_tokens = list(token_seqs)  # copy
refined_qed = base_qed.copy(); refined_sa = base_sa.copy(); refined_canon = list(base_canon)
n_improved = 0; n_attempted = 0
for iter_i in range(R_ITER):
    for i in range(N):
        if parse_cache[i] is None: continue
        parsed, span = parse_cache[i]
        n_attempted += 1
        cands = ao_refill_span(parsed, span, K=K_CAND, temperature=1.0, top_k=30)
        best_qed = base_qed[i]; best_sa = base_sa[i]; best_toks = refined_tokens[i]; best_canon = refined_canon[i]
        for cand_toks in cands:
            c2, m2 = smi_mol_from_tokens(cand_toks)
            if m2 is None: continue
            q2, s2 = score_qed_sa(m2)
            # Accept if (qed > 0.6 and sa < 4) and improves over current
            if q2 > best_qed and s2 < 4.0:
                best_qed = q2; best_sa = s2; best_toks = cand_toks; best_canon = c2
        if best_qed > base_qed[i] + 1e-6:
            refined_tokens[i] = best_toks
            refined_qed[i] = best_qed
            refined_sa[i]  = best_sa
            refined_canon[i] = best_canon
            n_improved += 1
        if (i+1) % 256 == 0:
            print(f'[treemask]   iter {iter_i+1} progress: {i+1}/{N}  improved={n_improved}', flush=True)
print(f'[treemask] refine: {n_improved}/{n_attempted} samples improved  '
      f'in {time.time()-t0:.1f}s', flush=True)

def score_token_seqs(seqs):
    canon = []
    for toks in seqs:
        try:
            smi = rep.decode(rep.detokenize(toks))
            if smi:
                m = Chem.MolFromSmiles(smi)
                canon.append(Chem.MolToSmiles(m) if m is not None else None)
            else:
                canon.append(None)
        except Exception:
            canon.append(None)
    valid = [c for c in canon if c]
    if not valid: return dict(validity=0.,unique=0.,quality=0.,qed=0.,sa=0.,div=0.,genmol=0.)
    mols = [Chem.MolFromSmiles(c) for c in valid]
    qeds = np.array([QED.qed(m) for m in mols])
    sas  = np.array([sascorer.calculateScore(m) for m in mols])
    qmask = (qeds > 0.6) & (sas < 4.0)
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024) for m in mols]
    rng = random.Random(0); sims=[]; n=len(fps); pairs=min(5000, n*(n-1)//2)
    for _ in range(pairs):
        i, j = rng.sample(range(n), 2); sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    total = len(canon); validity = len(valid)/total
    unique_rate = len(set(valid))/len(valid); quality = float(qmask.mean())
    return dict(validity=validity*100, unique=unique_rate*100, quality=quality*100,
                qed=float(qeds.mean()), sa=float(sas.mean()),
                div=(1.0 - float(np.mean(sims))) if sims else 0.0,
                genmol=validity*unique_rate*quality*100)

def fmt(name, s):
    return (f'[{name:<36s}] q={s["quality"]:>6.2f}%  u={s["unique"]:>6.2f}%  '
            f'GenMol={s["genmol"]:>6.2f}%  d={s["div"]:.3f}  QED={s["qed"]:.3f}  SA={s["sa"]:.2f}')

print('\n---- Results ----', flush=True)
s_base = score_token_seqs(token_seqs)
print(fmt('baseline AR', s_base))
s_ref = score_token_seqs(refined_tokens)
print(fmt(f'tree mask-regen (K={K_CAND}, R={R_ITER})', s_ref))

res = dict(
    baseline=s_base,
    refined=s_ref,
    n_attempted=n_attempted,
    n_improved=n_improved,
    K=K_CAND, R=R_ITER,
)
(out_dir / 'treemask_results.json').write_text(json.dumps(res, indent=2))
print(f'\n[treemask] wrote {out_dir}/treemask_results.json', flush=True)
PY
echo "[remote] end=$(date)"
