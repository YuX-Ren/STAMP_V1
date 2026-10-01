#!/usr/bin/env python3
"""RePRA-style robustness analysis for pretrained STAMP/SMILES encoders.

Generalises Activity Cliffs (AC) and Scaffold Hopping (SH) from the
structure-activity context to the representation-property context, following
Zhang et al. *Can pretrained models really learn better molecular
representations for AI-aided drug discovery?* (JCIM 2024,
DOI: 10.1021/acs.jcim.3c01707; reference code:
https://github.com/ZiqiaoZhang/RePRA).

Implementation matches the paper's metric protocol:
    z-distance:  MinMax-normalized Euclidean distance, dist ∈ [0,1]
    sim:         1 - dist                              (Linear F_{d→s})
    y-distance:  MinMax-normalized Euclidean distance, diff ∈ [0,1]

Thresholds (per task):
    Classification:
        eps = 0.5
        delta_AC = mean(dist | y_i == y_j)
        delta_SH = mean(dist | y_i != y_j)
    Regression:
        d_m = median(dist)
        delta_AC = mean(dist | dist <  d_m)
        delta_SH = mean(dist | dist >= d_m)
        eps_AC   = mean(diff | dist <  d_m)
        eps_SH   = mean(diff | dist >= d_m)

Detection (paper code):
    AC noise (cliff):  (diff > eps[_AC]) AND (sim > 1 - delta_AC)
    SH noise (hop):    (diff < eps[_SH]) AND (sim < 1 - delta_SH)
    deviation = min(|diff - eps|, |sim - threshold|)

Scores:
    s_AD = e * dbar
        e   = 2 * (C_AC + C_SH) / total_pairs
        dbar = (sum of deviations of all noise pairs) / (C_AC + C_SH)
    s_IR = C_AC_PTM / C_AC_FP  +  C_SH_PTM / C_SH_FP
        ECFP self-comparison: s_IR = 2  (baseline)
        s_IR < 2  ⇒ PTM better than ECFP
        s_IR > 2  ⇒ PTM worse than ECFP
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.roberta_property import (  # noqa: E402
    load_pretrained_roberta,
    load_moleculenet_task,
    _MOLECULENET_TASKS,
)
from stamp.benchmark.common import canonicalize_smiles  # noqa: E402


# Paper's RePRA uses ALL label columns for multi-task datasets, not the single
# target_column we use for fine-tuning. This dict mirrors the paper's config
# (Utils.py from ZiqiaoZhang/RePRA).
_REPRA_TASKS_WANTED = {
    "clintox": ["FDA_APPROVED", "CT_TOX"],
    "sider": [
        "Hepatobiliary disorders",
        "Metabolism and nutrition disorders",
        "Product issues",
        "Eye disorders",
        "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders",
        "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders",
        "Surgical and medical procedures",
        "Vascular disorders",
        "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders",
        "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders",
        "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    ],
    "tox21": [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
    ],
}


def load_multilabel_full(
    task: str,
    data_dir: str,
) -> Tuple[List[str], np.ndarray, bool]:
    """Load deepchem MoleculeNet **whole dataset** (train+valid+test concatenated)
    with full multi-label vector. RePRA paper uses the entire dataset, not a
    held-out split, so we match that protocol here.

    Returns (smiles_list, labels, is_classification) where labels is shape
    (N, T) for T tasks. Rows with all-NaN labels are filled with column means.
    """
    import deepchem as dc
    info = _MOLECULENET_TASKS[task]
    is_classification = info["task_type"] == "classification"
    loaders = {
        "bbbp": dc.molnet.load_bbbp,
        "hiv": dc.molnet.load_hiv,
        "tox21": dc.molnet.load_tox21,
        "bace": dc.molnet.load_bace_classification,
        "clintox": dc.molnet.load_clintox,
        "sider": dc.molnet.load_sider,
        "muv": dc.molnet.load_muv,
        "esol": dc.molnet.load_delaney,
        "freesolv": dc.molnet.load_sampl,
        "lipophilicity": dc.molnet.load_lipo,
        "bace_regression": dc.molnet.load_bace_regression,
        "clearance": dc.molnet.load_clearance,
    }
    tasks_list, datasets, _ = loaders[task](
        featurizer=dc.feat.DummyFeaturizer(),
        splitter="scaffold", data_dir=data_dir, transformers=[],
    )
    # Concatenate ALL splits — paper uses whole dataset for RePRA.
    if task in _REPRA_TASKS_WANTED:
        wanted = _REPRA_TASKS_WANTED[task]
        col_idx = [tasks_list.index(c) for c in wanted if c in tasks_list]
    else:
        col_idx = list(range(len(tasks_list)))

    smiles_list: List[str] = []
    label_rows: List[np.ndarray] = []
    seen_smiles: set = set()  # dedup across splits
    for ds in datasets:  # train, valid, test
        for i in range(len(ds)):
            smi = str(ds.ids[i])
            can = canonicalize_smiles(smi, strip_salts=True)
            if can is None or can in seen_smiles:
                continue
            seen_smiles.add(can)
            y = ds.y[i]
            if not hasattr(y, "__len__"):
                y = np.array([y])
            else:
                y = np.asarray(y)
            if len(col_idx) > 0:
                y_sel = y[col_idx].astype(np.float32)
            else:
                y_sel = y.astype(np.float32)
            smiles_list.append(can)
            label_rows.append(y_sel)
    if not label_rows:
        return [], np.zeros((0, 0)), is_classification
    labels = np.stack(label_rows, axis=0).astype(np.float32)
    # Fill NaN with column-mean (paper uses FillMissingValueByDefault for Tox21/Toxcast).
    if np.isnan(labels).any():
        col_means = np.nanmean(labels, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        nan_mask = np.isnan(labels)
        for c in range(labels.shape[1]):
            labels[nan_mask[:, c], c] = col_means[c]
    return smiles_list, labels, is_classification


def ecfp4_matrix(smiles: Sequence[str], n_bits: int = 1024) -> np.ndarray:
    """Return (N, n_bits) uint8 ECFP4 fingerprints."""
    out = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = arr
    return out


def stamp_cls_matrix(
    pretrain_dir: str,
    smiles: Sequence[str],
    device: str = "cuda",
    batch_size: int = 64,
) -> np.ndarray:
    """Return (N, d_model) [CLS] (BOS) embeddings from a pretrained ckpt."""
    encoder, vocab, rep, cfg, dev = load_pretrained_roberta(pretrain_dir, device=device)
    encoder.eval()
    keep_idx: List[int] = []
    seqs: List[List[int]] = []
    for i, smi in enumerate(smiles):
        try:
            text = rep.encode(smi)
            toks = rep.tokenize(text)
            if not toks:
                continue
            ids = [vocab.bos_id] + vocab.encode_tokens(toks) + [vocab.eos_id]
            ids = ids[: cfg.max_seq_len]
            seqs.append(ids)
            keep_idx.append(i)
        except Exception:
            continue

    pad_id = vocab.pad_id
    out = np.full((len(smiles), cfg.d_model), np.nan, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            batch = seqs[start : start + batch_size]
            max_len = max(len(s) for s in batch)
            x = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=dev)
            for r, s in enumerate(batch):
                x[r, : len(s)] = torch.tensor(s, dtype=torch.long, device=dev)
            cls = encoder(x, return_cls=True).float().cpu().numpy()
            for r, embed in enumerate(cls):
                out[keep_idx[start + r]] = embed
    return out


def minmax_normed_eud(matrix: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance, MinMax-normalized to [0, 1].
    Returns condensed upper-triangle vector of length N*(N-1)/2.

    Memory-friendly: uses scipy's pdist (BLAS-backed) which never materializes
    the (N, N, D) outer-product tensor.
    """
    from scipy.spatial.distance import pdist
    if matrix.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    flat = pdist(matrix.astype(np.float64), metric="euclidean")
    if flat.size == 0:
        return flat
    lo, hi = flat.min(), flat.max()
    if hi - lo < 1e-12:
        return np.zeros_like(flat)
    return (flat - lo) / (hi - lo)


def thresholds_classification(pair_dist: np.ndarray, pair_label_diff: np.ndarray) -> Tuple[float, float, float]:
    same = pair_label_diff == 0
    diff_mask = pair_label_diff != 0
    delta_AC = float(pair_dist[same].mean()) if same.any() else 0.0
    delta_SH = float(pair_dist[diff_mask].mean()) if diff_mask.any() else 1.0
    eps = 0.5
    return delta_AC, delta_SH, eps


def thresholds_regression(pair_dist: np.ndarray, pair_diff: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (delta_AC, delta_SH, eps_AC, eps_SH) per paper Eqs. 3-4."""
    d_m = float(np.median(pair_dist))
    close = pair_dist < d_m
    far = pair_dist >= d_m
    delta_AC = float(pair_dist[close].mean()) if close.any() else 0.0
    delta_SH = float(pair_dist[far].mean()) if far.any() else 1.0
    eps_AC = float(pair_diff[close].mean()) if close.any() else float(np.median(pair_diff))
    eps_SH = float(pair_diff[far].mean()) if far.any() else float(np.median(pair_diff))
    return delta_AC, delta_SH, eps_AC, eps_SH


def repra_metrics(
    pair_dist: np.ndarray,
    pair_diff: np.ndarray,
    is_classification: bool,
) -> Dict[str, float]:
    """Compute (C_AC, C_SH, R_AC list, R_SH list) and s_AD per paper code."""
    sim = 1.0 - pair_dist
    diff = pair_diff
    if is_classification:
        delta_AC, delta_SH, eps = thresholds_classification(pair_dist, diff)
        eps_AC = eps_SH = eps
    else:
        delta_AC, delta_SH, eps_AC, eps_SH = thresholds_regression(pair_dist, diff)
    delta_AC_s = 1.0 - delta_AC
    delta_SH_s = 1.0 - delta_SH

    # AC: very-similar in z, but different in y
    ac_mask = (diff > eps_AC) & (sim > delta_AC_s)
    # SH: very-different in z, but similar in y
    sh_mask = (diff < eps_SH) & (sim < delta_SH_s)

    # Per-pair deviation = min(|diff - eps|, |sim - threshold|)
    ac_dev = np.minimum(np.abs(diff[ac_mask] - eps_AC), np.abs(sim[ac_mask] - delta_AC_s))
    sh_dev = np.minimum(np.abs(diff[sh_mask] - eps_SH), np.abs(sim[sh_mask] - delta_SH_s))

    C_AC = int(ac_mask.sum())
    C_SH = int(sh_mask.sum())
    total_pairs = int(pair_dist.size)
    e = 2.0 * (C_AC + C_SH) / max(total_pairs, 1)
    if (C_AC + C_SH) > 0:
        dbar = float((ac_dev.sum() + sh_dev.sum()) / (C_AC + C_SH))
    else:
        dbar = 0.0
    s_AD = e * dbar

    return {
        "C_AC": C_AC,
        "C_SH": C_SH,
        "total_pairs": total_pairs,
        "delta_AC": delta_AC,
        "delta_SH": delta_SH,
        "eps_AC": eps_AC,
        "eps_SH": eps_SH,
        "e": e,
        "dbar": dbar,
        "sAD": s_AD,
    }


def evaluate_one_task(
    smiles: List[str],
    labels: np.ndarray,  # shape (N, T)
    is_classification: bool,
    representations: Dict[str, np.ndarray],
) -> Dict[str, dict]:
    y = np.asarray(labels, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    # MinMax-normalize Euclidean distance over T-D label vectors per the paper.

    results: Dict[str, dict] = {}
    for name, z in representations.items():
        valid = ~np.isnan(z).any(axis=1)
        if valid.sum() < 10:
            results[name] = {"error": f"only {int(valid.sum())} valid embeddings"}
            continue
        z_v = z[valid].astype(np.float64)
        y_v = y[valid]
        n_v = len(y_v)
        iu_v = np.triu_indices(n_v, k=1)
        pdz = minmax_normed_eud(z_v)
        # Pair-wise label distance: Euclidean over multi-label vector,
        # then MinMax-normalize across all pairs (matches paper's CalDiffArray).
        diff_vec = y_v[iu_v[0]] - y_v[iu_v[1]]
        raw_dy = np.sqrt(np.sum(diff_vec * diff_vec, axis=-1))
        lo, hi = raw_dy.min(), raw_dy.max()
        pdy = (raw_dy - lo) / max(hi - lo, 1e-12)
        stats = repra_metrics(pdz, pdy, is_classification)
        stats["n_valid"] = int(valid.sum())
        stats["n_label_dims"] = int(y.shape[1])
        results[name] = stats
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--pretrain", action="append", required=True,
        help="Tag=path mapping; repeat for multiple ckpts.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-bits", type=int, default=1024)
    ap.add_argument("--ecfp-key", default="MorganFP",
                    help="Tag used as the ECFP baseline for s_IR.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pretrain_map: Dict[str, str] = {}
    for entry in args.pretrain:
        tag, path = entry.split("=", 1)
        pretrain_map[tag.strip()] = path.strip()

    all_results: Dict[str, Dict[str, dict]] = {}
    rows: List[dict] = []

    for task in args.tasks:
        print(f"\n========== {task} ==========", flush=True)
        smis, labels, is_classification = load_multilabel_full(task, args.data_dir)
        if not smis:
            print(f"  empty test set, skipping", flush=True)
            continue
        print(f"  test n={len(smis)}, classification={is_classification}, label dims={labels.shape[1]}", flush=True)

        reps: Dict[str, np.ndarray] = {}
        print("  computing ECFP4 ...", flush=True)
        reps[args.ecfp_key] = ecfp4_matrix(smis, n_bits=args.n_bits).astype(np.float64)
        for tag, path in pretrain_map.items():
            print(f"  computing {tag} embeddings from {path} ...", flush=True)
            reps[tag] = stamp_cls_matrix(path, smis, device=args.device, batch_size=args.batch_size)

        per_task = evaluate_one_task(smis, labels, is_classification, reps)
        # s_IR uses ECFP as baseline
        ecfp_stats = per_task[args.ecfp_key]
        for tag, stats in per_task.items():
            if "C_AC" not in stats:
                continue
            if tag == args.ecfp_key:
                stats["sIR"] = 2.0  # self-comparison
            else:
                ec_AC = max(ecfp_stats["C_AC"], 1)
                ec_SH = max(ecfp_stats["C_SH"], 1)
                stats["sIR"] = stats["C_AC"] / ec_AC + stats["C_SH"] / ec_SH

        all_results[task] = per_task
        for tag, stats in per_task.items():
            if "C_AC" not in stats:
                continue
            rows.append({
                "task": task,
                "type": "clf" if is_classification else "reg",
                "rep": tag,
                "n": stats["n_valid"],
                "C_AC": stats["C_AC"],
                "C_SH": stats["C_SH"],
                "sAD": stats["sAD"],
                "sIR": stats["sIR"],
            })
            print(
                f"  {tag:8s}: C_AC={stats['C_AC']:6d}  C_SH={stats['C_SH']:6d}  "
                f"sAD={stats['sAD']:.4f}  sIR={stats['sIR']:.3f}",
                flush=True,
            )

    (out_dir / "repra_full.json").write_text(json.dumps(all_results, indent=2, default=float))
    md = ["| Task | Type | Rep | n | C_AC | C_SH | sAD | sIR |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['task']} | {r['type']} | {r['rep']} | {r['n']} | {r['C_AC']} | {r['C_SH']} | {r['sAD']:.4f} | {r['sIR']:.3f} |")
    (out_dir / "repra_summary.md").write_text("\n".join(md) + "\n")
    print("\nSaved:", out_dir / "repra_full.json", "and", out_dir / "repra_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
