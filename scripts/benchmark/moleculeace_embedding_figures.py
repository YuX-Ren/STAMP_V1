#!/usr/bin/env python3
"""Embedding-space visualizations for MoleculeACE MMP/cliff analysis.

This script compares STAMP-MTR, ChemBERTa-2, and ECFP4 at the embedding level:
  - 2D UMAP/PCA panels colored by activity with high-delta MMP cliff edges.
  - Cliff vs non-cliff MMP pair distance distributions using RePRA-style
    MinMax-normalized embedding distances.
  - Binned |delta activity| vs embedding distance curves.

UMAP is visualization-only. Quantitative comparisons use RePRA-style
per-representation MinMax-normalized Euclidean distances, with raw Euclidean
distances retained for diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'user')}")
os.environ.setdefault("NUMBA_CACHE_DIR", f"/tmp/numba-cache-{os.environ.get('USER', 'user')}")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.common import canonicalize_smiles  # noqa: E402
from stamp.benchmark.roberta_property import load_pretrained_roberta  # noqa: E402


def load_moleculeace_task(data_dir: Path, task: str, *, split: str) -> pd.DataFrame:
    path = data_dir / f"{task}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    rows: List[dict] = []
    for idx, row in df.iterrows():
        if str(row.get("split", "")) != split:
            continue
        smi = canonicalize_smiles(str(row["smiles"]), strip_salts=True)
        if smi is None:
            continue
        y = float(row["y [pEC50/pKi]"])
        if not math.isfinite(y):
            continue
        rows.append(
            {
                "mol_id": f"{task}:{idx}",
                "row_index": int(idx),
                "smiles": smi,
                "activity": y,
                "cliff_mol": int(row.get("cliff_mol", 0)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No valid {split} rows for {task}")
    return out.drop_duplicates("mol_id").reset_index(drop=True)


def ecfp4_matrix(smiles: Sequence[str], n_bits: int) -> np.ndarray:
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out[i] = np.nan
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = arr.astype(np.float32)
    return out


def stamp_embeddings(pretrain_dir: Path, smiles: Sequence[str], *, device: str, batch_size: int) -> np.ndarray:
    encoder, vocab, rep, cfg, dev = load_pretrained_roberta(pretrain_dir, device=device)
    encoder.eval()
    keep_idx: List[int] = []
    seqs: List[List[int]] = []
    for i, smi in enumerate(smiles):
        try:
            toks = rep.tokenize(rep.encode(smi))
            if not toks:
                continue
            ids = [vocab.bos_id] + vocab.encode_tokens(toks) + [vocab.eos_id]
            seqs.append(ids[: cfg.max_seq_len])
            keep_idx.append(i)
        except Exception:
            continue
    out = np.full((len(smiles), cfg.d_model), np.nan, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            batch = seqs[start : start + batch_size]
            max_len = max(len(x) for x in batch)
            x = torch.full((len(batch), max_len), vocab.pad_id, dtype=torch.long, device=dev)
            for r, ids in enumerate(batch):
                x[r, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=dev)
            emb = encoder(x, return_cls=True).float().cpu().numpy()
            for r, vec in enumerate(emb):
                out[keep_idx[start + r]] = vec
    return out


def chemberta_embeddings(
    model_name: str,
    smiles: Sequence[str],
    *,
    device: str,
    batch_size: int,
    max_len: int,
    local_files_only: bool,
) -> np.ndarray:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    model.eval()
    out = np.full((len(smiles), int(model.config.hidden_size)), np.nan, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            batch = list(smiles[start : start + batch_size])
            enc = tok(batch, max_length=max_len, padding=True, truncation=True, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()
            out[start : start + len(batch)] = hidden
    return out


def fit_2d(emb: np.ndarray, *, method: str, seed: int, metric: str) -> Tuple[np.ndarray, str]:
    valid = ~np.isnan(emb).any(axis=1)
    x = emb[valid]
    if method == "umap":
        try:
            import umap

            reducer = umap.UMAP(
                n_neighbors=25,
                min_dist=0.1,
                metric=metric,
                random_state=seed,
            )
            xy_valid = reducer.fit_transform(x)
            used = "umap"
        except Exception:
            x2 = StandardScaler(with_mean=True, with_std=True).fit_transform(x.astype(np.float64))
            xy_valid = PCA(n_components=2, random_state=seed).fit_transform(x2)
            used = "pca_fallback"
    else:
        x2 = StandardScaler(with_mean=True, with_std=True).fit_transform(x.astype(np.float64))
        xy_valid = PCA(n_components=2, random_state=seed).fit_transform(x2)
        used = "pca"
    xy = np.full((len(emb), 2), np.nan, dtype=np.float32)
    xy[valid] = xy_valid.astype(np.float32)
    return xy, used


def pair_euclidean_distances(emb: np.ndarray, pairs: pd.DataFrame, index_by_id: Dict[str, int]) -> np.ndarray:
    out = []
    for _, row in pairs.iterrows():
        ia = index_by_id[row["mol_a_id"]]
        ib = index_by_id[row["mol_b_id"]]
        a = emb[ia]
        b = emb[ib]
        if np.isnan(a).any() or np.isnan(b).any():
            out.append(np.nan)
            continue
        out.append(float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64))))
    return np.asarray(out, dtype=np.float32)


def repra_distance_bounds(emb: np.ndarray) -> Tuple[float, float, int]:
    """RePRA-style normalization bounds from all valid molecule pairs."""
    from scipy.spatial.distance import pdist

    valid = ~np.isnan(emb).any(axis=1)
    x = emb[valid]
    if len(x) < 2:
        return math.nan, math.nan, 0
    flat = pdist(x.astype(np.float64), metric="euclidean")
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return math.nan, math.nan, 0
    return float(flat.min()), float(flat.max()), int(flat.size)


def _minmax_norm(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values, math.nan, math.nan
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not math.isfinite(lo) or not math.isfinite(hi):
        return np.full_like(values, np.nan, dtype=np.float64), lo, hi
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float64), lo, hi
    return (values - lo) / (hi - lo), lo, hi


def apply_repra_norm(dist: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = np.full_like(dist, np.nan, dtype=np.float32)
    finite = np.isfinite(dist)
    if not finite.any() or not math.isfinite(lo) or not math.isfinite(hi):
        return out
    if hi - lo < 1e-12:
        out[finite] = 0.0
    else:
        out[finite] = np.clip((dist[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out


def repra_regression_stats(pair_dist: np.ndarray, pair_diff: np.ndarray) -> Dict[str, float]:
    """RePRA regression metrics on normalized z/y pair distances."""
    pair_dist = np.asarray(pair_dist, dtype=np.float64)
    pair_diff = np.asarray(pair_diff, dtype=np.float64)
    finite = np.isfinite(pair_dist) & np.isfinite(pair_diff)
    pair_dist = pair_dist[finite]
    pair_diff = pair_diff[finite]
    if pair_dist.size == 0:
        return {
            "C_AC": 0,
            "C_SH": 0,
            "total_pairs": 0,
            "delta_AC": math.nan,
            "delta_SH": math.nan,
            "eps_AC": math.nan,
            "eps_SH": math.nan,
            "e": math.nan,
            "dbar": math.nan,
            "sAD": math.nan,
        }

    sim = 1.0 - pair_dist
    d_m = float(np.median(pair_dist))
    close = pair_dist < d_m
    far = pair_dist >= d_m
    delta_AC = float(pair_dist[close].mean()) if close.any() else 0.0
    delta_SH = float(pair_dist[far].mean()) if far.any() else 1.0
    eps_AC = float(pair_diff[close].mean()) if close.any() else float(np.median(pair_diff))
    eps_SH = float(pair_diff[far].mean()) if far.any() else float(np.median(pair_diff))

    ac_mask = (pair_diff > eps_AC) & (sim > (1.0 - delta_AC))
    sh_mask = (pair_diff < eps_SH) & (sim < (1.0 - delta_SH))
    ac_dev = np.minimum(np.abs(pair_diff[ac_mask] - eps_AC), np.abs(sim[ac_mask] - (1.0 - delta_AC)))
    sh_dev = np.minimum(np.abs(pair_diff[sh_mask] - eps_SH), np.abs(sim[sh_mask] - (1.0 - delta_SH)))

    C_AC = int(ac_mask.sum())
    C_SH = int(sh_mask.sum())
    total_pairs = int(pair_dist.size)
    e = 2.0 * (C_AC + C_SH) / max(total_pairs, 1)
    if C_AC + C_SH > 0:
        dbar = float((ac_dev.sum() + sh_dev.sum()) / (C_AC + C_SH))
    else:
        dbar = 0.0
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
        "sAD": e * dbar,
    }


def repra_regression_for_embedding(emb: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Run RePRA regression analysis on all valid molecules for one representation."""
    from scipy.spatial.distance import pdist

    valid = (~np.isnan(emb).any(axis=1)) & np.isfinite(y)
    x = emb[valid].astype(np.float64)
    y_v = y[valid].astype(np.float64)[:, None]
    if len(x) < 2:
        out = repra_regression_stats(np.array([]), np.array([]))
        out.update({"n_valid": int(len(x)), "raw_z_min": math.nan, "raw_z_max": math.nan, "raw_y_min": math.nan, "raw_y_max": math.nan})
        return out

    raw_z = pdist(x, metric="euclidean")
    raw_y = pdist(y_v, metric="euclidean")
    z_dist, z_lo, z_hi = _minmax_norm(raw_z)
    y_diff, y_lo, y_hi = _minmax_norm(raw_y)
    out = repra_regression_stats(z_dist, y_diff)
    out.update(
        {
            "n_valid": int(len(x)),
            "raw_z_min": z_lo,
            "raw_z_max": z_hi,
            "raw_y_min": y_lo,
            "raw_y_max": y_hi,
        }
    )
    return out


def load_test_mmp_pairs(pair_csv: Path, task: str, mol_ids: set[str], *, max_edges: int) -> pd.DataFrame:
    cols = ["task", "mol_a_id", "mol_b_id", "split_a", "split_b", "delta_y", "ecfp4_tanimoto_distance"]
    df = pd.read_csv(pair_csv, usecols=lambda c: c in set(cols))
    df = df[
        (df["task"] == task)
        & (df["split_a"] == "test")
        & (df["split_b"] == "test")
        & df["mol_a_id"].isin(mol_ids)
        & df["mol_b_id"].isin(mol_ids)
    ].copy()
    df["delta_y"] = df["delta_y"].astype(float)
    df = df.sort_values("delta_y", ascending=False).drop_duplicates(["mol_a_id", "mol_b_id"])
    if max_edges > 0 and len(df) > max_edges:
        # Keep all pairs for metrics elsewhere; this function controls visual edge count only if requested by caller.
        pass
    return df.reset_index(drop=True)


def ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(values[np.isfinite(values)])
    if len(x) == 0:
        return x, x
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def plot_umap_panels(
    records: pd.DataFrame,
    coords: Dict[str, np.ndarray],
    pairs: pd.DataFrame,
    *,
    out_path: Path,
    max_edges: int,
    title: str,
) -> None:
    reps = list(coords)
    idx = {mid: i for i, mid in enumerate(records["mol_id"])}
    edge_pairs = pairs.sort_values("delta_y", ascending=False).head(max_edges)
    fig, axes = plt.subplots(1, len(reps), figsize=(5.2 * len(reps), 4.7), squeeze=False)
    y = records["activity"].to_numpy(dtype=float)
    for ax, rep in zip(axes[0], reps):
        xy = coords[rep]
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=y, s=24, cmap="viridis", alpha=0.88, edgecolors="none")
        for _, row in edge_pairs.iterrows():
            ia = idx.get(row["mol_a_id"])
            ib = idx.get(row["mol_b_id"])
            if ia is None or ib is None or np.isnan(xy[[ia, ib]]).any():
                continue
            ax.plot([xy[ia, 0], xy[ib, 0]], [xy[ia, 1], xy[ib, 1]], color="#d11f4a", alpha=0.38, linewidth=0.8)
        ax.set_title(rep)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.82, label="activity pEC50/pKi")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def distance_label(distance_col: str) -> str:
    if distance_col == "repra_norm_distance":
        return "RePRA normalized embedding distance"
    if distance_col == "embedding_distance":
        return "raw Euclidean embedding distance"
    return distance_col.replace("_", " ")


def plot_distance_ecdf(pair_metrics: pd.DataFrame, out_path: Path, *, distance_col: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    colors = {"STAMP-MTR": "#2f6f73", "ChemBERTa-2": "#c46a2b", "ECFP4": "#4f5f9f"}
    for rep, sub in pair_metrics.groupby("rep", sort=False):
        for label, style in [("cliff", "-"), ("noncliff", "--")]:
            vals = sub.loc[sub["pair_type"] == label, distance_col].to_numpy(dtype=float)
            x, y = ecdf(vals)
            ax.plot(x, y, linestyle=style, color=colors.get(rep), label=f"{rep} {label}")
    ax.set_xlabel(distance_label(distance_col))
    ax.set_ylabel("ECDF")
    ax.set_title("Cliff vs non-cliff MMP pair distances")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_binned_delta(pair_metrics: pd.DataFrame, out_path: Path, *, distance_col: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    colors = {"STAMP-MTR": "#2f6f73", "ChemBERTa-2": "#c46a2b", "ECFP4": "#4f5f9f"}
    for rep, sub in pair_metrics.groupby("rep", sort=False):
        sub = sub[np.isfinite(sub[distance_col])].copy()
        if len(sub) < 10:
            continue
        sub["bin"] = pd.qcut(sub[distance_col], q=min(10, len(sub) // 5), duplicates="drop")
        grouped = sub.groupby("bin", observed=True).agg(
            distance=(distance_col, "median"),
            delta_y=("delta_y", "median"),
        )
        ax.plot(grouped["distance"], grouped["delta_y"], marker="o", color=colors.get(rep), label=rep)
    ax.set_xlabel(f"binned median {distance_label(distance_col)}")
    ax.set_ylabel("median |delta activity|")
    ax.set_title("MMP activity difference vs embedding distance")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def summarize_pair_metrics(pair_metrics: pd.DataFrame, *, distance_col: str) -> Dict[str, object]:
    out: Dict[str, object] = {"representations": {}}
    for rep, sub in pair_metrics.groupby("rep", sort=False):
        cliff = sub[sub["pair_type"] == "cliff"][distance_col].dropna()
        non = sub[sub["pair_type"] == "noncliff"][distance_col].dropna()
        stats = {
            "n_cliff_pairs": int(len(cliff)),
            "n_noncliff_pairs": int(len(non)),
            "median_cliff_distance": float(cliff.median()) if len(cliff) else math.nan,
            "median_noncliff_distance": float(non.median()) if len(non) else math.nan,
            "cliff_to_noncliff_median_ratio": (
                float(cliff.median() / max(non.median(), 1e-12)) if len(cliff) and len(non) else math.nan
            ),
        }
        all_sub = sub.dropna(subset=[distance_col, "delta_y"])
        if len(all_sub) >= 3:
            stats[f"spearman_{distance_col}_vs_abs_delta_activity"] = float(
                all_sub[[distance_col, "delta_y"]].corr(method="spearman").iloc[0, 1]
            )
        out["representations"][rep] = stats
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="CHEMBL204_Ki")
    ap.add_argument("--moleculeace-dir", default="/scratch/yuxuan.ren/data/moleculeace")
    ap.add_argument("--mmp-pairs-csv", default="/scratch/yuxuan.ren/data/pubchem10m/moleculeace_mmp_edit_locality_509253.hopper-m-02_20260616_155324/final/mmp_edit_locality_pairs.csv")
    ap.add_argument("--stamp-pretrain-dir", default="/scratch/yuxuan.ren/data/pubchem10m/pretrain_stamp_top4666_mtr")
    ap.add_argument("--chemberta-model", default="seyonec/PubChem10M_SMILES_BPE_60k")
    ap.add_argument("--chemberta-local-files-only", action="store_true")
    ap.add_argument("--output-dir", default="results/moleculeace_embedding_figures")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--ecfp-bits", type=int, default=2048)
    ap.add_argument("--projection", choices=["umap", "pca"], default="umap")
    ap.add_argument("--cliff-delta-y", type=float, default=1.0)
    ap.add_argument("--noncliff-delta-y", type=float, default=0.25)
    ap.add_argument("--max-umap-edges", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_moleculeace_task(Path(args.moleculeace_dir), args.task, split="test")
    smiles = records["smiles"].tolist()
    mol_ids = set(records["mol_id"])
    idx = {mid: i for i, mid in enumerate(records["mol_id"])}
    pairs = load_test_mmp_pairs(Path(args.mmp_pairs_csv), args.task, mol_ids, max_edges=0)
    if pairs.empty:
        raise ValueError(f"No test-test MMP pairs found for {args.task}")

    print(f"[data] task={args.task} test_mols={len(records)} test_mmp_pairs={len(pairs)}", flush=True)
    emb: Dict[str, np.ndarray] = {}
    print("[embed] STAMP-MTR", flush=True)
    emb["STAMP-MTR"] = stamp_embeddings(Path(args.stamp_pretrain_dir), smiles, device=args.device, batch_size=args.batch_size)
    print("[embed] ChemBERTa-2", flush=True)
    emb["ChemBERTa-2"] = chemberta_embeddings(
        args.chemberta_model,
        smiles,
        device=args.device,
        batch_size=args.batch_size,
        max_len=args.max_len,
        local_files_only=bool(args.chemberta_local_files_only),
    )
    print("[embed] ECFP4", flush=True)
    emb["ECFP4"] = ecfp4_matrix(smiles, n_bits=args.ecfp_bits)
    np.savez_compressed(out_dir / "embeddings.npz", **emb)
    records.to_csv(out_dir / f"{args.task}_test_records.csv", index=False)

    coords: Dict[str, np.ndarray] = {}
    projection_used: Dict[str, str] = {}
    for rep, mat in emb.items():
        metric = "jaccard" if rep == "ECFP4" and args.projection == "umap" else "cosine"
        xy, used = fit_2d(mat, method=args.projection, seed=args.seed, metric=metric)
        coords[rep] = xy
        projection_used[rep] = used
    np.savez_compressed(out_dir / "projection_2d.npz", **coords)

    pair_rows: List[dict] = []
    distance_bounds: Dict[str, Dict[str, object]] = {}
    for rep, mat in emb.items():
        lo, hi, n_bound_pairs = repra_distance_bounds(mat)
        distance_bounds[rep] = {
            "raw_euclidean_min": lo,
            "raw_euclidean_max": hi,
            "n_all_test_pairs_for_bounds": n_bound_pairs,
        }
        dist = pair_euclidean_distances(mat, pairs, idx)
        norm_dist = apply_repra_norm(dist, lo, hi)
        for i, row in pairs.reset_index(drop=True).iterrows():
            dy = float(row["delta_y"])
            if dy >= args.cliff_delta_y:
                pair_type = "cliff"
            elif dy <= args.noncliff_delta_y:
                pair_type = "noncliff"
            else:
                pair_type = "mid"
            pair_rows.append(
                {
                    "rep": rep,
                    "mol_a_id": row["mol_a_id"],
                    "mol_b_id": row["mol_b_id"],
                    "delta_y": dy,
                    "ecfp4_tanimoto_distance": float(row["ecfp4_tanimoto_distance"]),
                    "embedding_distance": float(dist[i]),
                    "repra_norm_distance": float(norm_dist[i]),
                    "pair_type": pair_type,
                }
            )
    pair_metrics = pd.DataFrame(pair_rows)
    pair_metrics.to_csv(out_dir / "pair_embedding_distances.csv", index=False)

    repra_rows: List[dict] = []
    activity = records["activity"].to_numpy(dtype=float)
    for rep, mat in emb.items():
        stats = repra_regression_for_embedding(mat, activity)
        repra_rows.append({"rep": rep, **stats})
    repra_df = pd.DataFrame(repra_rows)
    if "ECFP4" in set(repra_df["rep"]):
        ecfp_row = repra_df[repra_df["rep"] == "ECFP4"].iloc[0]
        ecfp_ac = max(int(ecfp_row["C_AC"]), 1)
        ecfp_sh = max(int(ecfp_row["C_SH"]), 1)
        sir_values = []
        for _, row in repra_df.iterrows():
            if row["rep"] == "ECFP4":
                sir_values.append(2.0)
            else:
                sir_values.append(float(row["C_AC"] / ecfp_ac + row["C_SH"] / ecfp_sh))
        repra_df["sIR"] = sir_values
    else:
        repra_df["sIR"] = math.nan
    repra_df.to_csv(out_dir / "repra_regression_metrics.csv", index=False)

    cliff_pairs = pairs[pairs["delta_y"] >= args.cliff_delta_y].copy()
    plot_umap_panels(
        records,
        coords,
        cliff_pairs,
        out_path=out_dir / f"{args.task}_embedding_umap_cliff_edges.png",
        max_edges=args.max_umap_edges,
        title=f"{args.task} test embeddings; red edges: MMP cliffs |delta y| >= {args.cliff_delta_y}",
    )
    summary_pairs = pair_metrics[pair_metrics["pair_type"].isin(["cliff", "noncliff"])]
    distance_col = "repra_norm_distance"
    plot_distance_ecdf(
        summary_pairs,
        out_dir / f"{args.task}_cliff_noncliff_repra_norm_distance_ecdf.png",
        distance_col=distance_col,
    )
    plot_binned_delta(
        pair_metrics,
        out_dir / f"{args.task}_activity_delta_vs_repra_norm_distance.png",
        distance_col=distance_col,
    )

    norm_summary = summarize_pair_metrics(summary_pairs, distance_col=distance_col)
    raw_summary = summarize_pair_metrics(summary_pairs, distance_col="embedding_distance")

    summary = {
        "task": args.task,
        "n_test_molecules": int(len(records)),
        "n_test_mmp_pairs": int(len(pairs)),
        "cliff_delta_y": float(args.cliff_delta_y),
        "noncliff_delta_y": float(args.noncliff_delta_y),
        "projection_requested": args.projection,
        "projection_used": projection_used,
        "distance_column_for_quantitative_plots": distance_col,
        "distance_diagnostic_note": (
            "Median cliff/noncliff distances are a visualization diagnostic. "
            "They are not RePRA sIR and should not be compared to MoleculeNet RePRA tables."
        ),
        "distance_normalization": (
            "RePRA-style per-representation MinMax-normalized Euclidean distance; "
            "min/max are fitted on all valid test molecule pairs for the selected MoleculeACE task."
        ),
        "distance_bounds": distance_bounds,
        "representations": norm_summary["representations"],
        "raw_euclidean_representations": raw_summary["representations"],
        "repra_regression_scope": (
            "Selected MoleculeACE task test split, single regression label, all valid molecule pairs; "
            "sIR uses ECFP4 C_AC/C_SH as baseline with ECFP4 self-comparison fixed to 2.0."
        ),
        "repra_regression_metrics": repra_df.set_index("rep").to_dict(orient="index"),
    }
    (out_dir / "embedding_figure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved embedding figures to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
