#!/usr/bin/env python3
"""MoleculeACE matched-molecular-pair edit-locality benchmark.

For each MoleculeACE task, this script mines single-cut matched molecular
pairs (MMPs), then compares chemical distance against token edit distance for
STAMP, SAFE, and SMILES-BPE representations.

Outputs:
    mmp_edit_locality_pairs.csv
    mmp_edit_locality_summary.json
    edit_locality_scatter.png
    edit_locality_box_by_chem_bin.png
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMMPA

RDLogger.DisableLog("rdApp.*")
try:
    from loguru import logger as _safe_logger

    _safe_logger.disable("safe")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.common import canonicalize_smiles  # noqa: E402
from stamp.benchmark.representations import (  # noqa: E402
    RepresentationError,
    SAFERepresentation,
    STAMPRepresentation,
)


_SMILES_TOKEN_RE = re.compile(
    r"(\[[^\[\]]+\]|Br|Cl|Si|Se|Li|Na|Ca|Al|Mg|Zn|Sn|Ag|Fe|As|Pt|Au|Cu|Mn|Co|Ni|Pb|Hg|Bi|Kr|Xe|He|Ne|Ar|%\d{2}|[A-Za-z]|[0-9]|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|\*|\$)"
)
_MMPA_PATTERN = "[#6+0;!$(*=,#[!#6])]!@!=!#[*]"


@dataclass(frozen=True)
class MoleculeRecord:
    task: str
    mol_id: str
    row_index: int
    smiles: str
    split: str
    y: float
    cliff_mol: int


@dataclass(frozen=True)
class MMPContextRecord:
    mol: MoleculeRecord
    context: str
    variable: str
    context_heavy_atoms: int
    variable_heavy_atoms: int


def _tokenize_smiles_lex(smiles: str) -> List[str]:
    toks: List[str] = []
    pos = 0
    for m in _SMILES_TOKEN_RE.finditer(smiles):
        if m.start() > pos:
            toks.extend(list(smiles[pos : m.start()]))
        toks.append(m.group(0))
        pos = m.end()
    if pos < len(smiles):
        toks.extend(list(smiles[pos:]))
    return toks


def _heavy_atoms(fragment_smiles: str) -> int:
    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


def _fragment_parts(chains: str) -> List[str]:
    parts = [p for p in chains.split(".") if p]
    return sorted(parts)


def _fragment_records(
    mol: Chem.Mol,
    *,
    max_cut_bonds: int,
    all_context_sides: bool,
    min_context_heavy_atoms: int,
    max_variable_heavy_atoms: int,
) -> List[Tuple[str, str, int, int]]:
    try:
        cuts = rdMMPA.FragmentMol(mol, 1, 1, max_cut_bonds, _MMPA_PATTERN, False)
    except TypeError:
        cuts = rdMMPA.FragmentMol(mol, maxCuts=1, maxCutBonds=max_cut_bonds, resultsAsMols=False)

    out: List[Tuple[str, str, int, int]] = []
    seen: set[Tuple[str, str]] = set()
    for _core, chains in cuts:
        parts = _fragment_parts(str(chains))
        if len(parts) < 2:
            continue
        part_sizes = [_heavy_atoms(p) for p in parts]
        if all_context_sides:
            context_indices = range(len(parts))
        else:
            # Prefer the largest side as the constant context. This avoids
            # treating a methyl/phenyl side chain as the scaffold for huge pairs.
            max_size = max(part_sizes)
            context_indices = [i for i, size in enumerate(part_sizes) if size == max_size]
        for idx in context_indices:
            context = parts[idx]
            variable_parts = [p for j, p in enumerate(parts) if j != idx]
            variable = ".".join(sorted(variable_parts))
            context_ha = part_sizes[idx]
            variable_ha = sum(part_sizes[j] for j in range(len(parts)) if j != idx)
            if context_ha < min_context_heavy_atoms:
                continue
            if max_variable_heavy_atoms > 0 and variable_ha > max_variable_heavy_atoms:
                continue
            key = (context, variable)
            if key in seen:
                continue
            seen.add(key)
            out.append((context, variable, context_ha, variable_ha))
    return out


def load_moleculeace_records(
    data_dir: Path,
    *,
    strip_salts: bool,
    tasks: Optional[Sequence[str]] = None,
) -> Dict[str, List[MoleculeRecord]]:
    task_filter = set(tasks or [])
    task_records: Dict[str, List[MoleculeRecord]] = {}
    for csv_path in sorted(data_dir.glob("CHEMBL*.csv")):
        task = csv_path.stem
        if task_filter and task not in task_filter:
            continue
        df = pd.read_csv(csv_path)
        seen_smiles: set[str] = set()
        records: List[MoleculeRecord] = []
        for row_idx, row in df.iterrows():
            smiles = canonicalize_smiles(str(row["smiles"]), strip_salts=strip_salts)
            if smiles is None or smiles in seen_smiles:
                continue
            seen_smiles.add(smiles)
            y_value = row.get("y [pEC50/pKi]", row.get("y", np.nan))
            records.append(
                MoleculeRecord(
                    task=task,
                    mol_id=f"{task}:{row_idx}",
                    row_index=int(row_idx),
                    smiles=smiles,
                    split=str(row.get("split", "")),
                    y=float(y_value) if pd.notna(y_value) else math.nan,
                    cliff_mol=int(row.get("cliff_mol", 0)) if pd.notna(row.get("cliff_mol", 0)) else 0,
                )
            )
        if records:
            task_records[task] = records
    return task_records


def enumerate_mmp_pairs(
    task_records: Dict[str, List[MoleculeRecord]],
    *,
    max_cut_bonds: int,
    all_context_sides: bool,
    keep_context_duplicates: bool,
    min_context_heavy_atoms: int,
    max_variable_heavy_atoms: int,
    max_pairs_per_context: int,
    max_pairs_total: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    pair_rows: List[dict] = []

    for task, records in task_records.items():
        n_pairs_before = len(pair_rows)
        by_context: Dict[str, List[MMPContextRecord]] = defaultdict(list)
        for rec in records:
            mol = Chem.MolFromSmiles(rec.smiles)
            if mol is None:
                continue
            for context, variable, context_ha, variable_ha in _fragment_records(
                mol,
                max_cut_bonds=max_cut_bonds,
                all_context_sides=all_context_sides,
                min_context_heavy_atoms=min_context_heavy_atoms,
                max_variable_heavy_atoms=max_variable_heavy_atoms,
            ):
                by_context[context].append(
                    MMPContextRecord(
                        mol=rec,
                        context=context,
                        variable=variable,
                        context_heavy_atoms=context_ha,
                        variable_heavy_atoms=variable_ha,
                    )
                )

        task_pairs: Dict[Tuple[str, str], dict] = {}
        for context, hits in by_context.items():
            # Same molecule can hit the same context multiple ways; keep one
            # record per molecule/context/variable to avoid self-pair inflation.
            uniq: Dict[Tuple[str, str], MMPContextRecord] = {}
            for hit in hits:
                uniq[(hit.mol.mol_id, hit.variable)] = hit
            hits = list(uniq.values())
            if len(hits) < 2:
                continue
            combinations = list(itertools.combinations(hits, 2))
            if max_pairs_per_context > 0 and len(combinations) > max_pairs_per_context:
                combinations = rng.sample(combinations, max_pairs_per_context)
            for a, b in combinations:
                if a.mol.mol_id == b.mol.mol_id or a.variable == b.variable:
                    continue
                pair_key = tuple(sorted((a.mol.mol_id, b.mol.mol_id)))
                row = {
                    "task": task,
                    "mol_a_id": a.mol.mol_id,
                    "mol_b_id": b.mol.mol_id,
                    "smiles_a": a.mol.smiles,
                    "smiles_b": b.mol.smiles,
                    "split_a": a.mol.split,
                    "split_b": b.mol.split,
                    "y_a": a.mol.y,
                    "y_b": b.mol.y,
                    "delta_y": abs(a.mol.y - b.mol.y) if math.isfinite(a.mol.y) and math.isfinite(b.mol.y) else math.nan,
                    "cliff_a": a.mol.cliff_mol,
                    "cliff_b": b.mol.cliff_mol,
                    "context": context,
                    "variable_a": a.variable,
                    "variable_b": b.variable,
                    "context_heavy_atoms": min(a.context_heavy_atoms, b.context_heavy_atoms),
                    "variable_a_heavy_atoms": a.variable_heavy_atoms,
                    "variable_b_heavy_atoms": b.variable_heavy_atoms,
                    "n_mmp_contexts": 1,
                }
                if keep_context_duplicates:
                    pair_rows.append(row)
                else:
                    existing = task_pairs.get(pair_key)
                    if existing is None:
                        task_pairs[pair_key] = row
                    else:
                        existing["n_mmp_contexts"] += 1

        if not keep_context_duplicates:
            pair_rows.extend(task_pairs.values())
        n_pairs_after = len(pair_rows)
        n_pairs_task = n_pairs_after - n_pairs_before if keep_context_duplicates else len(task_pairs)
        print(
            f"{task}: molecules={len(records)} contexts={len(by_context)} pairs={n_pairs_task}",
            flush=True,
        )

    if max_pairs_total > 0 and len(pair_rows) > max_pairs_total:
        pair_rows = rng.sample(pair_rows, max_pairs_total)
    return pd.DataFrame(pair_rows)


def _ecfp4(smiles: str, n_bits: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)


def _tanimoto_distance(a_smiles: str, b_smiles: str, n_bits: int) -> float:
    a = _ecfp4(a_smiles, n_bits)
    b = _ecfp4(b_smiles, n_bits)
    if a is None or b is None:
        return math.nan
    return 1.0 - float(DataStructs.TanimotoSimilarity(a, b))


def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (0 if token_a == token_b else 1),
                )
            )
        previous = current
    return previous[-1]


class TokenizerWrapper:
    name: str

    def tokens(self, smiles: str) -> List[str]:
        raise NotImplementedError


class STAMPTokenizerWrapper(TokenizerWrapper):
    name = "stamp"

    def __init__(
        self,
        vocab_path: str,
        *,
        force_ring_aware_encode: bool,
        strict_vocab: bool,
        expand_oov_smiles: bool,
    ) -> None:
        self.rep = STAMPRepresentation(
            vocab_path=vocab_path,
            force_ring_aware_encode=force_ring_aware_encode,
            strict_vocab=strict_vocab,
        )
        self.expand_oov_smiles = bool(expand_oov_smiles)
        self.vocab_tokens = self._load_vocab_tokens(Path(vocab_path))

    @staticmethod
    def _load_vocab_tokens(path: Path) -> set[str]:
        toks: set[str] = set()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                toks.add(s.split()[0])
        return toks

    def tokens(self, smiles: str) -> List[str]:
        text = self.rep.encode(smiles)
        tokens = self.rep.tokenize(text)
        if not self.expand_oov_smiles:
            return tokens
        out: List[str] = []
        for tok in tokens:
            if tok in self.vocab_tokens or self.rep.is_special_token(tok):
                out.append(tok)
            else:
                pieces = _tokenize_smiles_lex(tok)
                out.extend([f"SMI:{p}" for p in pieces] or [tok])
        return out


class SAFETokenizerWrapper(TokenizerWrapper):
    name = "safe"

    def __init__(self) -> None:
        self.rep = SAFERepresentation()

    def tokens(self, smiles: str) -> List[str]:
        return self.rep.tokenize(self.rep.encode(smiles))


class SmilesBPETokenizerWrapper(TokenizerWrapper):
    name = "smiles_bpe"

    def __init__(self, model_name_or_path: str, *, local_files_only: bool) -> None:
        self._backend = None
        model_path = self._resolve_hf_snapshot(model_name_or_path)
        if model_path is not None and (model_path / "vocab.json").exists() and (model_path / "merges.txt").exists():
            try:
                from tokenizers import ByteLevelBPETokenizer

                self._backend = ByteLevelBPETokenizer(
                    str(model_path / "vocab.json"),
                    str(model_path / "merges.txt"),
                    lowercase=False,
                    add_prefix_space=False,
                )
                return
            except Exception:
                self._backend = None
        try:
            from transformers import AutoTokenizer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "SMILES-BPE requires either `tokenizers` with local vocab.json/merges.txt "
                "or a working `transformers` installation."
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )

    @staticmethod
    def _resolve_hf_snapshot(model_name_or_path: str) -> Optional[Path]:
        path = Path(model_name_or_path).expanduser()
        if path.exists():
            return path
        if "/" not in model_name_or_path:
            return None
        cache_name = "models--" + model_name_or_path.replace("/", "--")
        roots = [
            Path(os.environ.get("HF_HOME", "")).expanduser() / "hub" if os.environ.get("HF_HOME") else None,
            Path.home() / ".cache" / "huggingface" / "hub",
            Path("/home/svu/yuxuan.ren/.cache/huggingface/hub"),
            Path("/nfs/home/svu/yuxuan.ren/.cache/huggingface/hub"),
        ]
        for root in [r for r in roots if r is not None]:
            repo = root / cache_name
            ref = repo / "refs" / "main"
            if ref.exists():
                snapshot = repo / "snapshots" / ref.read_text(encoding="utf-8").strip()
                if snapshot.exists():
                    return snapshot
            snapshots = repo / "snapshots"
            if snapshots.exists():
                candidates = sorted([p for p in snapshots.iterdir() if p.is_dir()])
                if candidates:
                    return candidates[-1]
        return None

    def tokens(self, smiles: str) -> List[str]:
        if self._backend is not None:
            return [str(x) for x in self._backend.encode(smiles).ids]
        encoded = self.tokenizer(smiles, add_special_tokens=False)
        ids = encoded.get("input_ids", [])
        return [str(x) for x in ids]


def build_tokenizers(args: argparse.Namespace) -> Dict[str, TokenizerWrapper]:
    tokenizers: Dict[str, TokenizerWrapper] = {}
    wanted = set(args.representations)
    if "stamp" in wanted:
        tokenizers["stamp"] = STAMPTokenizerWrapper(
            args.stamp_vocab,
            force_ring_aware_encode=args.stamp_force_ring_aware_encode,
            strict_vocab=args.stamp_strict_vocab,
            expand_oov_smiles=not args.no_stamp_expand_oov_smiles,
        )
    if "safe" in wanted:
        tokenizers["safe"] = SAFETokenizerWrapper()
    if "smiles_bpe" in wanted:
        tokenizers["smiles_bpe"] = SmilesBPETokenizerWrapper(
            args.smiles_bpe_model,
            local_files_only=args.smiles_bpe_local_files_only,
        )
    return tokenizers


def add_distances(df: pd.DataFrame, tokenizers: Dict[str, TokenizerWrapper], *, n_bits: int) -> pd.DataFrame:
    if df.empty:
        return df

    all_smiles = sorted(set(df["smiles_a"]).union(df["smiles_b"]))
    fp_cache = {s: _ecfp4(s, n_bits) for s in all_smiles}
    token_cache: Dict[str, Dict[str, Optional[List[str]]]] = {name: {} for name in tokenizers}
    token_error: Dict[str, Dict[str, str]] = {name: {} for name in tokenizers}

    for name, tok in tokenizers.items():
        for smiles in all_smiles:
            try:
                token_cache[name][smiles] = tok.tokens(smiles)
            except Exception as exc:
                token_cache[name][smiles] = None
                token_error[name][smiles] = f"{type(exc).__name__}: {exc}"

    chem_dist: List[float] = []
    for a, b in zip(df["smiles_a"], df["smiles_b"]):
        fa, fb = fp_cache.get(a), fp_cache.get(b)
        if fa is None or fb is None:
            chem_dist.append(math.nan)
        else:
            chem_dist.append(1.0 - float(DataStructs.TanimotoSimilarity(fa, fb)))
    df = df.copy()
    df["ecfp4_tanimoto_distance"] = chem_dist

    for name in tokenizers:
        edits: List[float] = []
        norm_edits: List[float] = []
        len_a_values: List[float] = []
        len_b_values: List[float] = []
        ok_values: List[bool] = []
        error_values: List[str] = []
        for a, b in zip(df["smiles_a"], df["smiles_b"]):
            ta = token_cache[name].get(a)
            tb = token_cache[name].get(b)
            if ta is None or tb is None:
                edits.append(math.nan)
                norm_edits.append(math.nan)
                len_a_values.append(math.nan)
                len_b_values.append(math.nan)
                ok_values.append(False)
                error_values.append(token_error[name].get(a) or token_error[name].get(b) or "")
                continue
            dist = levenshtein(ta, tb)
            denom = max(len(ta), len(tb), 1)
            edits.append(float(dist))
            norm_edits.append(float(dist) / denom)
            len_a_values.append(float(len(ta)))
            len_b_values.append(float(len(tb)))
            ok_values.append(True)
            error_values.append("")
        df[f"{name}_edit_distance"] = edits
        df[f"{name}_norm_edit_distance"] = norm_edits
        df[f"{name}_len_a"] = len_a_values
        df[f"{name}_len_b"] = len_b_values
        df[f"{name}_ok"] = ok_values
        df[f"{name}_error"] = error_values
    return df


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(x, y, nan_policy="omit").correlation)
    except Exception:
        xr = pd.Series(x).rank(method="average").to_numpy()
        yr = pd.Series(y).rank(method="average").to_numpy()
        return _pearson(xr, yr)


def _display_rep_name(rep: str) -> str:
    return {
        "stamp": "STAMP tokenizer",
        "safe": "SAFE tokenizer",
        "smiles_bpe": "SMILES-BPE tokenizer",
    }.get(rep, rep)


def summarize(df: pd.DataFrame, representations: Sequence[str]) -> dict:
    summary: dict = {
        "benchmark_type": "token_edit_locality_no_model",
        "model_training_required": False,
        "comparison_scope": (
            "Tokenizer/sequence-representation locality only: ECFP4 chemical "
            "distance versus Levenshtein edit distance over representation tokens. "
            "Do not interpret SAFE/SMILES-BPE rows as trained encoder or property-prediction baselines."
        ),
        "n_pairs": int(len(df)),
        "n_tasks": int(df["task"].nunique()) if "task" in df else 0,
        "pairs_by_task": {str(k): int(v) for k, v in Counter(df["task"]).items()} if "task" in df else {},
        "representations": {},
    }
    chem = df["ecfp4_tanimoto_distance"].to_numpy(dtype=float) if not df.empty else np.array([])
    for name in representations:
        col = f"{name}_norm_edit_distance"
        if col not in df:
            continue
        y = df[col].to_numpy(dtype=float)
        valid = np.isfinite(chem) & np.isfinite(y)
        rep_stats = {
            "valid_pairs": int(valid.sum()),
            "tokenization_failures": int((~df.get(f"{name}_ok", pd.Series([], dtype=bool))).sum()),
        }
        if valid.any():
            rep_stats.update(
                {
                    "mean_norm_edit_distance": float(np.mean(y[valid])),
                    "median_norm_edit_distance": float(np.median(y[valid])),
                    "mean_edit_distance": float(np.nanmean(df[f"{name}_edit_distance"].to_numpy(dtype=float))),
                    "median_edit_distance": float(np.nanmedian(df[f"{name}_edit_distance"].to_numpy(dtype=float))),
                    "pearson_chem_vs_norm_edit": _pearson(chem[valid], y[valid]),
                    "spearman_chem_vs_norm_edit": _spearman(chem[valid], y[valid]),
                }
            )
            low = valid & (chem <= 0.3)
            if low.any():
                rep_stats["median_norm_edit_at_ecfp_distance_le_0.3"] = float(np.median(y[low]))
                rep_stats["mean_norm_edit_at_ecfp_distance_le_0.3"] = float(np.mean(y[low]))
                rep_stats["n_ecfp_distance_le_0.3"] = int(low.sum())
        summary["representations"][name] = rep_stats
    return summary


def write_figures(df: pd.DataFrame, representations: Sequence[str], out_dir: Path, *, max_plot_points: int, seed: int) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        return
    rng = np.random.default_rng(seed)
    plot_df = df
    if max_plot_points > 0 and len(df) > max_plot_points:
        idx = rng.choice(len(df), size=max_plot_points, replace=False)
        plot_df = df.iloc[np.sort(idx)].copy()

    reps = [r for r in representations if f"{r}_norm_edit_distance" in df.columns]
    if not reps:
        return

    fig, axes = plt.subplots(1, len(reps), figsize=(5.2 * len(reps), 4.2), squeeze=False)
    for ax, rep in zip(axes[0], reps):
        x = plot_df["ecfp4_tanimoto_distance"].to_numpy(dtype=float)
        y = plot_df[f"{rep}_norm_edit_distance"].to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid], y[valid], s=7, alpha=0.18, linewidths=0)
        ax.set_title(_display_rep_name(rep))
        ax.set_xlabel("ECFP4 Tanimoto distance")
        ax.set_ylabel("normalized token edit distance")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "edit_locality_scatter.png", dpi=220)
    plt.close(fig)

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    fig, axes = plt.subplots(1, len(reps), figsize=(5.2 * len(reps), 4.4), squeeze=False)
    for ax, rep in zip(axes[0], reps):
        values: List[np.ndarray] = []
        chem = df["ecfp4_tanimoto_distance"].to_numpy(dtype=float)
        edit = df[f"{rep}_norm_edit_distance"].to_numpy(dtype=float)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = np.isfinite(chem) & np.isfinite(edit) & (chem >= lo) & (chem < hi)
            values.append(edit[mask])
        ax.boxplot(values, tick_labels=labels, showfliers=False)
        ax.set_title(_display_rep_name(rep))
        ax.set_xlabel("ECFP4 distance bin")
        ax.set_ylabel("normalized token edit distance")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "edit_locality_box_by_chem_bin.png", dpi=220)
    plt.close(fig)

    medians = []
    for rep in reps:
        chem = df["ecfp4_tanimoto_distance"].to_numpy(dtype=float)
        edit = df[f"{rep}_norm_edit_distance"].to_numpy(dtype=float)
        mask = np.isfinite(chem) & np.isfinite(edit) & (chem <= 0.3)
        medians.append(float(np.median(edit[mask])) if mask.any() else math.nan)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar([_display_rep_name(r) for r in reps], medians, color=["#2f6f73", "#c46a2b", "#4f5f9f"][: len(reps)])
    ax.set_ylabel("median normalized edit distance")
    ax.set_title("Edit locality at ECFP4 distance <= 0.3")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "edit_locality_low_chem_distance_bar.png", dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--moleculeace-dir", default="/scratch/yuxuan.ren/data/moleculeace")
    ap.add_argument("--output-dir", default="results/moleculeace_mmp_edit_locality")
    ap.add_argument("--input-csv", default=None, help="Existing pairs CSV; only regenerate summary/figures.")
    ap.add_argument("--append-distances", action="store_true", help="With --input-csv, compute/add distances for --representations.")
    ap.add_argument("--skip-plots", action="store_true", help="Write CSV/summary but do not import matplotlib or save figures.")
    ap.add_argument("--tasks", nargs="*", default=None, help="Optional MoleculeACE task stems, e.g. CHEMBL1862_Ki.")
    ap.add_argument("--representations", nargs="+", default=["stamp", "safe", "smiles_bpe"], choices=["stamp", "safe", "smiles_bpe"])
    ap.add_argument("--stamp-vocab", default="/scratch/yuxuan.ren/data/pubchem10m/vocab_psvae_ring_v10000_n20k/vocab_top4666.txt")
    ap.add_argument("--stamp-force-ring-aware-encode", action="store_true")
    ap.add_argument("--stamp-strict-vocab", action="store_true")
    ap.add_argument("--no-stamp-expand-oov-smiles", action="store_true")
    ap.add_argument("--smiles-bpe-model", default="seyonec/PubChem10M_SMILES_BPE_60k")
    ap.add_argument("--smiles-bpe-local-files-only", action="store_true")
    ap.add_argument("--ecfp-bits", type=int, default=2048)
    ap.add_argument("--strip-salts", action="store_true", help="Default preserves salts/fragments.")
    ap.add_argument("--max-cut-bonds", type=int, default=20)
    ap.add_argument("--all-context-sides", action="store_true")
    ap.add_argument("--keep-context-duplicates", action="store_true")
    ap.add_argument("--min-context-heavy-atoms", type=int, default=5)
    ap.add_argument("--max-variable-heavy-atoms", type=int, default=0, help="0 disables filtering.")
    ap.add_argument("--max-pairs-per-context", type=int, default=0, help="0 keeps all pairs per context.")
    ap.add_argument("--max-pairs-total", type=int, default=0, help="0 keeps all pairs.")
    ap.add_argument("--max-plot-points", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=13)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_csv:
        pairs = pd.read_csv(args.input_csv)
        print(f"loaded existing pairs={len(pairs)} from {args.input_csv}", flush=True)
        if args.append_distances:
            tokenizers = build_tokenizers(args)
            pairs = add_distances(pairs, tokenizers, n_bits=args.ecfp_bits)
        else:
            tokenizers = {name: None for name in args.representations}
    else:
        records = load_moleculeace_records(
            Path(args.moleculeace_dir),
            strip_salts=args.strip_salts,
            tasks=args.tasks,
        )
        print(f"loaded tasks={len(records)} molecules={sum(len(v) for v in records.values())}", flush=True)

        pairs = enumerate_mmp_pairs(
            records,
            max_cut_bonds=args.max_cut_bonds,
            all_context_sides=args.all_context_sides,
            keep_context_duplicates=args.keep_context_duplicates,
            min_context_heavy_atoms=args.min_context_heavy_atoms,
            max_variable_heavy_atoms=args.max_variable_heavy_atoms,
            max_pairs_per_context=args.max_pairs_per_context,
            max_pairs_total=args.max_pairs_total,
            seed=args.seed,
        )
        print(f"mined pairs={len(pairs)}", flush=True)

        tokenizers = build_tokenizers(args)
        pairs = add_distances(pairs, tokenizers, n_bits=args.ecfp_bits)

    csv_path = out_dir / "mmp_edit_locality_pairs.csv"
    pairs.to_csv(csv_path, index=False)
    summary = summarize(pairs, list(tokenizers))
    summary.update(
        {
            "moleculeace_dir": str(Path(args.moleculeace_dir).resolve()),
            "stamp_vocab": args.stamp_vocab if "stamp" in tokenizers else None,
            "stamp_strict_vocab": bool(args.stamp_strict_vocab),
            "stamp_expand_oov_smiles": not bool(args.no_stamp_expand_oov_smiles),
            "strip_salts": bool(args.strip_salts),
            "min_context_heavy_atoms": int(args.min_context_heavy_atoms),
            "max_variable_heavy_atoms": int(args.max_variable_heavy_atoms),
            "keep_context_duplicates": bool(args.keep_context_duplicates),
        }
    )
    (out_dir / "mmp_edit_locality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.skip_plots:
        write_figures(pairs, list(tokenizers), out_dir, max_plot_points=args.max_plot_points, seed=args.seed)
    print(f"saved {csv_path}", flush=True)
    print(f"saved {out_dir / 'mmp_edit_locality_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
