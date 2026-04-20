#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED
from rdkit.Chem import rdFMCS

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.common import canonicalize_smiles  # noqa: E402
from stamp.benchmark.lm import load_trained_model  # noqa: E402
from stamp.benchmark.metrics import _internal_diversity, _load_sa_scorer  # noqa: E402
from stamp.benchmark.tasks import (  # noqa: E402
    _build_full_sequence_mask_row,
    _build_group_keep_row,
    _encode_completion_prompt_tokens,
    _encode_stamp_groups_from_smiles,
    _sample_ao_infill_sequences,
)

GENMOL_BASELINES: Dict[str, Dict[str, Dict[str, float]]] = {
    "GenMol": {
        "linker_design": {"validity": 100.0, "uniqueness": 83.7, "quality": 21.9, "diversity": 0.547, "distance": 0.563},
        "motif_extension": {"validity": 82.9, "uniqueness": 77.5, "quality": 30.1, "diversity": 0.617, "distance": 0.682},
        "scaffold_decoration": {"validity": 96.6, "uniqueness": 82.7, "quality": 31.8, "diversity": 0.591, "distance": 0.651},
        "superstructure_generation": {"validity": 97.5, "uniqueness": 83.6, "quality": 34.8, "diversity": 0.599, "distance": 0.762},
    },
    "GenMol V2": {
        "linker_design": {"validity": 100.0, "uniqueness": 76.6, "quality": 18.4, "diversity": 0.512, "distance": 0.539},
        "motif_extension": {"validity": 99.4, "uniqueness": 84.5, "quality": 49.0, "diversity": 0.626, "distance": 0.659},
        "scaffold_decoration": {"validity": 99.2, "uniqueness": 90.5, "quality": 39.7, "diversity": 0.571, "distance": 0.604},
        "superstructure_generation": {"validity": 99.7, "uniqueness": 89.8, "quality": 39.0, "diversity": 0.551, "distance": 0.769},
    },
}


def _strip_dummy_atoms(mol: Chem.Mol) -> Optional[Chem.Mol]:
    rw = Chem.RWMol(mol)
    to_remove = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(idx)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        try:
            Chem.SanitizeMol(
                out,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            )
        except Exception:
            return None
    return out


def _drop_stereo(mol: Chem.Mol) -> Chem.Mol:
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        if atom.HasProp("_CIPCode"):
            atom.ClearProp("_CIPCode")
    for bond in out.GetBonds():
        bond.SetStereo(Chem.rdchem.BondStereo.STEREONONE)
    return out


def _canonical_reparse(mol: Chem.Mol) -> Optional[Chem.Mol]:
    try:
        smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def _relaxed_match(source_mol: Chem.Mol, query_mol: Chem.Mol) -> set[int]:
    variants = [query_mol, _drop_stereo(query_mol)]
    reparsed = _canonical_reparse(query_mol)
    if reparsed is not None:
        variants.append(reparsed)
    reparsed_nostereo = _canonical_reparse(_drop_stereo(query_mol))
    if reparsed_nostereo is not None:
        variants.append(reparsed_nostereo)

    for q in variants:
        if q is None or q.GetNumAtoms() == 0:
            continue
        try:
            match = source_mol.GetSubstructMatch(q, useChirality=False)
        except Exception:
            match = ()
        if match:
            return set(int(x) for x in match)

    try:
        params = rdFMCS.MCSParameters()
        params.Timeout = 5
        params.AtomCompareParameters.MatchValences = False
        params.BondCompareParameters.RingMatchesRingOnly = True
        res = rdFMCS.FindMCS([source_mol, _drop_stereo(query_mol)], parameters=params)
        if res.smartsString:
            mcs = Chem.MolFromSmarts(res.smartsString)
            if mcs is not None and mcs.GetNumAtoms() >= max(1, math.floor(query_mol.GetNumAtoms() * 0.5)):
                match = source_mol.GetSubstructMatch(mcs)
                if match:
                    return set(int(x) for x in match)
    except Exception:
        pass
    return set()


def _match_query_part_atoms(source_mol: Chem.Mol, part_smiles: str) -> set[int]:
    q = Chem.MolFromSmiles(part_smiles)
    if q is None:
        q = Chem.MolFromSmiles(part_smiles, sanitize=False)
    if q is None:
        return set()
    q = _strip_dummy_atoms(q)
    if q is None or q.GetNumAtoms() == 0:
        return set()
    return _relaxed_match(source_mol, q)


def _match_query_atoms(source_mol: Chem.Mol, query_smiles: str) -> set[int]:
    atoms: set[int] = set()
    for part in [p for p in str(query_smiles).split(".") if p]:
        atoms.update(_match_query_part_atoms(source_mol, part))
    return atoms


def _group_adjacency(groups) -> Dict[int, set[int]]:
    adj: Dict[int, set[int]] = {int(g.frag_id): set() for g in groups}
    for g in groups:
        gid = int(g.frag_id)
        pid = g.parent_frag_id
        if pid is None:
            continue
        adj[gid].add(int(pid))
        adj[int(pid)].add(gid)
    return adj


def _keep_group_indices_for_atoms(groups, atom_set: set[int]) -> list[int]:
    keep: list[int] = []
    for idx, group in enumerate(groups):
        group_atoms = set(int(x) for x in group.atom_indices)
        if group_atoms and group_atoms.issubset(atom_set):
            keep.append(idx)
    if keep:
        return keep
    for idx, group in enumerate(groups):
        group_atoms = set(int(x) for x in group.atom_indices)
        if group_atoms and (len(group_atoms & atom_set) / max(1, len(group_atoms))) >= 0.5:
            keep.append(idx)
    if keep:
        return keep
    for idx, group in enumerate(groups):
        group_atoms = set(int(x) for x in group.atom_indices)
        if group_atoms & atom_set:
            keep.append(idx)
    return keep


def _group_is_in_vocab(group, vocab) -> bool:
    return all(tok in vocab.stoi for tok in group.tokens)


def _build_genmol_prompt_row(rep, vocab, source_smiles: str, task_name: str, fragment_smiles: str) -> Optional[dict[str, Any]]:
    encoded = _encode_stamp_groups_from_smiles(rep, source_smiles)
    if encoded is None:
        return None
    source_mol, source_tokens, groups = encoded
    parts = [p for p in str(fragment_smiles).split(".") if p]
    keep_atoms = _match_query_atoms(source_mol, fragment_smiles)
    if not keep_atoms:
        return None
    keep_indices = _keep_group_indices_for_atoms(groups, keep_atoms)
    if not keep_indices:
        return None
    keep_indices = [idx for idx in keep_indices if _group_is_in_vocab(groups[idx], vocab)]
    if not keep_indices:
        return None
    if task_name == "linker_design" and len(parts) >= 2 and len(set(keep_indices)) == len(groups):
        part_keep_sets = []
        for part in parts[:2]:
            part_atoms = _match_query_part_atoms(source_mol, part)
            if not part_atoms:
                continue
            keep_set = {idx for idx in _keep_group_indices_for_atoms(groups, part_atoms) if _group_is_in_vocab(groups[idx], vocab)}
            if keep_set:
                part_keep_sets.append(keep_set)
        if len(part_keep_sets) >= 2:
            adj = _group_adjacency(groups)
            boundary_mask: set[int] = set()
            for a in part_keep_sets[0]:
                for b in part_keep_sets[1]:
                    if b in adj.get(int(a), set()):
                        boundary_mask.update([int(a), int(b)])
            if not boundary_mask:
                boundary_mask = min(part_keep_sets, key=len)
            row = _build_full_sequence_mask_row(
                rep,
                source_smiles=source_smiles,
                source_tokens=source_tokens,
                groups=groups,
                mask_group_indices=sorted(boundary_mask),
                task=task_name,
            )
            if row is None:
                return None
            row["query_smiles"] = fragment_smiles
            row["n_kept_groups"] = int(len(groups) - len(boundary_mask))
            row["fallback_mode"] = "linker_boundary_mask"
            return row
    row = _build_group_keep_row(rep, source_smiles, source_tokens, groups, keep_indices)
    if row is None:
        return None
    row["task"] = task_name
    row["query_smiles"] = fragment_smiles
    row["n_kept_groups"] = int(len(keep_indices))
    return row


def _mean_tanimoto_distance(original_smiles: str, smiles_list: list[str]) -> float:
    if not smiles_list:
        return float("nan")
    mol = Chem.MolFromSmiles(original_smiles)
    if mol is None:
        return float("nan")
    fps_ref = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024))
    if not fps:
        return float("nan")
    dists = DataStructs.BulkTanimotoSimilarity(fps_ref, fps, returnDistance=True)
    return float(sum(dists) / len(dists))


def _safe_mean(values: list[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate AO-GPT on GenMol original fragment benchmark.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--fragments-csv", default="/tmp/genmol_repo/data/fragments.csv")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    RDLogger.DisableLog("rdApp.*")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, vocab, rep, dev = load_trained_model(args.model_dir, device=args.device)
    if getattr(rep, "name", "") != "stamp":
        raise ValueError("This benchmark script currently expects a STAMP model.")

    df = pd.read_csv(args.fragments_csv)
    sa = _load_sa_scorer()
    tasks = ["linker_design", "motif_extension", "scaffold_decoration", "superstructure_generation"]
    all_results: Dict[str, Any] = {"model_dir": str(args.model_dir), "fragments_csv": str(args.fragments_csv), "num_samples": int(args.num_samples), "temperature": float(args.temperature), "top_k": int(args.top_k), "tasks": {}}

    for task in tasks:
        validity: list[float] = []
        uniqueness: list[float] = []
        diversity: list[float] = []
        distance: list[float] = []
        quality: list[float] = []
        per_case: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            built = _build_genmol_prompt_row(rep, vocab, row["smiles"], task, row[task])
            if built is None:
                validity.append(0.0)
                uniqueness.append(0.0)
                diversity.append(0.0)
                distance.append(float("nan"))
                quality.append(0.0)
                per_case.append({"name": row["name"], "built": False})
                continue
            prompt_tokens = _encode_completion_prompt_tokens(rep, built["masked_repr"], is_repr=True)
            seqs = _sample_ao_infill_sequences(
                model,
                vocab,
                prompt_tokens=prompt_tokens,
                n_samples=int(args.num_samples),
                temperature=float(args.temperature),
                top_k=int(args.top_k),
                device=dev,
            )
            decoded: list[str] = []
            for toks in seqs:
                full_repr = rep.detokenize(toks)
                try:
                    s = rep.decode(full_repr)
                except Exception:
                    s = ""
                if s:
                    decoded.append(s)
            valid_canonical = [canonicalize_smiles(s, require_single_component=True) for s in decoded]
            valid_smiles = [s for s in valid_canonical if s is not None]
            valid_smiles = [str(s) for s in valid_smiles]
            validity.append(len(valid_smiles) / max(1, int(args.num_samples)))
            if valid_smiles:
                uniq = list(dict.fromkeys(valid_smiles))
                uniqueness.append(len(uniq) / len(valid_smiles))
                diversity.append(_internal_diversity(uniq, max_pairs=5000))
                distance.append(_mean_tanimoto_distance(row["smiles"], uniq))
                qual_count = 0
                if sa is not None:
                    for s in uniq:
                        mol = Chem.MolFromSmiles(s)
                        if mol is None:
                            continue
                        try:
                            sa_val = float(sa.calculateScore(mol))
                        except Exception:
                            continue
                        qed_val = float(QED.qed(mol))
                        if sa_val <= 4.0 and qed_val >= 0.6:
                            qual_count += 1
                quality.append(qual_count / max(1, int(args.num_samples)))
                per_case.append(
                    {
                        "name": row["name"],
                        "built": True,
                        "prompt_repr": built["prompt_repr"],
                        "masked_repr": built["masked_repr"],
                        "query_smiles": built["query_smiles"],
                        "validity": validity[-1],
                        "uniqueness": uniqueness[-1],
                        "diversity": diversity[-1],
                        "distance": distance[-1],
                        "quality": quality[-1],
                    }
                )
            else:
                uniqueness.append(0.0)
                diversity.append(0.0)
                distance.append(float("nan"))
                quality.append(0.0)
                per_case.append(
                    {
                        "name": row["name"],
                        "built": True,
                        "prompt_repr": built["prompt_repr"],
                        "masked_repr": built["masked_repr"],
                        "query_smiles": built["query_smiles"],
                        "validity": 0.0,
                        "uniqueness": 0.0,
                        "diversity": 0.0,
                        "distance": float("nan"),
                        "quality": 0.0,
                    }
                )

        summary = {
            "validity": 100.0 * _safe_mean(validity),
            "uniqueness": 100.0 * _safe_mean(uniqueness),
            "quality": 100.0 * _safe_mean(quality),
            "diversity": _safe_mean(diversity),
            "distance": _safe_mean(distance),
            "n_cases": int(len(per_case)),
            "genmol_baseline": GENMOL_BASELINES["GenMol"].get(task),
            "genmol_v2_baseline": GENMOL_BASELINES["GenMol V2"].get(task),
            "cases": per_case,
        }
        all_results["tasks"][task] = summary

    (out_dir / "genmol_fragment_benchmark.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {m: v for m, v in task_res.items() if m in ['validity','uniqueness','quality','diversity','distance']} for k, task_res in all_results['tasks'].items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
