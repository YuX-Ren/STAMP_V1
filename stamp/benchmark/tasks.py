from __future__ import annotations

import csv
import json
import math
import os
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdchem import BondType
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .common import canonicalize_smiles, iterable_with_progress
from .lm import LMConfig, NextTokenDataset, TinyDecoderLM, TokenVocab, _ids_with_special_tokens, _resolve_amp_mode, _resolve_torch_device, _set_seed, build_dynamic_lm_collate_fn, load_trained_model, sample_model_sequences
from .representations import RepresentationError, UnsupportedRepresentation, build_representation

_COMP_WORKER_REP = None
_COMP_WORKER_CFG: tuple[str, str, Optional[str], Optional[str]] | None = None
_EVAL_WORKER_REP = None
_MULTI_COMPONENT_PROMPT_SEP = "<DOT>"


def _scaffold_row_from_smiles(smiles: str, min_scaffold_heavy_atoms: int) -> Optional[tuple[str, str, str]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None or scaf.GetNumAtoms() == 0:
        return None
    if scaf.GetNumHeavyAtoms() < min_scaffold_heavy_atoms:
        return None
    scaf_s = Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=True)
    can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if scaf_s == can:
        return None
    scaf_prompt_mol = _add_scaffold_attachment_dummies(mol, scaf)
    if scaf_prompt_mol is None:
        return None
    scaf_prompt = Chem.MolToSmiles(scaf_prompt_mol, canonical=True, isomericSmiles=True)
    return can, scaf_s, scaf_prompt


def _build_scaffold_worker(item: tuple[str, int]) -> Optional[tuple[str, str, str]]:
    smiles, min_scaffold_heavy_atoms = item
    return _scaffold_row_from_smiles(smiles, min_scaffold_heavy_atoms)


def _add_scaffold_attachment_dummies(mol: Chem.Mol, scaffold: Chem.Mol) -> Optional[Chem.Mol]:
    match = mol.GetSubstructMatch(scaffold)
    if not match:
        return None
    match_set = set(match)
    rw = Chem.RWMol(scaffold)
    attach_sites: list[int] = []
    for scaf_atom_idx, mol_atom_idx in enumerate(match):
        atom = mol.GetAtomWithIdx(mol_atom_idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in match_set:
                attach_sites.append(scaf_atom_idx)
    if not attach_sites:
        return None
    for scaf_atom_idx in attach_sites:
        dummy = Chem.Atom(0)
        dummy.SetIsotope(0)
        d_idx = rw.AddAtom(dummy)
        rw.AddBond(scaf_atom_idx, d_idx, BondType.SINGLE)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def _remap_dummy_isotopes(mol: Chem.Mol, mapping: dict[int, int]) -> Optional[Chem.Mol]:
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        iso = int(atom.GetIsotope())
        if iso not in mapping:
            return None
        atom.SetIsotope(int(mapping[iso]))
    return out


def _relabel_fragment_dummy(mol: Chem.Mol, label: int) -> Optional[Chem.Mol]:
    out = Chem.Mol(mol)
    dummies = [a for a in out.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        return None
    dummies[0].SetIsotope(int(label))
    return out


def _relabel_linker_dummies(mol: Chem.Mol) -> Optional[Chem.Mol]:
    out = Chem.Mol(mol)
    dummies = [a for a in out.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        return None
    dummies = sorted(dummies, key=lambda a: a.GetIdx())
    dummies[0].SetIsotope(1)
    dummies[1].SetIsotope(2)
    return out


def _normalize_linker_triplet(
    frag_a: Chem.Mol,
    frag_b: Chem.Mol,
    linker: Chem.Mol,
) -> Optional[tuple[str, str, str, str, str, str]]:
    if sum(1 for a in frag_a.GetAtoms() if a.GetAtomicNum() == 0) != 1:
        return None
    if sum(1 for a in frag_b.GetAtoms() if a.GetAtomicNum() == 0) != 1:
        return None
    if sum(1 for a in linker.GetAtoms() if a.GetAtomicNum() == 0) != 2:
        return None

    fa_plain = _strip_dummy_atoms(frag_a)
    fb_plain = _strip_dummy_atoms(frag_b)
    lk_plain = _strip_dummy_atoms(linker)
    if fa_plain is None or fb_plain is None or lk_plain is None:
        return None
    if fa_plain.GetNumAtoms() == 0 or fb_plain.GetNumAtoms() == 0 or lk_plain.GetNumAtoms() == 0:
        return None

    fa_plain_s = Chem.MolToSmiles(fa_plain, canonical=True, isomericSmiles=True)
    fb_plain_s = Chem.MolToSmiles(fb_plain, canonical=True, isomericSmiles=True)
    if fb_plain_s < fa_plain_s:
        frag_a, frag_b = frag_b, frag_a
        fa_plain_s, fb_plain_s = fb_plain_s, fa_plain_s

    fa_m = _relabel_fragment_dummy(frag_a, label=1)
    fb_m = _relabel_fragment_dummy(frag_b, label=2)
    lk_m = _relabel_linker_dummies(linker)
    if fa_m is None or fb_m is None or lk_m is None:
        return None

    fa_ap_s = Chem.MolToSmiles(fa_m, canonical=True, isomericSmiles=True)
    fb_ap_s = Chem.MolToSmiles(fb_m, canonical=True, isomericSmiles=True)
    lk_ap_s = Chem.MolToSmiles(lk_m, canonical=True, isomericSmiles=True)
    lk_plain_s = Chem.MolToSmiles(lk_plain, canonical=True, isomericSmiles=True)
    return fa_plain_s, fb_plain_s, lk_plain_s, fa_ap_s, fb_ap_s, lk_ap_s


def _assemble_labeled_linker_fragments(
    frag_a_ap: str,
    frag_b_ap: str,
    linker_ap: str,
) -> Optional[str]:
    ma = Chem.MolFromSmiles(frag_a_ap)
    mb = Chem.MolFromSmiles(frag_b_ap)
    ml = Chem.MolFromSmiles(linker_ap)
    if ma is None or mb is None or ml is None:
        return None

    combined = Chem.CombineMols(Chem.CombineMols(ma, mb), ml)
    rw = Chem.RWMol(combined)
    by_iso: dict[int, list[int]] = {1: [], 2: []}
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() in by_iso:
            by_iso[int(atom.GetIsotope())].append(atom.GetIdx())
    if len(by_iso[1]) != 2 or len(by_iso[2]) != 2:
        return None

    to_remove: list[int] = []
    for iso in (1, 2):
        d1, d2 = by_iso[iso]
        a1_nbr = rw.GetAtomWithIdx(d1).GetNeighbors()
        a2_nbr = rw.GetAtomWithIdx(d2).GetNeighbors()
        if len(a1_nbr) != 1 or len(a2_nbr) != 1:
            return None
        a1 = a1_nbr[0].GetIdx()
        a2 = a2_nbr[0].GetIdx()
        b1 = rw.GetBondBetweenAtoms(d1, a1)
        b2 = rw.GetBondBetweenAtoms(d2, a2)
        if b1 is None or b2 is None:
            return None
        bt = b1.GetBondType() if b1.GetBondType() == b2.GetBondType() else BondType.SINGLE
        if rw.GetBondBetweenAtoms(a1, a2) is None:
            rw.AddBond(a1, a2, bt)
        to_remove.extend([d1, d2])

    for idx in sorted(set(to_remove), reverse=True):
        rw.RemoveAtom(idx)

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    if len(Chem.GetMolFrags(out)) != 1:
        return None
    return Chem.MolToSmiles(out, canonical=True, isomericSmiles=True)


def _linker_row_from_smiles(smiles: str) -> Optional[tuple[str, str, str, str, str, str, str]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    brics_bonds = []
    for (a, b), _ in BRICS.FindBRICSBonds(mol):
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is not None:
            brics_bonds.append(bond.GetIdx())

    if len(brics_bonds) < 2:
        return None

    for b1, b2 in combinations(brics_bonds[:8], 2):
        frag_mol = Chem.FragmentOnBonds(mol, [b1, b2], addDummies=True)
        frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
        if len(frags) != 3:
            continue

        linker_idx = -1
        for i, fm in enumerate(frags):
            n_dummy = sum(1 for a in fm.GetAtoms() if a.GetAtomicNum() == 0)
            if n_dummy == 2:
                linker_idx = i
                break
        if linker_idx < 0:
            continue

        other = [i for i in range(3) if i != linker_idx]
        fa = frags[other[0]]
        fb = frags[other[1]]
        lk = frags[linker_idx]

        normalized = _normalize_linker_triplet(fa, fb, lk)
        if normalized is None:
            continue
        fa_s, fb_s, lk_s, fa_ap, fb_ap, lk_ap = normalized

        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return can, fa_s, fb_s, lk_s, fa_ap, fb_ap, lk_ap

    return None


def _build_linker_worker(smiles: str) -> Optional[tuple[str, str, str, str, str, str, str]]:
    return _linker_row_from_smiles(smiles)


def _tokenize_completion_row_local(
    rep,
    row: dict[str, str],
    prompt_key: str,
    target_key: str,
    prompt_template: Optional[str],
    target_template: Optional[str],
) -> Optional[list[str]]:
    try:
        prompt_raw = prompt_template.format(**row) if prompt_template is not None else row[prompt_key]
        target_raw = target_template.format(**row) if target_template is not None else row[target_key]
        prompt_tokens = _encode_completion_prompt_tokens(rep, prompt_raw)
        target_text = rep.encode(target_raw)
        target_tokens = rep.tokenize(target_text)
    except Exception:
        return None
    if not prompt_tokens or not target_tokens:
        return None
    return prompt_tokens + ["<SEP>"] + target_tokens


def _encode_completion_prompt_tokens(rep, prompt_raw: str) -> list[str]:
    # STAMP cannot encode disconnected molecules (e.g. "frag_a.frag_b") in one pass.
    # For linker prompts, encode each component separately and stitch token streams.
    if getattr(rep, "name", "") == "stamp" and "." in prompt_raw:
        parts = [part for part in prompt_raw.split(".") if part]
        if len(parts) >= 2:
            out: list[str] = []
            for part in parts:
                part_tokens = rep.tokenize(rep.encode(part))
                if not part_tokens:
                    raise RepresentationError("Empty prompt component after STAMP tokenization.")
                if out:
                    out.append(_MULTI_COMPONENT_PROMPT_SEP)
                out.extend(part_tokens)
            return out

    return rep.tokenize(rep.encode(prompt_raw))


def _init_completion_worker(
    representation: str,
    prompt_key: str,
    target_key: str,
    prompt_template: Optional[str],
    target_template: Optional[str],
) -> None:
    global _COMP_WORKER_REP, _COMP_WORKER_CFG
    _COMP_WORKER_REP = build_representation(representation)
    _COMP_WORKER_CFG = (prompt_key, target_key, prompt_template, target_template)


def _tokenize_completion_row_worker(row: dict[str, str]) -> Optional[list[str]]:
    rep = _COMP_WORKER_REP
    cfg = _COMP_WORKER_CFG
    if rep is None or cfg is None:
        raise RuntimeError("Completion tokenize worker is not initialized.")
    prompt_key, target_key, prompt_template, target_template = cfg
    return _tokenize_completion_row_local(
        rep,
        row,
        prompt_key=prompt_key,
        target_key=target_key,
        prompt_template=prompt_template,
        target_template=target_template,
    )


def _tokenize_completion_records(
    records: Sequence[dict[str, str]],
    representation: str,
    prompt_key: str,
    target_key: str,
    prompt_template: Optional[str],
    target_template: Optional[str],
    show_progress: bool,
    desc: str,
    num_workers: int,
    chunksize: int,
) -> list[list[str]]:
    rep = build_representation(representation)
    if num_workers <= 0:
        num_workers = max(1, os.cpu_count() or 1)

    seqs: list[list[str]] = []
    if num_workers == 1:
        iterator = iterable_with_progress(
            records,
            show_progress=show_progress,
            total=len(records),
            desc=desc,
        )
        for row in iterator:
            toks = _tokenize_completion_row_local(
                rep,
                row,
                prompt_key=prompt_key,
                target_key=target_key,
                prompt_template=prompt_template,
                target_template=target_template,
            )
            if toks:
                seqs.append(toks)
        return seqs

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_completion_worker,
        initargs=(representation, prompt_key, target_key, prompt_template, target_template),
    ) as pool:
        mapped = pool.map(
            _tokenize_completion_row_worker,
            records,
            chunksize=max(1, int(chunksize)),
        )
        mapped = iterable_with_progress(
            mapped,
            show_progress=show_progress,
            total=len(records),
            desc=desc,
        )
        for toks in mapped:
            if toks:
                seqs.append(toks)
    return seqs


def _init_eval_worker(representation: str) -> None:
    global _EVAL_WORKER_REP
    _EVAL_WORKER_REP = build_representation(representation)


def _eval_scaffold_prediction_local(rep, pred_repr: str, scaffold: str) -> tuple[str, int, int]:
    pred_s = ""
    is_valid = False
    preserve = False
    try:
        pred_s = rep.decode(pred_repr)
        is_valid = canonicalize_smiles(pred_s, require_single_component=True) is not None
    except Exception:
        is_valid = False

    if is_valid:
        gmol = Chem.MolFromSmiles(pred_s)
        smol = Chem.MolFromSmiles(scaffold)
        if gmol is not None and smol is not None and gmol.HasSubstructMatch(smol):
            preserve = True
    return pred_s, int(is_valid), int(preserve)


def _eval_scaffold_prediction_worker(payload: tuple[str, str]) -> tuple[str, int, int]:
    rep = _EVAL_WORKER_REP
    if rep is None:
        raise RuntimeError("Eval worker is not initialized.")
    pred_repr, scaffold = payload
    return _eval_scaffold_prediction_local(rep, pred_repr=pred_repr, scaffold=scaffold)


def _eval_linker_prediction_local(
    rep,
    pred_repr: str,
    frag_a: str,
    frag_b: str,
) -> tuple[str, str, int, int, int]:
    pred_s = ""
    pred_linker = ""
    valid = False
    preserve = False
    connect_ok = False
    try:
        pred_s = rep.decode(pred_repr)
        pred_linker = pred_s
        valid = canonicalize_smiles(pred_s, require_single_component=True) is not None
    except Exception:
        valid = False
    if valid:
        gm = Chem.MolFromSmiles(pred_s)
        qa = Chem.MolFromSmiles(frag_a)
        qb = Chem.MolFromSmiles(frag_b)
        if gm is not None and qa is not None and qb is not None:
            has_a = gm.HasSubstructMatch(qa)
            has_b = gm.HasSubstructMatch(qb)
            preserve = has_a and has_b
            if preserve and len(Chem.GetMolFrags(gm)) == 1:
                connect_ok = True
    return pred_s, pred_linker, int(valid), int(preserve), int(connect_ok)


def _eval_linker_prediction_from_linker_local(
    rep,
    pred_repr: str,
    frag_a: str,
    frag_b: str,
    frag_a_ap: str,
    frag_b_ap: str,
) -> tuple[str, str, int, int, int]:
    pred_s = ""
    pred_linker = ""
    valid = False
    preserve = False
    connect_ok = False
    try:
        pred_linker = rep.decode(pred_repr)
    except Exception:
        return pred_s, pred_linker, int(valid), int(preserve), int(connect_ok)

    assembled = _assemble_labeled_linker_fragments(frag_a_ap=frag_a_ap, frag_b_ap=frag_b_ap, linker_ap=pred_linker)
    if assembled is None:
        return pred_s, pred_linker, int(valid), int(preserve), int(connect_ok)

    pred_s = assembled
    valid = canonicalize_smiles(pred_s, require_single_component=True) is not None
    if valid:
        gm = Chem.MolFromSmiles(pred_s)
        qa = Chem.MolFromSmiles(frag_a)
        qb = Chem.MolFromSmiles(frag_b)
        if gm is not None and qa is not None and qb is not None:
            has_a = gm.HasSubstructMatch(qa)
            has_b = gm.HasSubstructMatch(qb)
            preserve = has_a and has_b
            connect_ok = bool(preserve and len(Chem.GetMolFrags(gm)) == 1)
    return pred_s, pred_linker, int(valid), int(preserve), int(connect_ok)


def _eval_linker_prediction_worker(
    payload: tuple[str, str, str, str, str, str],
) -> tuple[str, str, int, int, int]:
    rep = _EVAL_WORKER_REP
    if rep is None:
        raise RuntimeError("Eval worker is not initialized.")
    mode, pred_repr, frag_a, frag_b, frag_a_ap, frag_b_ap = payload
    if mode == "linker":
        return _eval_linker_prediction_from_linker_local(
            rep,
            pred_repr=pred_repr,
            frag_a=frag_a,
            frag_b=frag_b,
            frag_a_ap=frag_a_ap,
            frag_b_ap=frag_b_ap,
        )
    return _eval_linker_prediction_local(rep, pred_repr=pred_repr, frag_a=frag_a, frag_b=frag_b)


def build_scaffold_completion_dataset(
    smiles_list: Sequence[str],
    output_path: str | Path,
    min_scaffold_heavy_atoms: int = 5,
    max_items: Optional[int] = None,
    show_progress: bool = False,
    num_workers: int = 1,
    chunksize: int = 128,
) -> dict[str, Any]:
    rows = smiles_list[: max_items if max_items is not None else len(smiles_list)]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "scaffold_smiles", "scaffold_prompt"])
        if num_workers <= 0:
            num_workers = max(1, os.cpu_count() or 1)

        if num_workers == 1:
            iterator = iterable_with_progress(
                rows,
                show_progress=show_progress,
                total=len(rows),
                desc="Build scaffold dataset",
            )
            for s in iterator:
                total += 1
                row = _scaffold_row_from_smiles(s, min_scaffold_heavy_atoms=min_scaffold_heavy_atoms)
                if row is None:
                    continue
                writer.writerow([row[0], row[1], row[2]])
                kept += 1
        else:
            payloads = [(s, int(min_scaffold_heavy_atoms)) for s in rows]
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                mapped = pool.map(
                    _build_scaffold_worker,
                    payloads,
                    chunksize=max(1, int(chunksize)),
                )
                mapped = iterable_with_progress(
                    mapped,
                    show_progress=show_progress,
                    total=len(rows),
                    desc="Build scaffold dataset",
                )
                for row in mapped:
                    total += 1
                    if row is None:
                        continue
                    writer.writerow([row[0], row[1], row[2]])
                    kept += 1

    summary = {
        "output": str(out),
        "total_input": total,
        "kept": kept,
        "keep_rate": kept / max(1, total),
    }
    (out.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _strip_dummy_atoms(mol: Chem.Mol) -> Optional[Chem.Mol]:
    rw = Chem.RWMol(mol)
    to_remove = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(idx)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def build_linker_design_dataset(
    smiles_list: Sequence[str],
    output_path: str | Path,
    max_items: Optional[int] = None,
    show_progress: bool = False,
    num_workers: int = 1,
    chunksize: int = 128,
) -> dict[str, Any]:
    rows = smiles_list[: max_items if max_items is not None else len(smiles_list)]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "smiles",
                "fragment_a",
                "fragment_b",
                "linker",
                "fragment_a_ap",
                "fragment_b_ap",
                "linker_ap",
            ]
        )
        if num_workers <= 0:
            num_workers = max(1, os.cpu_count() or 1)

        if num_workers == 1:
            iterator = iterable_with_progress(
                rows,
                show_progress=show_progress,
                total=len(rows),
                desc="Build linker dataset",
            )
            for s in iterator:
                total += 1
                row = _linker_row_from_smiles(s)
                if row is None:
                    continue
                writer.writerow([row[0], row[1], row[2], row[3], row[4], row[5], row[6]])
                kept += 1
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                mapped = pool.map(
                    _build_linker_worker,
                    rows,
                    chunksize=max(1, int(chunksize)),
                )
                mapped = iterable_with_progress(
                    mapped,
                    show_progress=show_progress,
                    total=len(rows),
                    desc="Build linker dataset",
                )
                for row in mapped:
                    total += 1
                    if row is None:
                        continue
                    writer.writerow([row[0], row[1], row[2], row[3], row[4], row[5], row[6]])
                    kept += 1

    summary = {
        "output": str(out),
        "total_input": total,
        "kept": kept,
        "keep_rate": kept / max(1, total),
    }
    (out.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _load_scaffold_rows(path: str | Path, max_rows: Optional[int] = None) -> list[dict[str, str]]:
    out = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                {
                    "smiles": row["smiles"],
                    "scaffold_smiles": row["scaffold_smiles"],
                    "scaffold_prompt": row.get("scaffold_prompt", row["scaffold_smiles"]),
                }
            )
            if max_rows is not None and len(out) >= max_rows:
                break
    return out


def _load_linker_rows(path: str | Path, max_rows: Optional[int] = None) -> list[dict[str, str]]:
    out = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                {
                    "smiles": row["smiles"],
                    "fragment_a": row["fragment_a"],
                    "fragment_b": row["fragment_b"],
                    "linker": row["linker"],
                    "fragment_a_ap": row.get("fragment_a_ap", row["fragment_a"]),
                    "fragment_b_ap": row.get("fragment_b_ap", row["fragment_b"]),
                    "linker_ap": row.get("linker_ap", row["linker"]),
                }
            )
            if max_rows is not None and len(out) >= max_rows:
                break
    return out


def train_completion_model(
    records: Sequence[dict[str, str]],
    prompt_key: str,
    target_key: str,
    representation: str,
    output_dir: str | Path,
    prompt_template: Optional[str] = None,
    target_template: Optional[str] = None,
    cfg: Optional[LMConfig] = None,
    show_progress: bool = True,
    device: Optional[str] = None,
    encode_workers: int = 1,
    encode_chunksize: int = 128,
) -> dict[str, Any]:
    rep = build_representation(representation)
    if isinstance(rep, UnsupportedRepresentation):
        raise RepresentationError(
            f"Cannot train completion model for unsupported placeholder: {representation}"
        )

    cfg = cfg or LMConfig()
    _set_seed(cfg.seed)
    dev = _resolve_torch_device(device)

    if encode_workers <= 0:
        encode_workers = max(1, os.cpu_count() or 1)
    print(
        f"[train-completion] tokenize_workers={encode_workers}, tokenize_chunksize={max(1, int(encode_chunksize))}",
        flush=True,
    )
    seqs = _tokenize_completion_records(
        records,
        representation=representation,
        prompt_key=prompt_key,
        target_key=target_key,
        prompt_template=prompt_template,
        target_template=target_template,
        show_progress=show_progress,
        desc=f"Tokenize completion[{representation}]",
        num_workers=encode_workers,
        chunksize=encode_chunksize,
    )

    if not seqs:
        raise ValueError("No valid completion records after tokenization.")

    n_valid = max(1, int(0.1 * len(seqs)))
    train_tok = seqs[:-n_valid]
    valid_tok = seqs[-n_valid:]

    vocab = TokenVocab.build(train_tok, min_freq=1)
    ds_train = NextTokenDataset(_ids_with_special_tokens(train_tok, vocab), cfg.max_seq_len, vocab.pad_id)
    ds_valid = NextTokenDataset(_ids_with_special_tokens(valid_tok, vocab), cfg.max_seq_len, vocab.pad_id)

    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    use_amp, amp_dtype, use_scaler, amp_label = _resolve_amp_mode(cfg.amp, dev)

    collate_fn = build_dynamic_lm_collate_fn(
        vocab.pad_id,
        pad_to_multiple_of=(8 if dev.type == "cuda" else None),
    )
    pin_memory = dev.type == "cuda"
    train_loader_kwargs: Dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "drop_last": False,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "num_workers": max(0, int(cfg.num_data_workers)),
    }
    valid_loader_kwargs: Dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "num_workers": max(0, int(cfg.num_data_workers)),
    }
    if cfg.num_data_workers > 0:
        pf = max(2, int(cfg.prefetch_factor))
        pw = bool(cfg.persistent_data_workers)
        train_loader_kwargs["prefetch_factor"] = pf
        train_loader_kwargs["persistent_workers"] = pw
        valid_loader_kwargs["prefetch_factor"] = pf
        valid_loader_kwargs["persistent_workers"] = pw
    loader_train = DataLoader(ds_train, **train_loader_kwargs)
    loader_valid = DataLoader(ds_valid, **valid_loader_kwargs)

    model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
    if cfg.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[misc]
            print("[train-completion] torch.compile enabled", flush=True)
        except Exception as exc:
            print(f"[train-completion] torch.compile unavailable/fallback: {exc}", flush=True)
    model.to(dev)
    n_model_params = int(sum(p.numel() for p in model.parameters()))
    if dev.type == "cuda":
        gpu_name = torch.cuda.get_device_name(dev)
        print(f"[train-completion] device={dev} ({gpu_name})", flush=True)
    else:
        print(f"[train-completion] device={dev}", flush=True)
    print(f"[train-completion] n_model_params={n_model_params} (~{n_model_params/1e6:.2f}M)", flush=True)
    train_effective_max_seq_len = max((len(x) for x, _ in ds_train.samples), default=0)
    valid_effective_max_seq_len = max((len(x) for x, _ in ds_valid.samples), default=0)
    print(
        f"[train-completion] dynamic_padding=batch_max (cfg_max_seq_len={cfg.max_seq_len}, "
        f"train_effective_max_seq_len={train_effective_max_seq_len}, "
        f"valid_effective_max_seq_len={valid_effective_max_seq_len})",
        flush=True,
    )
    print(
        f"[train-completion] amp={amp_label}, tf32={bool(cfg.tf32)}, "
        f"num_data_workers={max(0, int(cfg.num_data_workers))}, prefetch_factor={max(2, int(cfg.prefetch_factor))}",
        flush=True,
    )

    adamw_kwargs: Dict[str, Any] = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    if dev.type == "cuda":
        try:
            adamw_kwargs["fused"] = True
        except Exception:
            pass
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    except TypeError:
        adamw_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)  # pragma: no cover
    non_blocking = dev.type == "cuda"

    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        train_iter = iterable_with_progress(
            loader_train,
            show_progress=show_progress,
            total=len(loader_train),
            desc=f"Train completion[{representation}] ep{epoch}",
        )
        for x, y in train_iter:
            x = x.to(dev, non_blocking=non_blocking)
            y = y.to(dev, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                if use_amp
                else nullcontext()
            )
            with amp_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for x, y in loader_valid:
                x = x.to(dev, non_blocking=non_blocking)
                y = y.to(dev, non_blocking=non_blocking)
                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                    if use_amp
                    else nullcontext()
                )
                with amp_ctx:
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
                valid_losses.append(float(loss.item()))

        tr = float(np.mean(train_losses)) if train_losses else float("nan")
        va = float(np.mean(valid_losses)) if valid_losses else float("nan")
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr,
                "valid_loss": va,
                "train_ppl": float(math.exp(tr)) if not math.isnan(tr) else float("nan"),
                "valid_ppl": float(math.exp(va)) if not math.isnan(va) else float("nan"),
            }
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
            "cfg": vars(cfg),
            "representation": representation,
            "task": "completion",
        },
        out_dir / "model.pt",
    )
    (out_dir / "vocab.json").write_text(
        json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "representation": representation,
                "prompt_key": prompt_key,
                "target_key": target_key,
                "prompt_template": prompt_template,
                "target_template": target_template,
                "n_records": len(records),
                "cfg": vars(cfg),
                "device": str(dev),
                "n_model_params": n_model_params,
                "padding_mode": "dynamic_batch_max",
                "train_effective_max_seq_len": train_effective_max_seq_len,
                "valid_effective_max_seq_len": valid_effective_max_seq_len,
                "tokenize_workers": encode_workers,
                "tokenize_chunksize": max(1, int(encode_chunksize)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {"output_dir": str(out_dir), "history": history, "device": str(dev), "n_model_params": n_model_params}


def _extract_completion_suffix(tokens: Sequence[str]) -> list[str]:
    if "<SEP>" not in tokens:
        return list(tokens)
    i = tokens.index("<SEP>")
    return list(tokens[i + 1 :])


def evaluate_scaffold_completion(
    model_dir: str | Path,
    scaffold_csv: str | Path,
    output_dir: str | Path,
    max_eval: Optional[int] = 1000,
    n_samples_per_prompt: int = 1,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    show_progress: bool = True,
    device: str | None = None,
    eval_workers: int = 1,
    eval_chunksize: int = 256,
) -> dict[str, Any]:
    model, vocab, rep, dev = load_trained_model(model_dir, device=device)
    rows = _load_scaffold_rows(scaffold_csv, max_rows=max_eval)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    generated_rows = []
    recovered_smiles: list[str] = []
    if eval_workers <= 0:
        eval_workers = max(1, os.cpu_count() or 1)
    eval_chunksize = max(1, int(eval_chunksize))
    print(
        f"[eval-scaffold] eval_workers={eval_workers}, eval_chunksize={eval_chunksize}",
        flush=True,
    )

    iterator = iterable_with_progress(rows, show_progress=show_progress, total=len(rows), desc="Eval scaffold")
    for row in iterator:
        target = row["smiles"]
        scaffold = row["scaffold_smiles"]
        scaffold_prompt = row.get("scaffold_prompt", scaffold)

        try:
            prompt_tokens = _encode_completion_prompt_tokens(rep, scaffold_prompt) + ["<SEP>"]
        except Exception:
            continue

        seqs = sample_model_sequences(
            model,
            vocab,
            n_samples=n_samples_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            prompt_tokens=prompt_tokens,
            device=dev,
        )

        for toks in seqs:
            suffix = _extract_completion_suffix(toks)
            pred_repr = rep.detokenize(suffix)
            generated_rows.append(
                {
                    "target_smiles": target,
                    "scaffold_smiles": scaffold,
                    "pred_repr": pred_repr,
                    "pred_smiles": "",
                    "valid": 0,
                    "scaffold_preserved": 0,
                }
            )

    eval_payloads = [(r["pred_repr"], r["scaffold_smiles"]) for r in generated_rows]
    if eval_workers == 1:
        results = []
        for payload in iterable_with_progress(
            eval_payloads,
            show_progress=show_progress,
            total=len(eval_payloads),
            desc="Post-eval scaffold",
        ):
            results.append(_eval_scaffold_prediction_local(rep, pred_repr=payload[0], scaffold=payload[1]))
    else:
        with ProcessPoolExecutor(
            max_workers=eval_workers,
            initializer=_init_eval_worker,
            initargs=(rep.name,),
        ) as pool:
            mapped = pool.map(
                _eval_scaffold_prediction_worker,
                eval_payloads,
                chunksize=eval_chunksize,
            )
            mapped = iterable_with_progress(
                mapped,
                show_progress=show_progress,
                total=len(eval_payloads),
                desc="Post-eval scaffold",
            )
            results = list(mapped)

    valid = 0
    preserved = 0
    success = 0
    for i, (pred_s, is_valid_i, preserve_i) in enumerate(results):
        generated_rows[i]["pred_smiles"] = pred_s
        generated_rows[i]["valid"] = int(is_valid_i)
        generated_rows[i]["scaffold_preserved"] = int(preserve_i)
        if is_valid_i:
            valid += 1
            recovered_smiles.append(pred_s)
        if preserve_i:
            preserved += 1
        if is_valid_i and preserve_i:
            success += 1

    n_total = len(generated_rows)
    unique_valid = len(
        set(
            canonicalize_smiles(s, require_single_component=True)
            for s in recovered_smiles
            if canonicalize_smiles(s, require_single_component=True) is not None
        )
    )

    metrics = {
        "n_total": n_total,
        "validity": valid / max(1, n_total),
        "scaffold_preservation_rate": preserved / max(1, valid),
        "completion_success_rate": success / max(1, n_total),
        "diversity": unique_valid / max(1, valid),
    }

    with (out / "scaffold_completion_predictions.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_smiles",
                "scaffold_smiles",
                "pred_repr",
                "pred_smiles",
                "valid",
                "scaffold_preserved",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(generated_rows)

    (out / "scaffold_completion_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def evaluate_linker_design(
    model_dir: str | Path,
    linker_csv: str | Path,
    output_dir: str | Path,
    max_eval: Optional[int] = 1000,
    n_samples_per_prompt: int = 1,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    show_progress: bool = True,
    device: str | None = None,
    eval_workers: int = 1,
    eval_chunksize: int = 256,
) -> dict[str, Any]:
    model, vocab, rep, dev = load_trained_model(model_dir, device=device)
    rows = _load_linker_rows(linker_csv, max_rows=max_eval)
    meta_target_key = ""
    meta_path = Path(model_dir) / "meta.json"
    if meta_path.exists():
        try:
            meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_target_key = str(meta_obj.get("target_key", "")).strip()
        except Exception:
            meta_target_key = ""
    linker_mode = meta_target_key in {"linker_ap", "linker_with_attachments"}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    generated_rows = []
    valid_smiles: list[str] = []
    if eval_workers <= 0:
        eval_workers = max(1, os.cpu_count() or 1)
    eval_chunksize = max(1, int(eval_chunksize))
    print(
        f"[eval-linker] eval_workers={eval_workers}, eval_chunksize={eval_chunksize}, mode={'linker' if linker_mode else 'full'}",
        flush=True,
    )

    iterator = iterable_with_progress(rows, show_progress=show_progress, total=len(rows), desc="Eval linker")
    for row in iterator:
        target = row["smiles"]
        frag_a = row["fragment_a"]
        frag_b = row["fragment_b"]
        frag_a_ap = row.get("fragment_a_ap", frag_a)
        frag_b_ap = row.get("fragment_b_ap", frag_b)
        prompt_a = frag_a_ap if linker_mode else frag_a
        prompt_b = frag_b_ap if linker_mode else frag_b
        prompt = f"{prompt_a}.{prompt_b}"

        try:
            prompt_tokens = _encode_completion_prompt_tokens(rep, prompt) + ["<SEP>"]
        except Exception:
            continue

        seqs = sample_model_sequences(
            model,
            vocab,
            n_samples=n_samples_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            prompt_tokens=prompt_tokens,
            device=dev,
        )

        for toks in seqs:
            suffix = _extract_completion_suffix(toks)
            pred_repr = rep.detokenize(suffix)
            generated_rows.append(
                {
                    "target_smiles": target,
                    "fragment_a": frag_a,
                    "fragment_b": frag_b,
                    "fragment_a_ap": frag_a_ap,
                    "fragment_b_ap": frag_b_ap,
                    "pred_repr": pred_repr,
                    "pred_linker": "",
                    "pred_smiles": "",
                    "valid": 0,
                    "fragment_preserved": 0,
                    "connection_success": 0,
                }
            )

    mode_name = "linker" if linker_mode else "full"
    eval_payloads = [
        (
            mode_name,
            r["pred_repr"],
            r["fragment_a"],
            r["fragment_b"],
            r["fragment_a_ap"],
            r["fragment_b_ap"],
        )
        for r in generated_rows
    ]
    if eval_workers == 1:
        results = []
        for payload in iterable_with_progress(
            eval_payloads,
            show_progress=show_progress,
            total=len(eval_payloads),
            desc="Post-eval linker",
        ):
            results.append(
                _eval_linker_prediction_local(
                    rep,
                    pred_repr=payload[1],
                    frag_a=payload[2],
                    frag_b=payload[3],
                )
                if payload[0] == "full"
                else _eval_linker_prediction_from_linker_local(
                    rep,
                    pred_repr=payload[1],
                    frag_a=payload[2],
                    frag_b=payload[3],
                    frag_a_ap=payload[4],
                    frag_b_ap=payload[5],
                )
            )
    else:
        with ProcessPoolExecutor(
            max_workers=eval_workers,
            initializer=_init_eval_worker,
            initargs=(rep.name,),
        ) as pool:
            mapped = pool.map(
                _eval_linker_prediction_worker,
                eval_payloads,
                chunksize=eval_chunksize,
            )
            mapped = iterable_with_progress(
                mapped,
                show_progress=show_progress,
                total=len(eval_payloads),
                desc="Post-eval linker",
            )
            results = list(mapped)

    n_valid = 0
    n_preserve = 0
    n_connect = 0
    for i, (pred_s, pred_linker, valid_i, preserve_i, connect_i) in enumerate(results):
        generated_rows[i]["pred_linker"] = pred_linker
        generated_rows[i]["pred_smiles"] = pred_s
        generated_rows[i]["valid"] = int(valid_i)
        generated_rows[i]["fragment_preserved"] = int(preserve_i)
        generated_rows[i]["connection_success"] = int(connect_i)
        if valid_i:
            n_valid += 1
            valid_smiles.append(pred_s)
        if preserve_i:
            n_preserve += 1
        if connect_i:
            n_connect += 1

    n_total = len(generated_rows)
    unique_valid = len(
        set(
            canonicalize_smiles(s, require_single_component=True)
            for s in valid_smiles
            if canonicalize_smiles(s, require_single_component=True) is not None
        )
    )

    metrics = {
        "n_total": n_total,
        "validity": n_valid / max(1, n_total),
        "fragment_preservation_rate": n_preserve / max(1, n_valid),
        "connection_success_rate": n_connect / max(1, n_total),
        "diversity": unique_valid / max(1, n_valid),
    }

    with (out / "linker_design_predictions.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_smiles",
                "fragment_a",
                "fragment_b",
                "fragment_a_ap",
                "fragment_b_ap",
                "pred_repr",
                "pred_linker",
                "pred_smiles",
                "valid",
                "fragment_preserved",
                "connection_success",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(generated_rows)

    (out / "linker_design_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics
