#!/usr/bin/env python3
"""Build FC-STAMP rows from GenMol fragment tasks.

Two construction modes are supported.

``splice`` builds strict single-hole fill-in-the-middle rows:

    [TASK] left [INFILL] right [GEN] middle [EOS]

The generated middle is spliced back into the prompt before STAMP decoding.
Rows requiring multiple holes, or a leading hole before all fixed context, are
skipped by default because those are not clean GenMol constrained-generation
tasks.

``benchmark`` builds test-time in-place prompts without requiring a gold
target from GenMol:

* linker design: encode the two query fragments with attachment ports and put
  one infill span between them;
* motif extension: encode the query motif with an attachment port and append
  one tail infill span;
* scaffold decoration: encode the dummy-marked query scaffold, turn each
  dummy attachment into a STAMP joint plus an in-place infill hole;
* superstructure generation: add attachment ports to selected query atoms and
  append one tail infill span per prompt. Evaluation can aggregate several
  attach-site prompts for one query, which keeps the task source-free while
  increasing the generation support.

``completion`` remains available for debugging, but is not the default GenMol
benchmark mapping.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.common import canonicalize_smiles  # noqa: E402
from stamp.benchmark.representations import build_representation  # noqa: E402


GENMOL_TASKS = (
    "linker_design",
    "motif_extension",
    "scaffold_decoration",
    "superstructure_generation",
)

TASK_TOKEN_BY_GENMOL = {
    "linker_design": "[TASK_LINKER]",
    "motif_extension": "[TASK_EXTEND]",
    "scaffold_decoration": "[TASK_DECORATE]",
    "superstructure_generation": "[TASK_DECORATE]",
}

SINGLE_INFILL_TOKEN_BY_GENMOL = {
    "linker_design": "[INFILL_PATH]",
    "motif_extension": "[INFILL_TAIL]",
    "scaffold_decoration": "[INFILL_SUBTREE]",
    "superstructure_generation": "[INFILL_SUBTREE]",
}

FRAG_SEP = "[FRAG_SEP]"


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

    for query in variants:
        if query is None or query.GetNumAtoms() == 0:
            continue
        try:
            match = source_mol.GetSubstructMatch(query, useChirality=False)
        except Exception:
            match = ()
        if match:
            return {int(x) for x in match}

    try:
        params = rdFMCS.MCSParameters()
        params.Timeout = 5
        params.AtomCompareParameters.MatchValences = False
        params.BondCompareParameters.RingMatchesRingOnly = True
        res = rdFMCS.FindMCS([source_mol, _drop_stereo(query_mol)], parameters=params)
        if res.smartsString:
            mcs = Chem.MolFromSmarts(res.smartsString)
            min_atoms = max(1, math.floor(query_mol.GetNumAtoms() * 0.5))
            if mcs is not None and mcs.GetNumAtoms() >= min_atoms:
                match = source_mol.GetSubstructMatch(mcs)
                if match:
                    return {int(x) for x in match}
    except Exception:
        pass
    return set()


def _match_query_part_atoms(source_mol: Chem.Mol, part_smiles: str) -> set[int]:
    query = Chem.MolFromSmiles(part_smiles)
    if query is None:
        query = Chem.MolFromSmiles(part_smiles, sanitize=False)
    if query is None:
        return set()
    query = _strip_dummy_atoms(query)
    if query is None or query.GetNumAtoms() == 0:
        return set()
    return _relaxed_match(source_mol, query)


def _match_query_atoms(source_mol: Chem.Mol, query_smiles: str) -> set[int]:
    atoms: set[int] = set()
    for part in [p for p in str(query_smiles).split(".") if p]:
        atoms.update(_match_query_part_atoms(source_mol, part))
    return atoms


def _keep_group_indices_for_atoms(groups: Sequence[Any], atom_set: set[int]) -> list[int]:
    keep: list[int] = []
    for idx, group in enumerate(groups):
        group_atoms = {int(x) for x in group.atom_indices}
        if group_atoms and group_atoms.issubset(atom_set):
            keep.append(idx)
    if keep:
        return keep

    for idx, group in enumerate(groups):
        group_atoms = {int(x) for x in group.atom_indices}
        if group_atoms and (len(group_atoms & atom_set) / max(1, len(group_atoms))) >= 0.5:
            keep.append(idx)
    if keep:
        return keep

    for idx, group in enumerate(groups):
        group_atoms = {int(x) for x in group.atom_indices}
        if group_atoms & atom_set:
            keep.append(idx)
    return keep


def _group_tokens(groups: Sequence[Any], indices: Optional[Sequence[int]] = None) -> list[str]:
    if indices is None:
        selected = range(len(groups))
    else:
        selected = indices
    out: list[str] = []
    for idx in selected:
        out.extend(str(tok) for tok in groups[int(idx)].tokens)
    return out


def _detok(tokens: Sequence[str]) -> str:
    return " ".join(str(tok) for tok in tokens if str(tok))


def _is_stereo_token(tok: str) -> bool:
    return str(tok).startswith("[S_") and str(tok).endswith("]")


def _strip_stereo_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(tok) for tok in tokens if not _is_stereo_token(str(tok))]


def _strip_stereo_text(text: Any) -> str:
    return _detok(_strip_stereo_tokens(str(text or "").split()))


def _strip_stereo_from_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in [
        "stamp_smiles",
        "prompt_stamp",
        "target_stamp",
        "full_stamp",
        "masked_stamp",
        "left_stamp",
        "middle_stamp",
        "right_stamp",
    ]:
        if key in out:
            out[key] = _strip_stereo_text(out[key])
    if "prompt_stamp" in out:
        out["prompt_len"] = int(len(str(out["prompt_stamp"]).split()))
        out["label_start"] = int(out["prompt_len"])
    if "target_stamp" in out:
        out["target_len"] = int(len(str(out["target_stamp"]).split()))
    if "prompt_len" in out and "target_len" in out:
        out["loss_mask_spec"] = f"prompt0:{out['prompt_len']},target1:{out['target_len']}"
    if "hole_spans" in out:
        try:
            holes = json.loads(str(out["hole_spans"]))
        except Exception:
            holes = []
        if isinstance(holes, list):
            for hole in holes:
                if not isinstance(hole, dict):
                    continue
                middle = _strip_stereo_text(hole.get("middle_stamp", ""))
                hole["middle_stamp"] = middle
                hole["middle_len"] = int(len(middle.split()))
            out["hole_spans"] = json.dumps(holes, ensure_ascii=False)
    return out


def _contiguous_runs(indices: Sequence[int]) -> list[list[int]]:
    vals = sorted({int(x) for x in indices})
    if not vals:
        return []
    runs: list[list[int]] = []
    cur = [vals[0]]
    for value in vals[1:]:
        if value == cur[-1] + 1:
            cur.append(value)
        else:
            runs.append(cur)
            cur = [value]
    runs.append(cur)
    return runs


def _encode_stamp_groups(rep, smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        groups = rep.codec.encode_mol_to_groups(mol)
    except Exception:
        return None
    if not groups:
        return None
    source_tokens = _group_tokens(groups)
    return mol, source_tokens, groups


def _mol_from_smiles_relaxed(smiles: str) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
    return mol


def _cap_attach_with_h(piece_mol: Chem.Mol) -> tuple[Optional[Chem.Mol], list[int]]:
    attach_orig: list[int] = []
    for atom in piece_mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() > 1:
                attach_orig.append(int(nbr.GetIdx()))
                break
    if not attach_orig:
        return piece_mol, []

    rw = Chem.RWMol(piece_mol)
    for i, idx in enumerate(attach_orig):
        rw.GetAtomWithIdx(int(idx)).SetAtomMapNum(91 + i)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
            atom.SetIsotope(0)
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(False)
    capped_h = rw.GetMol()
    try:
        Chem.SanitizeMol(capped_h)
        capped = Chem.RemoveHs(capped_h)
    except Exception:
        return None, []
    attach_capped = [int(a.GetIdx()) for a in capped.GetAtoms() if 91 <= int(a.GetAtomMapNum()) <= 99]
    for atom in capped.GetAtoms():
        atom.SetAtomMapNum(0)
    return capped, attach_capped


def _dummy_attachment_neighbors(mol: Chem.Mol) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int, int]] = []
    fallback = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        nbr_idx: Optional[int] = None
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() > 1:
                nbr_idx = int(nbr.GetIdx())
                break
        if nbr_idx is None:
            continue
        label = int(atom.GetIsotope()) or int(atom.GetAtomMapNum())
        if label <= 0:
            fallback += 1
            label = 100000 + fallback
        pairs.append((label, int(atom.GetIdx()), nbr_idx))
    pairs.sort(key=lambda item: (item[0], item[1]))
    return [(dummy_idx, nbr_idx) for _label, dummy_idx, nbr_idx in pairs]


def _cap_dummy_scaffold_with_holes(piece_mol: Chem.Mol) -> tuple[Optional[Chem.Mol], list[int]]:
    attachments = _dummy_attachment_neighbors(piece_mol)
    if not attachments:
        return None, []

    rw = Chem.RWMol(piece_mol)
    for hole_idx, (_dummy_idx, nbr_idx) in enumerate(attachments):
        rw.GetAtomWithIdx(int(nbr_idx)).SetAtomMapNum(91 + int(hole_idx))
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
            atom.SetIsotope(0)
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(False)
    capped_h = rw.GetMol()
    try:
        Chem.SanitizeMol(capped_h)
        capped = Chem.RemoveHs(capped_h)
        mapped = Chem.MolToSmiles(capped, canonical=True, isomericSmiles=True)
        reparsed = Chem.MolFromSmiles(mapped)
    except Exception:
        return None, []
    if reparsed is None:
        return None, []

    attach_by_hole: dict[int, int] = {}
    for atom in reparsed.GetAtoms():
        amap = int(atom.GetAtomMapNum())
        if 91 <= amap <= 999:
            attach_by_hole[amap - 91] = int(atom.GetIdx())
        atom.SetAtomMapNum(0)
    if len(attach_by_hole) != len(attachments):
        return None, []
    return reparsed, [attach_by_hole[i] for i in range(len(attachments))]


def _encode_graph_groups_from_root(rep, fragments: Sequence[Any], edges: Sequence[Any], root: int) -> list[Any]:
    codec = rep.codec
    if len(fragments) == 1:
        frag = fragments[0]
        from stamp.core.stamp_codec import STAMPGroup

        return [
            STAMPGroup(
                frag_id=int(frag.frag_id),
                atom_indices=tuple(int(x) for x in frag.atom_indices),
                motif=str(frag.motif),
                tokens=(str(frag.motif),),
                parent_frag_id=None,
                subtree_group_count=1,
            )
        ]

    if len(edges) < len(fragments) - 1:
        raise ValueError("Reduced fragment graph is disconnected.")

    if len(edges) == len(fragments) - 1:
        adj = codec._build_adjacency(edges)
        tree_children, _tree_parent = codec._build_rooted_tree(int(root), adj)
        subtree_sizes = codec._compute_subtree_sizes(int(root), tree_children)
        out: list[Any] = []
        codec._emit_group_nodes(
            node=int(root),
            parent=None,
            prefix_tokens=[],
            out=out,
            fragments=fragments,
            adj=adj,
            tree_children=tree_children,
            subtree_sizes=subtree_sizes,
        )
        return out

    (
        tree_children,
        tree_parent,
        tree_parent_edge,
        back_edges,
        depth,
    ) = codec._build_spanning_tree_with_back_edges(
        root=int(root),
        graph_adj=codec._build_graph_adjacency(edges),
        edge_by_id={e.edge_id: e for e in edges},
        fragments=fragments,
    )
    subtree_sizes = codec._compute_subtree_sizes(int(root), tree_children)
    preorder = codec._compute_preorder_index(int(root), tree_children, subtree_sizes, fragments)
    ring_outputs, ring_closures = codec._organize_ring_back_edges(
        back_edges=back_edges,
        tree_parent=tree_parent,
        depth=depth,
        preorder=preorder,
    )
    out = []
    codec._emit_group_nodes_with_rings(
        node=int(root),
        parent=None,
        prefix_tokens=[],
        out=out,
        fragments=fragments,
        tree_children=tree_children,
        subtree_sizes=subtree_sizes,
        tree_parent_edge=tree_parent_edge,
        edge_by_id={e.edge_id: e for e in edges},
        ring_outputs=ring_outputs,
        ring_closures=ring_closures,
    )
    return out


def _insert_attach_joints(root_tokens: Sequence[str], attach_locals: Sequence[int], *, port_role: str) -> list[str]:
    attach_tokens = [f"[J_{int(idx)}]" for idx in attach_locals]
    toks = [str(tok) for tok in root_tokens]
    if port_role == "input":
        # The right linker fragment is decoded as a child. Its first joint must
        # consume the generated linker output port; internal child ports follow.
        return [toks[0], *attach_tokens, *toks[1:]]
    # The left linker/motif-extension fragment is decoded before the generated
    # span. Its external attachment port must remain after all internal child
    # groups have consumed their own ports.
    return [*toks, *attach_tokens]


def _encode_mol_with_attach(
    rep,
    mol: Chem.Mol,
    attach_atom_indices: Sequence[int],
    *,
    port_role: str = "output",
) -> Optional[list[str]]:
    if not attach_atom_indices:
        try:
            return str(rep.encode(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))).split()
        except Exception:
            return None

    rw = Chem.RWMol(mol)
    for i, idx in enumerate(attach_atom_indices):
        if int(idx) >= rw.GetNumAtoms():
            return None
        rw.GetAtomWithIdx(int(idx)).SetAtomMapNum(91 + i)
    try:
        mapped = Chem.MolToSmiles(rw.GetMol(), canonical=True, isomericSmiles=True)
        re_mol = Chem.MolFromSmiles(mapped)
    except Exception:
        return None
    if re_mol is None:
        return None

    attach_canon: list[int] = []
    for atom in re_mol.GetAtoms():
        if 91 <= int(atom.GetAtomMapNum()) <= 99:
            attach_canon.append(int(atom.GetIdx()))
        atom.SetAtomMapNum(0)

    try:
        fragments, edges = rep.codec._fragments_and_edges_from_mol(re_mol)
    except Exception:
        return None

    root_frag: Optional[int] = None
    attach_locals: list[int] = []
    for canon_idx in attach_canon:
        for frag in fragments:
            atoms = [int(x) for x in frag.atom_indices]
            if canon_idx not in atoms:
                continue
            if root_frag is None:
                root_frag = int(frag.frag_id)
            if int(frag.frag_id) != int(root_frag):
                return None
            local = frag.orig_to_local.get(int(canon_idx))
            if local is not None:
                attach_locals.append(int(local))
            break

    if root_frag is None or not attach_locals:
        return None

    try:
        groups = _encode_graph_groups_from_root(rep, fragments, edges, int(root_frag))
    except Exception:
        return None

    out: list[str] = []
    for pos, group in enumerate(groups):
        toks = [str(tok) for tok in group.tokens]
        if pos == 0:
            toks = _insert_attach_joints(toks, attach_locals, port_role=port_role)
        out.extend(toks)
    return out or None


def _encode_query_part_with_attach(rep, part_smiles: str, *, port_role: str = "output") -> Optional[list[str]]:
    mol = _mol_from_smiles_relaxed(part_smiles)
    if mol is None:
        return None
    capped, attach = _cap_attach_with_h(mol)
    if capped is None:
        return None
    return _encode_mol_with_attach(rep, capped, attach, port_role=port_role)


def _atom_can_accept_extra_single_bond(mol: Chem.Mol, atom_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(int(atom_idx))
    if atom.GetAtomicNum() <= 1:
        return False
    if int(atom.GetTotalNumHs()) > 0 or int(atom.GetNumImplicitHs()) > 0:
        return True
    try:
        rw = Chem.RWMol(mol)
        new_idx = rw.AddAtom(Chem.Atom(6))
        rw.AddBond(int(atom_idx), int(new_idx), Chem.BondType.SINGLE)
        test = rw.GetMol()
        Chem.SanitizeMol(test)
        return True
    except Exception:
        return False


def _encode_query_no_dummy_valid_attach_prompts(
    rep,
    query_smiles: str,
    *,
    max_sites: int,
) -> list[dict[str, Any]]:
    mol = _mol_from_smiles_relaxed(query_smiles)
    if mol is None:
        return []

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for atom in mol.GetAtoms():
        atom_idx = int(atom.GetIdx())
        if not _atom_can_accept_extra_single_bond(mol, atom_idx):
            continue
        toks = _encode_mol_with_attach(rep, mol, [atom_idx], port_role="output")
        if not toks:
            continue
        key = tuple(str(tok) for tok in toks)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "site_rank": int(len(out)),
                "attach_atom_idx": int(atom_idx),
                "prompt_tokens": list(key),
            }
        )
        if int(max_sites) > 0 and len(out) >= int(max_sites):
            break
    return out


def _encode_query_no_dummy_with_attach(
    rep,
    query_smiles: str,
    *,
    attach_rank: int = 0,
) -> Optional[list[str]]:
    mol = _mol_from_smiles_relaxed(query_smiles)
    if mol is None:
        return None
    heavy = [int(a.GetIdx()) for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy:
        return None
    attach_idx = heavy[int(attach_rank) % len(heavy)]
    return _encode_mol_with_attach(rep, mol, [attach_idx], port_role="output")


def _encode_dummy_scaffold_prompt(rep, query_smiles: str, *, task_name: str) -> Optional[dict[str, Any]]:
    mol = _mol_from_smiles_relaxed(query_smiles)
    if mol is None:
        return None
    capped, attach_atoms = _cap_dummy_scaffold_with_holes(mol)
    if capped is None or not attach_atoms:
        return None

    try:
        fragments, edges = rep.codec._fragments_and_edges_from_mol(capped)
        groups = rep.codec._encode_graph_groups(fragments, edges)
    except Exception:
        return None
    if not groups:
        return None

    group_pos_by_frag = {int(group.frag_id): pos for pos, group in enumerate(groups)}
    attach_by_group: dict[int, list[tuple[int, int]]] = {}
    for hole_idx, atom_idx in enumerate(attach_atoms):
        matched = False
        for frag in fragments:
            atoms = [int(x) for x in frag.atom_indices]
            if int(atom_idx) not in atoms:
                continue
            local = frag.orig_to_local.get(int(atom_idx))
            pos = group_pos_by_frag.get(int(frag.frag_id))
            if local is None or pos is None:
                return None
            attach_by_group.setdefault(int(pos), []).append((int(hole_idx), int(local)))
            matched = True
            break
        if not matched:
            return None

    n_holes = len(attach_atoms)
    group_tokens: list[list[str]] = []
    insert_after: dict[int, list[tuple[int, str]]] = {}
    holes: list[dict[str, Any]] = []
    for pos, group in enumerate(groups):
        toks = [str(tok) for tok in group.tokens]
        for hole_idx, local in sorted(attach_by_group.get(pos, []), key=lambda item: item[0]):
            toks.append(f"[J_{int(local)}]")
            infill = SINGLE_INFILL_TOKEN_BY_GENMOL[task_name] if n_holes == 1 else f"[INFILL_{hole_idx}]"
            end_pos = min(len(groups) - 1, int(pos) + int(getattr(group, "subtree_group_count", 1)) - 1)
            insert_after.setdefault(int(end_pos), []).append((int(pos), infill))
            holes.append(
                {
                    "hole_index": int(hole_idx),
                    "infill_token": infill,
                    "start_group": -1,
                    "end_group": -1,
                    "n_groups": 0,
                    "middle_stamp": "",
                    "middle_len": 0,
                    "attach_group": int(pos),
                    "attach_local": int(local),
                }
            )
        group_tokens.append(toks)

    prompt_body: list[str] = []
    for pos, toks in enumerate(group_tokens):
        prompt_body.extend(toks)
        for _attach_pos, infill in sorted(insert_after.get(pos, []), key=lambda item: (-item[0], item[1])):
            prompt_body.append(infill)

    if not prompt_body or not holes:
        return None
    return {
        "prompt_body_tokens": prompt_body,
        "holes": sorted(holes, key=lambda item: int(item["hole_index"])),
        "n_groups": int(len(groups)),
    }


def _encode_query_condition_tokens(rep, query_smiles: str) -> Optional[list[str]]:
    """Encode GenMol query fragments as condition tokens.

    Dummy atoms only mark the constraint boundary, so they are stripped before
    STAMP encoding. Multiple query fragments are separated by a control token
    and are never sent to the STAMP decoder directly.
    """
    out: list[str] = []
    for part in [p for p in str(query_smiles).split(".") if p]:
        query = Chem.MolFromSmiles(part) or Chem.MolFromSmiles(part, sanitize=False)
        if query is None:
            return None
        query = _strip_dummy_atoms(query)
        if query is None or query.GetNumAtoms() == 0:
            return None
        try:
            smi = Chem.MolToSmiles(query, canonical=True, isomericSmiles=True)
            toks = str(rep.encode(smi)).split()
        except Exception:
            return None
        if not toks:
            return None
        if out:
            out.append(FRAG_SEP)
        out.extend(toks)
    return out or None


def _matched_group_condition_tokens(groups: Sequence[Any], keep_indices: Sequence[int]) -> Optional[list[str]]:
    runs = _contiguous_runs(keep_indices)
    out: list[str] = []
    for run in runs:
        toks = _group_tokens(groups, run)
        if not toks:
            continue
        if out:
            out.append(FRAG_SEP)
        out.extend(toks)
    return out or None


def _build_prompt_and_target(
    groups: Sequence[Any],
    runs: Sequence[Sequence[int]],
    *,
    task_name: str,
) -> dict[str, Any]:
    n_holes = len(runs)
    run_by_start = {int(run[0]): (hole_idx, list(run)) for hole_idx, run in enumerate(runs)}
    prompt_body: list[str] = []
    holes: list[dict[str, Any]] = []
    i = 0
    while i < len(groups):
        item = run_by_start.get(i)
        if item is None:
            prompt_body.extend(str(tok) for tok in groups[i].tokens)
            i += 1
            continue

        hole_idx, run = item
        infill_token = (
            SINGLE_INFILL_TOKEN_BY_GENMOL[task_name]
            if n_holes == 1
            else f"[INFILL_{hole_idx}]"
        )
        middle_tokens = _group_tokens(groups, run)
        prompt_body.append(infill_token)
        holes.append(
            {
                "hole_index": int(hole_idx),
                "infill_token": infill_token,
                "start_group": int(run[0]),
                "end_group": int(run[-1] + 1),
                "n_groups": int(len(run)),
                "middle_stamp": _detok(middle_tokens),
                "middle_len": int(len(middle_tokens)),
            }
        )
        i = int(run[-1]) + 1

    task_token = TASK_TOKEN_BY_GENMOL[task_name]
    prompt_tokens = [task_token] + prompt_body + ["[GEN]"]
    if n_holes == 1:
        middle_tokens = _group_tokens(groups, runs[0])
        target_tokens = (middle_tokens if middle_tokens else ["[EMPTY]"]) + ["[EOS]"]
    else:
        target_tokens = []
        for hole in holes:
            middle = str(hole["middle_stamp"]).split()
            target_tokens.append(f"[FILL_{hole['hole_index']}]")
            target_tokens.extend(middle if middle else ["[EMPTY]"])
            target_tokens.append("[/FILL]")
        target_tokens.append("[EOS]")

    return {
        "task_token": task_token,
        "prompt_body_tokens": prompt_body,
        "prompt_tokens": prompt_tokens,
        "target_tokens": target_tokens,
        "holes": holes,
    }


def _build_completion_row(
    rep,
    *,
    name: str,
    source_smiles: str,
    task_name: str,
    query_smiles: str,
    split: str,
    strict_decode: bool,
) -> tuple[Optional[dict[str, Any]], str]:
    encoded = _encode_stamp_groups(rep, source_smiles)
    if encoded is None:
        return None, "encode_source_failed"
    source_mol, source_tokens, groups = encoded

    keep_atoms = _match_query_atoms(source_mol, query_smiles)
    keep_indices = _keep_group_indices_for_atoms(groups, keep_atoms) if keep_atoms else []
    query_tokens = _encode_query_condition_tokens(rep, query_smiles)
    query_source = "query_encode"
    if query_tokens is None:
        query_tokens = _matched_group_condition_tokens(groups, keep_indices)
        query_source = "matched_source_groups"
    if query_tokens is None:
        return None, "encode_query_failed"

    full_stamp = _detok(source_tokens)
    decoded_smiles = ""
    roundtrip_ok = False
    try:
        decoded_smiles = rep.decode(full_stamp)
        roundtrip_ok = canonicalize_smiles(decoded_smiles) == canonicalize_smiles(source_smiles)
    except Exception:
        if strict_decode:
            return None, "full_decode_failed"

    task_token = TASK_TOKEN_BY_GENMOL[task_name]
    prompt_tokens = [task_token] + query_tokens + ["[GEN]"]
    target_tokens = list(source_tokens) + ["[EOS]"]
    train_stamp = _detok(prompt_tokens + target_tokens)

    row = {
        "name": str(name),
        "task": task_name,
        "task_token": task_token,
        "source_smiles": str(source_smiles),
        "query_smiles": str(query_smiles),
        "split_random": str(split),
        "all_pass": bool(roundtrip_ok),
        "stamp_smiles": train_stamp,
        "prompt_stamp": _detok(prompt_tokens),
        "target_stamp": _detok(target_tokens),
        "prompt_len": int(len(prompt_tokens)),
        "target_len": int(len(target_tokens)),
        "label_start": int(len(prompt_tokens)),
        "loss_mask_spec": f"prompt0:{len(prompt_tokens)},target1:{len(target_tokens)}",
        "full_stamp": full_stamp,
        "masked_stamp": _detok(query_tokens),
        "left_stamp": _detok(query_tokens),
        "middle_stamp": full_stamp,
        "right_stamp": "",
        "decode_smiles": decoded_smiles,
        "roundtrip_ok": bool(roundtrip_ok),
        "n_groups": int(len(groups)),
        "n_keep_groups": int(len(set(keep_indices))),
        "n_mask_groups": int(0),
        "n_holes": int(0),
        "hole_spans": "[]",
        "keep_group_indices": json.dumps(sorted(set(int(x) for x in keep_indices))),
        "mask_group_indices": "[]",
        "construction_mode": "completion",
        "decode_mode": "generated_full",
        "condition_source": query_source,
    }
    return row, "ok"


def _branch_group_count(group: Any) -> Optional[int]:
    toks = [str(tok) for tok in group.tokens]
    if not toks:
        return None
    tok = toks[0]
    if not (tok.startswith("[B_") and tok.endswith("]")):
        return None
    try:
        return max(1, int(tok[len("[B_"):-1]))
    except ValueError:
        return None


def _make_benchmark_row(
    *,
    name: str,
    source_smiles: str,
    task_name: str,
    query_smiles: str,
    split: str,
    prompt_body_tokens: Sequence[str],
    infill_token: str,
    full_stamp: str = "",
    masked_middle_tokens: Optional[Sequence[str]] = None,
    n_groups: int = 0,
    keep_indices: Optional[Sequence[int]] = None,
    mask_indices: Optional[Sequence[int]] = None,
    row_suffix: str = "",
    holes_override: Optional[Sequence[dict[str, Any]]] = None,
) -> dict[str, Any]:
    task_token = TASK_TOKEN_BY_GENMOL[task_name]
    prompt_tokens = [task_token] + list(prompt_body_tokens) + ["[GEN]"]
    holes = list(holes_override) if holes_override is not None else [
        {
            "hole_index": 0,
            "infill_token": infill_token,
            "start_group": -1,
            "end_group": -1,
            "n_groups": int(len(mask_indices or [])),
            "middle_stamp": _detok(masked_middle_tokens or []),
            "middle_len": int(len(masked_middle_tokens or [])),
        }
    ]
    target_tokens = ["[EOS]"]
    row_name = str(name) + (f":{row_suffix}" if row_suffix else "")
    return {
        "name": row_name,
        "task": task_name,
        "task_token": task_token,
        "source_smiles": str(source_smiles),
        "query_smiles": str(query_smiles),
        "split_random": str(split),
        "all_pass": True,
        "stamp_smiles": _detok(prompt_tokens + target_tokens),
        "prompt_stamp": _detok(prompt_tokens),
        "target_stamp": _detok(target_tokens),
        "prompt_len": int(len(prompt_tokens)),
        "target_len": int(len(target_tokens)),
        "label_start": int(len(prompt_tokens)),
        "loss_mask_spec": f"prompt0:{len(prompt_tokens)},target1:{len(target_tokens)}",
        "full_stamp": str(full_stamp),
        "masked_stamp": _detok(prompt_body_tokens),
        "left_stamp": "",
        "middle_stamp": _detok(masked_middle_tokens or []),
        "right_stamp": "",
        "decode_smiles": "",
        "roundtrip_ok": True,
        "n_groups": int(n_groups),
        "n_keep_groups": int(len(keep_indices or [])),
        "n_mask_groups": int(len(mask_indices or [])),
        "n_holes": int(len(holes)),
        "hole_spans": json.dumps(holes, ensure_ascii=False),
        "keep_group_indices": json.dumps(list(keep_indices or [])),
        "mask_group_indices": json.dumps(list(mask_indices or [])),
        "construction_mode": "benchmark",
        "decode_mode": "splice",
    }


def _build_benchmark_rows(
    rep,
    *,
    name: str,
    source_smiles: str,
    task_name: str,
    query_smiles: str,
    split: str,
    strict_decode: bool,
    max_scaffold_subtrees: int,
    max_superstructure_attach_points: int,
    superstructure_attach_start: int,
) -> tuple[list[dict[str, Any]], str]:
    if task_name == "linker_design":
        parts = [p for p in str(query_smiles).split(".") if p]
        if len(parts) < 2:
            return [], "linker_need_two_fragments"
        left = _encode_query_part_with_attach(rep, parts[0], port_role="output")
        right = _encode_query_part_with_attach(rep, parts[1], port_role="input")
        if not left or not right:
            return [], "encode_query_failed"
        infill = SINGLE_INFILL_TOKEN_BY_GENMOL[task_name]
        prompt_body = list(left) + [infill] + list(right)
        return [
            _make_benchmark_row(
                name=name,
                source_smiles=source_smiles,
                task_name=task_name,
                query_smiles=query_smiles,
                split=split,
                prompt_body_tokens=prompt_body,
                infill_token=infill,
            )
        ], "ok"

    if task_name == "motif_extension":
        query = _encode_query_part_with_attach(rep, str(query_smiles))
        if not query:
            return [], "encode_query_failed"
        infill = SINGLE_INFILL_TOKEN_BY_GENMOL[task_name]
        return [
            _make_benchmark_row(
                name=name,
                source_smiles=source_smiles,
                task_name=task_name,
                query_smiles=query_smiles,
                split=split,
                prompt_body_tokens=list(query) + [infill],
                infill_token=infill,
            )
        ], "ok"

    if task_name == "superstructure_generation":
        rows: list[dict[str, Any]] = []
        limit = max(1, int(max_superstructure_attach_points))
        sites = _encode_query_no_dummy_valid_attach_prompts(
            rep,
            str(query_smiles),
            max_sites=limit + max(0, int(superstructure_attach_start)),
        )
        sites = sites[max(0, int(superstructure_attach_start)) : max(0, int(superstructure_attach_start)) + limit]
        for site in sites:
            query = list(site["prompt_tokens"])
            infill = SINGLE_INFILL_TOKEN_BY_GENMOL[task_name]
            rows.append(
                _make_benchmark_row(
                    name=name,
                    source_smiles=source_smiles,
                    task_name=task_name,
                    query_smiles=query_smiles,
                    split=split,
                    prompt_body_tokens=list(query) + [infill],
                    infill_token=infill,
                    row_suffix=f"attach{int(site['site_rank'])}",
                )
            )
        return (rows, "ok") if rows else ([], "encode_query_failed")

    if task_name == "scaffold_decoration":
        dummy_prompt = _encode_dummy_scaffold_prompt(rep, str(query_smiles), task_name=task_name)
        if dummy_prompt is not None:
            holes = list(dummy_prompt["holes"])
            infill = (
                str(holes[0]["infill_token"])
                if len(holes) == 1
                else SINGLE_INFILL_TOKEN_BY_GENMOL[task_name]
            )
            return [
                _make_benchmark_row(
                    name=name,
                    source_smiles=source_smiles,
                    task_name=task_name,
                    query_smiles=query_smiles,
                    split=split,
                    prompt_body_tokens=list(dummy_prompt["prompt_body_tokens"]),
                    infill_token=infill,
                    n_groups=int(dummy_prompt["n_groups"]),
                    holes_override=holes,
                )
            ], "ok"

        encoded = _encode_stamp_groups(rep, source_smiles)
        if encoded is None:
            return [], "encode_source_failed"
        source_mol, source_tokens, groups = encoded
        keep_atoms = _match_query_atoms(source_mol, query_smiles)
        keep_indices = set(_keep_group_indices_for_atoms(groups, keep_atoms)) if keep_atoms else set()
        full_stamp = _detok(source_tokens)
        try:
            rep.decode(full_stamp)
        except Exception:
            if strict_decode:
                return [], "full_decode_failed"

        candidate_runs: list[list[int]] = []
        for start, group in enumerate(groups):
            branch_n = _branch_group_count(group)
            if branch_n is None:
                continue
            end = start + branch_n
            if end > len(groups):
                continue
            run = list(range(start, end))
            if keep_indices and any(idx in keep_indices for idx in run):
                continue
            candidate_runs.append(run)
            if int(max_scaffold_subtrees) > 0 and len(candidate_runs) >= int(max_scaffold_subtrees):
                break
        if not candidate_runs:
            return [], "no_scaffold_subtree"

        run_by_start = {run[0]: (hole_idx, run) for hole_idx, run in enumerate(candidate_runs)}
        n_holes = len(candidate_runs)
        prompt_body: list[str] = []
        holes: list[dict[str, Any]] = []
        all_middle: list[str] = []
        i = 0
        while i < len(groups):
            item = run_by_start.get(i)
            if item is None:
                prompt_body.extend(str(tok) for tok in groups[i].tokens)
                i += 1
                continue
            hole_idx, run = item
            group_tokens = [str(tok) for tok in groups[i].tokens]
            if not group_tokens or not group_tokens[0].startswith("[B_"):
                i += 1
                continue
            infill = SINGLE_INFILL_TOKEN_BY_GENMOL[task_name] if n_holes == 1 else f"[INFILL_{hole_idx}]"
            middle = list(group_tokens[1:])
            for j in run[1:]:
                middle.extend(str(tok) for tok in groups[j].tokens)
            prompt_body.append(group_tokens[0])
            prompt_body.append(infill)
            all_middle.extend(middle)
            holes.append(
                {
                    "hole_index": int(hole_idx),
                    "infill_token": infill,
                    "start_group": int(run[0]),
                    "end_group": int(run[-1] + 1),
                    "n_groups": int(len(run)),
                    "middle_stamp": _detok(middle),
                    "middle_len": int(len(middle)),
                }
            )
            i = int(run[-1]) + 1

        task_token = TASK_TOKEN_BY_GENMOL[task_name]
        prompt_tokens = [task_token] + prompt_body + ["[GEN]"]
        target_tokens = ["[EOS]"]
        mask_indices = [idx for run in candidate_runs for idx in run]
        row = {
            "name": str(name),
            "task": task_name,
            "task_token": task_token,
            "source_smiles": str(source_smiles),
            "query_smiles": str(query_smiles),
            "split_random": str(split),
            "all_pass": True,
            "stamp_smiles": _detok(prompt_tokens + target_tokens),
            "prompt_stamp": _detok(prompt_tokens),
            "target_stamp": _detok(target_tokens),
            "prompt_len": int(len(prompt_tokens)),
            "target_len": int(len(target_tokens)),
            "label_start": int(len(prompt_tokens)),
            "loss_mask_spec": f"prompt0:{len(prompt_tokens)},target1:{len(target_tokens)}",
            "full_stamp": full_stamp,
            "masked_stamp": _detok(prompt_body),
            "left_stamp": "",
            "middle_stamp": _detok(all_middle),
            "right_stamp": "",
            "decode_smiles": "",
            "roundtrip_ok": True,
            "n_groups": int(len(groups)),
            "n_keep_groups": int(len(keep_indices)),
            "n_mask_groups": int(len(mask_indices)),
            "n_holes": int(len(holes)),
            "hole_spans": json.dumps(holes, ensure_ascii=False),
            "keep_group_indices": json.dumps(sorted(keep_indices)),
            "mask_group_indices": json.dumps(mask_indices),
            "construction_mode": "benchmark",
            "decode_mode": "splice",
        }
        return [row], "ok"

    return [], "unsupported_task"


def _build_fcstamp_row(
    rep,
    *,
    name: str,
    source_smiles: str,
    task_name: str,
    query_smiles: str,
    split: str,
    max_holes: int,
    single_hole_only: bool,
    allow_leading_hole: bool,
    strict_decode: bool,
) -> tuple[Optional[dict[str, Any]], str]:
    encoded = _encode_stamp_groups(rep, source_smiles)
    if encoded is None:
        return None, "encode_source_failed"
    source_mol, source_tokens, groups = encoded

    keep_atoms = _match_query_atoms(source_mol, query_smiles)
    if not keep_atoms:
        return None, "query_match_failed"
    keep_indices = _keep_group_indices_for_atoms(groups, keep_atoms)
    if not keep_indices:
        return None, "no_keep_groups"

    keep_set = set(int(x) for x in keep_indices)
    mask_indices = [idx for idx in range(len(groups)) if idx not in keep_set]
    if not mask_indices:
        return None, "no_missing_groups"
    runs = _contiguous_runs(mask_indices)
    if single_hole_only and len(runs) != 1:
        return None, "multi_hole"
    if max_holes > 0 and len(runs) > max_holes:
        return None, "too_many_holes"

    built = _build_prompt_and_target(groups, runs, task_name=task_name)
    if (
        not allow_leading_hole
        and built["prompt_body_tokens"]
        and str(built["prompt_body_tokens"][0]).startswith("[INFILL")
    ):
        return None, "leading_hole"
    full_stamp = _detok(source_tokens)
    prompt_stamp = _detok(built["prompt_tokens"])
    target_stamp = _detok(built["target_tokens"])
    train_stamp = _detok(list(built["prompt_tokens"]) + list(built["target_tokens"]))
    masked_stamp = _detok(built["prompt_body_tokens"])

    decoded_smiles = ""
    roundtrip_ok = False
    try:
        decoded_smiles = rep.decode(full_stamp)
        roundtrip_ok = canonicalize_smiles(decoded_smiles) == canonicalize_smiles(source_smiles)
    except Exception:
        if strict_decode:
            return None, "full_decode_failed"

    holes = built["holes"]
    single = holes[0] if len(holes) == 1 else None
    left_tokens: list[str] = []
    right_tokens: list[str] = []
    middle_tokens: list[str] = []
    if single is not None:
        start = int(single["start_group"])
        end = int(single["end_group"])
        left_tokens = _group_tokens(groups, range(0, start))
        middle_tokens = _group_tokens(groups, range(start, end))
        right_tokens = _group_tokens(groups, range(end, len(groups)))

    row = {
        "name": str(name),
        "task": task_name,
        "task_token": built["task_token"],
        "source_smiles": str(source_smiles),
        "query_smiles": str(query_smiles),
        "split_random": str(split),
        "all_pass": bool(roundtrip_ok),
        "stamp_smiles": train_stamp,
        "prompt_stamp": prompt_stamp,
        "target_stamp": target_stamp,
        "prompt_len": int(len(built["prompt_tokens"])),
        "target_len": int(len(built["target_tokens"])),
        "label_start": int(len(built["prompt_tokens"])),
        "loss_mask_spec": f"prompt0:{len(built['prompt_tokens'])},target1:{len(built['target_tokens'])}",
        "full_stamp": full_stamp,
        "masked_stamp": masked_stamp,
        "left_stamp": _detok(left_tokens),
        "middle_stamp": _detok(middle_tokens),
        "right_stamp": _detok(right_tokens),
        "decode_smiles": decoded_smiles,
        "roundtrip_ok": bool(roundtrip_ok),
        "n_groups": int(len(groups)),
        "n_keep_groups": int(len(keep_set)),
        "n_mask_groups": int(len(mask_indices)),
        "n_holes": int(len(holes)),
        "hole_spans": json.dumps(holes, ensure_ascii=False),
        "keep_group_indices": json.dumps(sorted(keep_set)),
        "mask_group_indices": json.dumps(mask_indices),
        "construction_mode": "splice",
        "decode_mode": "splice",
    }
    return row, "ok"


def _choose_split(rng: random.Random, valid_fraction: float) -> str:
    if valid_fraction <= 0:
        return "train"
    if rng.random() < valid_fraction:
        return "val"
    return "train"


def _write_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if output_path.suffix.lower() in {".csv", ".tsv"}:
        sep = "\t" if output_path.suffix.lower() == ".tsv" else ","
        df.to_csv(output_path, index=False, sep=sep)
    else:
        df.to_parquet(output_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Construct GenMol fragment tasks as FC-STAMP training rows."
    )
    parser.add_argument("--fragments-csv", default="third_party/genmol-read-local/data/fragments.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stamp-vocab", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(GENMOL_TASKS), choices=list(GENMOL_TASKS))
    parser.add_argument(
        "--construction",
        choices=("benchmark", "splice", "completion"),
        default="benchmark",
        help="benchmark = GenMol test-time in-place prompts; splice = source-derived single-hole FIM; completion = debugging full generation.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Limit input fragments rows; 0 = all.")
    parser.add_argument("--max-holes", type=int, default=1, help="Skip splice rows with more holes; <=0 = unlimited.")
    parser.add_argument(
        "--single-hole-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require one splice hole. Use --no-single-hole-only only for debugging multi-hole FIM.",
    )
    parser.add_argument(
        "--allow-leading-hole",
        action="store_true",
        help="Allow splice prompts whose first body token is an infill token.",
    )
    parser.add_argument("--max-scaffold-subtrees-per-case", type=int, default=0, help="0 = all branch-subtree prompts.")
    parser.add_argument("--max-superstructure-attach-points", type=int, default=1)
    parser.add_argument(
        "--superstructure-attach-start",
        type=int,
        default=0,
        help="First valid unique attach-site rank for benchmark superstructure prompts.",
    )
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strip-stereo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop [S_*] stereo tokens from generated benchmark rows.",
    )
    parser.add_argument("--allow-decode-failures", dest="strict_decode", action="store_false")
    parser.set_defaults(strict_decode=True)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    rng = random.Random(int(args.seed))
    rep = build_representation("stamp", stamp_vocab=args.stamp_vocab)
    df = pd.read_csv(args.fragments_csv)
    if int(args.max_rows) > 0:
        df = df.head(int(args.max_rows))

    rows: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    t0 = time.time()
    for _, item in df.iterrows():
        source_smiles = str(item.get("smiles", "") or "")
        name = str(item.get("name", "") or "")
        for task_name in args.tasks:
            query_value = item.get(task_name, "")
            if pd.isna(query_value) or not str(query_value).strip():
                skip_reasons["empty_query"] += 1
                continue
            split = _choose_split(rng, float(args.valid_fraction))
            if args.construction == "benchmark":
                built_rows, reason = _build_benchmark_rows(
                    rep,
                    name=name,
                    source_smiles=source_smiles,
                    task_name=task_name,
                    query_smiles=str(query_value),
                    split=split,
                    strict_decode=bool(args.strict_decode),
                    max_scaffold_subtrees=int(args.max_scaffold_subtrees_per_case),
                    max_superstructure_attach_points=int(args.max_superstructure_attach_points),
                    superstructure_attach_start=int(args.superstructure_attach_start),
                )
                if not built_rows:
                    skip_reasons[reason] += 1
                    continue
                rows.extend(built_rows)
                continue
            if args.construction == "completion":
                row, reason = _build_completion_row(
                    rep,
                    name=name,
                    source_smiles=source_smiles,
                    task_name=task_name,
                    query_smiles=str(query_value),
                    split=split,
                    strict_decode=bool(args.strict_decode),
                )
            else:
                row, reason = _build_fcstamp_row(
                    rep,
                    name=name,
                    source_smiles=source_smiles,
                    task_name=task_name,
                    query_smiles=str(query_value),
                    split=split,
                    max_holes=int(args.max_holes),
                    single_hole_only=bool(args.single_hole_only),
                    allow_leading_hole=bool(args.allow_leading_hole),
                    strict_decode=bool(args.strict_decode),
                )
            if row is None:
                skip_reasons[reason] += 1
                continue
            rows.append(row)

    if bool(args.strip_stereo):
        rows = [_strip_stereo_from_row(row) for row in rows]

    out = Path(args.out)
    _write_rows(rows, out)
    summary = {
        "fragments_csv": str(args.fragments_csv),
        "output_path": str(out),
        "stamp_vocab": str(args.stamp_vocab),
        "tasks": list(args.tasks),
        "construction": str(args.construction),
        "strip_stereo": bool(args.strip_stereo),
        "input_rows": int(len(df)),
        "output_rows": int(len(rows)),
        "skip_reasons": dict(skip_reasons),
        "n_by_task": dict(Counter(row["task"] for row in rows)),
        "n_by_split": dict(Counter(row["split_random"] for row in rows)),
        "n_by_holes": {str(k): int(v) for k, v in Counter(row["n_holes"] for row in rows).items()},
        "n_by_decode_mode": dict(Counter(str(row.get("decode_mode", "splice")) for row in rows)),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    summary_path = out.with_name(out.name + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
