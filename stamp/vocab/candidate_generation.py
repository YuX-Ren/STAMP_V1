from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from rdkit import Chem

from ..core.stamp_codec import STAMPEncoder
from .canonicalize import canonicalize_core, canonicalize_schema
from .stamp_vocab_types import CandidateMotif, PortInfo
from .tree_metrics import compute_tree_gain


def _bond_type_to_str(bond: Chem.Bond) -> str:
    if bond.GetIsAromatic():
        return "AROMATIC"
    btype = bond.GetBondType()
    if btype == Chem.BondType.SINGLE:
        return "SINGLE"
    if btype == Chem.BondType.DOUBLE:
        return "DOUBLE"
    if btype == Chem.BondType.TRIPLE:
        return "TRIPLE"
    return str(btype)


def _stereo_role(label: str | None) -> str:
    if label in {"R", "S"}:
        return "TETRA_REF"
    if label in {"E", "Z"}:
        return "EZ_REF"
    return "NONE"


def _local_env_code(atom: Chem.Atom) -> str:
    return (
        f"{atom.GetAtomicNum()}|"
        f"chg={atom.GetFormalCharge()}|"
        f"ar={int(atom.GetIsAromatic())}|"
        f"deg={atom.GetDegree()}"
    )


def generate_candidates(
    mol: Chem.Mol,
    encoder: STAMPEncoder | None = None,
) -> Tuple[List[CandidateMotif], Dict[str, int]]:
    """
    Generate STAMP motif candidates from a molecule via exocyclic cut decomposition.

    Returns:
      candidates, decomp_metrics
    """
    enc = encoder or STAMPEncoder()
    cut_bond_ids = enc._find_cut_bonds(mol)
    fragments, edges = enc._build_fragments_and_edges(mol, cut_bond_ids)

    atom_to_comp: Dict[int, int] = {}
    for frag in fragments:
        for atom_idx in frag.atom_indices:
            atom_to_comp[atom_idx] = frag.frag_id

    # Map local port -> stereo label captured from interface.
    stereo_labels: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for e in edges:
        if e.left_stereo_label is not None:
            stereo_labels[(e.left, e.left_local_atom)].append(e.left_stereo_label)
        if e.right_stereo_label is not None:
            stereo_labels[(e.right, e.right_local_atom)].append(e.right_stereo_label)

    ports_by_frag: Dict[int, List[PortInfo]] = defaultdict(list)
    for bond_id in cut_bond_ids:
        bond = mol.GetBondWithIdx(bond_id)
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        ca = atom_to_comp[a]
        cb = atom_to_comp[b]

        la = fragments[ca].orig_to_local[a]
        lb = fragments[cb].orig_to_local[b]

        labels_a = stereo_labels.get((ca, la), [])
        labels_b = stereo_labels.get((cb, lb), [])
        label_a = labels_a[0] if labels_a else None
        label_b = labels_b[0] if labels_b else None

        ports_by_frag[ca].append(
            PortInfo(
                local_atom_idx=la,
                external_bond_type=_bond_type_to_str(bond),
                local_env_code=_local_env_code(mol.GetAtomWithIdx(a)),
                stereo_role=_stereo_role(label_a),
            )
        )
        ports_by_frag[cb].append(
            PortInfo(
                local_atom_idx=lb,
                external_bond_type=_bond_type_to_str(bond),
                local_env_code=_local_env_code(mol.GetAtomWithIdx(b)),
                stereo_role=_stereo_role(label_b),
            )
        )

    candidates: List[CandidateMotif] = []
    for frag in fragments:
        core_hash, canonical_smarts, atom_count, bond_count = canonicalize_core(
            mol, frag.atom_indices
        )
        schema_hash, ordered_ports = canonicalize_schema(ports_by_frag.get(frag.frag_id, []))
        compression_gain = float(max(0, atom_count - 1))
        tree_gain = compute_tree_gain(mol, frag.atom_indices)

        candidates.append(
            CandidateMotif(
                atom_indices=tuple(frag.atom_indices),
                internal_subgraph_hash=core_hash,
                canonical_smarts=canonical_smarts,
                bond_count=bond_count,
                ports=ordered_ports,
                schema_hash=schema_hash,
                compression_gain=compression_gain,
                tree_gain=tree_gain,
                recoverability_score=1.0,
                ambiguity_penalty=0.0,
            )
        )

    decomp_metrics = {
        "num_fragments": len(fragments),
        "num_cut_bonds": len(cut_bond_ids),
        "num_edges": len(edges),
    }
    return candidates, decomp_metrics
