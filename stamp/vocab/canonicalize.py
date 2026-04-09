from __future__ import annotations

import hashlib
import json
from typing import List, Sequence, Tuple

from rdkit import Chem

from .stamp_vocab_types import PortInfo


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def canonicalize_core(mol: Chem.Mol, atom_indices: Sequence[int]) -> Tuple[str, str, int, int]:
    smiles = Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=list(atom_indices),
        canonical=True,
        isomericSmiles=True,
    )
    core_hash = _short_hash(smiles)

    atom_set = set(atom_indices)
    bond_count = 0
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in atom_set and b in atom_set:
            bond_count += 1

    return core_hash, smiles, len(atom_indices), bond_count


def canonicalize_schema(ports: Sequence[PortInfo]) -> Tuple[str, List[PortInfo]]:
    ordered = sorted(
        ports,
        key=lambda p: (
            p.local_atom_idx,
            p.external_bond_type,
            p.stereo_role,
            p.local_env_code,
        ),
    )

    normalized: List[PortInfo] = []
    for rank, port in enumerate(ordered):
        normalized.append(
            PortInfo(
                local_atom_idx=port.local_atom_idx,
                external_bond_type=port.external_bond_type,
                local_env_code=port.local_env_code,
                stereo_role=port.stereo_role,
                port_canonical_rank=rank,
            )
        )

    serializable = [
        {
            "local_atom_idx": p.local_atom_idx,
            "external_bond_type": p.external_bond_type,
            "local_env_code": p.local_env_code,
            "stereo_role": p.stereo_role,
            "port_canonical_rank": p.port_canonical_rank,
        }
        for p in normalized
    ]
    schema_hash = _short_hash(json.dumps(serializable, sort_keys=True, separators=(",", ":")))
    return schema_hash, normalized
