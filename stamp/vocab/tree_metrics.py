from __future__ import annotations

from typing import Sequence

from rdkit import Chem


def cycle_rank_of_subgraph(mol: Chem.Mol, atom_indices: Sequence[int]) -> int:
    atom_set = set(atom_indices)
    if not atom_set:
        return 0
    bond_count = 0
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in atom_set and b in atom_set:
            bond_count += 1
    # candidate motifs are connected in this pipeline
    return max(0, bond_count - len(atom_set) + 1)


def compute_tree_gain(mol: Chem.Mol, atom_indices: Sequence[int]) -> float:
    # Initial STAMP-v1 proxy: collapsing a motif removes its internal cycles.
    return float(cycle_rank_of_subgraph(mol, atom_indices))


def contracted_cycle_rank(num_nodes: int, num_edges: int, num_components: int = 1) -> int:
    return max(0, num_edges - num_nodes + num_components)
