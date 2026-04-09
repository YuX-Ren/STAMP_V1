from __future__ import annotations

from rdkit import Chem

from .stamp_codec import STAMPEncoder


def reconstruct_from_stamp(stamp: str, codec: STAMPEncoder | None = None) -> Chem.Mol:
    encoder = codec or STAMPEncoder()
    return encoder.decode_stamp_to_mol(stamp)


def is_exact_match(mol_a: Chem.Mol, mol_b: Chem.Mol) -> bool:
    sa = Chem.MolToSmiles(mol_a, canonical=True, isomericSmiles=True)
    sb = Chem.MolToSmiles(mol_b, canonical=True, isomericSmiles=True)
    return sa == sb
