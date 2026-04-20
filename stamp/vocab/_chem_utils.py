from __future__ import annotations

from typing import Dict, Optional

from rdkit import Chem


MAX_VALENCE: Dict[str, int] = {
    "B": 3,
    "Br": 1,
    "C": 4,
    "Cl": 1,
    "F": 1,
    "I": 1,
    "N": 5,
    "O": 2,
    "P": 5,
    "S": 6,
    "Na": 1,
}


def smi2mol(smiles: str, kekulize: bool = False, sanitize: bool = True) -> Optional[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        return None
    if kekulize:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol


def mol2smi(mol: Chem.Mol, canonical: bool = True, isomeric: bool = True) -> str:
    return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric)


def get_submol(mol: Chem.Mol, atom_indices, kekulize: bool = False) -> Chem.Mol:
    atom_indices = list(atom_indices)
    if len(atom_indices) == 1:
        atom = mol.GetAtomWithIdx(atom_indices[0])
        symbol = atom.GetSymbol()
        if symbol == "Si":
            seed = "[Si]"
        elif symbol == "Na":
            seed = "[Na+]"
        else:
            seed = symbol
        sub = smi2mol(seed, kekulize=kekulize)
        if sub is None:
            raise ValueError(f"Failed to seed single-atom submol for {symbol}")
        return sub

    aid_set = set(atom_indices)
    edge_indices = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        if bond.GetBeginAtomIdx() in aid_set and bond.GetEndAtomIdx() in aid_set:
            edge_indices.append(i)
    return Chem.PathToSubmol(mol, edge_indices)


def cnt_atom(smi: str, return_dict: bool = False):
    atom_dict = {atom: 0 for atom in MAX_VALENCE}
    i = 0
    while i < len(smi):
        symbol = smi[i].upper()
        next_char = smi[i + 1] if i + 1 < len(smi) else None
        if symbol == "B" and next_char == "r":
            symbol = "Br"
        elif symbol == "C" and next_char == "l":
            symbol = "Cl"
        if symbol in atom_dict:
            atom_dict[symbol] += 1
        i += 1
    if return_dict:
        return atom_dict
    return sum(atom_dict.values())
