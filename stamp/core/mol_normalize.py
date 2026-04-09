from __future__ import annotations

from rdkit import Chem


def normalize_mol(mol: Chem.Mol) -> Chem.Mol:
    """Return a sanitized, canonicalized RDKit molecule for deterministic processing."""
    if mol is None:
        raise ValueError("Input molecule is None.")

    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    normalized = Chem.MolFromSmiles(smiles)
    if normalized is None:
        raise ValueError("Failed to normalize molecule.")

    Chem.SanitizeMol(normalized)
    Chem.AssignStereochemistry(normalized, force=True, cleanIt=True)
    return normalized


def normalize_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return normalize_mol(mol)
