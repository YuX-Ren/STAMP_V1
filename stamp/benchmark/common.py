from __future__ import annotations

from typing import Iterable, Optional

from rdkit import Chem

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def iterable_with_progress(
    iterable: Iterable,
    show_progress: bool,
    total: Optional[int] = None,
    desc: str = "",
):
    if show_progress and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def canonicalize_smiles(smiles: str, require_single_component: bool = False) -> str | None:
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if mol.GetNumAtoms() == 0:
        return None
    if require_single_component and len(Chem.GetMolFrags(mol)) != 1:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def chemically_equivalent_smiles(
    smiles_a: str,
    smiles_b: str,
    ignore_stereo: bool = False,
) -> bool:
    if not smiles_a or not smiles_b:
        return False
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return False

    if ignore_stereo:
        Chem.RemoveStereochemistry(mol_a)
        Chem.RemoveStereochemistry(mol_b)

    # Fast path: canonical isomeric (or non-isomeric) SMILES equality.
    iso = not ignore_stereo
    can_a = Chem.MolToSmiles(mol_a, canonical=True, isomericSmiles=iso)
    can_b = Chem.MolToSmiles(mol_b, canonical=True, isomericSmiles=iso)
    if can_a == can_b:
        return True

    # Robust path: InChIKey collapses many equivalent valence/charge spellings.
    try:
        key_a = Chem.MolToInchiKey(mol_a)
        key_b = Chem.MolToInchiKey(mol_b)
        if key_a and key_b:
            return key_a == key_b
    except Exception:
        pass

    return False
