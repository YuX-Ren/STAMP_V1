"""Codec behavior used by the archived PMO optimizer.

The general STAMP codec remains the default for other tasks. PMO preserves
its original motif strictness and mapped-fragment atom ordering.
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
from rdkit import Chem
from .stamp_codec import Fragment, FragmentEdge, STAMPEncoder


class PMOEncoder(STAMPEncoder):
    def __init__(self, *args, motif_strict_vocab: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.motif_strict_vocab = motif_strict_vocab

    def encode_mol(self, mol: Chem.Mol) -> str:
        if mol.GetNumAtoms() == 0:
            raise ValueError("Empty molecule is not supported.")

        if len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError(
                "Disconnected molecules are not supported in one STAMP string."
            )

        fragments, edges = self._fragments_and_edges_from_mol(mol)
        if not fragments:
            raise ValueError("Failed to build fragments.")
        return " ".join(self._encode_graph_tokens(fragments, edges))

    def _fragments_and_edges_from_mol(
        self,
        mol: Chem.Mol,
    ) -> Tuple[List[Fragment], List[FragmentEdge]]:
        if self.motif_tokenizer is not None:
            # Capture CIP codes and bond E/Z on the aromatic form first.
            # Kekulization can flip CIP priorities and double-bond stereo
            # labels whose neighbors change bond order during aromatization;
            # the decoder rebuilds in aromatic form, so capture must agree
            # with the aromatic reference frame.
            aro_mol = Chem.Mol(mol)
            Chem.AssignStereochemistry(aro_mol, force=True, cleanIt=True)
            cip_by_idx: Dict[int, str] = {}
            for a in aro_mol.GetAtoms():
                if a.HasProp("_CIPCode"):
                    cip_by_idx[a.GetIdx()] = a.GetProp("_CIPCode")
            bond_stereo_by_idx: Dict[int, Tuple[int, Tuple[int, int]]] = {}
            for b in aro_mol.GetBonds():
                st = b.GetStereo()
                if st in (Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOZ):
                    atoms = tuple(b.GetStereoAtoms())
                    if len(atoms) == 2:
                        bond_stereo_by_idx[b.GetIdx()] = (int(st), atoms)

            work_mol = Chem.Mol(mol)
            if self.motif_tokenizer.kekulize:
                Chem.Kekulize(work_mol, clearAromaticFlags=True)
            Chem.AssignStereochemistry(work_mol, force=True, cleanIt=True)
            for idx, code in cip_by_idx.items():
                work_mol.GetAtomWithIdx(idx).SetProp("_CIPCode", code)
            for bond_idx, (st_int, atoms) in bond_stereo_by_idx.items():
                b = work_mol.GetBondWithIdx(bond_idx)
                b.SetStereoAtoms(atoms[0], atoms[1])
                b.SetStereo(Chem.rdchem.BondStereo(st_int))
            groups = self.motif_tokenizer.decompose(work_mol, strict_vocab=self.motif_strict_vocab)
            if getattr(self.motif_tokenizer, "allow_ring_single_edges", False):
                return self._build_fragments_and_edges_from_groups(work_mol, groups)
            cut_bond_ids = self._cut_bonds_from_groups(work_mol, groups)
            return self._build_fragments_and_edges(work_mol, cut_bond_ids)

        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        cut_bond_ids = self._find_cut_bonds(mol)
        return self._build_fragments_and_edges(mol, cut_bond_ids)

    @staticmethod
    def _fragment_motif_and_mapping(
        mol: Chem.Mol, atom_indices: Sequence[int]
    ) -> Tuple[str, Dict[int, int], int]:
        params = Chem.SmilesWriteParams()
        params.canonical = True
        params.doIsomericSmiles = True
        params.ignoreAtomMapNumbers = True

        motif = Chem.MolFragmentToSmiles(
            mol,
            params,
            atomsToUse=list(atom_indices),
        )

        atom_set = set(atom_indices)
        atom_map_numbers: Dict[int, int] = {}
        for idx in atom_indices:
            atom = mol.GetAtomWithIdx(idx)
            atom_map_numbers[idx] = atom.GetAtomMapNum()
            atom.SetAtomMapNum(idx + 1)

        try:
            mapped_smiles = Chem.MolFragmentToSmiles(
                mol,
                params,
                atomsToUse=list(atom_indices),
            )
        finally:
            for idx in atom_indices:
                mol.GetAtomWithIdx(idx).SetAtomMapNum(atom_map_numbers[idx])

        mapped_frag = Chem.MolFromSmiles(mapped_smiles, sanitize=False)
        if mapped_frag is None:
            raise ValueError("RDKit failed to parse mapped fragment SMILES.")

        orig_to_local: Dict[int, int] = {}
        heavy_count = 0
        for atom in mapped_frag.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num <= 0:
                continue
            orig_idx = map_num - 1
            if orig_idx in atom_set:
                orig_to_local[orig_idx] = atom.GetIdx()
            if atom.GetAtomicNum() > 1:
                heavy_count += 1

        return motif, orig_to_local, heavy_count
