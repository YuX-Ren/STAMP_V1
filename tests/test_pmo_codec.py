"""Chemical roundtrips for the codec used by the PMO baseline."""
import unittest

from rdkit import Chem

from stamp.core.pmo_codec import PMOEncoder


@unittest.skipUnless(
    hasattr(Chem.SmilesWriteParams(), "ignoreAtomMapNumbers"),
    "PMO requires RDKit 2025.09.4",
)
class PMOCodecRoundtripTests(unittest.TestCase):
    def test_roundtrip_preserves_connectivity_and_stereochemistry(self):
        encoder = PMOEncoder()
        for smiles in ["CCO", "CC(C)O", "CC1CCCCC1", "C[C@H](O)Cl", "O=C(O)c1ccccc1"]:
            with self.subTest(smiles=smiles):
                decoded = encoder.decode_stamp_to_mol(encoder.encode_smiles(smiles))
                self.assertEqual(
                    Chem.MolToSmiles(decoded, isomericSmiles=True),
                    Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True),
                )

    def test_fragment_mapping_preserves_input_atom_maps(self):
        molecule = Chem.MolFromSmiles("[CH3:7][CH2:8][OH:9]")
        before = Chem.MolToSmiles(molecule)
        PMOEncoder._fragment_motif_and_mapping(molecule, list(range(molecule.GetNumAtoms())))
        self.assertEqual(Chem.MolToSmiles(molecule), before)


if __name__ == "__main__":
    unittest.main()
