# STAMP models

Checkpoints, vocabularies and configurations:
[bi0s/STAMP_V1](https://huggingface.co/bi0s/STAMP_V1).
Methods and evaluation protocols: [paper](https://arxiv.org/abs/2610.18438).

| Model | Hugging Face directory |
| --- | --- |
| AO-GPT generation | `ao_generation` |
| Fragment infilling | `fragment_infill` |
| Scaffold decoration | `scaffold_decoration` |
| PubChem MLM encoder | `encoder_mlm` |
| PubChem MTR encoder | `encoder_mtr` |
| STAMP-MolT5 pretraining, 500k steps | `molt5_pretrain_500k` |
| Molecule-to-text, initialized from 500k | `molt5_mol2text_from500k` |
| Text-to-molecule, initialized from 500k | `molt5_text2mol_from500k` |

The repository also preserves two earlier MolT5 downstream versions named
`molt5_mol2text_from350k` and `molt5_text2mol_from350k`. These are separate
versions from the 500k initialization described in the paper's Methods.

Download all companion files for the selected model. AO-GPT and encoder `.pt`
files use the implementations in `stamp/benchmark/`; MolT5 directories use
the Transformers T5 format. Map saved training-system vocabulary paths to the
downloaded vocabulary files when loading on another computer.

Models are intended for molecular research. The source code is [MIT licensed](LICENSE);
upstream model and dataset terms apply separately.
