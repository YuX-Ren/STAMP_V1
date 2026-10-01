# Reproducibility and local assets

The repository keeps source code and lightweight documentation in Git. Datasets,
checkpoints, scheduler logs, generated figures, and result tables are local
artifacts and are excluded by `.gitignore`.

## Environments

`envs/stamp-train-rdkit-2025.09.4.yml` is the core codec/training environment.
Some research entry points use optional packages that are intentionally not
forced into the core environment:

| Workflow | Optional packages |
| --- | --- |
| Hugging Face pretraining and MolT5 | `transformers`, `datasets`, `huggingface_hub`, `sentencepiece` |
| MoleculeNet/property baselines | `deepchem`, `PyTDC` |
| Representation comparisons | `selfies`, `safe-mol`, `group-selfies` |
| Extended analysis | `umap-learn`, `datamol`, `loguru` |

Install only the extras required by the selected workflow. The CLI modules use
lazy imports where possible and report missing optional dependencies at runtime.

## Third-party checkouts

Third-party repositories are not vendored. The local experiments used these
checkouts under `third_party/`:

| Local directory | Upstream | Recorded commit |
| --- | --- | --- |
| `Eso-LMs` | `https://github.com/s-sahoo/Eso-LMs.git` | `515adf07ccf5ae71262df316adbfa41a295fab7c` |
| `genmol-pmo-clean` | `https://github.com/NVIDIA-Digital-Bio/genmol.git` | `add09fc83b7255bd09c797e527c0f4b51f5fb7c1` |
| `genmol-pmo-aonly` | `https://github.com/NVIDIA-Digital-Bio/genmol.git` | `add09fc83b7255bd09c797e527c0f4b51f5fb7c1` |
| `genmol-read-local` | `https://github.com/NVIDIA-Digital-Bio/genmol.git` | `add09fc83b7255bd09c797e527c0f4b51f5fb7c1` |

The representation layer also supports local checkouts of SAFE
(`https://github.com/datamol-io/safe.git`), Group SELFIES
(`https://github.com/aspuru-guzik-group/group-selfies.git`), and ChemicalGoF
(`https://github.com/f48r1/chemicalgof.git`). Follow each upstream repository's
license and installation instructions.

## Cluster paths

Files in `scripts/hpc/` are archived launch recipes. Defaults containing
`/scratch`, `/nfs`, or `/Project_Storage` describe the original cluster layout;
they are not portable paths. Prefer the environment-variable overrides exposed
by each script when adapting a run to another system.

The PMO deployment copy under `isolated/` and oracle pickle files under
`oracle/` are also local-only. See `pmo_project_storage_bundle.md` for the bundle
layout expected by those launchers.
