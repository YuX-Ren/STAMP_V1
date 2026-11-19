# Software and machine learning checklist mapping

This mapping follows the supplied software screenshot and the
`STAMP_Machine_Learning_Checklist.pdf` dated 16 November 2026. Repository
changes provide a runnable demo and reporting material. They do not change
the underlying experimental evidence or grant access to private assets.

## Software submission requirements

| Requirement | Repository evidence | Status |
| --- | --- | --- |
| Source code | `stamp/`, `scripts/` | Included |
| Small demonstration dataset | `examples/codec/molecules.txt` | Included; ten manually selected molecular strings |
| Software dependencies and OS versions | README System requirements; `envs/requirements-demo.txt`; training conda recipe | Demo pins and tested OS reported; research extras described in `docs/reproducibility.md` |
| Tested software versions | README System requirements | Local versions distinguished from the target training recipe |
| Non-standard hardware | README System requirements | CPU demo; research H200 use stated separately |
| Installation and typical time | README Installation | Commands included; desktop times explicitly estimated |
| Demo instructions and expected output | README Demo; `examples/codec/expected_roundtrips.tsv` | Included |
| Demo runtime | README Demo | 0.65 s measured locally; desktop estimate stated |
| Use on user data | README Instructions for use; demo `--input` and `--output` | Included |
| Quantitative result reproduction | `docs/experiments_runbook.md`, workflow reports | Protocols and ten selected model checkpoints available; exact study data require access |
| License | `LICENSE` | MIT, selected by repository owner |
| Code repository URL | `https://github.com/YuX-Ren/STAMP_V1` | Release source repository |
| Code functionality and pseudocode | `docs/software_functionality.md`, `docs/stamp.md` | Repository description provided; manuscript location remains an author declaration |

## Machine learning checklist

| Item | Evidence and scope |
| --- | --- |
| 1 Code availability and README | Source, installation and CPU demo in this repository. No compiled application or Code Ocean capsule is supplied. |
| 1 and 2B Exact study data and model access | See `docs/data_and_model_access.md`. Ten selected study checkpoints are public at `https://huggingface.co/bi0s/STAMP_V1`. Anonymous study dataset access returned HTTP 401 on 19 November 2026; the demo is available separately. |
| 2A Data sources | Workflow sources are mapped in `MODEL_CARD.md` and `docs/data_and_model_access.md`; the manuscript remains the study-wide source inventory. |
| 2C Dataset bias | `MODEL_CARD.md` describes benchmark-domain selection bias and the limits of matched-input comparisons. A repository statement does not establish that the submitted manuscript contains this discussion. |
| 2D Preprocessing | `stamp/benchmark/data.py`, `scripts/benchmark/prepare_pubchem_pretrain_splits.py`; parsing, canonicalization and deduplication before splitting. |
| 2E Multiple data sources | C4/ZINC mixture and source-specific construction described in the MolT5 reports and model card. |
| 3A Architecture | `MODEL_CARD.md` distinguishes causal/AO-GPT, encoders and T5. |
| 3B Model card | `MODEL_CARD.md` now provided for the study model families. |
| 3C and 3D Partitions | Task-specific train/validation/test or supplied benchmark partitions; model card and workflow protocols. |
| 3E Application relevance | Scaffold splits test unseen scaffolds; random and supplied benchmark splits serve the stated comparisons. |
| 3F Leakage prevention | Canonical deduplication and training-only vocabularies; target-specific decontamination script. Global exclusion from all pretraining data is not established. |
| 3G Interpretation | The supplied checklist points to Fig. 3 and manuscript/SI analyses; repository entry point `scripts/benchmark/analyze_edit_mechanism.py`. |
| 4A Metrics | `stamp/benchmark/metrics.py`, generation evaluators, property modules and translation report define task-specific metrics. |
| 4B Cross-validation | Fixed benchmark splits and repeated seeds do not constitute k-fold cross-validation; retain No unless additional evidence is supplied. |
| 4C Community benchmarks | ZINC, MoleculeNet, MoleculeACE, PMO and ChEBI-20 workflow code; S2-Bench is described in the supplied checklist/manuscript. |
| 4D Simple baselines | Hashed-token ridge and random-pair controls are described in the supplied checklist; do not infer their public artifact availability from the codec demo. |
| 4E Contemporary baselines | GenMol, ChemBERTa-2 and MolT5 comparisons in workflow reports; other comparisons follow the manuscript. |
| 4F Ablations | Manuscript/SI controls cited by the supplied checklist; code includes `scripts/benchmark/eval_subtree_crossover_operator.py`. |
| 4G Fully independent testing | Retain No: held-out partitions do not prove exclusion from all upstream training sources. |
| 5A and 5B Hardware and compute | README, model card and run reports state hardware, parallelization and selected run times; no total study compute/carbon accounting is claimed. |

## Remaining release requirements

Public or reviewer-authorized access to exact study splits must be established
before marking those data accessible. Ten selected study checkpoints and their
vocabularies/configurations have been released with a pinned revision and
SHA-256 manifest; see `docs/data_and_model_access.md`.

Research environment files do not fully lock every optional dependency used by
every experiment. Exact reproduction requires the corresponding run's
environment export as well as data and weights. The bundled CPU demo has its
own complete minimal dependency pins.

The submitted checklist and manuscript must be updated by the authors to
reflect the released repository version. A model card now exists, but private
data availability, independent testing and cross-validation must retain their
qualified status until supported by additional evidence.
