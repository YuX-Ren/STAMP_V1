# STAMP GPU Training Environment

The dedicated training environment file is:

- `envs/stamp-train-rdkit-2025.09.4.yml`

The critical version pin is:

- `rdkit=2025.09.4`

STAMP tokenization depends on RDKit canonical SMILES:

- motif tokens are RDKit canonical fragment SMILES
- canonical atom ordering changes can change `[J_i]` indices
- that changes exported STAMP sequences and therefore the learned token vocabulary

Because of that, keep the RDKit version fixed when comparing vocabularies or re-running training.

The GPU training runner in this repo now uses lightweight helper scripts that only import the STAMP diffusion path. That avoids unrelated optional benchmark dependencies such as `deepchem` when the goal is only vocabulary building and STAMP diffusion training.

## Create the environment

```bash
conda env create -f envs/stamp-train-rdkit-2025.09.4.yml
conda activate stamp-train-rdkit-2025_09_4
```

## Verify the pinned RDKit version

```bash
python -c "from rdkit import rdBase; print(rdBase.rdkitVersion)"
```

Expected output:

```text
2025.09.4
```

## Run all STAMP diffusion jobs from an allocated GPU shell

```bash
bash scripts/hpc/train_stamp_diffusion_datasets.sh
```

## Slurm workflow from `unites1`

Interactive debug shell:

```bash
srun -p debug --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=04:00:00 --pty bash
```

Inside that shell:

```bash
conda activate /playpen-shared/mufan/.miniforge3/envs/stamp-train-rdkit-2025_09_4
bash scripts/hpc/train_stamp_diffusion_datasets.sh
```

Batch submission for training:

```bash
sbatch scripts/hpc/train_stamp_diffusion_datasets.slurm
```

If you want a specific GPU family, override the partition:

```bash
sbatch -p a100 scripts/hpc/train_stamp_diffusion_datasets.slurm
sbatch -p ada scripts/hpc/train_stamp_diffusion_datasets.slurm
sbatch -p blackwell scripts/hpc/train_stamp_diffusion_datasets.slurm
```

The cluster guide says:

- `unites1` is the login/debug node
- regular training jobs should use Slurm, not direct SSH to compute nodes
- `debug` has higher priority but only 4 hours max
- a single user can run at most 8 GPUs at once
- conda environments should live on `/playpen-shared/`

The runner does the following per dataset:

- builds a STAMP motif vocabulary artifact first
- reuses existing train/valid files when available
- prepares train/valid/test splits for `zinc250k` and `qm9`
- trains a STAMP diffusion LM after the vocab step finishes

Dataset mapping:

- `moses`: `data/moses/train.smiles` and `data/moses/test.smiles`
- `guacamol`: `data/guacamol/train.smiles` and `data/guacamol/valid.smiles`
- `zinc250k`: prepared from `data/zinc250k.csv`
- `qm9`: prepared from `data/qm9/qm9_dataset.csv`
