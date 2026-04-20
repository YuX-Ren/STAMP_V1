"""Report per-split motif-OOV statistics for a PS-VAE STAMP vocab.

For every SMILES we run `PrincipalSubgraphTokenizer.decompose`, which produces a
list of atom-groups.  Each group has a canonical SMILES; if that SMILES is not in
the vocab, it is counted as OOV.  We report:

    - group-level OOV rate  = fraction of all decomposed groups that are OOV
    - mol-level OOV rate    = fraction of molecules with >=1 OOV group
    - mean_group_size       = avg atoms per group (lower => finer fragmentation)
    - atoms_per_mol         = mean heavy-atom count
    - num_groups_per_mol    = mean #groups per molecule
    - decode_failures       = molecules decompose() raised on (should be 0)

Splits evaluated: ZINC250k (single split), plus each MoleculeNet task's
train/valid/test triples via `load_moleculenet_task`.

Usage:
    python scripts/benchmark/report_moleculenet_oov.py \
        --vocab results/vocab/stamp_psvae_zinc250k_4096/vocab.txt \
        --zinc  data/zinc250k.csv \
        --tasks bbbp bace tox21 clintox sider esol freesolv lipophilicity \
        --output results/vocab/stamp_psvae_zinc250k_4096/oov_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger

from stamp.vocab.principal_subgraph import (
    MolInSubgraph,
    PrincipalSubgraphTokenizer,
    _pre_merge_bond_constraints,
)


def iter_zinc_smiles(path: Path, smiles_column: str = "smiles") -> List[str]:
    out: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = (row.get(smiles_column) or "").strip().strip('"').strip()
            if smi:
                out.append(smi)
    return out


def decompose_with_smis(
    mol: Chem.Mol,
    tokenizer: PrincipalSubgraphTokenizer,
    max_group_atoms: int = 0,
) -> List[Tuple[str, List[int]]]:
    """Run the tokenizer's greedy BPE decomposition AND capture the exact
    SMILES string PS-VAE uses for each final group.

    This mirrors `PrincipalSubgraphTokenizer.decompose()` but returns the
    internal `subgraphs_smis` map (the canonical MolFragmentToSmiles used
    during mining and vocab lookup), so downstream OOV checks query the
    vocab with the same string identity as the mining pass.

    If `max_group_atoms > 0`, any merge whose resulting motif has more than
    that many heavy atoms is skipped — forces finer fragmentation even
    with large vocabs.
    """
    work = MolInSubgraph(mol, kekulize=tokenizer.kekulize)
    _pre_merge_bond_constraints(
        work, mol,
        allow_ring_single_edges=tokenizer.allow_ring_single_edges,
        blessed_rings=getattr(tokenizer, "blessed_rings", None),
    )

    while True:
        nei_smis = work.get_nei_smis()
        best_smi: str | None = None
        best_freq = -1
        for s in nei_smis:
            if s not in tokenizer.vocab:
                continue
            atom_num, freq = tokenizer.vocab[s]
            if max_group_atoms > 0 and atom_num > max_group_atoms:
                continue
            if freq > best_freq:
                best_freq = freq
                best_smi = s
        if best_smi is None:
            break
        work.merge(best_smi)

    return work.get_smis_subgraphs()


def evaluate_split(
    smis: Iterable[str],
    tokenizer: PrincipalSubgraphTokenizer,
    name: str,
    max_group_atoms: int = 0,
) -> Dict[str, Any]:
    n_mols = 0
    n_valid = 0
    n_oov_mols = 0
    n_groups = 0
    n_oov_groups = 0
    total_atoms = 0
    total_groups = 0
    group_size_sum = 0
    decode_failures = 0

    for smi in smis:
        n_mols += 1
        try:
            mol = tokenizer.prepare_mol(smi)
        except Exception:
            decode_failures += 1
            continue

        try:
            groups = decompose_with_smis(mol, tokenizer, max_group_atoms=max_group_atoms)
        except Exception:
            decode_failures += 1
            continue

        n_valid += 1
        total_atoms += mol.GetNumHeavyAtoms()
        total_groups += len(groups)

        mol_has_oov = False
        for gsmi, atom_indices in groups:
            n_groups += 1
            group_size_sum += len(atom_indices)
            if gsmi not in tokenizer.vocab:
                n_oov_groups += 1
                mol_has_oov = True
        if mol_has_oov:
            n_oov_mols += 1

    valid_denom = max(1, n_valid)
    group_denom = max(1, n_groups)

    return {
        "split": name,
        "n_mols": n_mols,
        "n_valid": n_valid,
        "decode_failures": decode_failures,
        "mol_oov_rate": n_oov_mols / valid_denom,
        "group_oov_rate": n_oov_groups / group_denom,
        "mean_group_size": group_size_sum / group_denom,
        "mean_atoms_per_mol": total_atoms / valid_denom,
        "mean_groups_per_mol": total_groups / valid_denom,
        "n_oov_mols": n_oov_mols,
        "n_oov_groups": n_oov_groups,
    }


def load_moleculenet_splits(task: str) -> Dict[str, List[str]] | None:
    """Returns {"train":[...], "valid":[...], "test":[...]} or None on failure."""
    try:
        from stamp.benchmark.roberta_property import load_moleculenet_task
    except Exception as e:
        print(f"[warn] cannot import load_moleculenet_task: {e}", flush=True)
        return None

    try:
        data = load_moleculenet_task(task, data_dir=str(ROOT / "data"))
    except Exception as e:
        print(f"[warn] load_moleculenet_task({task}) failed: {e}", flush=True)
        return None

    out: Dict[str, List[str]] = {}
    for split in ("train", "valid", "test"):
        pairs: List[Tuple[str, float]] = data.get(split, [])
        out[split] = [smi for smi, _ in pairs]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--zinc", type=Path, default=ROOT / "data" / "zinc250k.csv")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["bbbp", "bace", "tox21", "clintox", "sider", "esol", "freesolv", "lipophilicity"],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zinc-limit", type=int, default=0, help="0 = full")
    parser.add_argument("--skip-zinc", action="store_true")
    parser.add_argument(
        "--max-group-atoms",
        type=int,
        default=0,
        help="0 = no cap; >0 forces BPE to skip merges producing motifs larger than this.",
    )
    parser.add_argument(
        "--force-ring-aware-encode",
        action="store_true",
        help="Override the vocab header and tokenize with ring-aware pre-merge (opens ring single bonds). Lets unseen rings be decomposed into sub-fragments instead of becoming one OOV group.",
    )
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")

    print(f"[load] loading tokenizer: {args.vocab}", flush=True)
    tokenizer = PrincipalSubgraphTokenizer(args.vocab)
    if args.force_ring_aware_encode:
        tokenizer.allow_ring_single_edges = True
        # Treat every vocab entry as a potential blessed ring:
        # pre_merge will keep a ring system intact iff its canonical SMILES
        # is already in the vocab (common rings) and open it (fragment) otherwise.
        tokenizer.blessed_rings = set(tokenizer.vocab.keys())
        print(
            f"[load] ring-aware encode ON; blessed set = full vocab ({len(tokenizer.blessed_rings)} entries)",
            flush=True,
        )
    print(f"[load] {len(tokenizer)} motifs, kekulize={tokenizer.kekulize}", flush=True)

    results: List[Dict[str, Any]] = []

    # ZINC250k itself.
    if (not args.skip_zinc) and args.zinc.exists():
        smis = iter_zinc_smiles(args.zinc)
        if args.zinc_limit > 0:
            smis = smis[: args.zinc_limit]
        print(f"[zinc] evaluating {len(smis)} SMILES from {args.zinc}", flush=True)
        results.append(evaluate_split(smis, tokenizer, max_group_atoms=args.max_group_atoms, name="zinc250k"))

    # MoleculeNet tasks.
    for task in args.tasks:
        splits = load_moleculenet_splits(task)
        if splits is None:
            results.append({"split": f"{task}/ERROR", "error": "load_failed"})
            continue
        for split_name, smis in splits.items():
            if not smis:
                continue
            name = f"{task}/{split_name}"
            print(f"[mnet] evaluating {name}: {len(smis)} SMILES", flush=True)
            try:
                results.append(evaluate_split(smis, tokenizer, max_group_atoms=args.max_group_atoms, name=name))
            except Exception:
                traceback.print_exc()
                results.append({"split": name, "error": "evaluate_failed"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vocab_path": str(args.vocab),
        "vocab_size": len(tokenizer),
        "kekulize": tokenizer.kekulize,
        "splits": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.output}", flush=True)

    print("\nsplit                       mols    mol_oov%  grp_oov%  g_size  g/mol")
    for r in results:
        if "error" in r:
            print(f"  {r['split']:<25}  ERROR ({r['error']})")
            continue
        print(
            f"  {r['split']:<25}  {r['n_valid']:>6}  "
            f"{r['mol_oov_rate']*100:>7.2f}  "
            f"{r['group_oov_rate']*100:>7.2f}  "
            f"{r['mean_group_size']:>5.2f}  "
            f"{r['mean_groups_per_mol']:>5.2f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
