"""PS-VAE vs legacy STAMP roundtrip benchmark on zinc250k.

Usage:
    python scripts/benchmark/psvae_roundtrip_test.py \
        --data data/zinc250k.csv \
        --mine-n 2000 --test-n 10000 --vocab-size 500 \
        --workers 8 --vocab-out results/psvae/zinc250k_vocab.txt
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

from rdkit import Chem, RDLogger

from stamp.core.stamp_codec import STAMPEncoder
from stamp.vocab.principal_subgraph import (
    PrincipalSubgraphTokenizer,
    PSVAEMineConfig,
    mine_principal_subgraphs,
)


def iter_smiles_csv(path: Path, limit: int | None = None) -> List[str]:
    out: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = (row.get("smiles") or "").strip().strip('"').strip()
            if not smi:
                continue
            out.append(smi)
            if limit is not None and len(out) >= limit:
                break
    return out


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def run_roundtrip(
    label: str,
    encoder: STAMPEncoder,
    smis: List[str],
    show_failures: int = 5,
) -> Tuple[int, int, List[Tuple[str, str, str]]]:
    ok = 0
    failures: List[Tuple[str, str, str]] = []
    errors: List[Tuple[str, str]] = []
    t0 = time.time()
    for smi in smis:
        try:
            stamp_str = encoder.encode_smiles(smi)
            decoded = encoder.decode_stamp(stamp_str)
            c1 = canonicalize(smi)
            c2 = canonicalize(decoded)
            if c1 is not None and c1 == c2:
                ok += 1
            else:
                failures.append((smi, stamp_str, decoded))
        except Exception as e:
            errors.append((smi, f"{type(e).__name__}: {e}"))
    dt = time.time() - t0
    total = len(smis)
    print(
        f"[{label}] {ok}/{total} pass  ({ok/total*100:.2f}%)  "
        f"failures={len(failures)} errors={len(errors)}  time={dt:.1f}s"
    )
    if show_failures > 0:
        for smi, stamp_str, decoded in failures[:show_failures]:
            print(f"   FAIL input: {smi}")
            print(f"        stamp: {stamp_str}")
            print(f"        dec  : {decoded}")
        for smi, err in errors[:show_failures]:
            print(f"   ERROR input: {smi}")
            print(f"          msg : {err}")
    return ok, total, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mine-n", type=int, default=2000)
    parser.add_argument("--test-n", type=int, default=10000)
    parser.add_argument("--vocab-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--vocab-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-mine", action="store_true")
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")

    print(f"Loading SMILES from {args.data} ...")
    all_smis = iter_smiles_csv(args.data, limit=max(args.mine_n, args.test_n) * 2)
    print(f"  loaded {len(all_smis)} valid rows")

    rng = random.Random(args.seed)
    rng.shuffle(all_smis)
    mine_smis = all_smis[: args.mine_n]
    test_smis = all_smis[args.mine_n : args.mine_n + args.test_n]
    print(f"  mining set: {len(mine_smis)}, test set: {len(test_smis)}")

    args.vocab_out.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_mine and args.vocab_out.exists():
        print(f"Skipping mine, reusing vocab at {args.vocab_out}")
    else:
        print(f"Mining PS-VAE vocab -> {args.vocab_out}")
        cfg = PSVAEMineConfig(
            vocab_size=args.vocab_size,
            kekulize=True,
            num_workers=args.workers,
            show_progress=True,
        )
        t0 = time.time()
        selected, _ = mine_principal_subgraphs(mine_smis, cfg, args.vocab_out)
        print(f"  mined {len(selected)} entries in {time.time()-t0:.1f}s")

    tokenizer = PrincipalSubgraphTokenizer(args.vocab_out)
    print(f"Loaded tokenizer: {len(tokenizer)} motifs, kekulize={tokenizer.kekulize}")

    enc_legacy = STAMPEncoder()
    enc_psvae = STAMPEncoder(motif_tokenizer=tokenizer)

    print("\n=== Legacy STAMP encoder (exocyclic_single) ===")
    run_roundtrip("legacy", enc_legacy, test_smis)

    print("\n=== PS-VAE enabled STAMP encoder ===")
    run_roundtrip("psvae ", enc_psvae, test_smis)

    return 0


if __name__ == "__main__":
    sys.exit(main())
