from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence

from .common import canonicalize_smiles, iterable_with_progress


def iter_smiles_from_file(
    input_path: str | Path,
    smiles_column: str = "smiles",
    max_mols: Optional[int] = None,
) -> Iterator[str]:
    path = Path(input_path)
    count = 0

    if path.suffix.lower() in {".smi", ".smiles", ".txt"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                yield s
                count += 1
                if max_mols is not None and count >= max_mols:
                    return
        return

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = (row.get(smiles_column) or "").strip()
            if not s:
                continue
            yield s
            count += 1
            if max_mols is not None and count >= max_mols:
                return


def load_smiles_list(
    input_path: str | Path,
    smiles_column: str = "smiles",
    max_mols: Optional[int] = None,
    canonicalize: bool = True,
    deduplicate: bool = True,
    show_progress: bool = False,
) -> List[str]:
    rows = iter_smiles_from_file(input_path, smiles_column=smiles_column, max_mols=max_mols)
    out: List[str] = []
    seen = set()

    for s in iterable_with_progress(rows, show_progress=show_progress, desc="Load smiles"):
        ss = canonicalize_smiles(s) if canonicalize else s
        if ss is None:
            continue
        if deduplicate:
            if ss in seen:
                continue
            seen.add(ss)
        out.append(ss)
    return out


def write_smiles_file(smiles_list: Sequence[str], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in smiles_list:
            f.write(s + "\n")


def prepare_dataset_splits(
    input_path: str | Path,
    output_dir: str | Path,
    smiles_column: str = "smiles",
    train_size: int = 100_000,
    valid_size: int = 10_000,
    test_size: int = 10_000,
    low_resource_sizes: Sequence[int] = (10_000, 50_000, 100_000),
    seed: int = 42,
    max_mols: Optional[int] = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    smiles = load_smiles_list(
        input_path,
        smiles_column=smiles_column,
        max_mols=max_mols,
        canonicalize=True,
        deduplicate=True,
        show_progress=show_progress,
    )

    required = train_size + valid_size + test_size
    if len(smiles) < required:
        raise ValueError(
            f"Not enough molecules after cleaning: got {len(smiles)}, need at least {required}."
        )

    rnd = random.Random(seed)
    rnd.shuffle(smiles)

    train = smiles[:train_size]
    valid = smiles[train_size : train_size + valid_size]
    test = smiles[train_size + valid_size : train_size + valid_size + test_size]

    out_dir = Path(output_dir)
    split_dir = out_dir / "splits"
    low_dir = split_dir / "low_resource"
    split_dir.mkdir(parents=True, exist_ok=True)
    low_dir.mkdir(parents=True, exist_ok=True)

    write_smiles_file(train, split_dir / "train.smiles")
    write_smiles_file(valid, split_dir / "valid.smiles")
    write_smiles_file(test, split_dir / "test.smiles")

    low_resource_summary: dict[str, int] = {}
    for size in low_resource_sizes:
        k = min(size, len(train))
        write_smiles_file(train[:k], low_dir / f"train_{size}.smiles")
        low_resource_summary[str(size)] = k

    summary = {
        "input": str(input_path),
        "output_dir": str(out_dir),
        "seed": seed,
        "total_clean_unique": len(smiles),
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
        "low_resource": low_resource_summary,
        "files": {
            "train": str(split_dir / "train.smiles"),
            "valid": str(split_dir / "valid.smiles"),
            "test": str(split_dir / "test.smiles"),
        },
    }
    (split_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
