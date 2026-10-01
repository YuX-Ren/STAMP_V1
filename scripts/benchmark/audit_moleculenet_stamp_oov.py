#!/usr/bin/env python3
"""Audit strict STAMP encoding coverage on MoleculeNet test splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.lm import TokenVocab  # noqa: E402
from stamp.benchmark.molformer_stamp import smiles_char_tokens, smiles_vocab_tokens_from_stamp_vocabs  # noqa: E402
from stamp.benchmark.multiscale_stamp import parse_csv_arg  # noqa: E402
from stamp.benchmark.representations import STAMPRepresentation  # noqa: E402
from stamp.benchmark.roberta_property import load_moleculenet_task  # noqa: E402
from scripts.benchmark.finetune_molformer_stamp_pp import load_task_rows  # noqa: E402
from scripts.benchmark.finetune_molformer_stamp_pp import TASKS as LOCAL_TASKS  # noqa: E402
from scripts.benchmark.finetune_molformer_stamp_pp import _find_task_file, _float_or_nan, _open_csv  # noqa: E402


MULTITASK_SETS = {"tox21", "clintox", "sider", "muv"}


def _read_vocab_tokens(path: str | Path) -> set[str]:
    toks: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if i == 0:
                continue
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                toks.add(parts[0])
    return toks


def _load_token_vocab_tokens(path: str | Path | None, scale_vocab_paths: list[str]) -> tuple[set[str], str]:
    if path is not None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "vocab_itos" in payload:
            return {str(x) for x in payload["vocab_itos"]}, f"scan_cache:{path}"
        if "summary" in payload and "vocab_itos" in payload["summary"]:
            return {str(x) for x in payload["summary"]["vocab_itos"]}, f"scan_cache:{path}"
        raise ValueError(f"Cannot find vocab_itos in {path}")

    toks = {TokenVocab.PAD, TokenVocab.BOS, TokenVocab.EOS, TokenVocab.UNK, "[BOS]", "[EOS]", "[CONCAT]"}
    for vp in scale_vocab_paths:
        toks.update(_read_vocab_tokens(vp))
    return toks, "scale_vocab_union_plus_dynamic_structural"


def _is_allowed_token(tok: str, token_vocab: set[str]) -> bool:
    return tok in token_vocab or TokenVocab._is_stamp_structural(tok)


def _rows_for_test(task: str, task_data: dict[str, Any]) -> list[tuple[str, Any]]:
    task_key = task.lower()
    if task_key in MULTITASK_SETS and task_data.get("test_multi"):
        return list(task_data["test_multi"])
    return list(task_data["test"])


def _scaffold_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def _load_task_rows_scaffold_fast(task: str, data_dir: Path, seed: int) -> dict[str, Any]:
    if task not in LOCAL_TASKS:
        raise ValueError(f"Unknown task for local CSV scaffold loader: {task}")
    info = dict(LOCAL_TASKS[task])
    path = _find_task_file(data_dir, info["file"])
    rows: list[tuple[str, list[float], str]] = []
    with _open_csv(path) as f:
        reader = csv.DictReader(f)
        targets = info["targets"]
        if targets is None:
            targets = [c for c in (reader.fieldnames or []) if c != info["smiles"]]
            info["targets"] = targets
        for row in reader:
            smi = (row.get(info["smiles"]) or "").strip()
            from stamp.benchmark.common import canonicalize_smiles

            can = canonicalize_smiles(smi, strip_salts=True)
            if can is None:
                continue
            y = [_float_or_nan(row.get(t)) for t in targets]
            if info["type"] == "classification":
                y = [v if v in (0.0, 1.0) else float("nan") for v in y]
                if not any(torch.isfinite(torch.tensor(v)).item() for v in y):
                    continue
            elif not torch.isfinite(torch.tensor(y[0])).item():
                continue
            split = str(row.get(info.get("split", ""), "")).strip().lower() if info.get("split") else ""
            rows.append((can, y, split))

    if any(s in {"train", "valid", "validation", "test"} for _, _, s in rows):
        train = [(s, y) for s, y, sp in rows if sp == "train"]
        valid = [(s, y) for s, y, sp in rows if sp in {"valid", "validation"}]
        test = [(s, y) for s, y, sp in rows if sp == "test"]
        if train and valid and test:
            return {"task": task, "task_type": info["type"], "targets": list(info["targets"]), "train": train, "valid": valid, "test": test, "source": "local_csv_predefined_split"}

    buckets: dict[str, list[tuple[str, list[float]]]] = {}
    for smi, y, _ in rows:
        buckets.setdefault(_scaffold_key(smi), []).append((smi, y))
    # Deterministic DeepChem-style scaffold split: largest scaffold buckets first.
    groups = sorted(buckets.values(), key=lambda g: (-len(g), g[0][0]))
    n = sum(len(g) for g in groups)
    train_cut = int(0.8 * n)
    valid_cut = int(0.9 * n)
    train: list[tuple[str, list[float]]] = []
    valid: list[tuple[str, list[float]]] = []
    test: list[tuple[str, list[float]]] = []
    for group in groups:
        if len(train) + len(group) <= train_cut:
            train.extend(group)
        elif len(train) + len(valid) + len(group) <= valid_cut:
            valid.extend(group)
        else:
            test.extend(group)
    if not valid and test:
        valid, test = test[:1], test[1:]
    if not test and valid:
        test, valid = valid[:1], valid[1:]
    return {"task": task, "task_type": info["type"], "targets": list(info["targets"]), "train": train, "valid": valid, "test": test, "source": "local_csv_murcko_scaffold_fast"}


def audit_task(
    *,
    task: str,
    data_dir: str,
    split_seed: int,
    reps: list[STAMPRepresentation],
    scale_names: list[str],
    token_vocab: set[str],
    max_rows: int,
    max_examples: int,
    loader: str,
    fallback_oov_to_smiles: bool,
) -> dict[str, Any]:
    if loader == "local_csv_fast":
        task_data = load_task_rows(task, Path(data_dir), split_seed)
        rows = list(task_data["test"])
    elif loader == "local_scaffold_fast":
        task_data = _load_task_rows_scaffold_fast(task, Path(data_dir), split_seed)
        rows = list(task_data["test"])
    elif loader == "deepchem":
        task_data = load_moleculenet_task(task, data_dir=data_dir, split_seed=split_seed)
        rows = _rows_for_test(task, task_data)
    else:
        raise ValueError(f"Unsupported loader: {loader}")
    if max_rows > 0:
        rows = rows[:max_rows]

    scale_fail_counts = Counter()
    fail_reasons = Counter()
    oov_counts = Counter()
    examples: list[dict[str, Any]] = []
    ok = 0
    total_tokens = 0
    total_oov = 0
    max_concat_len = 0

    for idx, row in enumerate(rows):
        smi = str(row[0])
        scale_texts: list[str] = []
        failed = False
        for scale_name, rep in zip(scale_names, reps):
            try:
                scale_texts.append(rep.encode(smi))
            except Exception as exc:
                failed = True
                scale_fail_counts[scale_name] += 1
                reason = f"{type(exc).__name__}: {str(exc)[:240]}"
                fail_reasons[reason] += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "kind": "encode_fail",
                            "row": idx,
                            "smiles": smi,
                            "scale": scale_name,
                            "reason": reason,
                        }
                    )
                break
        if failed:
            continue

        ok += 1
        concat_tokens: list[str] = []
        for i, text in enumerate(scale_texts):
            concat_tokens.extend(str(text).split())
            if i != len(scale_texts) - 1:
                concat_tokens.append("[CONCAT]")
        max_concat_len = max(max_concat_len, len(concat_tokens))
        total_tokens += len(concat_tokens)
        row_oov: list[str] = []
        for tok in concat_tokens:
            if _is_allowed_token(tok, token_vocab):
                continue
            if fallback_oov_to_smiles:
                try:
                    if all(_is_allowed_token(t, token_vocab) for t in smiles_char_tokens(tok)):
                        continue
                except ValueError:
                    pass
            row_oov.append(tok)
        if row_oov:
            total_oov += len(row_oov)
            oov_counts.update(row_oov)
            if len(examples) < max_examples:
                examples.append(
                    {
                        "kind": "token_oov",
                        "row": idx,
                        "smiles": smi,
                        "oov_tokens": sorted(set(row_oov))[:20],
                        "oov_occurrences": len(row_oov),
                    }
                )

    n = len(rows)
    failed = n - ok
    return {
        "task": task,
        "source": task_data.get("source", "unknown"),
        "loader": loader,
        "test_rows": n,
        "encode_ok": ok,
        "encode_failed": failed,
        "encode_fail_rate": failed / max(1, n),
        "scale_fail_counts": dict(scale_fail_counts),
        "top_fail_reasons": [{"reason": k, "count": v} for k, v in fail_reasons.most_common(10)],
        "token_oov_occurrences": total_oov,
        "token_oov_unique": len(oov_counts),
        "token_oov_rate": total_oov / max(1, total_tokens),
        "top_oov_tokens": [{"token": k, "count": v} for k, v in oov_counts.most_common(20)],
        "max_concat_tokens_no_bos_eos": max_concat_len,
        "examples": examples,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Audit strict STAMP coverage on MoleculeNet test splits.")
    p.add_argument("--data-dir", default="/scratch/yuxuan.ren/data/moleculenet")
    p.add_argument(
        "--tasks",
        default="bbbp,hiv,bace,clintox,tox21,sider,esol,freesolv,lipophilicity",
        help="Comma-separated MoleculeNet task names.",
    )
    p.add_argument(
        "--scale-vocabs",
        default=(
            "/Project_Storage/CFP-03/CFP03-CF-130/data/zinc_pretrain/vocab_merged/motif_vocab_top1k.txt,"
            "/Project_Storage/CFP-03/CFP03-CF-130/data/zinc_pretrain/vocab_merged/motif_vocab_top4k.txt,"
            "/Project_Storage/CFP-03/CFP03-CF-130/data/zinc_pretrain/vocab_merged/motif_vocab_top12k.txt"
        ),
    )
    p.add_argument("--scale-names", default="top1k,top4k,top12k")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument(
        "--loader",
        choices=["local_csv_fast", "local_scaffold_fast", "deepchem"],
        default="local_scaffold_fast",
        help="local_scaffold_fast uses RDKit Murcko scaffold split without DeepChem/TensorFlow startup.",
    )
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--max-examples", type=int, default=20)
    p.add_argument("--force-ring-aware-encode", action="store_true", help="Override vocab header and emit [Ring] closures.")
    p.add_argument(
        "--no-strict-vocab",
        action="store_false",
        dest="strict_vocab",
        help="Allow irreducible fragments outside the motif vocab instead of counting them as encode failures.",
    )
    p.add_argument(
        "--fallback-oov-to-smiles",
        action="store_true",
        help="Count a non-vocab motif as covered when all of its regex SMILES fallback tokens are in the model/token vocab.",
    )
    p.add_argument("--token-vocab-cache", default="", help="Optional _scan_cache_cams_decoder_v*.pt for exact model vocab OOV check.")
    p.add_argument("--output-json", default="")
    p.add_argument("--output-csv", default="")
    args = p.parse_args()

    RDLogger.DisableLog("rdApp.*")
    tasks = parse_csv_arg(args.tasks)
    scale_vocab_paths = parse_csv_arg(args.scale_vocabs)
    scale_names = parse_csv_arg(args.scale_names)
    if len(scale_vocab_paths) != len(scale_names):
        raise ValueError("--scale-vocabs and --scale-names must have the same length.")

    reps = [
        STAMPRepresentation(
            vocab_path=vp,
            force_ring_aware_encode=bool(args.force_ring_aware_encode),
            strict_vocab=bool(args.strict_vocab),
        )
        for vp in scale_vocab_paths
    ]
    token_vocab_cache = args.token_vocab_cache.strip() or None
    token_vocab, token_vocab_source = _load_token_vocab_tokens(token_vocab_cache, scale_vocab_paths)
    if args.fallback_oov_to_smiles:
        token_vocab.update(smiles_vocab_tokens_from_stamp_vocabs(scale_vocab_paths))
        token_vocab_source += "+smiles_fallback_tokens"

    results = []
    for task in tasks:
        try:
            result = audit_task(
                task=task,
                data_dir=args.data_dir,
                split_seed=int(args.split_seed),
                reps=reps,
                scale_names=scale_names,
                token_vocab=token_vocab,
                max_rows=int(args.max_rows),
                max_examples=int(args.max_examples),
                loader=str(args.loader),
                fallback_oov_to_smiles=bool(args.fallback_oov_to_smiles),
            )
        except Exception as exc:
            result = {
                "task": task,
                "error": f"{type(exc).__name__}: {exc}",
                "test_rows": 0,
                "encode_ok": 0,
                "encode_failed": 0,
                "encode_fail_rate": float("nan"),
                "token_oov_occurrences": 0,
                "token_oov_unique": 0,
                "token_oov_rate": float("nan"),
            }
        results.append(result)
        if "error" in result:
            print(f"{task}: ERROR {result['error']}", flush=True)
        else:
            print(
                f"{task}: rows={result['test_rows']} ok={result['encode_ok']} "
                f"fail={result['encode_failed']} ({result['encode_fail_rate']:.2%}) "
                f"token_oov={result['token_oov_occurrences']} unique={result['token_oov_unique']}",
                flush=True,
            )

    payload = {
        "data_dir": args.data_dir,
        "tasks": tasks,
        "scale_names": scale_names,
        "scale_vocabs": scale_vocab_paths,
        "force_ring_aware_encode": bool(args.force_ring_aware_encode),
        "strict_vocab": bool(args.strict_vocab),
        "fallback_oov_to_smiles": bool(args.fallback_oov_to_smiles),
        "loader": str(args.loader),
        "token_vocab_source": token_vocab_source,
        "results": results,
    }

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output_csv).open("w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(
                fout,
                fieldnames=[
                    "task",
                    "source",
                    "loader",
                    "test_rows",
                    "encode_ok",
                    "encode_failed",
                    "encode_fail_rate",
                    "token_oov_occurrences",
                    "token_oov_unique",
                    "token_oov_rate",
                    "max_concat_tokens_no_bos_eos",
                    "error",
                ],
            )
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
