#!/usr/bin/env python3
"""Run a CamS decoder MoleculeNet fine-tuning grid in one Python process.

This avoids the expensive shell-loop pattern of reloading the same checkpoint
and re-encoding the same split for every hyperparameter candidate.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark.finetune_cams_decoder_pp import (  # noqa: E402
    KPGT_TASK_ALIASES,
    _load_decoder_checkpoint_bundle,
    _load_kpgt_split_task,
    _prepare_task_data,
    _train_one_task,
)
from stamp.benchmark.multiscale_stamp import (  # noqa: E402
    MultiScaleSTAMPEncoder,
    parse_csv_arg,
    resolve_multiscale_vocab_paths,
)
from stamp.benchmark.representations import build_representation  # noqa: E402
from stamp.benchmark.roberta_property import load_moleculenet_task  # noqa: E402


TASKS_INFO = {
    "bbbp": "classification",
    "bace": "classification",
    "hiv": "classification",
    "tox21": "classification",
    "clintox": "classification",
    "sider": "classification",
    "muv": "classification",
    "toxcast": "classification",
    "estrogen": "classification",
    "metstab": "classification",
    "esol": "regression",
    "freesolv": "regression",
    "lipophilicity": "regression",
    "lipo": "regression",
}


def _strategy_values(strategy: str, args: argparse.Namespace) -> list[object]:
    if strategy == "normal":
        return [None]
    if strategy == "llrd":
        return list(args.llrd_decays)
    if strategy == "l2sp":
        return list(args.l2sp_weights)
    if strategy == "flag":
        return list(args.flag_step_sizes)
    if strategy == "reinit":
        return list(args.reinit_layers)
    raise ValueError(f"Unsupported strategy={strategy}")


def _iter_configs(args: argparse.Namespace) -> Iterator[dict]:
    for lr in args.lrs:
        for dropout in args.dropouts:
            for weight_decay in args.weight_decays:
                for fp_mode in args.fp_modes:
                    for batch_size in args.batch_sizes:
                        for strategy in args.strategies:
                            for strategy_value in _strategy_values(strategy, args):
                                cfg = {
                                    "lr": float(lr),
                                    "dropout": float(dropout),
                                    "weight_decay": float(weight_decay),
                                    "fp_mode": str(fp_mode),
                                    "batch_size": int(batch_size),
                                    "finetune_strategy": str(strategy),
                                    "llrd_decay_coef": float(args.llrd_decays[0]),
                                    "l2sp_weight": float(args.l2sp_weights[-1]),
                                    "flag_m": int(args.flag_m),
                                    "flag_step_size": float(args.flag_step_sizes[0]),
                                    "reinit_num_layers": int(args.reinit_layers[0]),
                                }
                                if strategy == "llrd":
                                    cfg["llrd_decay_coef"] = float(strategy_value)
                                elif strategy == "l2sp":
                                    cfg["l2sp_weight"] = float(strategy_value)
                                elif strategy == "flag":
                                    cfg["flag_step_size"] = float(strategy_value)
                                elif strategy == "reinit":
                                    cfg["reinit_num_layers"] = int(strategy_value)
                                yield cfg


def _tag(split: str, cfg: dict) -> str:
    strategy = cfg["finetune_strategy"]
    if strategy == "llrd":
        sv = cfg["llrd_decay_coef"]
    elif strategy == "l2sp":
        sv = cfg["l2sp_weight"]
    elif strategy == "flag":
        sv = cfg["flag_step_size"]
    elif strategy == "reinit":
        sv = cfg["reinit_num_layers"]
    else:
        sv = "none"
    text = (
        f"split-{split}_lr-{cfg['lr']:.0e}_do-{cfg['dropout']}_wd-{cfg['weight_decay']:.0e}"
        f"_fp-{cfg['fp_mode']}_bs-{cfg['batch_size']}_strategy-{strategy}_sv-{sv}"
    )
    return text.replace(".", "p").replace("-", "_")


def _reset_seeds(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_representation_for_pp(args: argparse.Namespace, bundle: dict):
    override_paths = parse_csv_arg(args.multiscale_vocabs)
    override_names = parse_csv_arg(args.multiscale_scale_names)
    if override_paths:
        if override_names and len(override_names) != len(override_paths):
            raise ValueError("--multiscale-scale-names must match --multiscale-vocabs length.")
        if not override_names:
            override_names = [f"s{i}" for i in range(len(override_paths))]
        rep = MultiScaleSTAMPEncoder(
            override_paths,
            scale_names=override_names,
            force_ring_aware_encode=bool(args.force_ring_aware_encode),
            strict_vocab=False,
        )
        return rep, f"multiscale_override[{','.join(override_names)}]"

    ckpt = bundle["ckpt"]
    model_dir = bundle["model_dir"]
    ms_vocab_files = list(ckpt.get("multiscale_vocab_files", []) or [])
    ms_scale_names = list(ckpt.get("multiscale_scale_names", []) or [])
    if ms_vocab_files:
        if not ms_scale_names:
            ms_scale_names = [f"s{i}" for i in range(len(ms_vocab_files))]
        rep = MultiScaleSTAMPEncoder(
            resolve_multiscale_vocab_paths(model_dir, ms_vocab_files),
            scale_names=ms_scale_names,
            force_ring_aware_encode=bool(args.force_ring_aware_encode),
            strict_vocab=False,
        )
        return rep, f"multiscale_ckpt[{','.join(ms_scale_names)}]"

    stamp_vocab_file = ckpt.get("stamp_vocab")
    stamp_vocab_path = None
    if stamp_vocab_file is not None:
        candidate = model_dir / str(stamp_vocab_file)
        if candidate.exists():
            stamp_vocab_path = str(candidate)
    rep = build_representation(str(ckpt.get("representation", "stamp")), stamp_vocab=stamp_vocab_path)
    if bool(args.force_ring_aware_encode) and hasattr(rep, "codec") and hasattr(rep.codec, "motif_tokenizer") and rep.codec.motif_tokenizer is not None:
        tok = rep.codec.motif_tokenizer
        tok.allow_ring_single_edges = True
        tok.blessed_rings = set(tok.vocab.keys())
    return rep, "single_scale"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--checkpoint-file", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", default=["bbbp"])
    p.add_argument("--splits", nargs="+", default=["scaffold-0", "scaffold-1", "scaffold-2"])
    p.add_argument("--split-source", choices=["deepchem", "kpgt"], default="kpgt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--fp-bits", type=int, default=2048)
    p.add_argument("--lrs", nargs="+", type=float, default=[1e-6, 3e-6, 1e-5, 3e-5])
    p.add_argument("--dropouts", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2])
    p.add_argument("--weight-decays", nargs="+", type=float, default=[0.0, 1e-6, 1e-4])
    p.add_argument("--fp-modes", nargs="+", default=["fp_token"])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[32])
    p.add_argument("--strategies", nargs="+", choices=["normal", "l2sp", "llrd", "flag", "reinit"], default=["normal"])
    p.add_argument("--llrd-decays", nargs="+", type=float, default=[0.95, 0.9, 0.85, 0.8])
    p.add_argument("--l2sp-weights", nargs="+", type=float, default=[1e-1, 1e-2, 1e-3, 1e-4])
    p.add_argument("--flag-step-sizes", nargs="+", type=float, default=[1e-3, 3e-3, 5e-3, 1e-2])
    p.add_argument("--flag-m", type=int, default=3)
    p.add_argument("--reinit-layers", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--lr-scheduler", choices=["constant", "polynomial"], default="polynomial")
    p.add_argument("--end-lr", type=float, default=1e-9)
    p.add_argument("--drop-last-train", action="store_true")
    p.add_argument("--grad-clip-norm", type=float, default=5.0)
    p.add_argument("--loss-normalization", choices=["valid_mean", "kpgt_mean"], default="kpgt_mean")
    p.add_argument("--multiscale-vocabs", default="")
    p.add_argument("--multiscale-scale-names", default="")
    p.add_argument("--force-ring-aware-encode", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    RDLogger.DisableLog("rdApp.*")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    bundle = _load_decoder_checkpoint_bundle(args.checkpoint_dir, args.checkpoint_file)
    vocab = bundle["vocab"]
    rep, mode_desc = _build_representation_for_pp(args, bundle)
    max_seq_len = int(args.max_seq_len or bundle["cfg"].max_seq_len)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configs = list(_iter_configs(args))
    print(
        f"[setup] d_model={bundle['cfg'].d_model} max_seq_len={max_seq_len} "
        f"input_mode={mode_desc} configs={len(configs)} tasks={args.tasks} splits={args.splits}",
        flush=True,
    )

    for task in args.tasks:
        if task not in TASKS_INFO:
            print(f"[skip] unknown task {task}", flush=True)
            continue
        for split in args.splits:
            t_split = time.time()
            try:
                if args.split_source == "kpgt":
                    d = _load_kpgt_split_task(task, args.data_dir, split)
                else:
                    d = load_moleculenet_task(task, data_dir=str(args.data_dir))
            except Exception as exc:
                print(f"[skip] task={task} split={split} load failed: {exc}", flush=True)
                continue
            data = _prepare_task_data(d, task, TASKS_INFO, rep, vocab, max_seq_len, args.fp_bits)
            print(
                f"[data] task={task} split={split} type={'classification' if data['is_class'] else 'regression'} "
                f"multi={data['is_multi']} n_tasks={data['n_tasks']} train={len(data['ids_tr'])} "
                f"valid={len(data['ids_va'])} test={len(data['ids_te'])} encode_stats={data['encode_stats']}",
                flush=True,
            )
            for idx, cfg in enumerate(configs, start=1):
                tag = _tag(split, cfg)
                out_json = args.output_dir / f"results_{task}_{tag}.json"
                if out_json.exists():
                    print(f"[skip] existing {out_json}", flush=True)
                    continue
                print(f"[run] task={task} split={split} config={idx}/{len(configs)} {cfg}", flush=True)
                _reset_seeds(args.seed, device)
                t0 = time.time()
                result = _train_one_task(
                    task,
                    data,
                    args.checkpoint_dir,
                    args.checkpoint_file,
                    device,
                    epochs=args.epochs,
                    lr=cfg["lr"],
                    weight_decay=cfg["weight_decay"],
                    batch_size=cfg["batch_size"],
                    dropout=cfg["dropout"],
                    patience=args.patience,
                    fp_bits=args.fp_bits,
                    fp_mode=cfg["fp_mode"],
                    warmup_ratio=args.warmup_ratio,
                    warmup_steps=args.warmup_steps,
                    lr_scheduler=args.lr_scheduler,
                    end_lr=args.end_lr,
                    drop_last_train=bool(args.drop_last_train),
                    grad_clip_norm=args.grad_clip_norm,
                    loss_normalization=args.loss_normalization,
                    finetune_strategy=cfg["finetune_strategy"],
                    llrd_decay_coef=cfg["llrd_decay_coef"],
                    l2sp_weight=cfg["l2sp_weight"],
                    flag_m=cfg["flag_m"],
                    flag_step_size=cfg["flag_step_size"],
                    reinit_num_layers=cfg["reinit_num_layers"],
                    checkpoint_bundle=bundle,
                )
                result["elapsed_sec"] = time.time() - t0
                result["split_source"] = args.split_source
                result["split_name"] = split
                payload = {
                    "checkpoint": str(args.checkpoint_dir),
                    "checkpoint_file": str(args.checkpoint_file) if args.checkpoint_file is not None else None,
                    "data_dir": str(args.data_dir),
                    "split_source": args.split_source,
                    "split_name": split,
                    "hyperparams": {
                        "epochs": args.epochs,
                        "lr": cfg["lr"],
                        "weight_decay": cfg["weight_decay"],
                        "batch_size": cfg["batch_size"],
                        "dropout": cfg["dropout"],
                        "patience": args.patience,
                        "fp_bits": args.fp_bits,
                        "fp_mode": cfg["fp_mode"],
                        "warmup_ratio": args.warmup_ratio,
                        "warmup_steps": args.warmup_steps,
                        "lr_scheduler": args.lr_scheduler,
                        "end_lr": args.end_lr,
                        "drop_last_train": args.drop_last_train,
                        "grad_clip_norm": args.grad_clip_norm,
                        "loss_normalization": args.loss_normalization,
                        "finetune_strategy": cfg["finetune_strategy"],
                        "llrd_decay_coef": cfg["llrd_decay_coef"],
                        "l2sp_weight": cfg["l2sp_weight"],
                        "flag_m": cfg["flag_m"],
                        "flag_step_size": cfg["flag_step_size"],
                        "reinit_num_layers": cfg["reinit_num_layers"],
                    },
                    "tasks": {task: result},
                }
                out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"[done] {out_json}", flush=True)
            print(f"[split_done] task={task} split={split} elapsed_sec={time.time() - t_split:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
