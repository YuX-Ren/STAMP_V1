#!/usr/bin/env python3
"""One-shot C4+ZINC valid_loss evaluation for a MolT5 mixed-pretrain checkpoint.

Loads the model/tokenizer from --ckpt-dir, reconstructs the same C4 + ZINC valid
loaders used during training (via `train_molt5_c4_zinc_mixed._make_loaders`),
runs `_evaluate`, and appends the result to `<output-dir>/eval_history.json`.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark.train_molt5_c4_zinc_mixed import (  # noqa: E402
    _evaluate,
    _make_loaders,
)


def _build_args(train_args_json: Path, overrides: dict) -> argparse.Namespace:
    with train_args_json.open("r", encoding="utf-8") as f:
        base = json.load(f)
    merged = dict(base)
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return argparse.Namespace(**merged)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True,
                    help="Path to a saved step_XXXXXXXX/ directory containing "
                         "model.safetensors + tokenizer.json.")
    ap.add_argument("--output-dir", required=True,
                    help="Run dir where eval_history.json lives; the new entry is appended.")
    ap.add_argument("--zinc-valid-parquet", default=None,
                    help="Override zinc valid glob; defaults to train_args.json value. "
                         "Point at a single known-good shard to avoid corrupt files.")
    ap.add_argument("--c4-valid-glob", default=None)
    ap.add_argument("--eval-max-batches", type=int, default=None,
                    help="Number of eval micro-batches per source; default from train_args.json (100).")
    ap.add_argument("--zinc-valid-max-rows", type=int, default=None)
    ap.add_argument("--c4-valid-max-rows", type=int, default=None)
    ap.add_argument("--num-data-workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", default=None,
                    help="Override amp mode (bf16/fp16/none); default from train_args.json.")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir).resolve()
    if not (ckpt_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"{ckpt_dir} missing model.safetensors")

    out_dir = Path(args.output_dir).resolve()
    train_args_json = out_dir / "train_args.json"
    if not train_args_json.exists():
        raise FileNotFoundError(f"{train_args_json} not found; needed to reproduce eval loaders.")

    overrides = {
        "zinc_valid_parquet": args.zinc_valid_parquet,
        "c4_valid_glob": args.c4_valid_glob,
        "eval_max_batches": args.eval_max_batches,
        "zinc_valid_max_rows": args.zinc_valid_max_rows,
        "c4_valid_max_rows": args.c4_valid_max_rows,
        "num_data_workers": args.num_data_workers,
        "seed": args.seed,
        "amp": args.amp,
    }
    run_args = _build_args(train_args_json, overrides)

    device = torch.device(args.device)

    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(str(ckpt_dir), local_files_only=True)
    model.to(device).eval()

    c4_valid_loader, zinc_valid_loader, c4_valid_ds, zinc_valid_ds = _make_loaders(
        run_args, tokenizer, train=False
    )

    step = 0
    latest_txt = out_dir / "checkpoints" / "latest_step.txt"
    if latest_txt.exists():
        stem = latest_txt.read_text(encoding="utf-8").strip()
        if stem.startswith("step_"):
            try:
                step = int(stem[len("step_"):])
            except ValueError:
                step = 0
    if step == 0 and ckpt_dir.name.startswith("step_"):
        try:
            step = int(ckpt_dir.name[len("step_"):])
        except ValueError:
            step = 0

    with torch.inference_mode():
        metrics = _evaluate(
            model=model,
            c4_loader=c4_valid_loader,
            zinc_loader=zinc_valid_loader,
            c4_dataset=c4_valid_ds,
            zinc_dataset=zinc_valid_ds,
            tokenizer=tokenizer,
            args=run_args,
            device=device,
            step=step,
        )

    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

    history_path = out_dir / "eval_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history = []
    metrics_out = copy.deepcopy(metrics)
    metrics_out["source"] = "one_shot_eval_molt5_ckpt"
    metrics_out["ckpt_dir"] = str(ckpt_dir)
    history.append(metrics_out)
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval-only] appended step={step} to {history_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
