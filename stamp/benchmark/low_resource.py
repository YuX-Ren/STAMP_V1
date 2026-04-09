from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from .denovo import run_denovo_sampling_eval
from .lm import LMConfig, train_language_model


def run_low_resource_experiments(
    split_dir: str | Path,
    output_dir: str | Path,
    representations: Sequence[str],
    sizes: Sequence[int] = (10_000, 50_000, 100_000),
    epochs: int = 5,
    batch_size: int = 64,
    max_seq_len: int = 256,
    n_samples: int = 2000,
    show_progress: bool = True,
    device: str | None = None,
    decode_batch_size: int = 256,
    encode_workers: int = 1,
    encode_chunksize: int = 128,
    eval_workers: int = 1,
    eval_chunksize: int = 256,
    num_data_workers: int = 0,
    prefetch_factor: int = 2,
    amp: str = "auto",
    torch_compile: bool = False,
    tf32: bool = True,
    model_type: str = "decoder",
    diffusion_steps: int = 128,
    diffusion_time_epsilon: float = 1e-3,
    diffusion_scheduler: str = "linear",
    diffusion_loss_weight: str = "scheduler",
    diffusion_remasking: str = "low_confidence",
    diffusion_stochastic_transfer: bool = False,
    diffusion_block_size: int = 0,
) -> dict[str, Any]:
    split_dir = Path(split_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    valid_smiles = [
        s.strip()
        for s in (split_dir / "valid.smiles").read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    test_smiles = [
        s.strip()
        for s in (split_dir / "test.smiles").read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]

    rows: list[dict[str, Any]] = []
    for rep_name in representations:
        for size in sizes:
            train_path = split_dir / "low_resource" / f"train_{size}.smiles"
            if not train_path.exists():
                continue
            train_smiles = [
                s.strip() for s in train_path.read_text(encoding="utf-8").splitlines() if s.strip()
            ]

            cfg = LMConfig(
                model_type=model_type,
                epochs=epochs,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_data_workers=num_data_workers,
                prefetch_factor=prefetch_factor,
                amp=amp,
                torch_compile=torch_compile,
                tf32=tf32,
                diffusion_steps=diffusion_steps,
                diffusion_time_epsilon=diffusion_time_epsilon,
                diffusion_scheduler=diffusion_scheduler,
                diffusion_loss_weight=diffusion_loss_weight,
                diffusion_remasking=diffusion_remasking,
                diffusion_stochastic_transfer=diffusion_stochastic_transfer,
                diffusion_block_size=diffusion_block_size,
            )
            model_dir = out / f"{rep_name}_train{size}"
            try:
                train_language_model(
                    train_smiles=train_smiles,
                    valid_smiles=valid_smiles,
                    representation=rep_name,
                    output_dir=model_dir,
                    cfg=cfg,
                    show_progress=show_progress,
                    device=device,
                    encode_workers=encode_workers,
                    encode_chunksize=encode_chunksize,
                )
                denovo_dir = model_dir / "denovo"
                metrics = run_denovo_sampling_eval(
                    model_dir=model_dir,
                    train_smiles=train_smiles,
                    reference_smiles=test_smiles,
                    output_dir=denovo_dir,
                    n_samples=n_samples,
                    show_progress=show_progress,
                    device=device,
                    decode_batch_size=decode_batch_size,
                    eval_workers=eval_workers,
                    eval_chunksize=eval_chunksize,
                )
                rows.append(
                    {
                        "representation": rep_name,
                        "train_size": size,
                        "validity": metrics.get("validity", float("nan")),
                        "novelty": metrics.get("novelty", float("nan")),
                        "internal_diversity": metrics.get("internal_diversity", float("nan")),
                        "quality": metrics.get("quality", float("nan")),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "representation": rep_name,
                        "train_size": size,
                        "validity": float("nan"),
                        "novelty": float("nan"),
                        "internal_diversity": float("nan"),
                        "quality": float("nan"),
                        "status": f"error: {exc}",
                    }
                )

    with (out / "low_resource_scaling.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "representation",
                "train_size",
                "validity",
                "novelty",
                "internal_diversity",
                "quality",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {"split_dir": str(split_dir), "output_dir": str(out), "rows": rows}
    (out / "low_resource_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
