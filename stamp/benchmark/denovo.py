from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from .common import canonicalize_smiles, iterable_with_progress
from .lm import load_trained_model, sample_model_sequences
from .metrics import evaluate_generated_smiles


def _is_probably_canonical_single_component(smiles_list: Sequence[str], sample_size: int = 2000) -> bool:
    if not smiles_list:
        return False
    n = min(sample_size, len(smiles_list))
    subset = smiles_list[:n]
    ok = 0
    total = 0
    for s in subset:
        c = canonicalize_smiles(s, require_single_component=True)
        if c is None:
            continue
        total += 1
        if c == s:
            ok += 1
    if total == 0:
        return False
    return (ok / total) >= 0.995


def run_denovo_sampling_eval(
    model_dir: str | Path,
    train_smiles: Sequence[str],
    reference_smiles: Sequence[str],
    output_dir: str | Path,
    n_samples: int = 10_000,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    diffusion_steps: int | None = None,
    diffusion_block_size: int | None = None,
    diffusion_remasking: str | None = None,
    diffusion_stochastic_transfer: bool | None = None,
    decode_batch_size: int = 256,
    reference_max_samples: int | None = None,
    assume_train_canonical: bool | None = None,
    eval_workers: int = 1,
    eval_chunksize: int = 256,
    show_progress: bool = True,
    device: str | None = None,
) -> dict[str, Any]:
    model, vocab, rep, dev = load_trained_model(model_dir, device=device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    batch_size = max(1, int(decode_batch_size))
    batches = []
    n_done = 0
    iterator = range(0, n_samples, batch_size)
    iterator = iterable_with_progress(
        iterator,
        show_progress=show_progress,
        total=(n_samples + batch_size - 1) // batch_size,
        desc="Sample de novo",
    )

    for _ in iterator:
        k = min(batch_size, n_samples - n_done)
        toks_batch = sample_model_sequences(
            model,
            vocab,
            n_samples=k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            prompt_tokens=None,
            device=dev,
            diffusion_steps=diffusion_steps,
            diffusion_block_size=diffusion_block_size,
            diffusion_remasking=diffusion_remasking,
            diffusion_stochastic_transfer=diffusion_stochastic_transfer,
        )
        batches.extend(toks_batch)
        n_done += k

    texts = [rep.detokenize(toks) for toks in batches]
    decoded_smiles = []
    for t in texts:
        try:
            s = rep.decode(t)
        except Exception:
            s = ""
        decoded_smiles.append(s)

    canonical_decoded = [
        canonicalize_smiles(s, require_single_component=True) if s else None
        for s in decoded_smiles
    ]
    assume_train = (
        _is_probably_canonical_single_component(train_smiles)
        if assume_train_canonical is None
        else bool(assume_train_canonical)
    )
    metrics = evaluate_generated_smiles(
        decoded_smiles,
        train_smiles=train_smiles,
        reference_smiles=reference_smiles,
        generated_canonical=canonical_decoded,
        assume_train_canonical=assume_train,
        reference_max_samples=reference_max_samples,
        eval_workers=eval_workers,
        eval_chunksize=eval_chunksize,
    )
    metrics.update(
        {
            "model_dir": str(model_dir),
            "representation": rep.name,
            "n_samples": n_samples,
            "temperature": temperature,
            "top_k": top_k,
            "diffusion_steps": diffusion_steps,
            "diffusion_block_size": diffusion_block_size,
            "diffusion_remasking": diffusion_remasking,
            "diffusion_stochastic_transfer": diffusion_stochastic_transfer,
            "decode_batch_size": batch_size,
            "device": str(dev),
            "assume_train_canonical": assume_train,
            "reference_max_samples": reference_max_samples,
            "eval_workers": eval_workers,
            "eval_chunksize": eval_chunksize,
        }
    )

    with (out / "denovo_samples.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["sample_id", "repr_text", "decoded_smiles", "valid"])
        for i, (text, smi, cano) in enumerate(zip(texts, decoded_smiles, canonical_decoded)):
            writer.writerow([i, text, smi, int(cano is not None)])

    (out / "denovo_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics
