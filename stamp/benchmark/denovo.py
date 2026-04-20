from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Optional, Sequence

from rdkit import Chem

from .common import canonicalize_smiles, iterable_with_progress
from .lm import _normalize_model_type, load_trained_model, sample_model_sequences
from .metrics import evaluate_generated_smiles
from .mcts_unified.edit_policy import AOEditPolicy
from .mcts_unified.subtree import edit_candidate_spans, parse_terminal_sequence

_REPEATED_B1_RE = re.compile(r"\[B_1\](?: \[B_1\])+")
_REPEATED_J_RE = re.compile(r"\[J_\d+\](?: \[J_\d+\]){2,}")


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


def _normalized_span_logprob(
    edit_policy: AOEditPolicy,
    tokens: Sequence[str],
    span_start: int,
    span_end: int,
) -> float:
    span_len = max(1, int(span_end) - int(span_start))
    return float(edit_policy.score_span(tokens, span_start, span_end)) / float(span_len)


def _ao_sequence_confidence(
    edit_policy: AOEditPolicy,
    parsed: Any,
    *,
    max_span_len: int | None,
    top_spans: int,
) -> float:
    spans = edit_candidate_spans(
        parsed,
        include_root=False,
        max_span_len=(None if max_span_len is None else max(1, int(max_span_len))),
    )
    if not spans:
        return 0.0
    scores = [
        _normalized_span_logprob(edit_policy, parsed.flat_tokens, span.token_start, span.token_end)
        for span in spans
    ]
    scores.sort()
    k = min(max(1, int(top_spans)), len(scores))
    return float(sum(scores[:k]) / float(k))


def _ao_refine_stamp_tokens(
    tokens: Sequence[str],
    *,
    codec: Any,
    edit_policy: AOEditPolicy,
    steps: int,
    top_spans: int,
    max_span_len: int | None,
    samples_per_span: int,
    temperature: float | None,
    top_k: int | None,
    min_gain: float = 0.0,
) -> tuple[list[str], dict[str, Any]]:
    original_tokens = list(tokens)
    current_tokens = list(tokens)
    changed = False
    stats: dict[str, Any] = {
        "changed": 0,
        "accepted_edits": 0,
        "steps_attempted": 0,
        "candidate_evals": 0,
        "invalid_candidates": 0,
        "parse_failures": 0,
        "score_gain_total": 0.0,
    }
    n_steps = max(0, int(steps))
    n_top_spans = max(1, int(top_spans))
    n_samples = max(1, int(samples_per_span))

    for _ in range(n_steps):
        parsed = parse_terminal_sequence(current_tokens, codec)
        if parsed is None:
            stats["parse_failures"] += 1
            break
        spans = edit_candidate_spans(
            parsed,
            include_root=False,
            max_span_len=(None if max_span_len is None else max(1, int(max_span_len))),
        )
        if not spans:
            break

        current_flat = list(parsed.flat_tokens)
        current_confidence = _ao_sequence_confidence(
            edit_policy,
            parsed,
            max_span_len=max_span_len,
            top_spans=n_top_spans,
        )
        span_scores = [
            (
                _normalized_span_logprob(edit_policy, current_flat, span.token_start, span.token_end),
                span,
            )
            for span in spans
        ]
        span_scores.sort(key=lambda item: item[0])
        best_gain = float("-inf")
        best_tokens: list[str] | None = None
        stats["steps_attempted"] += 1

        for base_score, span in span_scores[:n_top_spans]:
            for _ in range(n_samples):
                stats["candidate_evals"] += 1
                candidate_tokens = edit_policy.resample_span(
                    current_flat,
                    span.token_start,
                    span.token_end,
                    temperature=temperature,
                    top_k=top_k,
                )
                if list(candidate_tokens) == current_flat:
                    continue
                candidate_parsed = parse_terminal_sequence(candidate_tokens, codec)
                if candidate_parsed is None:
                    stats["invalid_candidates"] += 1
                    continue
                candidate_confidence = _ao_sequence_confidence(
                    edit_policy,
                    candidate_parsed,
                    max_span_len=max_span_len,
                    top_spans=n_top_spans,
                )
                gain = float(candidate_confidence - current_confidence)
                if gain > best_gain:
                    best_gain = gain
                    best_tokens = list(candidate_parsed.flat_tokens)

        if best_tokens is None or best_gain <= float(min_gain):
            break
        current_tokens = best_tokens
        changed = True
        stats["accepted_edits"] += 1
        stats["score_gain_total"] += float(best_gain)

    stats["changed"] = int(changed)
    stats["mean_score_gain"] = (
        float(stats["score_gain_total"]) / max(1, int(stats["accepted_edits"]))
        if int(stats["accepted_edits"]) > 0
        else 0.0
    )
    return (current_tokens if changed else original_tokens), stats


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
    diffusion_tokens_per_step: int | None = None,
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
    sample_lengths: Optional[Sequence[int]] = None,
    ao_decode_mode: str = "ar",
    ao_canvas_placeholder: str = "unk",
    ao_refine_steps: int = 0,
    ao_refine_top_spans: int = 1,
    ao_refine_max_span_len: int | None = 12,
    ao_refine_samples_per_span: int = 4,
    ao_refine_temperature: float | None = None,
    ao_refine_top_k: int | None = None,
    ao_refine_min_gain: float = 0.0,
) -> dict[str, Any]:
    model, vocab, rep, dev = load_trained_model(model_dir, device=device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ao_refine_enabled = int(ao_refine_steps) > 0
    ao_refine_stats: dict[str, Any] = {
        "enabled": bool(ao_refine_enabled),
        "steps": int(max(0, int(ao_refine_steps))),
        "top_spans": int(max(1, int(ao_refine_top_spans))),
        "max_span_len": (None if ao_refine_max_span_len is None else int(max(1, int(ao_refine_max_span_len)))),
        "samples_per_span": int(max(1, int(ao_refine_samples_per_span))),
        "temperature": (float(ao_refine_temperature) if ao_refine_temperature is not None else None),
        "top_k": (int(ao_refine_top_k) if ao_refine_top_k is not None else None),
        "min_gain": float(ao_refine_min_gain),
        "changed_samples": 0,
        "accepted_edits": 0,
        "candidate_evals": 0,
        "invalid_candidates": 0,
        "parse_failures": 0,
        "score_gain_total": 0.0,
    }
    ao_refine_per_sample: list[dict[str, Any]] = []
    ao_refine_policy: AOEditPolicy | None = None
    if ao_refine_enabled:
        model_type = _normalize_model_type(getattr(model.cfg, "model_type", "decoder"))
        if rep.name != "stamp" or not hasattr(rep, "codec"):
            raise ValueError("AO refine currently only supports STAMP representation.")
        if model_type not in {"ao_gpt", "ao_gpt_pp"}:
            raise ValueError(
                f"AO refine requires ao_gpt|ao_gpt_pp checkpoint, got model_type={model_type}."
            )
        ao_refine_policy = AOEditPolicy(
            model,
            vocab,
            device=dev,
            temperature=(
                float(ao_refine_temperature) if ao_refine_temperature is not None else float(temperature)
            ),
            top_k=(int(ao_refine_top_k) if ao_refine_top_k is not None else int(top_k)),
        )

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

    sampled_length_hist: dict[str, int] = {}
    for _ in iterator:
        k = min(batch_size, n_samples - n_done)
        if sample_lengths is None:
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
                diffusion_tokens_per_step=diffusion_tokens_per_step,
                diffusion_block_size=diffusion_block_size,
                diffusion_remasking=diffusion_remasking,
                diffusion_stochastic_transfer=diffusion_stochastic_transfer,
                ao_decode_mode=ao_decode_mode,
                ao_canvas_placeholder=ao_canvas_placeholder,
            )
        else:
            batch_lengths = [int(x) for x in sample_lengths[n_done : n_done + k]]
            toks_batch = []
            bucket_counts: dict[int, int] = {}
            for length in batch_lengths:
                bucket_counts[length] = bucket_counts.get(length, 0) + 1
            for length, count in sorted(bucket_counts.items()):
                sampled_length_hist[str(length)] = sampled_length_hist.get(str(length), 0) + int(count)
                toks_batch.extend(
                    sample_model_sequences(
                        model,
                        vocab,
                        n_samples=count,
                        max_new_tokens=max(1, int(length)),
                        temperature=temperature,
                        top_k=top_k,
                        prompt_tokens=None,
                        device=dev,
                        diffusion_steps=diffusion_steps,
                        diffusion_tokens_per_step=diffusion_tokens_per_step,
                        diffusion_block_size=diffusion_block_size,
                        diffusion_remasking=diffusion_remasking,
                        diffusion_stochastic_transfer=diffusion_stochastic_transfer,
                        ao_decode_mode=ao_decode_mode,
                        ao_canvas_placeholder=ao_canvas_placeholder,
                    )
                )
        if ao_refine_policy is not None:
            refined_batch = []
            for toks in toks_batch:
                refined_toks, refine_stats = _ao_refine_stamp_tokens(
                    toks,
                    codec=rep.codec,
                    edit_policy=ao_refine_policy,
                    steps=ao_refine_steps,
                    top_spans=ao_refine_top_spans,
                    max_span_len=ao_refine_max_span_len,
                    samples_per_span=ao_refine_samples_per_span,
                    temperature=ao_refine_temperature,
                    top_k=ao_refine_top_k,
                    min_gain=ao_refine_min_gain,
                )
                refined_batch.append(refined_toks)
                ao_refine_per_sample.append(refine_stats)
                ao_refine_stats["changed_samples"] += int(refine_stats["changed"])
                ao_refine_stats["accepted_edits"] += int(refine_stats["accepted_edits"])
                ao_refine_stats["candidate_evals"] += int(refine_stats["candidate_evals"])
                ao_refine_stats["invalid_candidates"] += int(refine_stats["invalid_candidates"])
                ao_refine_stats["parse_failures"] += int(refine_stats["parse_failures"])
                ao_refine_stats["score_gain_total"] += float(refine_stats["score_gain_total"])
            toks_batch = refined_batch
        batches.extend(toks_batch)
        n_done += k

    texts = [rep.detokenize(toks) for toks in batches]
    decoded_smiles = []
    strict_valid_flags: list[int] = []
    fallback_free_flags: list[int] = []
    fallback_totals: list[int] = []
    repeated_b1_flags: list[int] = []
    repeated_joint_flags: list[int] = []
    fallback_aggregate: dict[str, int] = {}
    for t in texts:
        repeated_b1_flags.append(int(bool(_REPEATED_B1_RE.search(t))))
        repeated_joint_flags.append(int(bool(_REPEATED_J_RE.search(t))))
        try:
            if rep.name == "stamp" and hasattr(rep, "codec"):
                mol, report = rep.codec.decode_stamp_to_mol(t, return_report=True)
                if len(Chem.GetMolFrags(mol)) != 1:
                    raise ValueError("Decoded STAMP is disconnected (multiple fragments).")
                s = rep.decode(t)
                report_dict = report.to_dict()
                fallback_totals.append(int(report_dict.get("total_fallbacks", 0)))
                fallback_free_flags.append(int(int(report_dict.get("total_fallbacks", 0)) == 0))
                for key, value in report_dict.items():
                    fallback_aggregate[key] = fallback_aggregate.get(key, 0) + int(value)
                try:
                    strict_mol = rep.codec.decode_stamp_to_mol(t, strict=True)
                    strict_valid_flags.append(int(len(Chem.GetMolFrags(strict_mol)) == 1))
                except Exception:
                    strict_valid_flags.append(0)
            else:
                s = rep.decode(t)
        except Exception:
            s = ""
            if rep.name == "stamp" and hasattr(rep, "codec"):
                fallback_totals.append(0)
                fallback_free_flags.append(0)
                strict_valid_flags.append(0)
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
            "ao_decode_mode": str(ao_decode_mode),
            "ao_canvas_placeholder": str(ao_canvas_placeholder),
            "diffusion_steps": diffusion_steps,
            "diffusion_tokens_per_step": diffusion_tokens_per_step,
            "diffusion_block_size": diffusion_block_size,
            "diffusion_remasking": diffusion_remasking,
            "diffusion_stochastic_transfer": diffusion_stochastic_transfer,
            "decode_batch_size": batch_size,
            "device": str(dev),
            "assume_train_canonical": assume_train,
            "reference_max_samples": reference_max_samples,
            "eval_workers": eval_workers,
            "eval_chunksize": eval_chunksize,
            "sample_length_mode": ("empirical_distribution" if sample_lengths is not None else "fixed_max_new_tokens"),
            "sampled_length_histogram": sampled_length_hist,
            "ao_refine": {
                **ao_refine_stats,
                "changed_rate": (
                    float(ao_refine_stats["changed_samples"]) / max(1, n_samples)
                    if ao_refine_enabled
                    else 0.0
                ),
                "mean_score_gain": (
                    float(ao_refine_stats["score_gain_total"]) / max(1, int(ao_refine_stats["accepted_edits"]))
                    if int(ao_refine_stats["accepted_edits"]) > 0
                    else 0.0
                ),
            },
        }
    )
    if rep.name == "stamp":
        metrics.update(
            {
                "strict_valid_count": int(sum(strict_valid_flags)),
                "strict_validity": (float(sum(strict_valid_flags)) / max(1, len(strict_valid_flags)) if strict_valid_flags else float("nan")),
                "fallback_free_count": int(sum(fallback_free_flags)),
                "fallback_free_rate": (float(sum(fallback_free_flags)) / max(1, len(fallback_free_flags)) if fallback_free_flags else float("nan")),
                "mean_decode_fallbacks": (float(sum(fallback_totals)) / max(1, len(fallback_totals)) if fallback_totals else float("nan")),
                "samples_with_repeated_B1_run": int(sum(repeated_b1_flags)),
                "repeated_B1_run_rate": float(sum(repeated_b1_flags)) / max(1, len(repeated_b1_flags)),
                "samples_with_repeated_joint_run": int(sum(repeated_joint_flags)),
                "repeated_joint_run_rate": float(sum(repeated_joint_flags)) / max(1, len(repeated_joint_flags)),
                "decode_fallback_totals": fallback_aggregate,
            }
        )

    with (out / "denovo_samples.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        header = ["sample_id", "repr_text", "decoded_smiles", "valid"]
        if ao_refine_enabled:
            header.extend(["ao_refined", "ao_refine_edits", "ao_refine_score_gain"])
        if rep.name == "stamp":
            header.extend(["strict_valid", "fallback_free", "decode_fallbacks", "repeated_B1_run", "repeated_joint_run"])
        writer.writerow(header)
        for i, (text, smi, cano) in enumerate(zip(texts, decoded_smiles, canonical_decoded)):
            row = [i, text, smi, int(cano is not None)]
            if ao_refine_enabled:
                refine_stats = ao_refine_per_sample[i]
                row.extend(
                    [
                        int(refine_stats["changed"]),
                        int(refine_stats["accepted_edits"]),
                        float(refine_stats["score_gain_total"]),
                    ]
                )
            if rep.name == "stamp":
                row.extend(
                    [
                        int(strict_valid_flags[i]),
                        int(fallback_free_flags[i]),
                        int(fallback_totals[i]),
                        int(repeated_b1_flags[i]),
                        int(repeated_joint_flags[i]),
                    ]
                )
            writer.writerow(row)

    (out / "denovo_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics
