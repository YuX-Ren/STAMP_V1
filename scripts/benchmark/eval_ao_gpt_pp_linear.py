"""Linear-probe evaluation of an AO-GPT-PP checkpoint on MoleculeNet.

Freezes the pretrained TinyDecoderLM, extracts mean-pooled last-layer hidden
states as molecule features, then fits sklearn LogisticRegression (classification)
or Ridge (regression) per task. Reports ROC-AUC / RMSE against the official
scaffold splits.

Usage:
    python scripts/benchmark/eval_ao_gpt_pp_linear.py \
        --checkpoint-dir results/ao_gpt_pp/cm31k_ring_small_v1 \
        --tasks bbbp bace tox21 clintox sider esol freesolv lipophilicity \
        --output results/ao_gpt_pp/cm31k_ring_small_v1/linear_probe.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from rdkit import RDLogger

from stamp.benchmark.lm import LMConfig, TokenVocab, load_trained_model
from stamp.benchmark.representations import build_representation
from stamp.benchmark.roberta_property import load_moleculenet_task


def _extract_features(
    smiles_list: Sequence[str],
    model,
    rep,
    vocab: TokenVocab,
    device: torch.device,
    batch_size: int = 64,
    max_seq_len: int = 128,
) -> Tuple[np.ndarray, List[int]]:
    """Mean-pool last-block hidden states over valid (non-pad) positions.

    For decoder-style AO checkpoints we mirror the training input convention:
    the model sees ``[BOS] + tokens`` and predicts the shifted targets, so the
    terminal ``EOS`` must NOT be included in the feature-extraction input.

    Returns (features [N, d_model], keep_indices list) — indices that
    tokenize-encode-forward successfully; unparseable SMILES are skipped.
    """
    captured: List[torch.Tensor] = []

    def hook(_module, _input, output):
        captured.append(output.detach())

    handle = model.blocks[-1].register_forward_hook(hook)

    feats: List[np.ndarray] = []
    keep: List[int] = []

    ids_per_mol: List[List[int]] = []
    for i, smi in enumerate(smiles_list):
        try:
            text = rep.encode(smi)
            toks = rep.tokenize(text)
            ids = [vocab.bos_id] + vocab.encode_tokens(toks)
            if len(ids) > max_seq_len:
                ids = ids[:max_seq_len]
            if len(ids) < 2:
                continue
            ids_per_mol.append(ids)
            keep.append(i)
        except Exception:
            continue

    model.eval()
    with torch.no_grad():
        for start in range(0, len(ids_per_mol), batch_size):
            batch = ids_per_mol[start : start + batch_size]
            lengths = [len(b) for b in batch]
            m = max(lengths)
            pad = vocab.pad_id
            padded = np.full((len(batch), m), pad, dtype=np.int64)
            for i, b in enumerate(batch):
                padded[i, : len(b)] = b
            x = torch.from_numpy(padded).to(device)
            captured.clear()
            _ = model(x, force_causal=True)
            # hook captured x shape [bsz, seqlen+1, d_model] (prepended None emb)
            feat = captured[-1][:, 1:, :]  # drop None prefix
            # Mean-pool over non-pad positions.
            mask = (x != pad).float().unsqueeze(-1)
            pooled = (feat * mask).sum(1) / mask.sum(1).clamp(min=1)
            feats.append(pooled.float().cpu().numpy())

    handle.remove()
    if not feats:
        return np.empty((0, model.cfg.d_model), dtype=np.float32), keep
    return np.concatenate(feats, axis=0), keep


def _fit_classification(X_train, y_train, X_test, y_test) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X_test)[:, 1]
    else:
        p = clf.decision_function(X_test)
    try:
        auc = roc_auc_score(y_test, p)
    except Exception:
        auc = float("nan")
    return {"roc_auc": float(auc)}


def _fit_classification_multitask(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_eval: np.ndarray,
    Y_eval: np.ndarray,
) -> Dict[str, float]:
    """Fit one LogisticRegression per column (sub-task), return mean AUC over
    columns that have both positive and negative labels in train + eval.
    NaN labels indicate missing and are masked per-subtask. Matches the
    MoleculeNet / MolCLR multi-task evaluation protocol.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    aucs: List[float] = []
    per_task: List[float] = []
    n_tasks = Y_train.shape[1]
    for j in range(n_tasks):
        y_tr = Y_train[:, j]
        y_ev = Y_eval[:, j]
        mask_tr = np.isfinite(y_tr) & ((y_tr == 0.0) | (y_tr == 1.0))
        mask_ev = np.isfinite(y_ev) & ((y_ev == 0.0) | (y_ev == 1.0))
        if mask_tr.sum() < 10 or mask_ev.sum() < 5:
            per_task.append(float("nan"))
            continue
        X_tr = X_train[mask_tr]
        y_tr2 = y_tr[mask_tr]
        X_ev = X_eval[mask_ev]
        y_ev2 = y_ev[mask_ev]
        if len(np.unique(y_tr2)) < 2 or len(np.unique(y_ev2)) < 2:
            per_task.append(float("nan"))
            continue
        try:
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            clf.fit(X_tr, y_tr2)
            p = clf.predict_proba(X_ev)[:, 1]
            auc = float(roc_auc_score(y_ev2, p))
        except Exception:
            auc = float("nan")
        per_task.append(auc)
        if not np.isnan(auc):
            aucs.append(auc)
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    return {
        "roc_auc_mean": mean_auc,
        "roc_auc_per_task": per_task,
        "n_tasks_scored": len(aucs),
        "n_tasks_total": n_tasks,
    }


def _fit_regression(X_train, y_train, X_test, y_test, y_std: float = 1.0) -> Dict[str, float]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    reg = Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    rmse_norm = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    # DeepChem z-scores regression labels; report the raw-scale RMSE that
    # MoleculeNet papers compare against.
    rmse_raw = rmse_norm * float(y_std)
    return {"rmse": rmse_raw, "rmse_normalized": rmse_norm}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=["bbbp", "bace", "tox21", "clintox", "sider", "esol", "freesolv", "lipophilicity"],
    )
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    RDLogger.DisableLog("rdApp.*")
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    print(f"[load] model from {args.checkpoint_dir}", flush=True)
    model, vocab, rep, _ = load_trained_model(str(args.checkpoint_dir), device=str(device))
    cfg = model.cfg
    # Force ring-aware encode to match pretraining (checkpoint's saved
    # vocab.txt header may not have the flag set).
    if hasattr(rep, "codec") and hasattr(rep.codec, "motif_tokenizer") and rep.codec.motif_tokenizer is not None:
        tok = rep.codec.motif_tokenizer
        tok.allow_ring_single_edges = True
        tok.blessed_rings = set(tok.vocab.keys())
        rep._force_ring_aware_encode = True
    print(f"[load] model_type={cfg.model_type}, d_model={cfg.d_model}, vocab={len(vocab.itos)}", flush=True)

    tasks_info = {
        "bbbp": "classification",
        "bace": "classification",
        "tox21": "classification",
        "clintox": "classification",
        "sider": "classification",
        "hiv": "classification",
        "muv": "classification",
        "esol": "regression",
        "freesolv": "regression",
        "lipophilicity": "regression",
    }

    # Multi-task benchmarks where the MoleculeNet / MolCLR protocol reports
    # the mean ROC-AUC across ALL sub-tasks (not just one column).
    multitask_datasets = {"tox21", "clintox", "sider", "muv"}

    results: Dict[str, Dict[str, Any]] = {}
    for task in args.tasks:
        if task not in tasks_info:
            print(f"[skip] unknown task {task}", flush=True)
            continue
        t0 = time.time()
        try:
            d = load_moleculenet_task(task, data_dir=str(ROOT / "data"))
        except Exception as e:
            print(f"[skip] {task}: load failed ({e})", flush=True)
            continue

        is_multitask = tasks_info[task] == "classification" and task in multitask_datasets and "train_multi" in d

        def _split_features(rows_key: str):
            rows = d[rows_key]
            smis = [item[0] for item in rows]
            feat, keep = _extract_features(
                smis, model, rep, vocab, device,
                batch_size=args.batch_size, max_seq_len=cfg.max_seq_len,
            )
            return feat, keep, rows

        if is_multitask:
            X_train, k_tr, rows_tr = _split_features("train_multi")
            X_valid, k_va, rows_va = _split_features("valid_multi")
            X_test, k_te, rows_te = _split_features("test_multi")
            # Stack per-subtask labels into [N, n_tasks] with NaN missing.
            n_tasks = max(len(r[1]) for r in rows_tr) if rows_tr else 1
            def _stack(rows, keep):
                if not keep:
                    return np.empty((0, n_tasks), dtype=float)
                arr = np.full((len(keep), n_tasks), np.nan, dtype=float)
                for out_i, in_i in enumerate(keep):
                    y_list = rows[in_i][1]
                    for j, v in enumerate(y_list[:n_tasks]):
                        arr[out_i, j] = float(v)
                return arr
            Y_train = _stack(rows_tr, k_tr)
            Y_valid = _stack(rows_va, k_va)
            Y_test = _stack(rows_te, k_te)

            valid_m = _fit_classification_multitask(X_train, Y_train, X_valid, Y_valid) if len(k_va) else {}
            test_m = _fit_classification_multitask(X_train, Y_train, X_test, Y_test) if len(k_te) else {}
            n_tr, n_va, n_te = len(rows_tr), len(rows_va), len(rows_te)
            metric_name, val_m, tst_m = "roc_auc_mean", valid_m, test_m
        else:
            # Single-task classification or regression.
            def _split_xy(rows_key: str):
                rows = d[rows_key]
                smis = [smi for smi, _ in rows]
                ys = np.array([y for _, y in rows], dtype=float)
                feat, keep = _extract_features(
                    smis, model, rep, vocab, device,
                    batch_size=args.batch_size, max_seq_len=cfg.max_seq_len,
                )
                ys_kept = ys[keep] if keep else np.array([])
                return feat, ys_kept, len(rows), len(keep)

            X_train, y_train, n_tr, k_tr = _split_xy("train")
            X_valid, y_valid, n_va, k_va = _split_xy("valid")
            X_test, y_test, n_te, k_te = _split_xy("test")

            if tasks_info[task] == "classification":
                valid_m = _fit_classification(X_train, y_train, X_valid, y_valid) if k_va else {}
                test_m = _fit_classification(X_train, y_train, X_test, y_test) if k_te else {}
            else:
                y_std = float(d.get("y_std", 1.0))
                valid_m = _fit_regression(X_train, y_train, X_valid, y_valid, y_std=y_std) if k_va else {}
                test_m = _fit_regression(X_train, y_train, X_test, y_test, y_std=y_std) if k_te else {}
            metric_name = "roc_auc" if tasks_info[task] == "classification" else "rmse"
            k_tr, k_va, k_te = k_tr, k_va, k_te  # noqa
            val_m, tst_m = valid_m, test_m

        elapsed = time.time() - t0
        results[task] = {
            "task_type": tasks_info[task],
            "multitask": is_multitask,
            "n_train": n_tr, "n_train_feat": k_tr if isinstance(k_tr, int) else len(k_tr),
            "n_valid": n_va, "n_valid_feat": k_va if isinstance(k_va, int) else len(k_va),
            "n_test": n_te, "n_test_feat": k_te if isinstance(k_te, int) else len(k_te),
            "valid": valid_m,
            "test": test_m,
            "elapsed_sec": elapsed,
        }
        print(
            f"[{task}] {tasks_info[task]}{'(multi)' if is_multitask else ''}  "
            f"valid {metric_name}={val_m.get(metric_name, float('nan')):.4f}  "
            f"test {metric_name}={tst_m.get(metric_name, float('nan')):.4f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint_dir),
        "model_type": cfg.model_type,
        "d_model": cfg.d_model,
        "vocab_size": len(vocab.itos),
        "tasks": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
