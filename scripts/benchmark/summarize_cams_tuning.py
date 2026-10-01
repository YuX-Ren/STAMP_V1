#!/usr/bin/env python3
"""Summarize per-task MoleculeNet tuning runs for CamS decoder fine-tuning."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _score_key(result: dict[str, Any]) -> float:
    value = float(result["best_valid"])
    if result.get("is_class", True):
        return value
    return -value


def _load_records(root: Path, task: str | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("results_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        tasks = payload.get("tasks", {})
        for task_name, result in tasks.items():
            if task is not None and task_name != task:
                continue
            hp = dict(payload.get("hyperparams", {}))
            rec = {
                "path": str(path),
                "task": task_name,
                "split": path.stem.split("_")[-1] if "split_name" not in result else result["split_name"],
                "best_valid": result.get("best_valid"),
                "best_test": result.get("best_test"),
                "best_epoch": result.get("best_epoch"),
                "is_class": bool(result.get("is_class", True)),
                "is_multi": bool(result.get("is_multi", False)),
                "n_tasks": int(result.get("n_tasks", 1)),
                "hyperparams": hp,
                "result": result,
            }
            if rec["best_valid"] is None or rec["best_test"] is None:
                continue
            if not math.isfinite(float(rec["best_valid"])) or not math.isfinite(float(rec["best_test"])):
                continue
            records.append(rec)
    return records


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--task", default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    records = _load_records(args.root, args.task)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        groups.setdefault((rec["task"], rec["split"]), []).append(rec)

    selected: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        best = max(group, key=_score_key)
        best["num_candidates"] = len(group)
        selected.append(best)

    by_task: dict[str, list[dict[str, Any]]] = {}
    for rec in selected:
        by_task.setdefault(rec["task"], []).append(rec)

    summary: dict[str, Any] = {"root": str(args.root), "num_records": len(records), "selected": selected, "tasks": {}}
    for task_name, rows in sorted(by_task.items()):
        vals = np.array([float(r["best_test"]) for r in rows], dtype=np.float64)
        summary["tasks"][task_name] = {
            "n_splits": len(rows),
            "mean_test": float(vals.mean()) if len(vals) else float("nan"),
            "std_test": float(vals.std(ddof=0)) if len(vals) else float("nan"),
            "splits": rows,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] {args.output}")
    for task_name, item in summary["tasks"].items():
        print(f"{task_name}: {item['mean_test']:.6f} +/- {item['std_test']:.6f} ({item['n_splits']} splits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
