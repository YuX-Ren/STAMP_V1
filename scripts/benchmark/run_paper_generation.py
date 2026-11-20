#!/usr/bin/env python3
"""Run a published generation task with downloaded STAMP checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TASKS = {
    "linker": ("fragment_infill", 100, 1.0, 4, 64),
    "motif": ("fragment_infill", 100, 1.0, 4, 64),
    "scaffold": ("scaffold_decoration", 100, 0.95, 2, 24),
    "superstructure": ("fragment_infill", 25, 1.0, 3, 64),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["denovo", *TASKS])
    parser.add_argument("--model-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Two samples and short decoding; not a benchmark run.")
    parser.add_argument("--dry-run", action="store_true", help="Print the native command without loading weights.")
    args = parser.parse_args()
    model_root = args.model_root.resolve()
    common = ["--output-dir", str(args.output_dir.resolve()), "--seed", str(args.seed), "--device", args.device]
    if args.task == "denovo":
        command = [sys.executable, str(ROOT / "scripts/benchmark/eval_denovo_taskaware.py"),
                   "--model-dir", str(model_root / "ao_generation"),
                   "--num-samples", "2" if args.smoke else "1024",
                   "--batch-size", "2" if args.smoke else "128",
                   "--max-new", "16" if args.smoke else "255",
                   "--temperature", "0.7", "--top-p", "0.95", "--top-k", "0",
                   "--ao-decode-mode", "ar", *common]
    else:
        model, batch, top_p, penalty, max_new = TASKS[args.task]
        command = [sys.executable, str(ROOT / "scripts/benchmark/eval_fcstamp_genmol.py"),
                   "--model-dir", str(model_root / model),
                   "--task-parquet", str(ROOT / "examples/fragments" / f"{args.task}.parquet"),
                   "--num-samples", "2" if args.smoke else "100",
                   "--batch-size", "2" if args.smoke else str(batch),
                   "--temperature", "1.3", "--top-p", str(top_p), "--top-k", "0",
                   "--diverse-prefix-penalty", str(penalty),
                   "--max-new", "16" if args.smoke else str(max_new), *common]
        if args.smoke:
            command += ["--max-cases", "1"]
        if args.task == "scaffold":
            command += ["--sequential-holes"]
        if args.task == "superstructure":
            command += ["--aggregate-attach-sites", "--samples-per-attach-site", "2" if args.smoke else "25"]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
