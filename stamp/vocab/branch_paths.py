from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def read_vocab_branch_mode(vocab_path: str | Path | None) -> Optional[str]:
    if not vocab_path:
        return None
    path = Path(vocab_path)
    if not path.exists():
        return None
    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        header = json.loads(first_line)
    except Exception:
        return None
    branch_mode = str(header.get("branch_mode", "")).strip()
    return branch_mode or None


def read_model_branch_mode(model_dir: str | Path | None) -> Optional[str]:
    if not model_dir:
        return None
    candidate = Path(model_dir) / "stamp_vocab.txt"
    return read_vocab_branch_mode(candidate)


def append_branch_subdir(output_dir: str | Path, branch_mode: str | None) -> Path:
    out = Path(output_dir)
    branch = (branch_mode or "").strip()
    if not branch:
        return out
    if out.name == branch:
        return out
    return out / branch


def resolve_branch_output_dir(
    output_dir: str | Path,
    *,
    stamp_vocab: str | Path | None = None,
    model_dir: str | Path | None = None,
) -> Path:
    branch_mode = read_vocab_branch_mode(stamp_vocab)
    if branch_mode is None:
        branch_mode = read_model_branch_mode(model_dir)
    return append_branch_subdir(output_dir, branch_mode)
