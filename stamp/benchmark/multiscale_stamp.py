from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .representations import STAMPRepresentation


def make_scale_tokens(scale_names: Sequence[str]) -> list[str]:
    return [f"<ms:{name}>" for name in scale_names]


def parse_csv_arg(text: str | None) -> list[str]:
    if text is None:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


class MultiScaleSTAMPEncoder:
    """Encode one molecule into multiple STAMP granularities and concatenate them.

    Each scale gets a leading marker token, e.g. ``<ms:fine>``. The resulting
    text is a plain whitespace-tokenized sequence that can be fed into the same
    decoder pretraining / finetuning pipeline as single-scale STAMP.
    """

    def __init__(
        self,
        vocab_paths: Sequence[str | Path],
        *,
        scale_names: Optional[Sequence[str]] = None,
        force_ring_aware_encode: bool = False,
        strict_vocab: bool = True,
    ) -> None:
        vocab_paths = [str(Path(p)) for p in vocab_paths]
        if not vocab_paths:
            raise ValueError("MultiScaleSTAMPEncoder requires at least one vocab path.")
        self.vocab_paths = vocab_paths
        if scale_names is None:
            scale_names = [f"s{i}" for i in range(len(vocab_paths))]
        if len(scale_names) != len(vocab_paths):
            raise ValueError("scale_names length must match vocab_paths length.")
        self.scale_names = [str(x) for x in scale_names]
        self.scale_tokens = make_scale_tokens(self.scale_names)
        self.representations = [
            STAMPRepresentation(
                vocab_path=vp,
                force_ring_aware_encode=force_ring_aware_encode,
                strict_vocab=strict_vocab,
            )
            for vp in self.vocab_paths
        ]

    def encode_scales(self, smiles: str) -> list[str]:
        return [rep.encode(smiles) for rep in self.representations]

    def encode(self, smiles: str) -> str:
        parts: list[str] = []
        for scale_tok, rep in zip(self.scale_tokens, self.representations):
            parts.append(scale_tok)
            parts.append(rep.encode(smiles))
        return " ".join(parts)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return [t for t in str(text).split() if t]


def save_multiscale_manifest(
    path: str | Path,
    *,
    scale_names: Sequence[str],
    vocab_files: Sequence[str],
    text_column: str = "stamp_multiscale",
) -> None:
    payload = {
        "scale_names": list(scale_names),
        "scale_tokens": make_scale_tokens(scale_names),
        "vocab_files": list(vocab_files),
        "text_column": str(text_column),
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_multiscale_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_multiscale_vocab_paths(model_dir: str | Path, vocab_files: Iterable[str]) -> list[str]:
    root = Path(model_dir)
    out: list[str] = []
    for vf in vocab_files:
        candidate = root / vf
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        out.append(str(candidate))
    return out
