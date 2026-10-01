#!/usr/bin/env python3
"""Tokenize a SMILES corpus with a STAMP/PS-VAE vocab and profile token-usage frequency.

Outputs:
    <output-dir>/token_counts.tsv     token \t count, sorted desc
    <output-dir>/seq_length_stats.json  sequence length percentiles
    <output-dir>/coverage.tsv         cumulative coverage curve (rank, cum_count, cum_frac)
    <output-dir>/summary.json         mol counts, OOV mols, encoding failures
    <output-dir>/zipf.png             log-log Zipf plot (if matplotlib available)

OOV semantics: a "failed" molecule is one where STAMPEncoder.encode_smiles raises.
PrincipalSubgraphTokenizer falls back to atom-level tokens when no vocab motif
matches; we still count those resulting tokens here.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


# --- worker globals --------------------------------------------------------
_W_ENCODER = None
_W_VOCAB_PATH = None


def _init_worker(vocab_path: str, force_ring_aware: bool) -> None:
    global _W_ENCODER, _W_VOCAB_PATH
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from stamp.benchmark.representations import STAMPRepresentation
    repr_ = STAMPRepresentation(
        vocab_path=vocab_path,
        force_ring_aware_encode=force_ring_aware,
    )
    _W_ENCODER = repr_
    _W_VOCAB_PATH = vocab_path


def _encode_one(smi: str) -> Tuple[Optional[List[str]], int]:
    """Returns (tokens, status). status: 0=ok, 1=encode-fail, 2=invalid-smiles."""
    try:
        s = _W_ENCODER.encode(smi)
        return _W_ENCODER.tokenize(s), 0
    except Exception:
        return None, 1


def iter_smiles(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to .smi (one SMILES per line).")
    ap.add_argument("--vocab", required=True, help="PS-VAE vocab.txt path.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunksize", type=int, default=512)
    ap.add_argument("--max-mols", type=int, default=0, help="0 = full corpus.")
    ap.add_argument("--force-ring-aware", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--progress-every", type=int, default=200_000)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    seq_lens: List[int] = []
    n_mols = 0
    n_fail = 0

    smis = iter_smiles(in_path)
    if args.max_mols > 0:
        from itertools import islice
        smis = islice(smis, args.max_mols)

    t0 = time.time()
    if args.workers <= 1:
        _init_worker(args.vocab, args.force_ring_aware)
        for smi in smis:
            toks, status = _encode_one(smi)
            n_mols += 1
            if status != 0 or toks is None:
                n_fail += 1
                continue
            counts.update(toks)
            seq_lens.append(len(toks))
            if n_mols % args.progress_every == 0:
                print(
                    f"[{n_mols:>10}] failed={n_fail} unique_tok={len(counts)} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(args.vocab, args.force_ring_aware),
        ) as pool:
            for toks, status in pool.imap_unordered(
                _encode_one, smis, chunksize=args.chunksize
            ):
                n_mols += 1
                if status != 0 or toks is None:
                    n_fail += 1
                    continue
                counts.update(toks)
                seq_lens.append(len(toks))
                if n_mols % args.progress_every == 0:
                    print(
                        f"[{n_mols:>10}] failed={n_fail} unique_tok={len(counts)} "
                        f"elapsed={time.time() - t0:.0f}s",
                        flush=True,
                    )

    elapsed = time.time() - t0
    print(f"Encoded {n_mols} mols, {n_fail} failures, {elapsed:.1f}s", flush=True)

    sorted_items = counts.most_common()
    with (out_dir / "token_counts.tsv").open("w", encoding="utf-8") as f:
        f.write("token\tcount\n")
        for tok, c in sorted_items:
            f.write(f"{tok}\t{c}\n")

    total_tok = sum(c for _, c in sorted_items)
    with (out_dir / "coverage.tsv").open("w", encoding="utf-8") as f:
        f.write("rank\ttoken\tcount\tcum_count\tcum_frac\n")
        cum = 0
        for rank, (tok, c) in enumerate(sorted_items, 1):
            cum += c
            f.write(f"{rank}\t{tok}\t{c}\t{cum}\t{cum / total_tok:.6f}\n")

    if seq_lens:
        seq_lens.sort()
        n = len(seq_lens)

        def pct(p: float) -> int:
            i = min(int(p * n), n - 1)
            return seq_lens[i]

        seq_stats = {
            "n_mols": n,
            "min": seq_lens[0],
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": seq_lens[-1],
            "mean": sum(seq_lens) / n,
        }
    else:
        seq_stats = {"n_mols": 0}
    (out_dir / "seq_length_stats.json").write_text(json.dumps(seq_stats, indent=2))

    summary = {
        "input": str(in_path),
        "vocab": str(args.vocab),
        "n_mols_total": n_mols,
        "n_encode_failures": n_fail,
        "n_mols_ok": n_mols - n_fail,
        "n_unique_tokens": len(counts),
        "n_total_token_occurrences": total_tok,
        "elapsed_seconds": elapsed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ranks = list(range(1, len(sorted_items) + 1))
            cnts = [c for _, c in sorted_items]
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
            ax[0].loglog(ranks, cnts)
            ax[0].set_xlabel("rank")
            ax[0].set_ylabel("count")
            ax[0].set_title("Zipf (token freq vs rank)")
            ax[0].grid(True, which="both", alpha=0.3)
            cum = 0
            cum_fracs = []
            for c in cnts:
                cum += c
                cum_fracs.append(cum / total_tok)
            ax[1].semilogx(ranks, cum_fracs)
            ax[1].set_xlabel("vocab size (top-K by freq)")
            ax[1].set_ylabel("cumulative token coverage")
            ax[1].set_title("Cumulative coverage")
            ax[1].grid(True, which="both", alpha=0.3)
            for tgt in [0.90, 0.95, 0.99, 0.995, 0.999]:
                k = next((r for r, f in zip(ranks, cum_fracs) if f >= tgt), None)
                if k is not None:
                    ax[1].axhline(tgt, color="grey", linestyle=":", linewidth=0.7)
                    ax[1].axvline(k, color="grey", linestyle=":", linewidth=0.7)
            fig.tight_layout()
            fig.savefig(out_dir / "zipf.png", dpi=120)
            print(f"Saved plot: {out_dir / 'zipf.png'}")
        except Exception as e:
            print(f"[warn] plotting failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
