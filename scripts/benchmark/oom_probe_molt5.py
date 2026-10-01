#!/usr/bin/env python3
"""OOM probe for the MolT5 c4+zinc pretrain config.

Loads t5-v1.1-base, adds all STAMP structural + motif tokens (matches the
real trainer's tokenizer setup), then runs a single forward+backward+step
at increasing per-GPU batch sizes until CUDA OOM. Reports the max batch
size that fits.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.hybrid_vocab import FCSTAMP_CONTROL_TOKENS  # noqa: E402


def _stamp_structural_tokens(max_index: int = 255) -> list[str]:
    toks = ["[BOS]", "[EOS]", "[CONCAT]", "[END]", "[BACK]", "[Ring]", ".", "[S_R]", "[S_S]", "[S_E]", "[S_Z]"]
    toks.extend(f"[J_{i}]" for i in range(max_index + 1))
    toks.extend(f"[B_{i}]" for i in range(max_index + 1))
    toks.extend(FCSTAMP_CONTROL_TOKENS)
    return toks


def _read_stamp_vocab_tokens(path: str) -> list[str]:
    toks: list[str] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if i == 0 and line.startswith("{"):
                continue
            toks.append(line.split("\t")[0])
    return toks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-model", default="/scratch/yuxuan.ren/models/t5-v1_1-base")
    ap.add_argument("--stamp-vocab", default="/scratch/yuxuan.ren/data/ckpt_stamp/zinc_curated_vocab_30k_legacy/motif_vocab.txt")
    ap.add_argument("--max-source-len", type=int, default=192)
    ap.add_argument("--max-target-len", type=int, default=96)
    ap.add_argument("--sizes", default="128,256,384,512,768,1024,1536,2048",
                    help="Comma-separated per-GPU batch sizes to probe, in ascending order.")
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    args = ap.parse_args()

    from transformers import AutoTokenizer, T5ForConditionalGeneration, AddedToken

    print(f"[probe] loading model + tokenizer from {args.init_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.init_model, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(args.init_model, local_files_only=True)

    # Add STAMP tokens exactly like the trainer does
    stamp_tokens = _stamp_structural_tokens(255) + _read_stamp_vocab_tokens(args.stamp_vocab)
    seen: set[str] = set()
    add: list[AddedToken] = []
    for tok in stamp_tokens:
        if tok in seen:
            continue
        seen.add(tok)
        if tokenizer.convert_tokens_to_ids(tok) != tokenizer.unk_token_id:
            continue
        add.append(AddedToken(tok, single_word=False, lstrip=False, rstrip=False, normalized=False, special=False))
    n_added = int(tokenizer.add_tokens(add, special_tokens=False))
    print(f"[probe] tokenizer size: base=32100  added={n_added}  total={len(tokenizer)}", flush=True)
    model.resize_token_embeddings(len(tokenizer))
    print(f"[probe] model params: {sum(p.numel() for p in model.parameters())/1e6:.1f} M", flush=True)

    device = torch.device("cuda")
    model.to(device).train()
    torch.backends.cuda.matmul.allow_tf32 = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def autocast_ctx():
        if args.amp == "bf16":
            return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.amp == "fp16":
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        from contextlib import nullcontext
        return nullcontext()

    def try_batch(bs: int) -> dict:
        # Random input/target ids in the added-token range for realistic tensor sizes.
        vocab_size = len(tokenizer)
        input_ids = torch.randint(5, vocab_size - 1, (bs, args.max_source_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.randint(5, vocab_size - 1, (bs, args.max_target_len), device=device)
        # A quarter of label tokens masked as -100 (loss ignore) to mimic sentinel padding.
        mask = torch.rand_like(labels.float()) < 0.15
        labels_masked = labels.clone()
        labels_masked[mask] = -100

        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        try:
            with autocast_ctx():
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_masked)
            out.loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as e:
            return {"ok": False, "error": "OOM", "detail": str(e)[:200]}
        except Exception as e:
            return {"ok": False, "error": type(e).__name__, "detail": str(e)[:200]}
        dt = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        reserved_gb = torch.cuda.max_memory_reserved(device) / 1e9
        return {"ok": True, "step_time_s": round(dt, 3), "peak_alloc_gb": round(peak_gb, 2), "peak_reserved_gb": round(reserved_gb, 2)}

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    results = []
    last_ok = None
    for bs in sizes:
        # Warm the compile/kernel path with a small step first if very first
        r = try_batch(bs)
        r["batch_size"] = bs
        results.append(r)
        if r["ok"]:
            last_ok = bs
            print(f"[probe] bs={bs:>5}  OK  peak_alloc={r['peak_alloc_gb']:.2f}GB  reserved={r['peak_reserved_gb']:.2f}GB  step={r['step_time_s']}s", flush=True)
        else:
            print(f"[probe] bs={bs:>5}  {r['error']}  {r['detail']}", flush=True)
            break
        # Explicit cleanup between probes
        gc.collect()
        torch.cuda.empty_cache()

    print()
    print("=== summary ===")
    print(json.dumps({
        "max_ok_batch_size_per_gpu": last_ok,
        "results": results,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
