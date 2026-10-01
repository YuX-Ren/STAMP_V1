#!/usr/bin/env python3
"""Parallel downloader for public IBM Box MoLFormer files.

Box shared links redirect to a temporary public.boxcloud.com URL. This helper
resolves that URL, downloads byte ranges concurrently, then concatenates the
parts into the final zip. It is intentionally stdlib-only.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError


SHARED_NAME = "j7nrr80v8gswev01gngati5onpq8pbfd"
FILES = {
    "pubchem": ("f_1095723242548", "pubchem_smiles_canonical.zip", 1128405509),
    "zinc": ("f_1096127714688", "ZINC15_smiles.zip", 9544479800),
    "finetune": ("f_1095699036312", "finetune_datasets.zip", 27974444),
    "pretrained": ("f_1099206797207", "Pretrained MoLFormer.zip", 3603429774),
}


def _shared_url(file_id: str) -> str:
    return (
        "https://ibm.ent.box.com/index.php?"
        f"rm=box_download_shared_file&shared_name={SHARED_NAME}&file_id={file_id}"
    )


def _resolve_direct_url(url: str) -> tuple[str, int | None]:
    opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler)
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "curl/8"})
    try:
        with opener.open(req, timeout=60) as resp:
            return resp.geturl(), int(resp.headers.get("Content-Length") or 0) or None
    except HTTPError as exc:
        if exc.code in {403, 405}:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "curl/8"})
            with opener.open(req, timeout=60) as resp:
                total = resp.headers.get("Content-Range", "").split("/")[-1]
                return resp.geturl(), int(total) if total.isdigit() else None
        raise


def _curl_range(direct_url: str, out: Path, start: int, end: int, retries: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size == end - start + 1:
        return
    tmp = out.with_suffix(out.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    cmd = [
        "curl",
        "-sS",
        "-L",
        "--fail",
        "--retry",
        str(retries),
        "--retry-delay",
        "5",
        "-H",
        f"Range: bytes={start}-{end}",
        "-o",
        str(tmp),
        direct_url,
    ]
    subprocess.run(cmd, check=True)
    got = tmp.stat().st_size
    expected = end - start + 1
    if got != expected:
        raise RuntimeError(f"bad part size for {out.name}: got {got}, expected {expected}")
    os.replace(tmp, out)


def _concat(parts: list[Path], out: Path) -> None:
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("wb") as w:
        for part in parts:
            with part.open("rb") as r:
                shutil.copyfileobj(r, w, length=64 * 1024 * 1024)
    os.replace(tmp, out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--file", choices=sorted(FILES), required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--connections", type=int, default=8)
    p.add_argument("--chunk-mib", type=int, default=512)
    p.add_argument("--retries", type=int, default=8)
    p.add_argument("--keep-parts", action="store_true")
    args = p.parse_args()

    file_id, filename, known_size = FILES[args.file]
    out = args.out_dir / filename
    if out.exists() and out.stat().st_size == known_size:
        print(f"[skip] already complete: {out} ({known_size} bytes)", flush=True)
        return 0

    direct_url, size = _resolve_direct_url(_shared_url(file_id))
    size = int(size or known_size)
    print(f"[resolve] {filename} size={size} url={direct_url[:96]}...", flush=True)
    part_dir = args.out_dir / f".{filename}.parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk = max(1, int(args.chunk_mib)) * 1024 * 1024
    ranges: list[tuple[int, int, Path]] = []
    idx = 0
    for start in range(0, size, chunk):
        end = min(size - 1, start + chunk - 1)
        ranges.append((start, end, part_dir / f"part_{idx:05d}"))
        idx += 1

    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, int(args.connections))) as ex:
        futs = {
            ex.submit(_curl_range, direct_url, part, start, end, args.retries): (start, end, part)
            for start, end, part in ranges
        }
        for fut in cf.as_completed(futs):
            start, end, part = futs[fut]
            fut.result()
            done += end - start + 1
            rate = done / max(1e-9, time.time() - t0) / (1024 * 1024)
            print(f"[part] {part.name} complete {done/size*100:.1f}% rate={rate:.1f} MiB/s", flush=True)

    parts = [part for _, _, part in ranges]
    _concat(parts, out)
    if out.stat().st_size != size:
        raise RuntimeError(f"bad final size: got {out.stat().st_size}, expected {size}")
    if not args.keep_parts:
        shutil.rmtree(part_dir)
    print(f"[done] {out} {out.stat().st_size} bytes elapsed={time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
