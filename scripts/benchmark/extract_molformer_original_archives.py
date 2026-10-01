#!/usr/bin/env python3
"""Extract selected IBM MoLFormer archives into a reusable raw-data layout."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def _extract(zip_path: Path, out_dir: Path, *, overwrite: bool) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    print(f"[extract] {zip_path} -> {out_dir}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = out_dir / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists() and target.stat().st_size == info.file_size and not overwrite:
                print(f"[skip] {target}", flush=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                while True:
                    buf = src.read(64 * 1024 * 1024)
                    if not buf:
                        break
                    dst.write(buf)
            print(f"[write] {target} {target.stat().st_size} bytes", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--downloads-dir", type=Path, required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--extract-zinc", action="store_true", help="Also extract all 1718 ZINC .smi files.")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    _extract(args.downloads_dir / "pubchem_smiles_canonical.zip", args.raw_dir, overwrite=args.overwrite)
    finetune_zip = args.downloads_dir / "finetune_datasets.zip"
    if finetune_zip.exists():
        _extract(finetune_zip, args.raw_dir, overwrite=args.overwrite)
    else:
        print(f"[warn] missing optional finetune zip: {finetune_zip}", flush=True)
    if args.extract_zinc:
        _extract(args.downloads_dir / "ZINC15_smiles.zip", args.raw_dir, overwrite=args.overwrite)
    else:
        print("[info] ZINC stays compressed; encoder can stream assigned members from the zip.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
