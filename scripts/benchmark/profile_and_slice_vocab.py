"""Profile real motif usage frequency on STAMP-encoded shards, then truncate
the source vocab to top-K motifs by usage."""
import argparse, glob, time, json
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq


def is_structural(t: str) -> bool:
    return t in ("[Ring]", "[END]", "[BACK]") or (t.startswith("[") and t.endswith("]") and (t.startswith("[J_") or t.startswith("[B_") or t.startswith("[S_")))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards-glob", required=True, help="Glob pattern for STAMP parquet shards")
    p.add_argument("--source-vocab", required=True, help="Original (large) motif vocab to slice")
    p.add_argument("--output-vocab", required=True)
    p.add_argument("--top-k", type=int, default=12000)
    args = p.parse_args()

    shards = sorted(glob.glob(args.shards_glob))
    print(f"[profile] {len(shards)} shards", flush=True)

    freq = Counter()
    t0 = time.time(); n_mols = 0
    for i, shard in enumerate(shards):
        tbl = pq.read_table(shard, columns=["stamp"])
        col = tbl.column("stamp").to_pylist()
        for s in col:
            if not s: continue
            n_mols += 1
            for tok in s.split():
                if not is_structural(tok):
                    freq[tok] += 1
        if (i + 1) % 10 == 0 or (i + 1) == len(shards):
            print(f"  shard {i+1}/{len(shards)}  cum_mols={n_mols:,}  unique_motifs={len(freq):,}  ({time.time()-t0:.0f}s)", flush=True)

    # Load source vocab to get atom_count for each motif
    entries = {}
    with open(args.source_vocab) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3: continue
            smi, ac, _ = parts
            entries[smi] = int(ac)

    # Top K by usage
    top = freq.most_common(args.top_k)
    print(f"\n[profile] {len(top):,} motifs in top-{args.top_k}; total unique = {len(freq):,}")

    # Filter to motifs present in source vocab (avoid drift)
    out_entries = []
    skipped = 0
    for smi, real_freq in top:
        if smi in entries:
            out_entries.append((smi, entries[smi], real_freq))
        else:
            skipped += 1
    print(f"[profile] {len(out_entries)} kept, {skipped} skipped (not in source vocab)")

    # Sort by real_freq descending (then atom_count desc, then alphabet)
    out_entries.sort(key=lambda x: (-x[2], -x[1], x[0]))

    Path(args.output_vocab).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_vocab, "w") as f:
        f.write(header)  # preserve original JSON header
        for smi, ac, real_freq in out_entries:
            f.write(f"{smi}\t{ac}\t{real_freq}\n")
    print(f"[profile] wrote {args.output_vocab}: {len(out_entries)} motifs")

    # Coverage stats
    total_freq_in_vocab = sum(real_freq for _, _, real_freq in out_entries)
    total_freq_all = sum(freq.values())
    print(f"[profile] vocab coverage: {total_freq_in_vocab/total_freq_all*100:.2f}% of all motif occurrences")


if __name__ == "__main__":
    main()
