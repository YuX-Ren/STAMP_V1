"""Prepare linker_subtree training data from STAMP pretrain parquet.

For each drug:
  1. STAMP-encode → motif groups (motif tree)
  2. BRICS-cut: sample pairs of cleavable bonds → 3 fragments
  3. Identify linker fragment (the one with 2 attach points)
  4. Map linker atoms back to original mol → identify which motifs are in linker
  5. Save linker_motif_indices as new parquet column

Output parquet has original columns + new `linker_subtrees` column = JSON list of lists,
each inner list is one valid linker subtree (motif indices).

At training time, sample one subtree per epoch.
"""
import argparse, json, time, sys, random
import itertools
from pathlib import Path
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
RDLogger.DisableLog('rdApp.*')


def init_worker(stamp_vocab_path):
    """Init each worker process with its own rep instance (RDKit/torch state)."""
    sys.path.insert(0, '/home/svu/yuxuan.ren/STAMP_V1.5')
    from stamp.benchmark.representations import build_representation
    global REP
    REP = build_representation('stamp', stamp_vocab=stamp_vocab_path)


def find_linker_subtrees(args):
    """For one drug SMILES, return list of valid linker_motif_index lists.
    args = (smiles, max_pairs, max_linker_motifs)
    Returns JSON-encoded list of lists, e.g. "[[3], [2,4]]" or empty "[]"."""
    smiles, max_pairs, max_linker_motifs = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return "[]"

    # Find BRICS bonds
    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    if len(brics_bonds) < 2: return "[]"

    # Use canonical mol for atom indexing consistency
    canon_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    canon_mol = Chem.MolFromSmiles(canon_smi)
    if canon_mol is None: return "[]"

    # Re-find BRICS bonds on canon_mol (atom indices may differ)
    brics_bonds_canon = list(BRICS.FindBRICSBonds(canon_mol))
    if len(brics_bonds_canon) < 2: return "[]"

    # Get STAMP groups
    try:
        groups = REP.codec.encode_mol_to_groups(canon_mol)
    except Exception:
        return "[]"
    n_groups = len(groups)
    if n_groups < 3: return "[]"  # need at least 3 motifs for a sub-mid-tail

    # Mark canon atoms with map nums (1-indexed)
    canon_marked = Chem.RWMol(canon_mol)
    for i, a in enumerate(canon_marked.GetAtoms()):
        a.SetAtomMapNum(i + 1)
    canon_marked_mol = canon_marked.GetMol()

    # Sample bond pairs
    pairs = list(itertools.combinations(brics_bonds_canon, 2))
    if len(pairs) > max_pairs:
        random.shuffle(pairs)
        pairs = pairs[:max_pairs]

    results = []
    for b1, b2 in pairs:
        try:
            cut_mol = BRICS.BreakBRICSBonds(canon_marked_mol, [b1, b2], sanitize=True)
            frags = list(Chem.GetMolFrags(cut_mol, asMols=True))
            if len(frags) != 3: continue

            # Identify linker fragment (= fragment with 2 attach [*] atoms)
            attach_counts = [sum(1 for a in f.GetAtoms() if a.GetAtomicNum() == 0) for f in frags]
            if attach_counts.count(2) != 1: continue
            linker_idx = attach_counts.index(2)

            # Get linker atoms' original (canon) indices via map nums
            linker_atoms_orig = set()
            for a in frags[linker_idx].GetAtoms():
                if a.GetAtomicNum() == 0: continue  # skip [*]
                mn = a.GetAtomMapNum()
                if mn > 0: linker_atoms_orig.add(mn - 1)

            if not linker_atoms_orig: continue

            # Find motifs entirely within linker_atoms
            linker_motif_indices = []
            for gi, g in enumerate(groups):
                if set(g.atom_indices).issubset(linker_atoms_orig):
                    linker_motif_indices.append(gi)

            # Filter: linker should have 1 to max_linker_motifs motifs
            if 1 <= len(linker_motif_indices) <= max_linker_motifs:
                # Avoid first / last motif (usually root or terminal — tricky topology)
                if 0 in linker_motif_indices or (n_groups - 1) in linker_motif_indices:
                    continue
                results.append(linker_motif_indices)
        except Exception:
            continue

    # Dedup (in case different cuts give same linker subtree)
    seen = set(); uniq = []
    for r in results:
        key = tuple(sorted(r))
        if key not in seen:
            seen.add(key); uniq.append(sorted(r))

    return json.dumps(uniq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stamp-vocab", required=True)
    ap.add_argument("--smiles-col", default="std_smiles")
    ap.add_argument("--max-pairs", type=int, default=5)
    ap.add_argument("--max-linker-motifs", type=int, default=5)
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=10000)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    pf = pq.ParquetFile(args.inp)
    schema = pf.schema_arrow
    # Add new column
    new_field = pa.field("linker_subtrees", pa.string())
    new_schema = pa.schema(list(schema) + [new_field])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.out, new_schema, compression="snappy")

    t0 = time.time()
    total = 0; with_subtrees = 0

    with Pool(args.n_workers, initializer=init_worker, initargs=(args.stamp_vocab,)) as pool:
        for batch in pf.iter_batches(batch_size=args.batch_size):
            df = batch.to_pandas()
            smiles_list = df[args.smiles_col].fillna("").tolist()
            tasks = [(s, args.max_pairs, args.max_linker_motifs) for s in smiles_list]
            subtrees_list = pool.map(find_linker_subtrees, tasks)
            df["linker_subtrees"] = subtrees_list
            with_subtrees += sum(1 for s in subtrees_list if s != "[]")
            total += len(df)
            writer.write_table(pa.Table.from_pandas(df, schema=new_schema, preserve_index=False))
            if total % (args.batch_size * 5) == 0:
                rate = total / max(1, time.time() - t0)
                pct = with_subtrees / total * 100
                eta = (pf.metadata.num_rows - total) / rate if rate > 0 else 0
                print(f"  {total:>10,}  with_subtrees={with_subtrees:>10,} ({pct:.1f}%)  rate={rate:.0f}/s  ETA={eta:.0f}s", flush=True)
            if args.max_rows > 0 and total >= args.max_rows: break

    writer.close()
    print(f"\nDone. {total:,} rows processed, {with_subtrees:,} ({with_subtrees/total*100:.1f}%) with subtrees ({time.time()-t0:.0f}s)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
