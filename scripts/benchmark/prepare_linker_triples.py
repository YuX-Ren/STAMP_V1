"""Prepare (pieceA_stamp, pieceB_stamp, linker_stamp) triples from pretrain
parquet for SAFE-GPT-style prefix fine-tuning.

For each drug:
  1. BRICS cut to get 3 fragments (pieceA, linker, pieceB), each with [*] marker
  2. Encode each fragment as STAMP with [J_attach] marker(s)
  3. Save token sequences

Output parquet has new columns:
  - pieceA_stamp: space-joined STAMP tokens of pieceA fragment
  - pieceB_stamp: same for pieceB
  - linker_stamp: linker fragment STAMP tokens (with 2 attach markers)
"""
import argparse, json, time, sys, random, itertools
from pathlib import Path
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
RDLogger.DisableLog('rdApp.*')


def init_worker(stamp_vocab_path):
    sys.path.insert(0, '/home/svu/yuxuan.ren/STAMP_V1.5')
    from stamp.benchmark.representations import build_representation
    from stamp.benchmark.hybrid_vocab import is_stamp_structural
    global REP, IS_STAMP_STRUCT
    REP = build_representation('stamp', stamp_vocab=stamp_vocab_path)
    IS_STAMP_STRUCT = is_stamp_structural


def encode_fragment_with_attach(frag_mol):
    """Encode a fragment mol (with one or more [*] atoms) as STAMP with [J_attach]
    marker(s) inserted at the attach atom positions in the canonical STAMP encoding.

    Returns: list of STAMP tokens (str), or None if encoding fails.
    """
    # Find attach atoms (heavy atoms bonded to [*])
    attach_orig_indices = []
    for a in frag_mol.GetAtoms():
        if a.GetAtomicNum() == 0:  # [*]
            for n in a.GetNeighbors():
                if n.GetAtomicNum() > 1:
                    attach_orig_indices.append(n.GetIdx())
                    break

    if not attach_orig_indices:
        return None

    # Mark attach atoms with map nums (use 91, 92, ... — won't collide)
    rwmol = Chem.RWMol(frag_mol)
    for i, idx in enumerate(attach_orig_indices):
        rwmol.GetAtomWithIdx(idx).SetAtomMapNum(91 + i)

    # Remove all [*] atoms
    star_idxs = [a.GetIdx() for a in rwmol.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(star_idxs, reverse=True):
        rwmol.RemoveAtom(idx)

    capped = rwmol.GetMol()
    try:
        canon_with_map = Chem.MolToSmiles(capped, canonical=True, isomericSmiles=True)
    except Exception:
        return None

    re_mol = Chem.MolFromSmiles(canon_with_map)
    if re_mol is None:
        return None

    # Find attach atoms in canon order
    attach_canon = []  # list of (canon_idx, marker_id)
    for i, a in enumerate(re_mol.GetAtoms()):
        mn = a.GetAtomMapNum()
        if 91 <= mn <= 99:
            attach_canon.append((i, mn))

    # Strip map nums for clean encoding
    for a in re_mol.GetAtoms():
        a.SetAtomMapNum(0)
    canon_clean = Chem.MolToSmiles(re_mol, canonical=True, isomericSmiles=True)

    # Get STAMP encoding
    try:
        text = REP.encode(canon_clean)
    except Exception:
        return None
    base_toks = text.split()

    # Build mapping: canon atom idx → motif idx (groups)
    marked_mol = Chem.MolFromSmiles(canon_with_map)
    if marked_mol is None:
        return None
    try:
        groups = REP.codec.encode_mol_to_groups(marked_mol)
    except Exception:
        return None

    # For each attach atom, find its motif idx + local heavy idx
    attach_info = []
    for canon_idx, marker_id in attach_canon:
        # Find the same atom in marked_mol (look up by map num)
        mn = marker_id
        marked_idx = None
        for i, a in enumerate(marked_mol.GetAtoms()):
            if a.GetAtomMapNum() == mn:
                marked_idx = i
                break
        if marked_idx is None:
            continue
        for gi, g in enumerate(groups):
            if marked_idx in g.atom_indices:
                heavy = [i for i in g.atom_indices if marked_mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
                if marked_idx in heavy:
                    attach_info.append((gi, heavy.index(marked_idx)))
                break

    if not attach_info:
        return None

    # Walk base_toks; for each motif we encounter, after its joints, insert the
    # [J_<local>] markers for all attach atoms in this motif.
    # attach_info: list of (motif_idx, local_heavy_idx)
    by_motif = {}
    for mi, li in attach_info:
        by_motif.setdefault(mi, []).append(li)

    out = []
    motif_count = 0
    i = 0
    while i < len(base_toks):
        t = base_toks[i]
        if not IS_STAMP_STRUCT(t):
            this_motif = motif_count
            out.append(t); i += 1
            while i < len(base_toks) and (base_toks[i].startswith("[J_") or
                                            base_toks[i].startswith("[S_") or
                                            base_toks[i] == "[Ring]"):
                out.append(base_toks[i]); i += 1
            # Append additional [J_<local>] for any attach atoms in this motif
            if this_motif in by_motif:
                for li in by_motif[this_motif]:
                    out.append(f"[J_{li}]")
            motif_count += 1
        else:
            out.append(t); i += 1
    return out


def process_drug(args):
    """For one drug, do BRICS cut + encode triples. Returns JSON-encoded list of
    triples [{pieceA, pieceB, linker}, ...]."""
    smiles, max_pairs, max_linker_motifs = args
    if not smiles: return "[]"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return "[]"

    # Early reject: count BRICS bonds quickly
    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    if len(brics_bonds) < 2: return "[]"

    # Use input mol directly (assume input parquet std_smiles is already canonical)
    canon_mol = mol
    brics_canon = brics_bonds

    pairs = list(itertools.combinations(brics_canon, 2))
    if len(pairs) > max_pairs:
        random.shuffle(pairs); pairs = pairs[:max_pairs]

    results = []
    for b1, b2 in pairs:
        try:
            cut_mol = BRICS.BreakBRICSBonds(canon_mol, [b1, b2], sanitize=True)
            frags = list(Chem.GetMolFrags(cut_mol, asMols=True))
            if len(frags) != 3: continue

            # Identify linker (2 attach), pieces (1 attach each)
            attach_counts = [sum(1 for a in f.GetAtoms() if a.GetAtomicNum() == 0) for f in frags]
            if attach_counts.count(2) != 1: continue
            linker_idx = attach_counts.index(2)
            piece_idxs = [i for i in range(3) if i != linker_idx]
            pieceA = frags[piece_idxs[0]]
            pieceB = frags[piece_idxs[1]]
            linker = frags[linker_idx]

            # Filter linker size
            n_linker_heavy = sum(1 for a in linker.GetAtoms() if a.GetAtomicNum() > 1)
            if n_linker_heavy > max_linker_motifs * 6: continue  # rough size cap

            # Encode each
            pa_toks = encode_fragment_with_attach(pieceA)
            pb_toks = encode_fragment_with_attach(pieceB)
            link_toks = encode_fragment_with_attach(linker)
            if not all([pa_toks, pb_toks, link_toks]): continue

            results.append({
                "pieceA": " ".join(pa_toks),
                "pieceB": " ".join(pb_toks),
                "linker": " ".join(link_toks),
            })
        except Exception:
            continue

    return json.dumps(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stamp-vocab", required=True)
    ap.add_argument("--smiles-col", default="std_smiles")
    ap.add_argument("--max-pairs", type=int, default=3)
    ap.add_argument("--max-linker-motifs", type=int, default=5)
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=5000)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    pf = pq.ParquetFile(args.inp)
    schema = pf.schema_arrow
    new_field = pa.field("linker_triples", pa.string())
    new_schema = pa.schema(list(schema) + [new_field])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.out, new_schema, compression="snappy")

    t0 = time.time()
    total = 0; with_triples = 0; total_triples = 0
    with Pool(args.n_workers, initializer=init_worker, initargs=(args.stamp_vocab,)) as pool:
        for batch in pf.iter_batches(batch_size=args.batch_size):
            df = batch.to_pandas()
            smiles_list = df[args.smiles_col].fillna("").tolist()
            tasks = [(s, args.max_pairs, args.max_linker_motifs) for s in smiles_list]
            # imap + chunksize for pipelined dispatch (preserves order)
            chunksize = max(1, len(tasks) // (args.n_workers * 4))
            triples_list = list(pool.imap(process_drug, tasks, chunksize=chunksize))
            df["linker_triples"] = triples_list
            for s in triples_list:
                tr = json.loads(s)
                if tr:
                    with_triples += 1
                    total_triples += len(tr)
            total += len(df)
            writer.write_table(pa.Table.from_pandas(df, schema=new_schema, preserve_index=False))
            if total % (args.batch_size * 5) == 0:
                rate = total / max(1, time.time() - t0)
                pct = with_triples / total * 100
                print(f"  {total:>10,}  with_triples={with_triples:>10,} ({pct:.1f}%)  total_triples={total_triples:>10,}  rate={rate:.0f}/s", flush=True)
            if args.max_rows > 0 and total >= args.max_rows: break

    writer.close()
    print(f"\nDone. {total:,} rows, {with_triples:,} with triples ({with_triples/total*100:.1f}%), {total_triples:,} total triples ({time.time()-t0:.0f}s)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
