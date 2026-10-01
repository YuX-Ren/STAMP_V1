"""Prepare (prefix_stamp, completion_stamp) pairs for GenMol-style fragment tasks.

Tasks:
  motif_ext   — BRICS cut at 1 bond → smaller fragment = piece (with [J_x] marker),
                 full drug = completion.
  scaffold_dec — Murcko scaffold = prefix (with [J_x] markers at attach atoms),
                 full drug = completion.
  super_gen    — random connected subgraph (40-70% of heavy atoms) = prefix
                 (no attach markers), full drug = completion.

Output parquet has new column `task_pairs` (JSON list of dicts {"prefix":..., "completion":...}).
"""
import argparse, json, time, sys, random, itertools
from pathlib import Path
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')


def init_worker(stamp_vocab_path):
    sys.path.insert(0, '/home/svu/yuxuan.ren/STAMP_V1.5')
    from stamp.benchmark.representations import build_representation
    from stamp.benchmark.hybrid_vocab import is_stamp_structural
    global REP, IS_STAMP_STRUCT
    REP = build_representation('stamp', stamp_vocab=stamp_vocab_path)
    IS_STAMP_STRUCT = is_stamp_structural


def encode_with_attach(frag_mol, attach_atom_indices):
    """Encode a mol fragment as STAMP tokens, inserting [J_<local>] markers at the
    given heavy-atom indices (in the input mol's atom ordering)."""
    if not attach_atom_indices:
        # No attach: just encode plain (used by super_gen)
        try:
            text = REP.encode(Chem.MolToSmiles(frag_mol, canonical=True, isomericSmiles=True))
            return text.split()
        except Exception:
            return None

    # Map atoms by atom map num (use 91+i, then strip on canonicalization)
    rwmol = Chem.RWMol(frag_mol)
    for i, idx in enumerate(attach_atom_indices):
        rwmol.GetAtomWithIdx(idx).SetAtomMapNum(91 + i)
    capped = rwmol.GetMol()
    try:
        canon_with_map = Chem.MolToSmiles(capped, canonical=True, isomericSmiles=True)
    except Exception:
        return None
    re_mol = Chem.MolFromSmiles(canon_with_map)
    if re_mol is None: return None
    attach_canon = []
    for i, a in enumerate(re_mol.GetAtoms()):
        mn = a.GetAtomMapNum()
        if 91 <= mn <= 99:
            attach_canon.append((i, mn))
    for a in re_mol.GetAtoms(): a.SetAtomMapNum(0)
    canon_clean = Chem.MolToSmiles(re_mol, canonical=True, isomericSmiles=True)
    try: text = REP.encode(canon_clean)
    except Exception: return None
    base_toks = text.split()
    try: groups = REP.codec.encode_mol_to_groups(re_mol)
    except Exception: return None
    attach_info = []
    for canon_idx, _mn in attach_canon:
        for gi, g in enumerate(groups):
            if canon_idx in g.atom_indices:
                heavy = [i for i in g.atom_indices if re_mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
                if canon_idx in heavy:
                    attach_info.append((gi, heavy.index(canon_idx)))
                break
    by_motif = {}
    for mi, li in attach_info:
        by_motif.setdefault(mi, []).append(li)
    out = []; mc = 0; i = 0
    while i < len(base_toks):
        t = base_toks[i]
        if not IS_STAMP_STRUCT(t):
            this_motif = mc
            out.append(t); i += 1
            while i < len(base_toks) and (base_toks[i].startswith("[J_") or
                                            base_toks[i].startswith("[S_") or
                                            base_toks[i] == "[Ring]"):
                out.append(base_toks[i]); i += 1
            if this_motif in by_motif:
                for li in by_motif[this_motif]:
                    out.append(f"[J_{li}]")
            mc += 1
        else:
            out.append(t); i += 1
    return out


def encode_full_mol(mol):
    try:
        text = REP.encode(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        return text.split()
    except Exception:
        return None


def cap_attach_with_h(piece_mol):
    """Replace [*] dummy atoms with explicit H, sanitize, RemoveHs.
    Returns (capped_mol, attach_heavy_indices_in_capped).
    Mirrors linker_safegpt_eval encode_piece_with_attach for prep."""
    attach_orig = []
    for a in piece_mol.GetAtoms():
        if a.GetAtomicNum() == 0:
            for n in a.GetNeighbors():
                if n.GetAtomicNum() > 1:
                    attach_orig.append(n.GetIdx()); break
    if not attach_orig: return piece_mol, []
    rw = Chem.RWMol(piece_mol)
    for i, idx in enumerate(attach_orig):
        rw.GetAtomWithIdx(idx).SetAtomMapNum(91 + i)
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetAtomicNum(1); a.SetIsotope(0); a.SetFormalCharge(0); a.SetNoImplicit(False)
    capped_h = rw.GetMol()
    try: Chem.SanitizeMol(capped_h)
    except: return None, []
    try: capped = Chem.RemoveHs(capped_h)
    except: return None, []
    # Find map-num atoms in capped (after H removal indices may have changed)
    attach_capped = [i for i, a in enumerate(capped.GetAtoms()) if 91 <= a.GetAtomMapNum() <= 99]
    for a in capped.GetAtoms(): a.SetAtomMapNum(0)
    return capped, attach_capped


def task_motif_ext(mol, smiles, max_pairs=3):
    """BRICS-cut at 1 bond → smaller fragment = piece. Full drug = completion."""
    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    if not brics_bonds: return []
    if len(brics_bonds) > max_pairs:
        random.shuffle(brics_bonds); brics_bonds = brics_bonds[:max_pairs]
    full_toks = encode_full_mol(mol)
    if full_toks is None: return []
    out = []
    for b in brics_bonds:
        try:
            cut = BRICS.BreakBRICSBonds(mol, [b], sanitize=True)
            frags = list(Chem.GetMolFrags(cut, asMols=True))
            if len(frags) != 2: continue
            # Pick the smaller fragment as "piece"
            small, large = sorted(frags, key=lambda f: f.GetNumHeavyAtoms())[:2]
            n_small_heavy = sum(1 for a in small.GetAtoms() if a.GetAtomicNum() > 1)
            if n_small_heavy < 3 or n_small_heavy > 25: continue
            capped, attach_capped = cap_attach_with_h(small)
            if capped is None or not attach_capped: continue
            piece_toks = encode_with_attach(capped, attach_capped)
            if piece_toks is None: continue
            out.append({"prefix": " ".join(piece_toks), "completion": " ".join(full_toks)})
        except Exception:
            continue
    return out


def task_scaffold_dec(mol, smiles):
    """Murcko scaffold = prefix. Full drug = completion."""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return []
    if scaf is None or scaf.GetNumHeavyAtoms() < 4: return []
    n_scaf = scaf.GetNumHeavyAtoms()
    n_full = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if n_scaf > n_full * 0.85: return []  # too much scaffold, no decoration room
    full_toks = encode_full_mol(mol)
    if full_toks is None: return []
    # For scaffold_dec, the scaffold has no attach markers — model has to
    # connect decoration onto whatever atom. We encode plain (no [J_x]).
    scaf_toks = encode_full_mol(scaf)
    if scaf_toks is None: return []
    return [{"prefix": " ".join(scaf_toks), "completion": " ".join(full_toks)}]


def task_super_gen(mol, smiles, max_pairs=3):
    """Random connected substructure (40-70% atoms) = prefix. Full mol = completion."""
    n_full = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if n_full < 8: return []
    full_toks = encode_full_mol(mol)
    if full_toks is None: return []
    out = []
    for _ in range(max_pairs):
        # Pick a random subset via BFS from a random start atom
        target_n = random.randint(int(n_full * 0.4), int(n_full * 0.7))
        start = random.choice([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1])
        visited = {start}; frontier = [start]
        while frontier and len(visited) < target_n:
            v = frontier.pop(0)
            neighs = [n.GetIdx() for n in mol.GetAtomWithIdx(v).GetNeighbors() if n.GetAtomicNum() > 1]
            random.shuffle(neighs)
            for n in neighs:
                if n not in visited:
                    visited.add(n); frontier.append(n)
                    if len(visited) >= target_n: break
        if len(visited) < 4: continue
        sub = Chem.PathToSubmol(mol, [b.GetIdx() for b in mol.GetBonds()
                                       if b.GetBeginAtomIdx() in visited and b.GetEndAtomIdx() in visited])
        if sub is None or sub.GetNumHeavyAtoms() < 4: continue
        try: Chem.SanitizeMol(sub)
        except:
            try: Chem.SanitizeMol(sub, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except: continue
        sub_toks = encode_full_mol(sub)
        if sub_toks is None: continue
        out.append({"prefix": " ".join(sub_toks), "completion": " ".join(full_toks)})
    return out


def process_drug(args):
    smiles, task, max_pairs = args
    if not smiles: return "[]"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return "[]"
    if task == "motif_ext":
        results = task_motif_ext(mol, smiles, max_pairs=max_pairs)
    elif task == "scaffold_dec":
        results = task_scaffold_dec(mol, smiles)
    elif task == "super_gen":
        results = task_super_gen(mol, smiles, max_pairs=max_pairs)
    else:
        return "[]"
    return json.dumps(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stamp-vocab", required=True)
    ap.add_argument("--task", choices=["motif_ext", "scaffold_dec", "super_gen"], required=True)
    ap.add_argument("--smiles-col", default="std_smiles")
    ap.add_argument("--max-pairs", type=int, default=3)
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=5000)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    pf = pq.ParquetFile(args.inp)
    schema = pf.schema_arrow
    new_field = pa.field("task_pairs", pa.string())
    new_schema = pa.schema(list(schema) + [new_field])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.out, new_schema, compression="snappy")

    t0 = time.time()
    total = 0; with_pairs = 0; total_pairs = 0
    with Pool(args.n_workers, initializer=init_worker, initargs=(args.stamp_vocab,)) as pool:
        for batch in pf.iter_batches(batch_size=args.batch_size):
            df = batch.to_pandas()
            smiles_list = df[args.smiles_col].fillna("").tolist()
            tasks_args = [(s, args.task, args.max_pairs) for s in smiles_list]
            chunksize = max(1, len(tasks_args) // (args.n_workers * 4))
            pairs_list = list(pool.imap(process_drug, tasks_args, chunksize=chunksize))
            df["task_pairs"] = pairs_list
            for s in pairs_list:
                pl = json.loads(s)
                if pl: with_pairs += 1; total_pairs += len(pl)
            total += len(df)
            writer.write_table(pa.Table.from_pandas(df, schema=new_schema, preserve_index=False))
            if total % (args.batch_size * 5) == 0:
                rate = total / max(1, time.time() - t0)
                print(f"  {total:>10,}  with_pairs={with_pairs:>10,} ({with_pairs/total*100:.1f}%)  total_pairs={total_pairs:>10,}  rate={rate:.0f}/s", flush=True)
            if args.max_rows > 0 and total >= args.max_rows: break

    writer.close()
    print(f"\nDone [{args.task}]. {total:,} rows, {with_pairs:,} with pairs ({with_pairs/total*100:.1f}%), {total_pairs:,} total pairs ({time.time()-t0:.0f}s)")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
