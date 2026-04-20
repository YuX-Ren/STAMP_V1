from __future__ import annotations

import json
import multiprocessing as mp
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem

from ._chem_utils import MAX_VALENCE, cnt_atom, smi2mol

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


def _maybe_tqdm(iterable: Iterable, enabled: bool, total: Optional[int], desc: str):
    if enabled and _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc)
    return iterable


class MolInSubgraph:
    """Port of PS-VAE MolInSubgraph with isomeric canonical SMILES."""

    def __init__(self, mol: Chem.Mol, kekulize: bool = False) -> None:
        self.mol = mol
        self.kekulize = kekulize
        self.subgraphs: Dict[int, Dict[int, str]] = {}
        self.subgraphs_smis: Dict[int, str] = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()
            self.subgraphs[idx] = {idx: symbol}
            self.subgraphs_smis[idx] = symbol
        self.inversed_index: Dict[int, int] = {}
        for aid in range(mol.GetNumAtoms()):
            self.inversed_index[aid] = aid
        self.upid_cnt = mol.GetNumAtoms()
        self.dirty = True
        self.smi2pids: Dict[str, List[Tuple[int, int]]] = {}

    def _get_nei_subgraphs(self) -> Tuple[List[Dict[int, str]], List[Tuple[int, int]]]:
        nei_subgraphs: List[Dict[int, str]] = []
        merge_pids: List[Tuple[int, int]] = []
        for key, subgraph in self.subgraphs.items():
            local_nei_pid = set()
            for aid in subgraph:
                atom = self.mol.GetAtomWithIdx(aid)
                for nei in atom.GetNeighbors():
                    nei_idx = nei.GetIdx()
                    if nei_idx in subgraph or nei_idx > aid:
                        continue
                    local_nei_pid.add(self.inversed_index[nei_idx])
            for nei_pid in local_nei_pid:
                new_subgraph = copy(subgraph)
                new_subgraph.update(self.subgraphs[nei_pid])
                nei_subgraphs.append(new_subgraph)
                merge_pids.append((key, nei_pid))
        return nei_subgraphs, merge_pids

    def get_nei_smis(self) -> List[str]:
        if self.dirty:
            nei_subgraphs, merge_pids = self._get_nei_subgraphs()
            self.smi2pids = {}
            nei_smis: List[str] = []
            for i, subgraph in enumerate(nei_subgraphs):
                atom_ids = sorted(subgraph.keys())
                if len(atom_ids) == 1:
                    smi = self.mol.GetAtomWithIdx(atom_ids[0]).GetSymbol()
                else:
                    smi = Chem.MolFragmentToSmiles(
                        self.mol,
                        atomsToUse=atom_ids,
                        canonical=True,
                        isomericSmiles=True,
                    )
                nei_smis.append(smi)
                self.smi2pids.setdefault(smi, []).append(merge_pids[i])
            self.dirty = False
            return nei_smis
        return list(self.smi2pids.keys())

    def merge(self, smi: str) -> None:
        if self.dirty:
            self.get_nei_smis()
        if smi not in self.smi2pids:
            return
        for pid1, pid2 in self.smi2pids[smi]:
            if pid1 in self.subgraphs and pid2 in self.subgraphs:
                merged = dict(self.subgraphs[pid1])
                merged.update(self.subgraphs[pid2])
                new_pid = self.upid_cnt
                self.subgraphs[new_pid] = merged
                self.subgraphs_smis[new_pid] = smi
                for aid in merged:
                    self.inversed_index[aid] = new_pid
                del self.subgraphs[pid1]
                del self.subgraphs[pid2]
                del self.subgraphs_smis[pid1]
                del self.subgraphs_smis[pid2]
                self.upid_cnt += 1
        self.dirty = True

    def get_smis_subgraphs(self) -> List[Tuple[str, List[int]]]:
        res: List[Tuple[str, List[int]]] = []
        for pid in self.subgraphs_smis:
            smi = self.subgraphs_smis[pid]
            idxs = list(self.subgraphs[pid].keys())
            res.append((smi, idxs))
        return res


def _compute_ring_system_smiles(mol: Chem.Mol) -> Dict[int, str]:
    """Return {atom_idx: ring_system_canonical_smiles} for atoms belonging
    to a multi-atom ring cluster. Ring clusters are the union-find closure
    over all in-ring bonds (fuses shared-atom and shared-edge rings).
    """
    n = mol.GetNumAtoms()
    rparent = list(range(n))

    def rfind(x: int) -> int:
        while rparent[x] != x:
            rparent[x] = rparent[rparent[x]]
            x = rparent[x]
        return x

    def runion(a: int, b: int) -> None:
        ra, rb = rfind(a), rfind(b)
        if ra != rb:
            rparent[ra] = rb

    for bond in mol.GetBonds():
        if bond.IsInRing():
            runion(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    clusters: Dict[int, List[int]] = {}
    for aid in range(n):
        if mol.GetAtomWithIdx(aid).IsInRing():
            clusters.setdefault(rfind(aid), []).append(aid)

    atom_to_smi: Dict[int, str] = {}
    for _, aids in clusters.items():
        if len(aids) < 2:
            continue
        try:
            smi = Chem.MolFragmentToSmiles(
                mol, atomsToUse=aids, canonical=True, isomericSmiles=True,
            )
        except Exception:
            continue
        for a in aids:
            atom_to_smi[a] = smi
    return atom_to_smi


def _pre_merge_bond_constraints(
    work: MolInSubgraph,
    mol: Chem.Mol,
    *,
    allow_ring_single_edges: bool = False,
    blessed_rings: Optional[set] = None,
) -> None:
    """Pre-merge atoms connected by decoder-unsafe bonds (union-find).

    Three modes:
    - ``blessed_rings`` provided: frequency-gated. Ring single bonds are
      pre-merged ONLY if the containing ring system's canonical SMILES is
      in ``blessed_rings``. Uncommon rings stay open so BPE can learn to
      split them into sub-fragments.
    - ``allow_ring_single_edges=True`` and no blessed_rings: all ring
      single bonds left open (aggressive, BPE must rebuild every ring).
    - Default: original STAMP/PS-VAE. All ring bonds + non-single bonds
      pre-merged; greedy BPE only sees exocyclic single-bond merges.
    Non-single bonds are always pre-merged because STAMP cannot
    reconstruct them across motif boundaries.
    """
    n = mol.GetNumAtoms()

    atom_to_ring_smi: Dict[int, str] = {}
    if blessed_rings is not None:
        atom_to_ring_smi = _compute_ring_system_smiles(mol)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if bond.GetBondType() != Chem.BondType.SINGLE:
            union(a, b)
            continue
        if not bond.IsInRing():
            continue  # exocyclic single, never pre-merge
        # In-ring single bond: decide per policy.
        if blessed_rings is not None:
            smi_a = atom_to_ring_smi.get(a)
            if smi_a and smi_a in blessed_rings:
                union(a, b)
            # else: leave open — rare ring, BPE will fragment it
            continue
        if allow_ring_single_edges:
            continue  # aggressive mode: open all ring single bonds
        # Default: union all ring bonds
        union(a, b)

    clusters: Dict[int, List[int]] = {}
    for aid in range(n):
        clusters.setdefault(find(aid), []).append(aid)

    for root, aids in clusters.items():
        if len(aids) <= 1:
            continue
        smi = Chem.MolFragmentToSmiles(
            mol, atomsToUse=aids, canonical=True, isomericSmiles=True,
        )
        first_pid = work.inversed_index[aids[0]]
        for aid in aids[1:]:
            other_pid = work.inversed_index[aid]
            if other_pid == first_pid:
                continue
            merged = dict(work.subgraphs[first_pid])
            merged.update(work.subgraphs[other_pid])
            new_pid = work.upid_cnt
            work.subgraphs[new_pid] = merged
            work.subgraphs_smis[new_pid] = smi
            for a in merged:
                work.inversed_index[a] = new_pid
            del work.subgraphs[first_pid]
            del work.subgraphs[other_pid]
            if first_pid in work.subgraphs_smis:
                del work.subgraphs_smis[first_pid]
            if other_pid in work.subgraphs_smis:
                del work.subgraphs_smis[other_pid]
            work.upid_cnt += 1
            first_pid = new_pid
    work.dirty = True


def _freq_cnt(mol: MolInSubgraph) -> Tuple[Dict[str, int], MolInSubgraph]:
    freqs: Dict[str, int] = {}
    for smi in mol.get_nei_smis():
        freqs[smi] = freqs.get(smi, 0) + 1
    return freqs, mol


def _ring_freq_worker(args: Tuple[List[str], bool]) -> Dict[str, int]:
    """Count distinct ring-system canonical SMILES per molecule on a shard."""
    smis_shard, kekulize = args
    local: Dict[str, int] = {}
    for smi in smis_shard:
        try:
            rd = smi2mol(smi, kekulize=kekulize)
            if rd is None:
                continue
            atom_to_smi = _compute_ring_system_smiles(rd)
            for ring_smi in set(atom_to_smi.values()):
                local[ring_smi] = local.get(ring_smi, 0) + 1
        except Exception:
            pass
    return local


def compute_ring_frequencies(
    smis: List[str],
    kekulize: bool,
    num_workers: int,
) -> Dict[str, int]:
    """Aggregate ring-system SMILES frequencies across a SMILES corpus
    (parallel over ``num_workers``). Returns ring-smi → #mols containing it.
    """
    if num_workers > 1 and len(smis) > 10000:
        shards = [smis[i::num_workers] for i in range(num_workers)]
        args = [(shard, kekulize) for shard in shards]
        with mp.Pool(num_workers) as pool:
            partials = pool.map(_ring_freq_worker, args)
        out: Dict[str, int] = {}
        for p in partials:
            for k, v in p.items():
                out[k] = out.get(k, 0) + v
        return out
    return _ring_freq_worker((smis, kekulize))


def _seed_freq_worker(args: Tuple) -> Dict[str, int]:
    """Count pre-merge seed fragment frequencies on a SMILES shard.

    args tuple: (smis_shard, kekulize, allow_ring_single_edges, blessed_rings_or_None)
    """
    smis_shard, kekulize, allow_ring_single_edges, blessed_rings = args
    local: Dict[str, int] = {}
    for smi in smis_shard:
        try:
            rd = smi2mol(smi, kekulize=kekulize)
            if rd is None:
                continue
            tmp = MolInSubgraph(rd, kekulize=kekulize)
            _pre_merge_bond_constraints(
                tmp,
                rd,
                allow_ring_single_edges=allow_ring_single_edges,
                blessed_rings=blessed_rings,
            )
            for frag_smi in tmp.subgraphs_smis.values():
                local[frag_smi] = local.get(frag_smi, 0) + 1
        except Exception:
            pass
    return local


# ---------------------------------------------------------------------------
# Persistent-worker helpers for parallel BPE mining.
# Workers hold their molecule shard in process-local memory; only freq dicts
# and merge-smi strings are communicated, avoiding per-iteration pickling of
# MolInSubgraph objects.
# ---------------------------------------------------------------------------

def _bpe_worker_loop(
    smiles_shard: List[str],
    kekulize: bool,
    allow_ring_single_edges: bool,
    blessed_rings: Optional[set],
    cmd_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    mols: List[MolInSubgraph] = []
    for smi in smiles_shard:
        try:
            rd = smi2mol(smi, kekulize=kekulize)
            if rd is not None:
                work = MolInSubgraph(rd, kekulize=kekulize)
                _pre_merge_bond_constraints(
                    work,
                    rd,
                    allow_ring_single_edges=allow_ring_single_edges,
                    blessed_rings=blessed_rings,
                )
                mols.append(work)
        except Exception:
            pass
    result_queue.put(len(mols))

    # Incremental freq tracking: maintain a running total and only
    # recompute for molecules whose subgraph state changed (dirty).
    cached_freqs: Dict[str, int] = {}
    per_mol_freqs: List[Dict[str, int]] = [{} for _ in mols]

    while True:
        cmd = cmd_queue.get()
        if cmd is None:
            break
        if cmd == "FREQ":
            for i, mol in enumerate(mols):
                if not mol.dirty and per_mol_freqs[i]:
                    continue
                old = per_mol_freqs[i]
                for k, v in old.items():
                    cached_freqs[k] = cached_freqs.get(k, 0) - v
                new_f: Dict[str, int] = {}
                for smi in mol.get_nei_smis():
                    new_f[smi] = new_f.get(smi, 0) + 1
                per_mol_freqs[i] = new_f
                for k, v in new_f.items():
                    cached_freqs[k] = cached_freqs.get(k, 0) + v
            result_queue.put(dict(cached_freqs))
        else:
            for mol in mols:
                mol.merge(cmd)
            result_queue.put(True)


@dataclass
class PSVAEMineConfig:
    vocab_size: int = 500
    kekulize: bool = True
    num_workers: int = 1
    show_progress: bool = True
    max_mining_mols: int = 0
    allow_ring_single_edges: bool = False
    # Keep the default mining behavior frequency-based. This is not a closure
    # mechanism; forcing all observed seeds into vocab simply inflates the
    # vocabulary and is distinct from CamS-style SAVC.
    min_seed_freq: int = 2
    # Frequency-gated ring pre-merge. When > 0 and allow_ring_single_edges=True,
    # ring systems occurring in >= this many molecules are pre-merged as whole
    # seeds (benzene, pyridine, ...), while rarer rings are left open for BPE
    # to decompose along their single bonds. 0 disables gating — flag becomes
    # the old "open all ring single bonds" behavior.
    ring_blessed_freq_threshold: int = 0
    # Early-stop BPE when the best merge candidate's frequency falls below
    # this threshold. 0 = disabled (run to vocab_size). Useful to cap
    # long-tail rare merges without training for hours.
    min_bpe_merge_freq: int = 0
    # Optional preloaded set of ring-system canonical SMILES to use directly
    # as blessed rings, bypassing the auto-frequency computation. When set
    # alongside allow_ring_single_edges=True, pre_merge treats these rings
    # as whole seeds and opens all other ring single bonds for BPE.
    blessed_rings_preloaded: Optional[frozenset] = None


def mine_principal_subgraphs(
    smiles_iter: Iterable[str],
    config: PSVAEMineConfig,
    output_path: str | Path,
) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    """
    PS-VAE principal subgraph mining with BPE-style merges.

    Uses isomericSmiles=True for canonicalization so stereo information is
    preserved as distinct motifs.

    When ``num_workers > 1``, molecules are sharded across persistent worker
    processes.  Only frequency dicts (str→int) and merge-smi strings are
    communicated per iteration — MolInSubgraph objects are never pickled,
    which is the key to scaling to large corpora (200k+ molecules).
    """
    import random as _rng

    smis = [s.strip() for s in smiles_iter if s and s.strip()]
    if config.max_mining_mols > 0 and len(smis) > config.max_mining_mols:
        _rng.seed(42)
        smis = _rng.sample(smis, config.max_mining_mols)
        if config.show_progress:
            print(f"[mine] subsampled {len(smis)} mols for BPE mining", flush=True)

    # Optionally build the blessed-rings set first, so common rings get
    # pre-merged as whole seeds while rare rings are opened for BPE.
    # Priority: explicit preloaded set > frequency-gated auto-compute.
    blessed_rings: Optional[set] = None
    if config.allow_ring_single_edges and getattr(config, "blessed_rings_preloaded", None):
        blessed_rings = set(config.blessed_rings_preloaded)
        if config.show_progress:
            print(
                f"[ring] using {len(blessed_rings)} preloaded blessed rings "
                f"(frequency precompute skipped)",
                flush=True,
            )
    elif config.allow_ring_single_edges and config.ring_blessed_freq_threshold > 0:
        if config.show_progress:
            print(
                f"[ring] computing ring-system freqs on {len(smis)} mols (parallel)",
                flush=True,
            )
        ring_freqs = compute_ring_frequencies(
            smis, config.kekulize, max(1, config.num_workers)
        )
        blessed_rings = {
            smi for smi, fr in ring_freqs.items()
            if fr >= config.ring_blessed_freq_threshold
        }
        if config.show_progress:
            print(
                f"[ring] {len(blessed_rings)}/{len(ring_freqs)} ring systems "
                f"passed freq ≥ {config.ring_blessed_freq_threshold} "
                f"(rest will be opened for BPE)",
                flush=True,
            )

    # Collect seed fragments from the configured pre-merge regime.
    # Parallelised across ``num_workers`` so full-corpus (ChEMBL-scale)
    # mining is not bottlenecked by serial RDKit parsing.
    details: Dict[str, Tuple[int, int]] = {}
    if config.num_workers > 1 and len(smis) > 10000:
        shards = [smis[i::config.num_workers] for i in range(config.num_workers)]
        args = [
            (shard, config.kekulize, config.allow_ring_single_edges, blessed_rings)
            for shard in shards
        ]
        with mp.Pool(config.num_workers) as pool:
            partials = pool.map(_seed_freq_worker, args)
        seed_frag_freq: Dict[str, int] = {}
        for p in partials:
            for k, v in p.items():
                seed_frag_freq[k] = seed_frag_freq.get(k, 0) + v
    else:
        seed_frag_freq = {}
        for smi_str in smis:
            try:
                rd = smi2mol(smi_str, kekulize=config.kekulize)
                if rd is None:
                    continue
                tmp = MolInSubgraph(rd, kekulize=config.kekulize)
                _pre_merge_bond_constraints(
                    tmp,
                    rd,
                    allow_ring_single_edges=config.allow_ring_single_edges,
                    blessed_rings=blessed_rings,
                )
                for frag_smi in tmp.subgraphs_smis.values():
                    seed_frag_freq[frag_smi] = seed_frag_freq.get(frag_smi, 0) + 1
            except Exception:
                continue

    selected_smis: List[str] = []
    for frag_smi, freq in sorted(seed_frag_freq.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq < config.min_seed_freq:
            continue
        selected_smis.append(frag_smi)
        details[frag_smi] = (cnt_atom(frag_smi), freq)

    if config.show_progress:
        total_seeds = len(seed_frag_freq)
        print(
            f"[mine] {len(selected_smis)}/{total_seeds} seed fragments pass min_seed_freq={config.min_seed_freq}",
            flush=True,
        )

    seed_count = len(selected_smis)
    add_len = max(0, config.vocab_size - len(selected_smis))

    if config.num_workers > 1:
        selected_smis, details = _mine_parallel(
            smis, config, details, selected_smis, add_len, blessed_rings
        )
    else:
        selected_smis, details = _mine_single(
            smis, config, details, selected_smis, add_len, blessed_rings
        )

    # Preserve insertion order before the frequency sort so downstream
    # tooling can prefix-slice smaller vocabs from a larger mining run
    # (seeds in seed-phase order, then BPE merges in greedy merge order).
    merge_history = list(selected_smis)

    selected_smis.sort(key=lambda x: (-details[x][1], -details[x][0], x))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    branch_mode = "ring_single_edges" if config.allow_ring_single_edges else "legacy_tree"
    with output_path.open("w", encoding="utf-8") as fout:
        fout.write(
            json.dumps(
                {
                    "branch_mode": branch_mode,
                    "kekulize": config.kekulize,
                    "isomeric": True,
                    "allow_ring_single_edges": config.allow_ring_single_edges,
                }
            )
            + "\n"
        )
        for smi in selected_smis:
            atom_num, freq = details[smi]
            fout.write(f"{smi}\t{atom_num}\t{freq}\n")

    history_path = output_path.with_suffix(output_path.suffix + ".history.json")
    with history_path.open("w", encoding="utf-8") as fout:
        json.dump(
            {
                "branch_mode": branch_mode,
                "kekulize": config.kekulize,
                "isomeric": True,
                "allow_ring_single_edges": config.allow_ring_single_edges,
                "seed_count": seed_count,
                "min_seed_freq": config.min_seed_freq,
                "ring_blessed_freq_threshold": config.ring_blessed_freq_threshold,
                "merge_order": merge_history,
            },
            fout,
            ensure_ascii=False,
        )

    # Persist blessed_rings alongside the vocab so downstream
    # PrincipalSubgraphTokenizer can reproduce the same pre-merge behavior.
    if blessed_rings is not None:
        bpath = output_path.with_suffix(output_path.suffix + ".blessed_rings.json")
        with bpath.open("w", encoding="utf-8") as fout:
            json.dump(
                {
                    "ring_blessed_freq_threshold": config.ring_blessed_freq_threshold,
                    "blessed_rings": sorted(blessed_rings),
                },
                fout,
                ensure_ascii=False,
            )

    return selected_smis, details


def _mine_single(
    smis: List[str],
    config: PSVAEMineConfig,
    details: Dict[str, Tuple[int, int]],
    selected_smis: List[str],
    add_len: int,
    blessed_rings: Optional[set] = None,
) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    mols: List[MolInSubgraph] = []
    for smi in _maybe_tqdm(smis, enabled=config.show_progress, total=len(smis), desc="Init mols"):
        try:
            rd = smi2mol(smi, kekulize=config.kekulize)
            if rd is None:
                continue
            work = MolInSubgraph(rd, kekulize=config.kekulize)
            _pre_merge_bond_constraints(
                work,
                rd,
                allow_ring_single_edges=config.allow_ring_single_edges,
                blessed_rings=blessed_rings,
            )
            mols.append(work)
        except Exception:
            continue

    pbar = _maybe_tqdm(range(add_len), enabled=config.show_progress, total=add_len, desc="BPE merge")
    pbar_iter = iter(pbar)
    cached_freqs: Dict[str, int] = {}
    per_mol_freqs: List[Dict[str, int]] = [{} for _ in mols]
    for _ in range(add_len):
        for i, mol in enumerate(mols):
            if not mol.dirty and per_mol_freqs[i]:
                continue
            old = per_mol_freqs[i]
            for k, v in old.items():
                cached_freqs[k] = cached_freqs.get(k, 0) - v
            new_f: Dict[str, int] = {}
            for smi in mol.get_nei_smis():
                new_f[smi] = new_f.get(smi, 0) + 1
            per_mol_freqs[i] = new_f
            for k, v in new_f.items():
                cached_freqs[k] = cached_freqs.get(k, 0) + v

        freqs = {k: v for k, v in cached_freqs.items() if v > 0}
        if not freqs:
            break
        merge_smi, max_cnt = max(freqs.items(), key=lambda kv: kv[1])
        if max_cnt <= 0:
            break
        if config.min_bpe_merge_freq > 0 and max_cnt < config.min_bpe_merge_freq:
            if config.show_progress:
                print(
                    f"[mine] early-stop: best merge freq {max_cnt} < "
                    f"min_bpe_merge_freq={config.min_bpe_merge_freq}",
                    flush=True,
                )
            break

        for mol in mols:
            mol.merge(merge_smi)

        if merge_smi not in details:
            selected_smis.append(merge_smi)
            details[merge_smi] = (cnt_atom(merge_smi), max_cnt)
        try:
            next(pbar_iter)
        except StopIteration:
            pass

    return selected_smis, details


def _mine_parallel(
    smis: List[str],
    config: PSVAEMineConfig,
    details: Dict[str, Tuple[int, int]],
    selected_smis: List[str],
    add_len: int,
    blessed_rings: Optional[set] = None,
) -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    n_workers = config.num_workers
    shard_size = (len(smis) + n_workers - 1) // n_workers
    shards = [smis[i * shard_size : (i + 1) * shard_size] for i in range(n_workers)]

    cmd_queues: List[mp.Queue] = [mp.Queue() for _ in range(n_workers)]
    result_queue: mp.Queue = mp.Queue()

    workers: List[mp.Process] = []
    for i in range(n_workers):
        p = mp.Process(
            target=_bpe_worker_loop,
            args=(
                shards[i],
                config.kekulize,
                config.allow_ring_single_edges,
                blessed_rings,
                cmd_queues[i],
                result_queue,
            ),
            daemon=True,
        )
        p.start()
        workers.append(p)

    total_mols = 0
    for _ in range(n_workers):
        total_mols += result_queue.get()
    if config.show_progress:
        print(f"[mine] {total_mols} mols across {n_workers} workers", flush=True)

    pbar = _maybe_tqdm(range(add_len), enabled=config.show_progress, total=add_len, desc="BPE merge")
    pbar_iter = iter(pbar)

    try:
        for _ in range(add_len):
            for q in cmd_queues:
                q.put("FREQ")
            freqs: Dict[str, int] = {}
            for _ in range(n_workers):
                worker_freq = result_queue.get()
                for key, value in worker_freq.items():
                    freqs[key] = freqs.get(key, 0) + value

            if not freqs:
                break
            merge_smi, max_cnt = max(freqs.items(), key=lambda kv: kv[1])
            if max_cnt <= 0:
                break
            if config.min_bpe_merge_freq > 0 and max_cnt < config.min_bpe_merge_freq:
                if config.show_progress:
                    print(
                        f"[mine] early-stop: best merge freq {max_cnt} < "
                        f"min_bpe_merge_freq={config.min_bpe_merge_freq}",
                        flush=True,
                    )
                break

            for q in cmd_queues:
                q.put(merge_smi)
            for _ in range(n_workers):
                result_queue.get()

            if merge_smi not in details:
                selected_smis.append(merge_smi)
                details[merge_smi] = (cnt_atom(merge_smi), max_cnt)
            try:
                next(pbar_iter)
            except StopIteration:
                pass
    finally:
        for q in cmd_queues:
            q.put(None)
        for p in workers:
            p.join(timeout=10)

    return selected_smis, details


class PrincipalSubgraphTokenizer:
    """
    Greedy PS-VAE tokenizer. Decomposes a molecule into principal subgraphs
    by repeatedly merging the neighboring candidate with the highest vocab
    frequency until no candidate is in-vocab.
    """

    def __init__(self, vocab_path: str | Path) -> None:
        path = Path(vocab_path)
        with path.open("r", encoding="utf-8") as fin:
            lines = fin.read().strip().split("\n")
        header = json.loads(lines[0])
        self.branch_mode: str = str(header.get("branch_mode", "legacy_tree"))
        self.kekulize: bool = bool(header.get("kekulize", True))
        self.isomeric: bool = bool(header.get("isomeric", True))
        self.allow_ring_single_edges: bool = bool(header.get("allow_ring_single_edges", False))

        self.vocab: Dict[str, Tuple[int, int]] = {}
        self.idx2smi: List[str] = []
        self.smi2idx: Dict[str, int] = {}
        for line in lines[1:]:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            smi, atom_num, freq = parts
            self.vocab[smi] = (int(atom_num), int(freq))
            self.smi2idx[smi] = len(self.idx2smi)
            self.idx2smi.append(smi)

        # Optional frequency-gated blessed rings sibling file: if present,
        # pre-merge only those rings as seeds; rarer rings stay open (to be
        # decomposed along their single bonds during tokenization).
        self.blessed_rings: Optional[set] = None
        bpath = path.with_suffix(path.suffix + ".blessed_rings.json")
        if bpath.exists():
            try:
                data = json.loads(bpath.read_text(encoding="utf-8"))
                self.blessed_rings = set(data.get("blessed_rings", []))
            except Exception:
                self.blessed_rings = None

    def __len__(self) -> int:
        return len(self.idx2smi)

    def _greedy_decompose_work(self, mol: Chem.Mol) -> MolInSubgraph:
        if mol is None:
            raise ValueError("Input mol is None")
        if len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError("PrincipalSubgraphTokenizer requires a connected molecule.")

        work = MolInSubgraph(mol, kekulize=self.kekulize)
        _pre_merge_bond_constraints(
            work,
            mol,
            allow_ring_single_edges=self.allow_ring_single_edges,
            blessed_rings=self.blessed_rings,
        )

        while True:
            nei_smis = work.get_nei_smis()
            best_smi: Optional[str] = None
            best_freq = -1
            for smi in nei_smis:
                if smi not in self.vocab:
                    continue
                freq = self.vocab[smi][1]
                if freq > best_freq:
                    best_freq = freq
                    best_smi = smi
            if best_smi is None:
                break
            work.merge(best_smi)
        return work

    def decompose_with_smis(
        self,
        mol: Chem.Mol,
        *,
        strict_vocab: bool = True,
    ) -> List[Tuple[str, List[int]]]:
        """
        Returns: ``[(canonical_fragment_smiles, atom_indices), ...]`` for the
        tokenizer's final principal subgraphs.

        Before the greedy BPE merge, decoder-unsafe bonds are pre-merged into
        irreducible groups. In legacy mode this includes ring bonds, so the
        reduced graph stays tree-structured. In ring-aware mode only non-single
        bonds are pre-merged, allowing ring single bonds to survive as
        boundaries that STAMP later serializes with ``[Ring]`` closures.

        When ``strict_vocab=True``, every final group must already be in the
        provided vocab. This prevents silently emitting a brand-new motif token
        for an irreducible fragment that the tokenizer could not reduce any
        further.
        """
        work = self._greedy_decompose_work(mol)
        groups = [(smi, sorted(idxs)) for smi, idxs in work.get_smis_subgraphs()]
        if strict_vocab:
            oov = sorted({smi for smi, _ in groups if smi not in self.vocab})
            if oov:
                preview = ", ".join(oov[:5])
                if len(oov) > 5:
                    preview += ", ..."
                raise ValueError(
                    "PrincipalSubgraphTokenizer produced irreducible fragments outside vocab: "
                    f"{preview}. Under the current STAMP grammar, these seeds cannot be "
                    "recovered by single-atom fallback alone because motif boundaries only "
                    "support tree-structured single-bond attachments."
                )
        return groups

    def decompose(
        self,
        mol: Chem.Mol,
        *,
        strict_vocab: bool = True,
    ) -> List[List[int]]:
        groups = self.decompose_with_smis(mol, strict_vocab=strict_vocab)
        return [idxs for _, idxs in groups]

    def decompose_smiles(
        self,
        smiles: str,
        *,
        strict_vocab: bool = True,
    ) -> List[List[int]]:
        mol = smi2mol(smiles, kekulize=self.kekulize)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return self.decompose(mol, strict_vocab=strict_vocab)

    def prepare_mol(self, smiles: str) -> Chem.Mol:
        """Convenience: build an RDKit mol in the same kekulize mode as the vocab."""
        mol = smi2mol(smiles, kekulize=self.kekulize)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return mol
