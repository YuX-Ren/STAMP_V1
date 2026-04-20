---
name: mp-persistent-worker-rdkit-bpe
description: |
  Fix extreme slowness in Python multiprocessing BPE-style iterative algorithms over
  RDKit molecules (or any large non-picklable objects). Use when: (1) pool.map over
  molecules takes 30-60s per BPE iteration at 200k+ corpus size, (2) profiling shows
  pickle serialization dominates wall time, (3) iterative map-reduce loops where workers
  need mutable state across iterations. Solution: persistent worker processes with
  command queues — workers hold molecule shards in local memory, only freq dicts (str->int)
  and merge-smi strings are communicated. Achieves near-linear speedup (3.7x on 4 cores)
  and eliminates per-iteration serialization entirely.
author: Claude Code
version: 1.0.0
date: 2026-04-16
---

# Persistent-Worker Pattern for Iterative BPE over RDKit Molecules

## Problem

Python's `multiprocessing.Pool.map(fn, mols)` serializes (pickles) every molecule object
to workers and back on **every iteration** of the BPE loop. For a corpus of 200k molecules
(each containing an RDKit `Chem.Mol` plus subgraph tracking dicts), this produces ~36s of
pure serialization overhead per iteration, making a 500-step BPE run take hours instead
of minutes.

## Context / Trigger Conditions

- Running PS-VAE-style principal subgraph mining (`graph_bpe` / `mine_principal_subgraphs`)
- Corpus size > 10k molecules
- `pool.map(_freq_cnt, mols)` where `_freq_cnt` returns `(freq_dict, mol)` — the mol
  object round-trips through pickle on every call
- tqdm shows 30+ seconds per BPE iteration even on multi-core machines
- The actual computation (canonical SMILES, neighbor enumeration) is fast; the bottleneck
  is data movement

Symptoms:
- `BPE merge: 1%|... [03:43<4:49:19, 35.87s/it]` at 200k mols
- CPU utilization is low despite multiple workers (workers spend most time in pickle)
- Single-process mode is paradoxically faster than multi-process for small corpora

## Solution

Replace `Pool.map` with **persistent worker processes** that hold their molecule shard
in local memory across all BPE iterations. Communication is limited to:

- **Main → Workers**: command strings (`"FREQ"` or `merge_smi` or `None` to stop)
- **Workers → Main**: frequency dicts (`Dict[str, int]`) or ack (`True`)

### Architecture

```
Main Process                     Worker 0          Worker 1         ...
    |                               |                 |
    |-- smiles_shard[0] ----------->|  (init: build MolInSubgraph locally)
    |-- smiles_shard[1] -------------------------------->|
    |                               |                 |
    |== BPE Loop ==                 |                 |
    |-- "FREQ" -------------------->|                 |
    |-- "FREQ" -------------------------------->      |
    |<-- {smi: count, ...} ---------|                 |
    |<-- {smi: count, ...} ----------------------|    |
    |   (aggregate freqs, pick merge_smi)         |
    |-- merge_smi ----------------->|                 |
    |-- merge_smi ------------------------------>     |
    |<-- True ----------------------|                 |
    |<-- True -----------------------------------|    |
    |== End Loop ==                 |                 |
    |-- None ---------------------->|  (exit)         |
    |-- None -------------------------------->  (exit)|
```

### Key implementation details

1. Workers receive **SMILES strings** (not MolInSubgraph objects) at startup and build
   molecules locally — zero RDKit Mol pickling even at init.
2. Use `mp.Process` + `mp.Queue` (not `Pool`) for full control over worker lifecycle.
3. Workers are daemon processes with cleanup in a `finally` block.
4. The same `MolInSubgraph.merge(smi)` mutation happens in each worker's local shard —
   the BPE algorithm is embarrassingly parallel for the freq-counting step and requires
   only a broadcast for the merge step.

### Code pattern

```python
def _bpe_worker_loop(smiles_shard, kekulize, cmd_queue, result_queue):
    # Build molecules locally from SMILES strings — no pickling
    mols = []
    for smi in smiles_shard:
        mol = smi2mol(smi, kekulize=kekulize)
        if mol is not None:
            mols.append(MolInSubgraph(mol, kekulize=kekulize))
    result_queue.put(len(mols))  # signal ready

    while True:
        cmd = cmd_queue.get()
        if cmd is None:
            break
        if cmd == "FREQ":
            freqs = {}
            for mol in mols:
                for smi in mol.get_nei_smis():
                    freqs[smi] = freqs.get(smi, 0) + 1
            result_queue.put(freqs)
        else:  # cmd is a merge_smi string
            for mol in mols:
                mol.merge(cmd)
            result_queue.put(True)
```

## Verification

- Compare vocab output of single-process vs multi-process: `sorted(s1) == sorted(s2)` ✓
- Measure wall time: 2k mols, 30 vocab → 16.7s (1 worker) vs 4.5s (4 workers) = **3.7x speedup**
- At 20k mols, 100 vocab, 16 workers: **16s** (vs estimated 55s+ with old Pool.map approach)
- Roundtrip test unchanged: 99.997% pass rate on zinc250k

## Example

```python
# Old (slow): pickles 200k MolInSubgraph objects per iteration
pool = mp.Pool(num_workers)
for _ in range(add_len):
    res_list = pool.map(_freq_cnt, mols)        # 200k objects pickled TO workers
    freqs, mols = {}, []                         # 200k objects pickled BACK
    for freq, mol in res_list:
        mols.append(mol)
        for k, v in freq.items():
            freqs[k] = freqs.get(k, 0) + v
    merge_smi = max(freqs, key=freqs.get)
    for mol in mols:
        mol.merge(merge_smi)                     # mutation on main process

# New (fast): zero mol pickling after init
cmd_queues = [mp.Queue() for _ in range(n_workers)]
result_queue = mp.Queue()
workers = [mp.Process(target=_bpe_worker_loop,
                      args=(shards[i], kekulize, cmd_queues[i], result_queue))
           for i in range(n_workers)]
for p in workers:
    p.start()
# ... BPE loop sends "FREQ" / merge_smi strings only
```

## Notes

- This pattern applies to ANY iterative algorithm where workers hold mutable state
  across iterations and the per-iteration communication is small (strings, dicts)
  compared to the state (molecules, graphs, large objects).
- The pattern is NOT specific to RDKit — it applies to any unpicklable or expensive-
  to-pickle objects (PyTorch models, large numpy arrays, database connections).
- For PS-VAE specifically: the `MolInSubgraph.dirty` flag ensures that only molecules
  affected by the merge recompute their neighbor candidates in the next iteration,
  providing natural load balancing as later BPE steps affect fewer molecules.
- Queue-based communication adds ~1ms overhead per round-trip, negligible compared
  to the 30s+ pickle overhead it replaces.
