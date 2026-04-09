from __future__ import annotations

import csv
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from rdkit import Chem

from ..core.mol_normalize import normalize_mol
from ..core.reconstruction import is_exact_match
from ..core.stamp_codec import STAMPEncoder
from .candidate_generation import generate_candidates
from .stamp_vocab_types import (
    BuildConfig,
    CandidateStat,
    CoreVocabEntry,
    PortInfo,
    SchemaVocabEntry,
)
from .statistics import aggregate_core_stats, score_candidate_stats
from .tree_metrics import contracted_cycle_rank

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    _tqdm = None


_WORKER_ENCODER: Optional[STAMPEncoder] = None
_WORKER_CORE_ID_BY_HASH: Optional[Dict[str, int]] = None
_WORKER_SCHEMA_ID_BY_HASH: Optional[Dict[str, int]] = None


def _maybe_tqdm(
    iterable: Iterable,
    enabled: bool,
    total: Optional[int],
    desc: str,
):
    if enabled and _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc)
    return iterable


class StampVocabBuilder:
    def __init__(self, config: Optional[BuildConfig] = None) -> None:
        self.config = config or BuildConfig()
        self.encoder = STAMPEncoder()

        self.candidate_stats: Dict[Tuple[str, str], CandidateStat] = {}
        self.core_vocab: List[CoreVocabEntry] = []
        self.schema_vocab: List[SchemaVocabEntry] = []
        self.core_schema_map: Dict[str, List[str]] = {}

        self._core_id_by_hash: Dict[str, int] = {}
        self._schema_id_by_hash: Dict[str, int] = {}

    def fit(
        self,
        mols: Iterable[Chem.Mol],
        show_progress: bool = False,
        total: Optional[int] = None,
        progress_desc: str = "Build vocab",
    ) -> "StampVocabBuilder":
        candidate_stats: Dict[Tuple[str, str], CandidateStat] = {}

        for mol in _maybe_tqdm(mols, enabled=show_progress, total=total, desc=progress_desc):
            normalized = normalize_mol(mol)
            candidates, _ = generate_candidates(normalized, self.encoder)

            seen = set()
            for cand in candidates:
                key = (cand.internal_subgraph_hash, cand.schema_hash)
                if key not in candidate_stats:
                    candidate_stats[key] = CandidateStat(
                        core_hash=cand.internal_subgraph_hash,
                        schema_hash=cand.schema_hash,
                        canonical_smarts=cand.canonical_smarts,
                        atom_count=len(cand.atom_indices),
                        bond_count=cand.bond_count,
                        port_definition=cand.ports,
                    )
                st = candidate_stats[key]
                st.occ_count += 1
                st.compression_gain_sum += cand.compression_gain
                st.tree_gain_sum += cand.tree_gain
                st.recoverability_sum += cand.recoverability_score
                st.ambiguity_sum += cand.ambiguity_penalty
                seen.add(key)

            for key in seen:
                candidate_stats[key].mol_coverage += 1

        score_candidate_stats(candidate_stats, self.config)
        self.candidate_stats = candidate_stats
        self._select_vocab()
        return self

    def fit_smiles(
        self,
        smiles_iter: Iterable[str],
        show_progress: bool = False,
        total: Optional[int] = None,
        progress_desc: str = "Build vocab",
        num_workers: int = 1,
        chunksize: int = 64,
    ) -> "StampVocabBuilder":
        candidate_stats: Dict[Tuple[str, str], CandidateStat] = {}

        if num_workers <= 1:
            stream = _maybe_tqdm(
                smiles_iter, enabled=show_progress, total=total, desc=progress_desc
            )
            for smiles in stream:
                entries = _fit_worker(smiles)
                self._merge_fit_entries(candidate_stats, entries)
        else:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_fit_worker,
            ) as pool:
                mapped = pool.map(_fit_worker, smiles_iter, chunksize=chunksize)
                mapped = _maybe_tqdm(
                    mapped, enabled=show_progress, total=total, desc=progress_desc
                )
                for entries in mapped:
                    self._merge_fit_entries(candidate_stats, entries)

        score_candidate_stats(candidate_stats, self.config)
        self.candidate_stats = candidate_stats
        self._select_vocab()
        return self

    @staticmethod
    def _merge_fit_entries(
        candidate_stats: Dict[Tuple[str, str], CandidateStat],
        entries: List[dict],
    ) -> None:
        for entry in entries:
            key = (entry["core_hash"], entry["schema_hash"])
            if key not in candidate_stats:
                candidate_stats[key] = CandidateStat(
                    core_hash=entry["core_hash"],
                    schema_hash=entry["schema_hash"],
                    canonical_smarts=entry["canonical_smarts"],
                    atom_count=entry["atom_count"],
                    bond_count=entry["bond_count"],
                    port_definition=[_port_from_dict(p) for p in entry["port_definition"]],
                )
            st = candidate_stats[key]
            st.occ_count += entry["occ_count"]
            st.mol_coverage += entry["mol_coverage"]
            st.compression_gain_sum += entry["compression_gain_sum"]
            st.tree_gain_sum += entry["tree_gain_sum"]
            st.recoverability_sum += entry["recoverability_sum"]
            st.ambiguity_sum += entry["ambiguity_sum"]

    def _select_vocab(self) -> None:
        core_stats = aggregate_core_stats(self.candidate_stats)

        core_candidates: List[Tuple[str, dict]] = []
        for core_hash, st in core_stats.items():
            avg_score = st["score_sum"] / max(1, st["schema_count"])
            if st["frequency"] < self.config.min_core_freq:
                continue
            if avg_score < self.config.min_core_score:
                continue
            core_candidates.append((core_hash, st))

        core_candidates.sort(
            key=lambda x: (
                -x[1]["frequency"],
                -x[1]["score_sum"] / max(1, x[1]["schema_count"]),
                x[0],
            )
        )
        if self.config.max_vocab_size is not None:
            core_candidates = core_candidates[: self.config.max_vocab_size]

        selected_core_hashes = {core_hash for core_hash, _ in core_candidates}

        # Select schemas under selected cores.
        schema_selected: Dict[str, CandidateStat] = {}
        core_to_schema_freq: Dict[str, Dict[str, int]] = defaultdict(dict)
        for (core_hash, schema_hash), st in self.candidate_stats.items():
            if core_hash not in selected_core_hashes:
                continue
            if st.occ_count < self.config.min_schema_freq:
                continue
            schema_selected[schema_hash] = st
            core_to_schema_freq[core_hash][schema_hash] = st.occ_count

        schema_items = sorted(
            schema_selected.items(),
            key=lambda x: (-x[1].occ_count, x[0]),
        )

        self.schema_vocab = []
        self._schema_id_by_hash = {}
        for schema_id, (schema_hash, st) in enumerate(schema_items):
            entry = SchemaVocabEntry(
                schema_id=schema_id,
                schema_hash=schema_hash,
                num_ports=len(st.port_definition),
                port_definition=st.port_definition,
                frequency=st.occ_count,
            )
            self.schema_vocab.append(entry)
            self._schema_id_by_hash[schema_hash] = schema_id

        # Build core vocab and core->schema map only for selected schemas.
        self.core_vocab = []
        self.core_schema_map = {}
        self._core_id_by_hash = {}

        for core_id, (core_hash, st) in enumerate(core_candidates):
            schema_freq = {
                s_hash: freq
                for s_hash, freq in core_to_schema_freq.get(core_hash, {}).items()
                if s_hash in self._schema_id_by_hash
            }
            if not schema_freq:
                continue

            # Representative candidate stat for atom/bond/core-string fields.
            rep_stat = None
            for (c_hash, s_hash), cand_stat in self.candidate_stats.items():
                if c_hash == core_hash and s_hash in schema_freq:
                    rep_stat = cand_stat
                    break
            if rep_stat is None:
                continue

            entry = CoreVocabEntry(
                core_id=len(self.core_vocab),
                core_hash=core_hash,
                canonical_smarts=rep_stat.canonical_smarts,
                atom_count=rep_stat.atom_count,
                bond_count=rep_stat.bond_count,
                frequency=st["frequency"],
                schemas=schema_freq,
            )
            self.core_vocab.append(entry)
            self._core_id_by_hash[core_hash] = entry.core_id
            self.core_schema_map[core_hash] = sorted(schema_freq.keys())

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        core_payload = [
            {
                "core_id": e.core_id,
                "core_hash": e.core_hash,
                "canonical_smarts": e.canonical_smarts,
                "atom_count": e.atom_count,
                "bond_count": e.bond_count,
                "frequency": e.frequency,
                "schemas": e.schemas,
            }
            for e in self.core_vocab
        ]
        schema_payload = [
            {
                "schema_id": e.schema_id,
                "schema_hash": e.schema_hash,
                "num_ports": e.num_ports,
                "ports": [_port_to_dict(p) for p in e.port_definition],
                "frequency": e.frequency,
            }
            for e in self.schema_vocab
        ]

        (out / "core_vocab.json").write_text(
            json.dumps(core_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "schema_vocab.json").write_text(
            json.dumps(schema_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "core_schema_map.json").write_text(
            json.dumps(self.core_schema_map, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        config_payload = {
            "min_core_freq": self.config.min_core_freq,
            "min_schema_freq": self.config.min_schema_freq,
            "min_core_score": self.config.min_core_score,
            "max_vocab_size": self.config.max_vocab_size,
            "alpha": self.config.alpha,
            "beta": self.config.beta,
            "gamma": self.config.gamma,
            "delta": self.config.delta,
            "eta": self.config.eta,
            "lambda_cov": self.config.lambda_cov,
            "num_cores": len(self.core_vocab),
            "num_schemas": len(self.schema_vocab),
        }
        (out / "vocab_config.json").write_text(
            json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "StampVocabBuilder":
        in_dir = Path(path)
        builder = cls()

        core_payload = json.loads((in_dir / "core_vocab.json").read_text(encoding="utf-8"))
        schema_payload = json.loads((in_dir / "schema_vocab.json").read_text(encoding="utf-8"))
        core_schema_map = json.loads((in_dir / "core_schema_map.json").read_text(encoding="utf-8"))

        builder.core_vocab = [
            CoreVocabEntry(
                core_id=e["core_id"],
                core_hash=e["core_hash"],
                canonical_smarts=e["canonical_smarts"],
                atom_count=e["atom_count"],
                bond_count=e["bond_count"],
                frequency=e["frequency"],
                schemas=e.get("schemas", {}),
            )
            for e in core_payload
        ]
        builder.schema_vocab = [
            SchemaVocabEntry(
                schema_id=e["schema_id"],
                schema_hash=e["schema_hash"],
                num_ports=e["num_ports"],
                port_definition=[_port_from_dict(p) for p in e.get("ports", [])],
                frequency=e["frequency"],
            )
            for e in schema_payload
        ]
        builder.core_schema_map = core_schema_map

        builder._core_id_by_hash = {e.core_hash: e.core_id for e in builder.core_vocab}
        builder._schema_id_by_hash = {e.schema_hash: e.schema_id for e in builder.schema_vocab}
        return builder

    def export_decomposition_records(
        self,
        mols: Iterable[Chem.Mol],
        output_jsonl: str | Path,
        show_progress: bool = False,
        total: Optional[int] = None,
        progress_desc: str = "Export records",
    ) -> None:
        out_path = Path(output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        iterable = _maybe_tqdm(mols, enabled=show_progress, total=total, desc=progress_desc)
        with out_path.open("w", encoding="utf-8") as f:
            for idx, mol in enumerate(iterable):
                normalized = normalize_mol(mol)
                candidates, decomp_metrics = generate_candidates(normalized, self.encoder)

                selected = []
                fallback_atoms = []
                covered = set()

                for cand in candidates:
                    core_id = self._core_id_by_hash.get(cand.internal_subgraph_hash)
                    schema_id = self._schema_id_by_hash.get(cand.schema_hash)
                    if core_id is not None and schema_id is not None:
                        selected.append(
                            {
                                "core_id": core_id,
                                "schema_id": schema_id,
                                "core_hash": cand.internal_subgraph_hash,
                                "schema_hash": cand.schema_hash,
                                "atom_indices": list(cand.atom_indices),
                            }
                        )
                    else:
                        fallback_atoms.extend(cand.atom_indices)

                    covered.update(cand.atom_indices)

                stamp = self.encoder.encode_mol(normalized)
                reconstructed = self.encoder.decode_stamp_to_mol(stamp)

                num_atoms = normalized.GetNumAtoms()
                num_bonds = normalized.GetNumBonds()
                num_components = len(Chem.GetMolFrags(normalized))
                original_cycle_rank = max(0, num_bonds - num_atoms + num_components)
                contracted_rank = contracted_cycle_rank(
                    decomp_metrics["num_fragments"], decomp_metrics["num_edges"], num_components
                )

                record = {
                    "record_id": idx,
                    "canonical_smiles": Chem.MolToSmiles(
                        normalized, canonical=True, isomericSmiles=True
                    ),
                    "selected_motifs": selected,
                    "atom_coverage": len(covered),
                    "fallback_positions": sorted(set(fallback_atoms)),
                    "reconstruction_success": is_exact_match(normalized, reconstructed),
                    "token_count": len(stamp.split()),
                    "tree_complexity": {
                        "num_fragments": decomp_metrics["num_fragments"],
                        "num_cut_bonds": decomp_metrics["num_cut_bonds"],
                        "num_back_edges": contracted_rank,
                        "original_cycle_rank": original_cycle_rank,
                        "contracted_cycle_rank": contracted_rank,
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def export_decomposition_records_smiles(
        self,
        smiles_iter: Iterable[str],
        output_jsonl: str | Path,
        show_progress: bool = False,
        total: Optional[int] = None,
        progress_desc: str = "Export records",
        num_workers: int = 1,
        chunksize: int = 64,
    ) -> None:
        out_path = Path(output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as f:
            if num_workers <= 1:
                _init_export_worker(self._core_id_by_hash, self._schema_id_by_hash)
                stream = _maybe_tqdm(
                    enumerate(smiles_iter),
                    enabled=show_progress,
                    total=total,
                    desc=progress_desc,
                )
                for idx, smiles in stream:
                    record = _export_record_worker((idx, smiles))
                    if record is not None:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    initializer=_init_export_worker,
                    initargs=(self._core_id_by_hash, self._schema_id_by_hash),
                ) as pool:
                    indexed_smiles = enumerate(smiles_iter)
                    mapped = pool.map(_export_record_worker, indexed_smiles, chunksize=chunksize)
                    mapped = _maybe_tqdm(
                        mapped,
                        enabled=show_progress,
                        total=total,
                        desc=progress_desc,
                    )
                    for record in mapped:
                        if record is not None:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _port_to_dict(port: PortInfo) -> dict:
    return {
        "local_atom_idx": port.local_atom_idx,
        "external_bond_type": port.external_bond_type,
        "local_env_code": port.local_env_code,
        "stereo_role": port.stereo_role,
        "port_canonical_rank": port.port_canonical_rank,
    }


def _port_from_dict(data: dict) -> PortInfo:
    return PortInfo(
        local_atom_idx=data["local_atom_idx"],
        external_bond_type=data["external_bond_type"],
        local_env_code=data.get("local_env_code", ""),
        stereo_role=data.get("stereo_role", "NONE"),
        port_canonical_rank=data.get("port_canonical_rank", 0),
    )


def _get_worker_encoder() -> STAMPEncoder:
    global _WORKER_ENCODER
    if _WORKER_ENCODER is None:
        _WORKER_ENCODER = STAMPEncoder()
    return _WORKER_ENCODER


def _init_fit_worker() -> None:
    global _WORKER_ENCODER
    _WORKER_ENCODER = STAMPEncoder()


def _fit_worker(smiles: str) -> List[dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    try:
        normalized = normalize_mol(mol)
    except Exception:
        return []

    candidates, _ = generate_candidates(normalized, _get_worker_encoder())
    per_key: Dict[Tuple[str, str], dict] = {}
    for cand in candidates:
        key = (cand.internal_subgraph_hash, cand.schema_hash)
        if key not in per_key:
            per_key[key] = {
                "core_hash": cand.internal_subgraph_hash,
                "schema_hash": cand.schema_hash,
                "canonical_smarts": cand.canonical_smarts,
                "atom_count": len(cand.atom_indices),
                "bond_count": cand.bond_count,
                "port_definition": [_port_to_dict(p) for p in cand.ports],
                "occ_count": 0,
                "mol_coverage": 1,
                "compression_gain_sum": 0.0,
                "tree_gain_sum": 0.0,
                "recoverability_sum": 0.0,
                "ambiguity_sum": 0.0,
            }
        item = per_key[key]
        item["occ_count"] += 1
        item["compression_gain_sum"] += cand.compression_gain
        item["tree_gain_sum"] += cand.tree_gain
        item["recoverability_sum"] += cand.recoverability_score
        item["ambiguity_sum"] += cand.ambiguity_penalty

    return list(per_key.values())


def _init_export_worker(core_id_by_hash: Dict[str, int], schema_id_by_hash: Dict[str, int]) -> None:
    global _WORKER_ENCODER, _WORKER_CORE_ID_BY_HASH, _WORKER_SCHEMA_ID_BY_HASH
    _WORKER_ENCODER = STAMPEncoder()
    _WORKER_CORE_ID_BY_HASH = dict(core_id_by_hash)
    _WORKER_SCHEMA_ID_BY_HASH = dict(schema_id_by_hash)


def _export_record_worker(payload: Tuple[int, str]) -> Optional[dict]:
    idx, smiles = payload
    if _WORKER_CORE_ID_BY_HASH is None or _WORKER_SCHEMA_ID_BY_HASH is None:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        normalized = normalize_mol(mol)
    except Exception:
        return None

    encoder = _get_worker_encoder()
    candidates, decomp_metrics = generate_candidates(normalized, encoder)

    selected = []
    fallback_atoms = []
    covered = set()

    for cand in candidates:
        core_id = _WORKER_CORE_ID_BY_HASH.get(cand.internal_subgraph_hash)
        schema_id = _WORKER_SCHEMA_ID_BY_HASH.get(cand.schema_hash)
        if core_id is not None and schema_id is not None:
            selected.append(
                {
                    "core_id": core_id,
                    "schema_id": schema_id,
                    "core_hash": cand.internal_subgraph_hash,
                    "schema_hash": cand.schema_hash,
                    "atom_indices": list(cand.atom_indices),
                }
            )
        else:
            fallback_atoms.extend(cand.atom_indices)
        covered.update(cand.atom_indices)

    stamp = encoder.encode_mol(normalized)
    reconstructed = encoder.decode_stamp_to_mol(stamp)

    num_atoms = normalized.GetNumAtoms()
    num_bonds = normalized.GetNumBonds()
    num_components = len(Chem.GetMolFrags(normalized))
    original_cycle_rank = max(0, num_bonds - num_atoms + num_components)
    contracted_rank = contracted_cycle_rank(
        decomp_metrics["num_fragments"], decomp_metrics["num_edges"], num_components
    )

    return {
        "record_id": idx,
        "canonical_smiles": Chem.MolToSmiles(normalized, canonical=True, isomericSmiles=True),
        "selected_motifs": selected,
        "atom_coverage": len(covered),
        "fallback_positions": sorted(set(fallback_atoms)),
        "reconstruction_success": is_exact_match(normalized, reconstructed),
        "token_count": len(stamp.split()),
        "tree_complexity": {
            "num_fragments": decomp_metrics["num_fragments"],
            "num_cut_bonds": decomp_metrics["num_cut_bonds"],
            "num_back_edges": contracted_rank,
            "original_cycle_rank": original_cycle_rank,
            "contracted_cycle_rank": contracted_rank,
        },
    }


def iter_smiles_from_file(
    input_path: str | Path,
    smiles_column: str = "smiles",
    max_mols: Optional[int] = None,
) -> Iterator[str]:
    path = Path(input_path)
    count = 0

    if path.suffix.lower() in {".smi", ".smiles", ".txt"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                smi = line.strip().split()[0] if line.strip() else ""
                if not smi:
                    continue
                yield smi
                count += 1
                if max_mols is not None and count >= max_mols:
                    return
        return

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = (row.get(smiles_column) or "").strip()
            if not smi:
                continue
            yield smi
            count += 1
            if max_mols is not None and count >= max_mols:
                return


def iter_mols_from_smiles_file(
    input_path: str | Path,
    smiles_column: str = "smiles",
    max_mols: Optional[int] = None,
) -> Iterator[Chem.Mol]:
    for smi in iter_smiles_from_file(
        input_path,
        smiles_column=smiles_column,
        max_mols=max_mols,
    ):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        yield mol
