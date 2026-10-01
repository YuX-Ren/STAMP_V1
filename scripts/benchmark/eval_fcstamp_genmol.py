#!/usr/bin/env python3
"""Evaluate a FC-STAMP FIM model on GenMol fragment prompts.

The model receives:

    [TASK] left [INFILL] right [GEN]

It generates a missing STAMP span. The evaluator splices that span back into
the masked prompt, decodes the resulting standard STAMP sequence, and reports
GenMol-style validity/uniqueness/quality/diversity/distance metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, QED

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stamp.benchmark.common import canonicalize_smiles  # noqa: E402
from stamp.benchmark.hybrid_vocab import HybridVocab, is_fcstamp_control_token, is_stamp_structural  # noqa: E402
from stamp.benchmark.lm import LMConfig, LlamaDecoderLM, TinyDecoderLM, TokenVocab, _resolve_torch_device, _sample_next_tokens  # noqa: E402
from stamp.benchmark.metrics import _internal_diversity, _load_sa_scorer  # noqa: E402
from stamp.benchmark.representations import build_representation  # noqa: E402


GENMOL_BASELINES: dict[str, dict[str, dict[str, float]]] = {
    "GenMol": {
        "linker_design": {"validity": 100.0, "uniqueness": 83.7, "quality": 21.9, "diversity": 0.547, "distance": 0.563},
        "motif_extension": {"validity": 82.9, "uniqueness": 77.5, "quality": 30.1, "diversity": 0.617, "distance": 0.682},
        "scaffold_decoration": {"validity": 96.6, "uniqueness": 82.7, "quality": 31.8, "diversity": 0.591, "distance": 0.651},
        "superstructure_generation": {"validity": 97.5, "uniqueness": 83.6, "quality": 34.8, "diversity": 0.599, "distance": 0.762},
    },
    "GenMol V2": {
        "linker_design": {"validity": 100.0, "uniqueness": 76.6, "quality": 18.4, "diversity": 0.512, "distance": 0.539},
        "motif_extension": {"validity": 99.4, "uniqueness": 84.5, "quality": 49.0, "diversity": 0.626, "distance": 0.659},
        "scaffold_decoration": {"validity": 99.2, "uniqueness": 90.5, "quality": 39.7, "diversity": 0.571, "distance": 0.604},
        "superstructure_generation": {"validity": 99.7, "uniqueness": 89.8, "quality": 39.0, "diversity": 0.551, "distance": 0.769},
    },
}


def _load_vocab(path: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "atomic_motifs" in payload and "motif_expansion" in payload:
        return HybridVocab.from_dict(payload)
    return TokenVocab.from_dict(payload)


def _load_model(model_dir: str | Path, device: str) -> tuple[torch.nn.Module, Any, Any, torch.device]:
    model_dir = Path(model_dir)
    ckpt = torch.load(model_dir / "model.pt", map_location="cpu")
    cfg = LMConfig(**ckpt["cfg"])
    model_type = str(ckpt.get("model_type", cfg.model_type)).strip().lower()
    cfg.model_type = model_type
    vocab = _load_vocab(model_dir / "vocab.json")
    if model_type == "llama":
        model = LlamaDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
    else:
        model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
    state = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    dev = _resolve_torch_device(device)
    model.to(dev)
    model.eval()

    stamp_vocab_file = ckpt.get("stamp_vocab")
    stamp_vocab_path = None
    if stamp_vocab_file:
        candidate = model_dir / str(stamp_vocab_file)
        if candidate.exists():
            stamp_vocab_path = str(candidate)
    rep = build_representation(str(ckpt.get("representation", "stamp")), stamp_vocab=stamp_vocab_path)
    return model, vocab, rep, dev


def _load_task_sampling_config(value: str | None) -> dict[str, dict[str, float]]:
    if not value:
        return {}
    text = str(value).strip()
    path = Path(text)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--task-sampling-json must be a JSON object or a path to one")
    out: dict[str, dict[str, float]] = {}
    for task, cfg in payload.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"Task sampling config for {task!r} must be an object")
        vals: dict[str, float] = {}
        for key in ("temperature", "top_p", "top_k", "diverse_prefix_penalty"):
            if key in cfg:
                vals[key] = float(cfg[key])
        out[str(task)] = vals
    return out


def _encode_tokens(vocab: Any, tokens: Sequence[str]) -> list[int]:
    if hasattr(vocab, "encode_stream"):
        return list(vocab.encode_stream(tokens, training=False))
    return list(vocab.encode_tokens(tokens))


def _decode_ids(vocab: Any, ids: Sequence[int]) -> list[str]:
    out: list[str] = []
    for idx in ids:
        i = int(idx)
        out.append(vocab.itos[i] if 0 <= i < len(vocab.itos) else "<unk>")
    return out


def _is_stereo_token(tok: str) -> bool:
    return str(tok).startswith("[S_") and str(tok).endswith("]")


def _strip_stereo_tokens(tokens: Sequence[str]) -> list[str]:
    return [str(tok) for tok in tokens if not _is_stereo_token(str(tok))]


def _sample_generated_tokens(
    model: torch.nn.Module,
    vocab: Any,
    prompt_tokens: Sequence[str],
    *,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    diverse_prefix_penalty: float,
    device: torch.device,
) -> list[list[str]]:
    prompt_ids = _encode_tokens(vocab, prompt_tokens)
    prefix = [int(vocab.bos_id)] + prompt_ids
    ids = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0).repeat(int(n_samples), 1)
    finished = torch.zeros(int(n_samples), dtype=torch.bool, device=device)
    cfg = getattr(model, "cfg", None)
    max_seq_len = int(getattr(cfg, "max_seq_len", 256))
    model_type = str(getattr(cfg, "model_type", "decoder")).strip().lower()
    use_ao_orders = model_type in {"ao_gpt", "ao_gpt_pp"} and hasattr(model, "target_pos_emb")
    generated_prefixes: list[tuple[int, ...]] = [tuple() for _ in range(int(n_samples))]

    for _ in range(int(max_new_tokens)):
        ctx = ids[:, -max_seq_len:]
        with torch.no_grad():
            if use_ao_orders:
                seq_len = ctx.size(1)
                orders = torch.arange(seq_len, device=device).unsqueeze(0).expand(ctx.size(0), -1)
                logits = model(ctx, orders=orders)[:, -1, :]
            else:
                logits = model(ctx)[:, -1, :]
        for suppressed_id in (vocab.pad_id, vocab.bos_id, getattr(vocab, "mask_id", -1), vocab.unk_id):
            if 0 <= int(suppressed_id) < logits.size(-1):
                logits[:, int(suppressed_id)] = float("-inf")
        if float(diverse_prefix_penalty) > 0:
            sampled: list[torch.Tensor] = []
            seen_by_prefix: dict[tuple[int, ...], set[int]] = {}
            for row_idx in range(logits.size(0)):
                row_logits = logits[row_idx : row_idx + 1].clone()
                if not bool(finished[row_idx].item()):
                    prefix_key = generated_prefixes[row_idx]
                    seen = seen_by_prefix.setdefault(prefix_key, set())
                    if seen:
                        row_logits[0, list(seen)] -= float(diverse_prefix_penalty)
                    row_next = _sample_next_tokens(
                        row_logits,
                        temperature=float(temperature),
                        top_k=int(top_k),
                        top_p=float(top_p),
                    )
                    seen.add(int(row_next.item()))
                else:
                    row_next = torch.tensor([int(vocab.eos_id)], dtype=torch.long, device=device)
                sampled.append(row_next)
            next_ids = torch.cat(sampled, dim=0)
        else:
            next_ids = _sample_next_tokens(
                logits,
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
            )
        eos_fill = torch.full_like(next_ids, int(vocab.eos_id))
        next_ids = torch.where(finished, eos_fill, next_ids)
        ids = torch.cat([ids, next_ids.unsqueeze(1)], dim=1)
        for row_idx, token_id in enumerate(next_ids.detach().cpu().tolist()):
            if not bool(finished[row_idx].item()) and int(token_id) != int(vocab.eos_id):
                generated_prefixes[row_idx] = generated_prefixes[row_idx] + (int(token_id),)
        finished = finished | (next_ids == int(vocab.eos_id))
        if bool(torch.all(finished).item()):
            break

    gen_ids = ids[:, len(prefix):].detach().cpu().tolist()
    out: list[list[str]] = []
    for row in gen_ids:
        if int(vocab.eos_id) in row:
            row = row[: row.index(int(vocab.eos_id))]
        out.append(_decode_ids(vocab, row))
    return out


def _tokens_to_stamp_span(tokens: Sequence[str]) -> list[str]:
    from stamp.benchmark.hybrid_vocab import BOS, EOS, MASK, PAD, UNK, is_condition_token

    special = {PAD, BOS, EOS, MASK, UNK, "<bos>", "<eos>", "<pad>", "<mask>", "<unk>"}
    out: list[str] = []
    buf: list[str] = []
    for tok in tokens:
        if tok == "[EOS]":
            break
        if tok == "[EMPTY]":
            continue
        if tok in special or is_condition_token(tok) or is_fcstamp_control_token(tok):
            if buf:
                out.append("".join(buf))
                buf = []
            continue
        if is_stamp_structural(tok):
            if buf:
                out.append("".join(buf))
                buf = []
            out.append(tok)
        else:
            buf.append(tok)
    if buf:
        out.append("".join(buf))
    return out


def _strip_after_control_eos(tokens: Sequence[str]) -> list[str]:
    vals = list(tokens)
    return vals[: vals.index("[EOS]")] if "[EOS]" in vals else vals


def _parse_fills(tokens: Sequence[str], holes: Sequence[dict[str, Any]]) -> Optional[dict[str, list[str]]]:
    body = _strip_after_control_eos(tokens)
    if len(holes) == 1:
        return {str(holes[0]["infill_token"]): _tokens_to_stamp_span(body)}

    fills: dict[int, list[str]] = {}
    cur_idx: Optional[int] = None
    cur: list[str] = []
    for tok in body:
        if tok.startswith("[FILL_") and tok.endswith("]"):
            if cur_idx is not None:
                fills[cur_idx] = _tokens_to_stamp_span(cur)
            try:
                cur_idx = int(tok[len("[FILL_"):-1])
            except ValueError:
                cur_idx = None
            cur = []
            continue
        if tok == "[/FILL]":
            if cur_idx is not None:
                fills[cur_idx] = _tokens_to_stamp_span(cur)
            cur_idx = None
            cur = []
            continue
        if cur_idx is not None:
            cur.append(tok)
    if cur_idx is not None:
        fills[cur_idx] = _tokens_to_stamp_span(cur)

    out: dict[str, list[str]] = {}
    for hole in holes:
        idx = int(hole["hole_index"])
        if idx not in fills:
            return None
        out[str(hole["infill_token"])] = fills[idx]
    return out


def _prompt_body(prompt_stamp: str) -> list[str]:
    toks = str(prompt_stamp).split()
    if toks and toks[0].startswith("[TASK_"):
        toks = toks[1:]
    if "[GEN]" in toks:
        toks = toks[: toks.index("[GEN]")]
    return toks


def _splice(prompt_stamp: str, fills: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for tok in _prompt_body(prompt_stamp):
        if tok in fills:
            out.extend(fills[tok])
        else:
            out.append(tok)
    return out


def _joint_specs(tokens: Sequence[str]) -> list[tuple[int, int, int]]:
    specs: list[tuple[int, int, int]] = []
    vals = [str(tok) for tok in tokens]
    i = 0
    while i < len(vals):
        tok = vals[i]
        if not (tok.startswith("[J_") and tok.endswith("]")):
            i += 1
            continue
        try:
            local = int(tok[3:-1])
        except ValueError:
            i += 1
            continue
        start = i
        i += 1
        if i < len(vals) and vals[i].startswith("[S_"):
            i += 1
        if i < len(vals) and vals[i] == "[Ring]":
            i += 1
        specs.append((start, i, local))
    return specs


def _stamp_group_ranges(tokens: Sequence[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    pending_start: Optional[int] = None
    vals = [str(tok) for tok in tokens]
    i = 0
    while i < len(vals):
        tok = vals[i]
        if tok.startswith("[INFILL"):
            i += 1
            continue
        if tok.startswith("[B_"):
            pending_start = i if pending_start is None else pending_start
            i += 1
            continue
        if tok.startswith("[J_") or tok.startswith("[S_") or tok in {"[BACK]", "[Ring]"}:
            i += 1
            continue
        start = i if pending_start is None else pending_start
        pending_start = None
        i += 1
        while i < len(vals) and (
            vals[i].startswith("[J_") or vals[i].startswith("[S_") or vals[i] == "[Ring]"
        ):
            i += 1
        ranges.append((start, i))
    return ranges


def _hole_prompt_order(prompt_stamp: str, holes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    body = _prompt_body(prompt_stamp)
    by_token = {str(hole["infill_token"]): dict(hole) for hole in holes}
    ordered: list[dict[str, Any]] = []
    for tok in body:
        if tok in by_token:
            ordered.append(by_token[tok])
    seen = {str(hole["infill_token"]) for hole in ordered}
    ordered.extend(dict(hole) for hole in holes if str(hole["infill_token"]) not in seen)
    return ordered


def _assign_hole_joint_positions(prompt_stamp: str, holes: Sequence[dict[str, Any]]) -> dict[str, set[int]]:
    body = _prompt_body(prompt_stamp)
    ranges = _stamp_group_ranges(body)
    holes_by_group: dict[int, list[dict[str, Any]]] = {}
    for hole in holes:
        if "attach_group" not in hole or "attach_local" not in hole:
            continue
        try:
            holes_by_group.setdefault(int(hole["attach_group"]), []).append(dict(hole))
        except Exception:
            continue

    assigned: dict[str, set[int]] = {}
    for group_idx, vals in holes_by_group.items():
        if group_idx < 0 or group_idx >= len(ranges):
            continue
        start, end = ranges[group_idx]
        group_tokens = body[start:end]
        specs = _joint_specs(group_tokens)
        used: set[int] = set()
        # External hole joints are appended by the benchmark constructor. Walk
        # holes/specs from the right to avoid removing an internal input joint
        # when it shares the same local atom index.
        for hole in sorted(vals, key=lambda item: int(item.get("hole_index", 0)), reverse=True):
            try:
                local = int(hole["attach_local"])
            except Exception:
                continue
            match_idx: Optional[int] = None
            for spec_idx in range(len(specs) - 1, -1, -1):
                if spec_idx in used:
                    continue
                spec_start, _spec_end, spec_local = specs[spec_idx]
                if int(spec_local) == local:
                    match_idx = spec_idx
                    break
            if match_idx is None:
                continue
            used.add(match_idx)
            spec_start, spec_end, _spec_local = specs[match_idx]
            assigned[str(hole["infill_token"])] = set(range(start + spec_start, start + spec_end))
    return assigned


def _parse_branch_count(tok: str) -> Optional[int]:
    if not (str(tok).startswith("[B_") and str(tok).endswith("]")):
        return None
    try:
        return int(str(tok)[3:-1])
    except ValueError:
        return None


def _span_group_count(tokens: Sequence[str]) -> int:
    vals = [str(tok) for tok in tokens if str(tok)]
    if not vals:
        return 0
    return max(1, len(_stamp_group_ranges(vals)))


def _branch_additions_for_fills(
    holes: Sequence[dict[str, Any]],
    fills: dict[str, list[str]],
    *,
    current_token: Optional[str] = None,
) -> dict[int, int]:
    out: dict[int, int] = {}
    for hole in holes:
        if "attach_group" not in hole:
            continue
        token = str(hole["infill_token"])
        add = 0
        if token in fills:
            add = _span_group_count(fills[token])
        elif current_token is not None and token == str(current_token):
            # The current placeholder will become at least one generated group.
            add = 1
        if add <= 0:
            continue
        try:
            group_idx = int(hole["attach_group"])
        except Exception:
            continue
        out[group_idx] = out.get(group_idx, 0) + int(add)
    return out


def _adjust_branch_counts_for_virtual_children(
    body: Sequence[str],
    additions_by_group: dict[int, int],
) -> list[str]:
    vals = [str(tok) for tok in body]
    if not additions_by_group:
        return vals
    ranges = _stamp_group_ranges(vals)
    out = list(vals)
    for group_idx, add in additions_by_group.items():
        if add <= 0 or group_idx < 0 or group_idx >= len(ranges):
            continue
        start, _end = ranges[int(group_idx)]
        current = _parse_branch_count(out[start])
        if current is None:
            continue
        out[start] = f"[B_{max(1, current + int(add))}]"
    return out


def _splice_with_branch_counts(
    prompt_stamp: str,
    holes: Sequence[dict[str, Any]],
    fills: dict[str, list[str]],
) -> list[str]:
    body = _prompt_body(prompt_stamp)
    additions = _branch_additions_for_fills(holes, fills)
    adjusted_body = _adjust_branch_counts_for_virtual_children(body, additions)
    out: list[str] = []
    for tok in adjusted_body:
        if tok in fills:
            out.extend(fills[tok])
        else:
            out.append(tok)
    return out


def _single_hole_step_prompt(
    prompt_stamp: str,
    holes: Sequence[dict[str, Any]],
    *,
    current_token: str,
    fills: dict[str, list[str]],
    joint_positions: dict[str, set[int]],
) -> list[str]:
    raw = str(prompt_stamp).split()
    task_token = raw[0] if raw and raw[0].startswith("[TASK_") else "[TASK_DECORATE]"
    body = _prompt_body(prompt_stamp)
    additions = _branch_additions_for_fills(holes, fills, current_token=current_token)
    body = _adjust_branch_counts_for_virtual_children(body, additions)
    hole_tokens = {str(hole["infill_token"]) for hole in holes}
    remove_positions: set[int] = set()
    for tok in hole_tokens:
        if tok != current_token and tok not in fills:
            remove_positions.update(joint_positions.get(tok, set()))

    out: list[str] = [task_token]
    for pos, tok in enumerate(body):
        if pos in remove_positions:
            continue
        if tok in hole_tokens:
            if tok in fills:
                out.extend(fills[tok])
            elif tok == current_token:
                out.append("[INFILL_SUBTREE]")
            else:
                continue
        else:
            out.append(tok)
    out.append("[GEN]")
    return out


def _sample_sequential_fills(
    model: torch.nn.Module,
    vocab: Any,
    prompt_stamp: str,
    holes: Sequence[dict[str, Any]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    diverse_prefix_penalty: float,
    device: torch.device,
) -> tuple[Optional[dict[str, list[str]]], str]:
    fills: dict[str, list[str]] = {}
    ordered_holes = _hole_prompt_order(prompt_stamp, holes)
    joint_positions = _assign_hole_joint_positions(prompt_stamp, holes)
    for hole in ordered_holes:
        token = str(hole["infill_token"])
        step_prompt = _single_hole_step_prompt(
            prompt_stamp,
            holes,
            current_token=token,
            fills=fills,
            joint_positions=joint_positions,
        )
        seqs = _sample_generated_tokens(
            model,
            vocab,
            step_prompt,
            n_samples=1,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            diverse_prefix_penalty=float(diverse_prefix_penalty),
            device=device,
        )
        raw = seqs[0] if seqs else []
        span = _tokens_to_stamp_span(_strip_after_control_eos(raw))
        if _contains_branch_token(span):
            return None, " ".join(raw)
        fills[token] = span
    return fills, " | ".join(" ".join(fills[str(hole["infill_token"])]) for hole in ordered_holes)


def _contains_branch_token(tokens: Sequence[str]) -> bool:
    return any(str(tok).startswith("[B_") and str(tok).endswith("]") for tok in tokens)


def _fixed_prompt_tokens(prompt_stamp: str) -> list[str]:
    fixed: list[str] = []
    for tok in _prompt_body(prompt_stamp):
        if tok.startswith("[INFILL"):
            continue
        fixed.append(tok)
    return fixed


def _parse_int_list_json(value: Any) -> list[int]:
    try:
        raw = json.loads(str(value))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for item in raw:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def _mol_from_stamp_tokens(rep: Any, tokens: Sequence[str]) -> Optional[Chem.Mol]:
    if not tokens:
        return None
    try:
        smiles = rep.decode(rep.detokenize(tokens))
    except Exception:
        return None
    return Chem.MolFromSmiles(str(smiles))


def _fixed_source_topology_query(
    rep: Any,
    *,
    source_smiles: str,
    mask_group_indices: Sequence[int],
) -> Optional[tuple[Chem.Mol, dict[int, int]]]:
    if not mask_group_indices:
        return None
    source = Chem.MolFromSmiles(str(source_smiles))
    if source is None:
        return None
    try:
        groups = rep.codec.encode_mol_to_groups(source)
    except Exception:
        return None
    valid_indices = [idx for idx in mask_group_indices if 0 <= int(idx) < len(groups)]
    if not valid_indices:
        return None

    mask_atoms: set[int] = set()
    for idx in valid_indices:
        mask_atoms.update(int(x) for x in groups[int(idx)].atom_indices)
    if not mask_atoms or len(mask_atoms) >= source.GetNumAtoms():
        return None

    boundary_counts: dict[int, int] = {}
    for atom in source.GetAtoms():
        atom_idx = int(atom.GetIdx())
        if atom_idx in mask_atoms:
            continue
        boundary_counts[atom_idx] = sum(1 for nbr in atom.GetNeighbors() if int(nbr.GetIdx()) in mask_atoms)

    rw = Chem.RWMol(source)
    for atom in rw.GetAtoms():
        atom.SetAtomMapNum(int(atom.GetIdx()) + 1)
    for atom_idx in sorted(mask_atoms, reverse=True):
        rw.RemoveAtom(int(atom_idx))
    query = rw.GetMol()
    try:
        query.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    return query, boundary_counts


def _fixed_prompt_topology_query(rep: Any, prompt_stamp: str) -> Optional[tuple[Chem.Mol, Optional[dict[int, int]]]]:
    query = _mol_from_stamp_tokens(rep, _fixed_prompt_tokens(prompt_stamp))
    if query is None or query.GetNumAtoms() == 0:
        return None
    return query, None


def _bond_key(a: int, b: int) -> tuple[int, int]:
    return (int(a), int(b)) if int(a) <= int(b) else (int(b), int(a))


def _topology_preserved(
    generated_smiles: str,
    query_payload: Optional[tuple[Chem.Mol, Optional[dict[int, int]]]],
) -> bool:
    if query_payload is None:
        return False
    query, boundary_counts = query_payload
    mol = Chem.MolFromSmiles(str(generated_smiles))
    if mol is None or query is None or query.GetNumAtoms() == 0:
        return False
    query_bonds = {
        _bond_key(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in query.GetBonds()
    }
    try:
        matches = mol.GetSubstructMatches(query, useChirality=False, uniquify=False)
    except Exception:
        return False
    for match in matches:
        matched_atoms = {int(x) for x in match}
        gen_to_query = {int(gen_idx): int(query_idx) for query_idx, gen_idx in enumerate(match)}
        exact_induced = True
        for bond in mol.GetBonds():
            a = int(bond.GetBeginAtomIdx())
            b = int(bond.GetEndAtomIdx())
            if a in matched_atoms and b in matched_atoms:
                if _bond_key(gen_to_query[a], gen_to_query[b]) not in query_bonds:
                    exact_induced = False
                    break
        if not exact_induced:
            continue

        if boundary_counts is None:
            return True

        boundary_ok = True
        for query_idx, gen_idx in enumerate(match):
            atom_map = int(query.GetAtomWithIdx(int(query_idx)).GetAtomMapNum())
            source_idx = atom_map - 1 if atom_map > 0 else int(query_idx)
            expected_external = int(boundary_counts.get(source_idx, 0))
            actual_external = sum(1 for nbr in mol.GetAtomWithIdx(int(gen_idx)).GetNeighbors() if int(nbr.GetIdx()) not in matched_atoms)
            if actual_external != expected_external:
                boundary_ok = False
                break
        if boundary_ok:
            return True
    return False


def _topology_query_for_case(rep: Any, row: Any, fixed_context_tokens: Sequence[str]) -> tuple[Optional[tuple[Chem.Mol, Optional[dict[int, int]]]], str]:
    if str(row.get("task", "")) == "linker_design":
        return None, "query_substructure"

    source_query = _fixed_source_topology_query(
        rep,
        source_smiles=str(row["source_smiles"]),
        mask_group_indices=_parse_int_list_json(row.get("mask_group_indices", "[]")),
    )
    if source_query is not None:
        return source_query, "source_fixed_topology"

    prompt_query = _mol_from_stamp_tokens(rep, fixed_context_tokens)
    if prompt_query is not None and prompt_query.GetNumAtoms() > 0:
        return (prompt_query, None), "fixed_prompt_topology"
    return None, "query_substructure"


def _quality_count(smiles: Sequence[str], sa: Any) -> int:
    n = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        try:
            if float(QED.qed(mol)) >= 0.6 and float(sa.calculateScore(mol)) <= 4.0:
                n += 1
        except Exception:
            continue
    return n


def _property_quality_pass(smiles: str, sa: Any) -> bool:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return False
    try:
        return float(QED.qed(mol)) >= 0.6 and float(sa.calculateScore(mol)) <= 4.0
    except Exception:
        return False


def _mean_tanimoto_distance(original_smiles: str, smiles_list: Sequence[str]) -> float:
    if not smiles_list:
        return float("nan")
    mol = Chem.MolFromSmiles(str(original_smiles))
    if mol is None:
        return float("nan")
    fp_ref = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
    fps = []
    for smi in smiles_list:
        m = Chem.MolFromSmiles(str(smi))
        if m is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024))
    if not fps:
        return float("nan")
    vals = DataStructs.BulkTanimotoSimilarity(fp_ref, fps, returnDistance=True)
    return float(sum(vals) / len(vals))


def _safe_mean(values: Sequence[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _strip_dummy_atoms(mol: Chem.Mol) -> Optional[Chem.Mol]:
    rw = Chem.RWMol(mol)
    for idx in sorted([a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0], reverse=True):
        rw.RemoveAtom(idx)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        try:
            Chem.SanitizeMol(out, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception:
            return None
    return out


def _query_preserved(generated_smiles: str, query_smiles: str) -> bool:
    mol = Chem.MolFromSmiles(str(generated_smiles))
    if mol is None:
        return False
    for part in [p for p in str(query_smiles).split(".") if p]:
        if re.search(r"\[(?:\d+)?\*\]", part):
            # For constrained linker-style queries, the dummy atom is the
            # required external attachment. Deleting it changes the attachment
            # atom valence/query H count and falsely rejects connected products.
            smarts = re.sub(r"\[(?:\d+)?\*\]", "[*]", part)
            query = Chem.MolFromSmarts(smarts)
        else:
            query = Chem.MolFromSmiles(part) or Chem.MolFromSmiles(part, sanitize=False)
            if query is not None:
                query = _strip_dummy_atoms(query)
        if query is None or query.GetNumAtoms() == 0:
            return False
        try:
            if not mol.HasSubstructMatch(query, useChirality=False):
                return False
        except Exception:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--task-parquet", default="data/genmol_fcstamp_infill.parquet")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--diverse-prefix-penalty",
        type=float,
        default=0.0,
        help=(
            "Sampling-only diversity penalty within each prompt batch. "
            "When samples share the same generated prefix, previously sampled next tokens "
            "for that prefix receive this logit penalty. Default 0 keeps baseline decoding."
        ),
    )
    parser.add_argument(
        "--task-sampling-json",
        default="",
        help=(
            "Optional JSON string or file path mapping task names to sampling overrides, "
            "for example {'linker_design': {'temperature': 1.3, 'top_p': 1.0}}."
        ),
    )
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--strip-stereo",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop [S_*] stereo tokens from prompts and generated spans before decoding.",
    )
    parser.add_argument(
        "--sequential-holes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For multi-hole splice prompts, fill one hole at a time with single-hole prompts.",
    )
    parser.add_argument(
        "--aggregate-attach-sites",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For superstructure_generation rows named case:attachN, split num-samples "
            "across all attach-site prompts for the same case, then merge decoded molecules "
            "before computing metrics."
        ),
    )
    parser.add_argument(
        "--samples-per-attach-site",
        type=int,
        default=0,
        help="When aggregating attach sites, sample this many molecules per site. 0 splits --num-samples across sites.",
    )
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_sampling = _load_task_sampling_config(args.task_sampling_json)

    model, vocab, rep, dev = _load_model(args.model_dir, args.device)
    df = pd.read_parquet(args.task_parquet)
    if int(args.max_cases) > 0:
        df = df.head(int(args.max_cases))
    sa = _load_sa_scorer()
    tasks = list(dict.fromkeys(str(x) for x in df["task"].tolist()))

    all_results: dict[str, Any] = {
        "model_dir": str(args.model_dir),
        "task_parquet": str(args.task_parquet),
        "num_samples": int(args.num_samples),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "diverse_prefix_penalty": float(args.diverse_prefix_penalty),
        "task_sampling": task_sampling,
        "max_new": int(args.max_new),
        "seed": int(args.seed),
        "strip_stereo": bool(args.strip_stereo),
        "sequential_holes": bool(args.sequential_holes),
        "aggregate_attach_sites": bool(args.aggregate_attach_sites),
        "samples_per_attach_site": int(args.samples_per_attach_site),
        "tasks": {},
    }
    pred_rows: list[dict[str, Any]] = []

    for task in tasks:
        sampling_cfg = task_sampling.get(str(task), {})
        task_temperature = float(sampling_cfg.get("temperature", float(args.temperature)))
        task_top_k = int(sampling_cfg.get("top_k", int(args.top_k)))
        task_top_p = float(sampling_cfg.get("top_p", float(args.top_p)))
        task_diverse_prefix_penalty = float(
            sampling_cfg.get("diverse_prefix_penalty", float(args.diverse_prefix_penalty))
        )
        task_df = df[df["task"] == task]
        validity: list[float] = []
        uniqueness: list[float] = []
        quality: list[float] = []
        constraint_quality: list[float] = []
        property_quality: list[float] = []
        diversity: list[float] = []
        distance: list[float] = []
        preservation: list[float] = []
        query_preservation: list[float] = []
        cases: list[dict[str, Any]] = []

        if bool(args.aggregate_attach_sites) and task == "superstructure_generation":
            grouped: dict[str, list[tuple[Any, Any]]] = {}
            for case_idx, row in task_df.iterrows():
                raw_name = str(row.get("name", case_idx))
                base_name = raw_name.split(":attach", 1)[0]
                grouped.setdefault(base_name, []).append((case_idx, row))
            case_groups: list[tuple[str, list[tuple[Any, Any, int]]]] = []
            for base_name, items in grouped.items():
                n_sites = max(1, len(items))
                if int(args.samples_per_attach_site) > 0:
                    per_site = int(args.samples_per_attach_site)
                    samples = [(case_idx, row, per_site) for case_idx, row in items]
                else:
                    base_n = int(args.num_samples) // n_sites
                    rem = int(args.num_samples) % n_sites
                    samples = [
                        (case_idx, row, base_n + (1 if site_idx < rem else 0))
                        for site_idx, (case_idx, row) in enumerate(items)
                    ]
                case_groups.append((base_name, samples))
        else:
            case_groups = [
                (str(row.get("name", case_idx)), [(case_idx, row, int(args.num_samples))])
                for case_idx, row in task_df.iterrows()
            ]

        for group_name, row_samples in case_groups:
            first_case_idx, first_row, _first_samples = row_samples[0]
            first_prompt_tokens = str(first_row["prompt_stamp"]).split()
            if bool(args.strip_stereo):
                first_prompt_tokens = _strip_stereo_tokens(first_prompt_tokens)
            first_prompt_stamp = " ".join(first_prompt_tokens)
            first_fixed_context_tokens = _fixed_prompt_tokens(first_prompt_stamp)
            topology_query, preservation_mode = _topology_query_for_case(rep, first_row, first_fixed_context_tokens)
            decoded: list[str] = []
            generated_raw: list[str] = []
            n_parse_fail = 0
            n_decode_fail = 0
            total_requested = 0
            prompt_stamps: list[str] = []
            site_sample_counts: dict[str, int] = {}

            for case_idx, row, row_num_samples in row_samples:
                if int(row_num_samples) <= 0:
                    continue
                site_name = str(row.get("name", case_idx))
                site_sample_counts[site_name] = int(row_num_samples)
                total_requested += int(row_num_samples)
                prompt_tokens = str(row["prompt_stamp"]).split()
                if bool(args.strip_stereo):
                    prompt_tokens = _strip_stereo_tokens(prompt_tokens)
                prompt_stamp = " ".join(prompt_tokens)
                prompt_stamps.append(prompt_stamp)
                holes = json.loads(str(row["hole_spans"]))
                decode_mode = str(row.get("decode_mode", "splice") or "splice")

                remaining = int(row_num_samples)
                while remaining > 0:
                    cur = min(int(args.batch_size), remaining)
                    remaining -= cur
                    if bool(args.sequential_holes) and decode_mode != "generated_full" and len(holes) > 1:
                        seqs: list[list[str]] = [[] for _ in range(cur)]
                        sequential_fills: list[tuple[Optional[dict[str, list[str]]], str]] = []
                        for _sample_idx in range(cur):
                            fills, raw = _sample_sequential_fills(
                                model,
                                vocab,
                                prompt_stamp,
                                holes,
                                max_new_tokens=int(args.max_new),
                                temperature=task_temperature,
                                top_k=task_top_k,
                                top_p=task_top_p,
                                diverse_prefix_penalty=task_diverse_prefix_penalty,
                                device=dev,
                            )
                            sequential_fills.append((fills, raw))
                    else:
                        seqs = _sample_generated_tokens(
                            model,
                            vocab,
                            prompt_tokens,
                            n_samples=cur,
                            max_new_tokens=int(args.max_new),
                            temperature=task_temperature,
                            top_k=task_top_k,
                            top_p=task_top_p,
                            diverse_prefix_penalty=task_diverse_prefix_penalty,
                            device=dev,
                        )
                        sequential_fills = [(None, "") for _ in seqs]

                    for item_idx, gen_tokens in enumerate(seqs):
                        generated_raw.append(" ".join(gen_tokens))
                        if decode_mode == "generated_full":
                            final_tokens = _tokens_to_stamp_span(_strip_after_control_eos(gen_tokens))
                            if not final_tokens:
                                n_parse_fail += 1
                                continue
                        else:
                            if bool(args.sequential_holes) and len(holes) > 1:
                                fills = sequential_fills[item_idx][0]
                                generated_raw[-1] = sequential_fills[item_idx][1]
                            else:
                                fills = _parse_fills(gen_tokens, holes)
                            if fills is None:
                                n_parse_fail += 1
                                continue
                            if task in {"linker_design", "scaffold_decoration"} and any(
                                _contains_branch_token(fill) for fill in fills.values()
                            ):
                                n_parse_fail += 1
                                continue
                            final_tokens = _splice_with_branch_counts(prompt_stamp, holes, fills)
                        if bool(args.strip_stereo):
                            final_tokens = _strip_stereo_tokens(final_tokens)
                        try:
                            smiles = rep.decode(rep.detokenize(final_tokens))
                        except Exception:
                            n_decode_fail += 1
                            continue
                        can = canonicalize_smiles(smiles, require_single_component=True)
                        if can is not None:
                            decoded.append(str(can))

            uniq = list(dict.fromkeys(decoded))
            n_valid = len(decoded)
            n_unique = len(uniq)
            n_property_quality = _quality_count(uniq, sa)
            query_preserved = [s for s in uniq if _query_preserved(s, str(first_row["query_smiles"]))]
            if topology_query is not None:
                topology_preserved = [s for s in uniq if _topology_preserved(s, topology_query)]
                preserved = topology_preserved
                n_preserved = len(topology_preserved)
            else:
                preserved = query_preserved
                n_preserved = len(query_preserved)
            n_quality = sum(1 for s in query_preserved if _property_quality_pass(s, sa))
            n_constraint_quality = sum(1 for s in preserved if _property_quality_pass(s, sa))
            denom = max(1, int(total_requested))
            validity.append(n_valid / denom)
            uniqueness.append(n_unique / max(1, n_valid) if n_valid else 0.0)
            quality.append(n_quality / denom)
            constraint_quality.append(n_constraint_quality / denom)
            property_quality.append(n_property_quality / denom)
            diversity.append(_internal_diversity(uniq, max_pairs=5000) if n_unique >= 2 else 0.0)
            distance.append(_mean_tanimoto_distance(str(first_row["source_smiles"]), uniq))
            preservation.append(n_preserved / max(1, n_unique) if n_unique else 0.0)
            query_preservation.append(len(query_preserved) / max(1, n_unique) if n_unique else 0.0)

            case = {
                "name": str(group_name),
                "task": task,
                "prompt_stamp": first_prompt_stamp,
                "prompt_stamps": prompt_stamps,
                "decode_mode": str(first_row.get("decode_mode", "splice") or "splice"),
                "temperature": task_temperature,
                "top_k": task_top_k,
                "top_p": task_top_p,
                "diverse_prefix_penalty": task_diverse_prefix_penalty,
                "fixed_context_tokens": " ".join(first_fixed_context_tokens),
                "preservation_mode": preservation_mode,
                "query_smiles": str(first_row["query_smiles"]),
                "source_smiles": str(first_row["source_smiles"]),
                "n_requested": int(total_requested),
                "n_attach_sites": int(len(row_samples)),
                "attach_site_samples": site_sample_counts,
                "n_valid": n_valid,
                "n_unique": n_unique,
                "n_property_quality": int(n_property_quality),
                "n_quality": int(n_quality),
                "n_constraint_quality": int(n_constraint_quality),
                "n_parse_fail": int(n_parse_fail),
                "n_decode_fail": int(n_decode_fail),
                "validity": validity[-1],
                "uniqueness": uniqueness[-1],
                "quality": quality[-1],
                "constraint_quality": constraint_quality[-1],
                "property_quality": property_quality[-1],
                "diversity": diversity[-1],
                "distance": distance[-1],
                "constraint_preservation": preservation[-1],
                "query_substructure_preservation": query_preservation[-1],
                "examples": uniq[:10],
                "raw_generations": generated_raw[:10],
            }
            cases.append(case)
            for smi in uniq[:50]:
                pred_rows.append({"task": task, "case": case["name"], "smiles": smi})
            print(
                f"[fcstamp-eval] task={task} case={case['name']} "
                f"valid={100*validity[-1]:.1f} uniq={100*uniqueness[-1]:.1f} "
                f"quality={100*quality[-1]:.1f} property_quality={100*property_quality[-1]:.1f} "
                f"constraint_quality={100*constraint_quality[-1]:.1f} preserve={100*preservation[-1]:.1f}",
                flush=True,
            )

        summary = {
            "validity": 100.0 * _safe_mean(validity),
            "uniqueness": 100.0 * _safe_mean(uniqueness),
            "quality": 100.0 * _safe_mean(quality),
            "constraint_quality": 100.0 * _safe_mean(constraint_quality),
            "property_quality": 100.0 * _safe_mean(property_quality),
            "diversity": _safe_mean(diversity),
            "distance": _safe_mean(distance),
            "constraint_preservation": 100.0 * _safe_mean(preservation),
            "query_substructure_preservation": 100.0 * _safe_mean(query_preservation),
            "n_cases": int(len(cases)),
            "sampling": {
                "temperature": task_temperature,
                "top_k": task_top_k,
                "top_p": task_top_p,
                "diverse_prefix_penalty": task_diverse_prefix_penalty,
            },
            "genmol_baseline": GENMOL_BASELINES["GenMol"].get(task),
            "genmol_v2_baseline": GENMOL_BASELINES["GenMol V2"].get(task),
            "cases": cases,
        }
        all_results["tasks"][task] = summary

    out_json = out_dir / f"fcstamp_genmol_eval_seed{int(args.seed)}.json"
    out_json.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(out_dir / f"fcstamp_genmol_predictions_seed{int(args.seed)}.tsv", sep="\t", index=False)
    compact = {
        task: {k: v for k, v in res.items() if k in {"validity", "uniqueness", "quality", "constraint_quality", "property_quality", "diversity", "distance", "constraint_preservation", "query_substructure_preservation", "n_cases"}}
        for task, res in all_results["tasks"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
