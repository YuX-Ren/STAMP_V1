from __future__ import annotations

import csv
import json
import math
import os
import random
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdchem import BondType
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .common import canonicalize_smiles, iterable_with_progress
from .lm import (
    LMConfig,
    NextTokenDataset,
    TinyDecoderLM,
    TokenVocab,
    _add_gumbel_noise,
    _apply_top_k_logits,
    _compute_ao_gpt_loss,
    _ids_with_special_tokens,
    _normalize_model_type,
    _resolve_amp_mode,
    _resolve_torch_device,
    _set_seed,
    build_dynamic_lm_collate_fn,
    load_trained_model,
    sample_model_sequences,
)
from .metrics import evaluate_generated_smiles
from .representations import RepresentationError, UnsupportedRepresentation, build_representation

_COMP_WORKER_REP = None
_COMP_WORKER_CFG: tuple[str, str, Optional[str], Optional[str], bool, bool] | None = None
_EVAL_WORKER_REP = None
_MULTI_COMPONENT_PROMPT_SEP = "<DOT>"


class MaskedSpanInfillDataset(Dataset):
    def __init__(self, masked_ids: Sequence[list[int]], label_ids: Sequence[list[int]], max_seq_len: int, pad_id: int) -> None:
        self.samples: list[tuple[list[int], list[int]]] = []
        self.pad_id = pad_id
        for x_ids, y_ids in zip(masked_ids, label_ids):
            if len(x_ids) != len(y_ids):
                continue
            ids_x = x_ids[:max_seq_len]
            ids_y = y_ids[:max_seq_len]
            if not ids_x:
                continue
            self.samples.append((ids_x, ids_y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.samples[idx]


def _scaffold_row_from_smiles(smiles: str, min_scaffold_heavy_atoms: int) -> Optional[tuple[str, str, str]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None or scaf.GetNumAtoms() == 0:
        return None
    if scaf.GetNumHeavyAtoms() < min_scaffold_heavy_atoms:
        return None
    scaf_s = Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=True)
    can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if scaf_s == can:
        return None
    scaf_prompt_mol = _add_scaffold_attachment_dummies(mol, scaf)
    if scaf_prompt_mol is None:
        return None
    scaf_prompt = Chem.MolToSmiles(scaf_prompt_mol, canonical=True, isomericSmiles=True)
    return can, scaf_s, scaf_prompt


def _build_scaffold_worker(item: tuple[str, int]) -> Optional[tuple[str, str, str]]:
    smiles, min_scaffold_heavy_atoms = item
    return _scaffold_row_from_smiles(smiles, min_scaffold_heavy_atoms)


def _add_scaffold_attachment_dummies(mol: Chem.Mol, scaffold: Chem.Mol) -> Optional[Chem.Mol]:
    match = mol.GetSubstructMatch(scaffold)
    if not match:
        return None
    match_set = set(match)
    rw = Chem.RWMol(scaffold)
    attach_sites: list[int] = []
    for scaf_atom_idx, mol_atom_idx in enumerate(match):
        atom = mol.GetAtomWithIdx(mol_atom_idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() not in match_set:
                attach_sites.append(scaf_atom_idx)
    if not attach_sites:
        return None
    for scaf_atom_idx in attach_sites:
        dummy = Chem.Atom(0)
        dummy.SetIsotope(0)
        d_idx = rw.AddAtom(dummy)
        rw.AddBond(scaf_atom_idx, d_idx, BondType.SINGLE)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def _remap_dummy_isotopes(mol: Chem.Mol, mapping: dict[int, int]) -> Optional[Chem.Mol]:
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        iso = int(atom.GetIsotope())
        if iso not in mapping:
            return None
        atom.SetIsotope(int(mapping[iso]))
    return out


def _relabel_fragment_dummy(mol: Chem.Mol, label: int) -> Optional[Chem.Mol]:
    out = Chem.Mol(mol)
    dummies = [a for a in out.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        return None
    dummies[0].SetIsotope(int(label))
    return out


def _relabel_linker_dummies(mol: Chem.Mol) -> Optional[Chem.Mol]:
    out = Chem.Mol(mol)
    dummies = [a for a in out.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 2:
        return None
    dummies = sorted(dummies, key=lambda a: a.GetIdx())
    dummies[0].SetIsotope(1)
    dummies[1].SetIsotope(2)
    return out


def _normalize_linker_triplet(
    frag_a: Chem.Mol,
    frag_b: Chem.Mol,
    linker: Chem.Mol,
) -> Optional[tuple[str, str, str, str, str, str]]:
    if sum(1 for a in frag_a.GetAtoms() if a.GetAtomicNum() == 0) != 1:
        return None
    if sum(1 for a in frag_b.GetAtoms() if a.GetAtomicNum() == 0) != 1:
        return None
    if sum(1 for a in linker.GetAtoms() if a.GetAtomicNum() == 0) != 2:
        return None

    fa_plain = _strip_dummy_atoms(frag_a)
    fb_plain = _strip_dummy_atoms(frag_b)
    lk_plain = _strip_dummy_atoms(linker)
    if fa_plain is None or fb_plain is None or lk_plain is None:
        return None
    if fa_plain.GetNumAtoms() == 0 or fb_plain.GetNumAtoms() == 0 or lk_plain.GetNumAtoms() == 0:
        return None

    fa_plain_s = Chem.MolToSmiles(fa_plain, canonical=True, isomericSmiles=True)
    fb_plain_s = Chem.MolToSmiles(fb_plain, canonical=True, isomericSmiles=True)
    if fb_plain_s < fa_plain_s:
        frag_a, frag_b = frag_b, frag_a
        fa_plain_s, fb_plain_s = fb_plain_s, fa_plain_s

    fa_m = _relabel_fragment_dummy(frag_a, label=1)
    fb_m = _relabel_fragment_dummy(frag_b, label=2)
    lk_m = _relabel_linker_dummies(linker)
    if fa_m is None or fb_m is None or lk_m is None:
        return None

    fa_ap_s = Chem.MolToSmiles(fa_m, canonical=True, isomericSmiles=True)
    fb_ap_s = Chem.MolToSmiles(fb_m, canonical=True, isomericSmiles=True)
    lk_ap_s = Chem.MolToSmiles(lk_m, canonical=True, isomericSmiles=True)
    lk_plain_s = Chem.MolToSmiles(lk_plain, canonical=True, isomericSmiles=True)
    return fa_plain_s, fb_plain_s, lk_plain_s, fa_ap_s, fb_ap_s, lk_ap_s


def _assemble_labeled_linker_fragments(
    frag_a_ap: str,
    frag_b_ap: str,
    linker_ap: str,
) -> Optional[str]:
    ma = Chem.MolFromSmiles(frag_a_ap)
    mb = Chem.MolFromSmiles(frag_b_ap)
    ml = Chem.MolFromSmiles(linker_ap)
    if ma is None or mb is None or ml is None:
        return None

    combined = Chem.CombineMols(Chem.CombineMols(ma, mb), ml)
    rw = Chem.RWMol(combined)
    by_iso: dict[int, list[int]] = {1: [], 2: []}
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetIsotope() in by_iso:
            by_iso[int(atom.GetIsotope())].append(atom.GetIdx())
    if len(by_iso[1]) != 2 or len(by_iso[2]) != 2:
        return None

    to_remove: list[int] = []
    for iso in (1, 2):
        d1, d2 = by_iso[iso]
        a1_nbr = rw.GetAtomWithIdx(d1).GetNeighbors()
        a2_nbr = rw.GetAtomWithIdx(d2).GetNeighbors()
        if len(a1_nbr) != 1 or len(a2_nbr) != 1:
            return None
        a1 = a1_nbr[0].GetIdx()
        a2 = a2_nbr[0].GetIdx()
        b1 = rw.GetBondBetweenAtoms(d1, a1)
        b2 = rw.GetBondBetweenAtoms(d2, a2)
        if b1 is None or b2 is None:
            return None
        bt = b1.GetBondType() if b1.GetBondType() == b2.GetBondType() else BondType.SINGLE
        if rw.GetBondBetweenAtoms(a1, a2) is None:
            rw.AddBond(a1, a2, bt)
        to_remove.extend([d1, d2])

    for idx in sorted(set(to_remove), reverse=True):
        rw.RemoveAtom(idx)

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    if len(Chem.GetMolFrags(out)) != 1:
        return None
    return Chem.MolToSmiles(out, canonical=True, isomericSmiles=True)


def _linker_row_from_smiles(smiles: str) -> Optional[tuple[str, str, str, str, str, str, str]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    brics_bonds = []
    for (a, b), _ in BRICS.FindBRICSBonds(mol):
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is not None:
            brics_bonds.append(bond.GetIdx())

    if len(brics_bonds) < 2:
        return None

    for b1, b2 in combinations(brics_bonds[:8], 2):
        frag_mol = Chem.FragmentOnBonds(mol, [b1, b2], addDummies=True)
        frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=True)
        if len(frags) != 3:
            continue

        linker_idx = -1
        for i, fm in enumerate(frags):
            n_dummy = sum(1 for a in fm.GetAtoms() if a.GetAtomicNum() == 0)
            if n_dummy == 2:
                linker_idx = i
                break
        if linker_idx < 0:
            continue

        other = [i for i in range(3) if i != linker_idx]
        fa = frags[other[0]]
        fb = frags[other[1]]
        lk = frags[linker_idx]

        normalized = _normalize_linker_triplet(fa, fb, lk)
        if normalized is None:
            continue
        fa_s, fb_s, lk_s, fa_ap, fb_ap, lk_ap = normalized

        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return can, fa_s, fb_s, lk_s, fa_ap, fb_ap, lk_ap

    return None


def _build_linker_worker(smiles: str) -> Optional[tuple[str, str, str, str, str, str, str]]:
    return _linker_row_from_smiles(smiles)


def _tokenize_completion_row_local(
    rep,
    row: dict[str, str],
    prompt_key: str,
    target_key: str,
    prompt_template: Optional[str],
    target_template: Optional[str],
    prompt_is_repr: bool = False,
    target_is_repr: bool = False,
) -> Optional[list[str]]:
    try:
        prompt_raw = prompt_template.format(**row) if prompt_template is not None else row[prompt_key]
        target_raw = target_template.format(**row) if target_template is not None else row[target_key]
        prompt_tokens = _encode_completion_prompt_tokens(rep, prompt_raw, is_repr=prompt_is_repr)
        target_tokens = _encode_completion_target_tokens(rep, target_raw, is_repr=target_is_repr)
    except Exception:
        return None
    if not prompt_tokens or not target_tokens:
        return None
    return prompt_tokens + ["<SEP>"] + target_tokens


def _encode_completion_target_tokens(rep, target_raw: str, is_repr: bool = False) -> list[str]:
    if is_repr:
        return [tok for tok in str(target_raw).split() if tok]
    target_text = rep.encode(target_raw)
    return rep.tokenize(target_text)


def _encode_completion_prompt_tokens(rep, prompt_raw: str, is_repr: bool = False) -> list[str]:
    if is_repr:
        return [tok for tok in str(prompt_raw).split() if tok]
    # STAMP cannot encode disconnected molecules (e.g. "frag_a.frag_b") in one pass.
    # For linker prompts, encode each component separately and stitch token streams.
    if getattr(rep, "name", "") == "stamp" and "." in prompt_raw:
        parts = [part for part in prompt_raw.split(".") if part]
        if len(parts) >= 2:
            out: list[str] = []
            for part in parts:
                part_tokens = rep.tokenize(rep.encode(part))
                if not part_tokens:
                    raise RepresentationError("Empty prompt component after STAMP tokenization.")
                if out:
                    out.append(_MULTI_COMPONENT_PROMPT_SEP)
                out.extend(part_tokens)
            return out

    return rep.tokenize(rep.encode(prompt_raw))


def _init_completion_worker(
    representation: str,
    stamp_vocab: Optional[str],
    prompt_key: str,
    target_key: str,
    prompt_template: Optional[str],
    target_template: Optional[str],
    prompt_is_repr: bool,
    target_is_repr: bool,
) -> None:
    global _COMP_WORKER_REP, _COMP_WORKER_CFG
    _COMP_WORKER_REP = build_representation(representation, stamp_vocab=stamp_vocab)
    _COMP_WORKER_CFG = (prompt_key, target_key, prompt_template, target_template, prompt_is_repr, target_is_repr)


def _tokenize_completion_row_worker(row: dict[str, str]) -> Optional[list[str]]:
    rep = _COMP_WORKER_REP
    cfg = _COMP_WORKER_CFG
    if rep is None or cfg is None:
        raise RuntimeError("Completion tokenize worker is not initialized.")
    prompt_key, target_key, prompt_template, target_template, prompt_is_repr, target_is_repr = cfg
    return _tokenize_completion_row_local(
        rep,
        row,
        prompt_key=prompt_key,
        target_key=target_key,
        prompt_template=prompt_template,
        target_template=target_template,
        prompt_is_repr=prompt_is_repr,
        target_is_repr=target_is_repr,
    )


def _tokenize_completion_records(
    records: Sequence[dict[str, str]],
    representation: str,
    stamp_vocab: Optional[str],
    prompt_key: str,
    target_key: str,
    prompt_template: Optional[str],
    target_template: Optional[str],
    prompt_is_repr: bool,
    target_is_repr: bool,
    show_progress: bool,
    desc: str,
    num_workers: int,
    chunksize: int,
) -> list[list[str]]:
    rep = build_representation(representation, stamp_vocab=stamp_vocab)
    if num_workers <= 0:
        num_workers = max(1, os.cpu_count() or 1)

    seqs: list[list[str]] = []
    if num_workers == 1:
        iterator = iterable_with_progress(
            records,
            show_progress=show_progress,
            total=len(records),
            desc=desc,
        )
        for row in iterator:
            toks = _tokenize_completion_row_local(
                rep,
                row,
                prompt_key=prompt_key,
                target_key=target_key,
                prompt_template=prompt_template,
                target_template=target_template,
                prompt_is_repr=prompt_is_repr,
                target_is_repr=target_is_repr,
            )
            if toks:
                seqs.append(toks)
        return seqs

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_completion_worker,
        initargs=(
            representation,
            stamp_vocab,
            prompt_key,
            target_key,
            prompt_template,
            target_template,
            prompt_is_repr,
            target_is_repr,
        ),
    ) as pool:
        mapped = pool.map(
            _tokenize_completion_row_worker,
            records,
            chunksize=max(1, int(chunksize)),
        )
        mapped = iterable_with_progress(
            mapped,
            show_progress=show_progress,
            total=len(records),
            desc=desc,
        )
        for toks in mapped:
            if toks:
                seqs.append(toks)
    return seqs


def train_ao_fragment_infill(
    records: Sequence[dict[str, str]],
    representation: str,
    output_dir: str | Path,
    cfg: Optional[LMConfig] = None,
    show_progress: bool = True,
    device: Optional[str] = None,
    stamp_vocab: Optional[str] = None,
) -> dict[str, Any]:
    rep = build_representation(representation, stamp_vocab=stamp_vocab)
    if isinstance(rep, UnsupportedRepresentation):
        raise RepresentationError(
            f"Cannot train AO fragment infill for unsupported placeholder: {representation}"
        )
    cfg = cfg or LMConfig(model_type="ao_gpt")
    cfg.model_type = "ao_gpt"
    _set_seed(cfg.seed)
    dev = _resolve_torch_device(device)

    masked_ids, label_ids, vocab = _encode_infill_records(records, rep, cfg.max_seq_len)
    if not masked_ids:
        raise ValueError("No valid infill records after encoding.")
    n_valid = max(1, int(0.1 * len(masked_ids)))
    if len(masked_ids) <= n_valid:
        n_valid = max(1, len(masked_ids) // 2)
    ds_train = MaskedSpanInfillDataset(masked_ids[:-n_valid], label_ids[:-n_valid], cfg.max_seq_len, vocab.pad_id)
    ds_valid = MaskedSpanInfillDataset(masked_ids[-n_valid:], label_ids[-n_valid:], cfg.max_seq_len, vocab.pad_id)

    collate_fn = build_dynamic_lm_collate_fn(vocab.pad_id, pad_to_multiple_of=(8 if dev.type == "cuda" else None))
    loader_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn, pin_memory=(dev.type == "cuda"), num_workers=max(0, int(cfg.num_data_workers)))
    loader_valid = DataLoader(ds_valid, batch_size=cfg.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn, pin_memory=(dev.type == "cuda"), num_workers=max(0, int(cfg.num_data_workers)))

    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    use_amp, amp_dtype, use_scaler, amp_label = _resolve_amp_mode(cfg.amp, dev)

    model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
    if cfg.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[misc]
        except Exception:
            pass
    model.to(dev)
    adamw_kwargs: Dict[str, Any] = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    if dev.type == "cuda":
        try:
            adamw_kwargs["fused"] = True
        except Exception:
            pass
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    except TypeError:
        adamw_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)  # pragma: no cover

    history: list[dict[str, float]] = []
    non_blocking = dev.type == "cuda"
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses: list[float] = []
        iterator = iterable_with_progress(loader_train, show_progress=show_progress, total=len(loader_train), desc=f"Train ao-infill[{representation}] ep{epoch}")
        for x, y in iterator:
            x = x.to(dev, non_blocking=non_blocking)
            y = y.to(dev, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if use_amp else nullcontext()
            with amp_ctx:
                loss = _compute_ao_infill_loss(model, x, y, random_ratio=1.0)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        valid_losses: list[float] = []
        with torch.no_grad():
            for x, y in loader_valid:
                x = x.to(dev, non_blocking=non_blocking)
                y = y.to(dev, non_blocking=non_blocking)
                amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True) if use_amp else nullcontext()
                with amp_ctx:
                    loss = _compute_ao_infill_loss(model, x, y, random_ratio=1.0)
                valid_losses.append(float(loss.item()))
        tr = float(np.mean(train_losses)) if train_losses else float("nan")
        va = float(np.mean(valid_losses)) if valid_losses else float("nan")
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr,
                "valid_loss": va,
                "train_ppl": float(math.exp(tr)) if not math.isnan(tr) else float("nan"),
                "valid_ppl": float(math.exp(va)) if not math.isnan(va) else float("nan"),
            }
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp_vocab_rel = None
    if stamp_vocab is not None:
        import shutil
        dest = out_dir / "stamp_vocab.txt"
        shutil.copy2(stamp_vocab, dest)
        stamp_vocab_rel = "stamp_vocab.txt"
    torch.save(
        {
            "model_state": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
            "cfg": vars(cfg),
            "representation": representation,
            "model_type": "ao_gpt",
            "stamp_vocab": stamp_vocab_rel,
            "task": "ao_fragment_infill",
        },
        out_dir / "model.pt",
    )
    (out_dir / "vocab.json").write_text(json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "representation": representation,
                "task": "ao_fragment_infill",
                "n_records": len(records),
                "n_train_samples": len(ds_train),
                "n_valid_samples": len(ds_valid),
                "cfg": vars(cfg),
                "device": str(dev),
                "stamp_vocab": stamp_vocab_rel,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"output_dir": str(out_dir), "history": history, "device": str(dev)}


def _stamp_fragment_groups(tokens: Sequence[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    pending_prefix: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok == "[END]":
            break
        if tok.startswith("[B_"):
            pending_prefix.append(tok)
            i += 1
            continue
        if tok.startswith("[J_") or tok.startswith("[S_") or tok in {"[BACK]", "[Ring]"}:
            if groups:
                groups[-1].append(tok)
            else:
                pending_prefix.append(tok)
            i += 1
            continue
        group = list(pending_prefix)
        pending_prefix.clear()
        group.append(tok)
        i += 1
        while i < n and (tokens[i].startswith("[J_") or tokens[i].startswith("[S_") or tokens[i] == "[Ring]"):
            group.append(tokens[i])
            i += 1
        groups.append(group)
    if pending_prefix and groups:
        groups[-1].extend(pending_prefix)
    return groups


def _mask_fragment_groups(
    groups: Sequence[Sequence[str]],
    rng,
    min_mask_groups: int,
    max_mask_groups: int,
) -> Optional[tuple[list[str], list[str], list[str], int, int]]:
    if not groups:
        return None
    n_groups = len(groups)
    k_max = min(max(1, int(max_mask_groups)), n_groups)
    k_min = min(max(1, int(min_mask_groups)), k_max)
    k = int(rng.randint(k_min, k_max))
    start = int(rng.randrange(0, n_groups - k + 1))
    end = start + k
    prompt_tokens: list[str] = []
    masked_tokens: list[str] = []
    target_tokens: list[str] = []
    for idx, group in enumerate(groups):
        if start <= idx < end:
            target_tokens.extend(group)
            masked_tokens.extend([TokenVocab.MASK] * len(group))
        else:
            prompt_tokens.extend(group)
            masked_tokens.extend(group)
            if idx + 1 == start:
                prompt_tokens.append(TokenVocab.MASK)
    if start == 0:
        prompt_tokens = [TokenVocab.MASK] + prompt_tokens
    return prompt_tokens, masked_tokens, target_tokens, start, k


def _collapse_mask_runs(tokens: Sequence[str]) -> list[str]:
    out: list[str] = []
    prev_mask = False
    for tok in tokens:
        is_mask = (tok == TokenVocab.MASK)
        if is_mask and prev_mask:
            continue
        out.append(tok)
        prev_mask = is_mask
    return out


def _build_group_keep_row(
    rep,
    source_smiles: str,
    source_tokens: Sequence[str],
    groups,
    keep_group_indices: Sequence[int],
) -> Optional[dict[str, Any]]:
    keep_set = set(int(i) for i in keep_group_indices)
    if not keep_set:
        return None
    masked_tokens: list[str] = []
    target_tokens: list[str] = []
    kept_any = False
    masked_any = False
    for idx, group in enumerate(groups):
        group_toks = list(group.tokens)
        if idx in keep_set:
            masked_tokens.extend(group_toks)
            kept_any = True
        else:
            masked_tokens.extend([TokenVocab.MASK] * len(group_toks))
            target_tokens.extend(group_toks)
            masked_any = True
    if not kept_any or not masked_any:
        return None
    return {
        "source_smiles": source_smiles,
        "source_repr": rep.detokenize(source_tokens),
        "prompt_repr": rep.detokenize(_collapse_mask_runs(masked_tokens)),
        "masked_repr": rep.detokenize(masked_tokens),
        "target_repr": rep.detokenize(target_tokens),
    }


def _encode_stamp_groups_from_smiles(rep, smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    groups = rep.codec.encode_mol_to_groups(mol)
    source_repr = rep.encode(smiles)
    source_tokens = rep.tokenize(source_repr)
    if len(groups) == 0:
        return None
    return mol, source_tokens, groups


def _build_linker_like_infill_row(
    rep,
    *,
    source_smiles: str,
    fragment_a_ap: str,
    fragment_b_ap: str,
    linker_ap: str,
    task: str,
) -> Optional[dict[str, Any]]:
    try:
        left_tokens = _encode_completion_prompt_tokens(rep, fragment_a_ap)
        right_tokens = _encode_completion_prompt_tokens(rep, fragment_b_ap)
        target_tokens = _encode_completion_target_tokens(rep, linker_ap)
    except Exception:
        return None
    if not left_tokens or not right_tokens or not target_tokens:
        return None
    masked_tokens = list(left_tokens) + [TokenVocab.MASK] * len(target_tokens) + list(right_tokens)
    prompt_tokens = list(left_tokens) + [TokenVocab.MASK] + list(right_tokens)
    return {
        "source_smiles": source_smiles,
        "source_repr": "",
        "prompt_repr": rep.detokenize(prompt_tokens),
        "masked_repr": rep.detokenize(masked_tokens),
        "target_repr": rep.detokenize(target_tokens),
        "task": task,
        "fragment_a_ap": fragment_a_ap,
        "fragment_b_ap": fragment_b_ap,
        "linker_ap": linker_ap,
    }


def _build_full_sequence_mask_row(
    rep,
    *,
    source_smiles: str,
    source_tokens: Sequence[str],
    groups,
    mask_group_indices: Sequence[int],
    task: str,
) -> Optional[dict[str, Any]]:
    mask_set = set(int(i) for i in mask_group_indices)
    if not mask_set:
        return None
    masked_tokens: list[str] = []
    target_tokens: list[str] = []
    any_masked = False
    for idx, group in enumerate(groups):
        group_toks = list(group.tokens)
        if idx in mask_set:
            masked_tokens.extend([TokenVocab.MASK] * len(group_toks))
            target_tokens.extend(group_toks)
            any_masked = True
        else:
            masked_tokens.extend(group_toks)
    if not any_masked:
        return None
    return {
        "source_smiles": source_smiles,
        "source_repr": rep.detokenize(source_tokens),
        "prompt_repr": rep.detokenize(_collapse_mask_runs(masked_tokens)),
        "masked_repr": rep.detokenize(masked_tokens),
        "target_repr": rep.detokenize(target_tokens),
        "task": task,
    }


def _build_linker_group_mask_row(
    rep,
    *,
    source_smiles: str,
    ring_only: bool = False,
    task_name: str = "linker_design",
) -> Optional[dict[str, Any]]:
    encoded = _encode_stamp_groups_from_smiles(rep, source_smiles)
    if encoded is None:
        return None
    mol, source_tokens, groups = encoded
    brics_bonds = []
    for (a, b), _ in BRICS.FindBRICSBonds(mol):
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is not None:
            brics_bonds.append(bond.GetIdx())
    if len(brics_bonds) < 2:
        return None
    n_atoms = mol.GetNumAtoms()
    for b1, b2 in combinations(brics_bonds[:8], 2):
        frag_mol = Chem.FragmentOnBonds(mol, [b1, b2], addDummies=True)
        frag_sets = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)
        linker_atoms: Optional[set[int]] = None
        for atom_ids in frag_sets:
            dummy_count = 0
            orig_atoms: set[int] = set()
            for aid in atom_ids:
                atom = frag_mol.GetAtomWithIdx(int(aid))
                if atom.GetAtomicNum() == 0:
                    dummy_count += 1
                elif int(aid) < n_atoms:
                    orig_atoms.add(int(aid))
            if dummy_count == 2 and orig_atoms:
                linker_atoms = orig_atoms
                break
        if not linker_atoms:
            continue
        if ring_only:
            ring_present = False
            for atom_idx in linker_atoms:
                if mol.GetAtomWithIdx(atom_idx).IsInRing():
                    ring_present = True
                    break
            if not ring_present:
                continue
        mask_group_indices: list[int] = []
        for idx, group in enumerate(groups):
            atom_set = set(int(x) for x in group.atom_indices)
            if atom_set and atom_set.issubset(linker_atoms):
                mask_group_indices.append(idx)
        if not mask_group_indices:
            for idx, group in enumerate(groups):
                atom_set = set(int(x) for x in group.atom_indices)
                if atom_set and (len(atom_set & linker_atoms) / max(1, len(atom_set))) >= 0.5:
                    mask_group_indices.append(idx)
        row = _build_full_sequence_mask_row(
            rep,
            source_smiles=source_smiles,
            source_tokens=source_tokens,
            groups=groups,
            mask_group_indices=mask_group_indices,
            task=task_name,
        )
        if row is not None:
            row["n_mask_groups"] = int(len(mask_group_indices))
            return row
    return None


def build_fragment_infill_dataset(
    smiles_list: Sequence[str],
    output_path: str | Path,
    representation: str = "stamp",
    stamp_vocab: Optional[str] = None,
    min_mask_groups: int = 1,
    max_mask_groups: int = 3,
    max_items: Optional[int] = None,
    seed: int = 42,
    show_progress: bool = True,
) -> dict[str, Any]:
    rep = build_representation(representation, stamp_vocab=stamp_vocab)
    if isinstance(rep, UnsupportedRepresentation):
        raise RepresentationError(
            f"Cannot build fragment infill dataset for unsupported placeholder: {representation}"
        )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    written = 0
    iterator = smiles_list[: max_items if max_items is not None else None]
    iterator = iterable_with_progress(
        iterator,
        show_progress=show_progress,
        total=(min(len(smiles_list), int(max_items)) if max_items is not None else len(smiles_list)),
        desc=f"Build infill[{representation}]",
    )
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_smiles",
                "source_repr",
                "prompt_repr",
                "masked_repr",
                "target_repr",
                "mask_start_group",
                "mask_group_count",
            ],
        )
        writer.writeheader()
        for smiles in iterator:
            try:
                source_repr = rep.encode(smiles)
                tokens = rep.tokenize(source_repr)
            except Exception:
                continue
            if not tokens:
                continue
            if getattr(rep, "name", "") == "stamp":
                groups = _stamp_fragment_groups(tokens)
            else:
                groups = [[tok] for tok in tokens]
            masked = _mask_fragment_groups(groups, rng, min_mask_groups=min_mask_groups, max_mask_groups=max_mask_groups)
            if masked is None:
                continue
            prompt_tokens, masked_tokens, target_tokens, start, k = masked
            if not prompt_tokens or not target_tokens:
                continue
            writer.writerow(
                {
                    "source_smiles": smiles,
                    "source_repr": rep.detokenize(tokens),
                    "prompt_repr": rep.detokenize(prompt_tokens),
                    "masked_repr": rep.detokenize(masked_tokens),
                    "target_repr": rep.detokenize(target_tokens),
                    "mask_start_group": int(start),
                    "mask_group_count": int(k),
                }
            )
            written += 1
    return {
        "output_path": str(out),
        "representation": representation,
        "n_rows": written,
        "min_mask_groups": int(min_mask_groups),
        "max_mask_groups": int(max_mask_groups),
        "seed": int(seed),
    }


def build_motif_extension_dataset(
    smiles_list: Sequence[str],
    output_path: str | Path,
    representation: str = "stamp",
    stamp_vocab: Optional[str] = None,
    max_items: Optional[int] = None,
    seed: int = 42,
    show_progress: bool = True,
) -> dict[str, Any]:
    rep = build_representation(representation, stamp_vocab=stamp_vocab)
    if getattr(rep, "name", "") != "stamp":
        raise RepresentationError("Motif extension dataset currently requires STAMP representation.")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written = 0
    rows = smiles_list[: max_items if max_items is not None else None]
    iterator = iterable_with_progress(rows, show_progress=show_progress, total=(min(len(smiles_list), int(max_items)) if max_items is not None else len(smiles_list)), desc="Build motif extension")
    with out.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["source_smiles", "source_repr", "prompt_repr", "masked_repr", "target_repr", "task", "kept_group_index", "kept_motif"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for smiles in iterator:
            encoded = _encode_stamp_groups_from_smiles(rep, smiles)
            if encoded is None:
                continue
            _, source_tokens, groups = encoded
            keep_idx = rng.randrange(len(groups))
            row = _build_group_keep_row(rep, smiles, source_tokens, groups, [keep_idx])
            if row is None:
                continue
            row.update({"task": "motif_extension", "kept_group_index": int(keep_idx), "kept_motif": groups[keep_idx].motif})
            writer.writerow(row)
            written += 1
    return {"output_path": str(out), "representation": representation, "task": "motif_extension", "n_rows": written, "seed": int(seed)}


def build_superstructure_generation_dataset(
    smiles_list: Sequence[str],
    output_path: str | Path,
    representation: str = "stamp",
    stamp_vocab: Optional[str] = None,
    max_items: Optional[int] = None,
    min_keep_groups: int = 1,
    max_keep_groups: int = 3,
    seed: int = 42,
    show_progress: bool = True,
) -> dict[str, Any]:
    rep = build_representation(representation, stamp_vocab=stamp_vocab)
    if getattr(rep, "name", "") != "stamp":
        raise RepresentationError("Superstructure generation dataset currently requires STAMP representation.")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written = 0
    rows = smiles_list[: max_items if max_items is not None else None]
    iterator = iterable_with_progress(rows, show_progress=show_progress, total=(min(len(smiles_list), int(max_items)) if max_items is not None else len(smiles_list)), desc="Build superstructure")
    with out.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["source_smiles", "source_repr", "prompt_repr", "masked_repr", "target_repr", "task", "subtree_root_group", "kept_group_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for smiles in iterator:
            encoded = _encode_stamp_groups_from_smiles(rep, smiles)
            if encoded is None:
                continue
            _, source_tokens, groups = encoded
            candidates = []
            for idx, group in enumerate(groups):
                if min_keep_groups <= int(group.subtree_group_count) <= max_keep_groups:
                    candidates.append((idx, int(group.subtree_group_count)))
            if not candidates:
                continue
            start_idx, width = rng.choice(candidates)
            keep_indices = list(range(int(start_idx), int(start_idx + width)))
            row = _build_group_keep_row(rep, smiles, source_tokens, groups, keep_indices)
            if row is None:
                continue
            row.update({"task": "superstructure_generation", "subtree_root_group": int(start_idx), "kept_group_count": int(width)})
            writer.writerow(row)
            written += 1
    return {
        "output_path": str(out),
        "representation": representation,
        "task": "superstructure_generation",
        "n_rows": written,
        "seed": int(seed),
        "min_keep_groups": int(min_keep_groups),
        "max_keep_groups": int(max_keep_groups),
    }


def build_scaffold_decoration_dataset(
    smiles_list: Sequence[str],
    output_path: str | Path,
    representation: str = "stamp",
    stamp_vocab: Optional[str] = None,
    min_scaffold_heavy_atoms: int = 5,
    max_items: Optional[int] = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    rep = build_representation(representation, stamp_vocab=stamp_vocab)
    if getattr(rep, "name", "") != "stamp":
        raise RepresentationError("Scaffold decoration dataset currently requires STAMP representation.")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    rows = smiles_list[: max_items if max_items is not None else None]
    iterator = iterable_with_progress(rows, show_progress=show_progress, total=(min(len(smiles_list), int(max_items)) if max_items is not None else len(smiles_list)), desc="Build scaffold decoration")
    with out.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["source_smiles", "source_repr", "prompt_repr", "masked_repr", "target_repr", "task", "n_kept_groups", "scaffold_smiles"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for smiles in iterator:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None or scaffold.GetNumAtoms() == 0 or scaffold.GetNumHeavyAtoms() < int(min_scaffold_heavy_atoms):
                continue
            match = mol.GetSubstructMatch(scaffold)
            if not match:
                continue
            scaffold_atoms = set(int(x) for x in match)
            encoded = _encode_stamp_groups_from_smiles(rep, smiles)
            if encoded is None:
                continue
            _, source_tokens, groups = encoded
            keep_indices: list[int] = []
            for idx, group in enumerate(groups):
                atom_set = set(int(x) for x in group.atom_indices)
                if atom_set and atom_set.issubset(scaffold_atoms):
                    keep_indices.append(idx)
            if not keep_indices:
                for idx, group in enumerate(groups):
                    atom_set = set(int(x) for x in group.atom_indices)
                    if atom_set and (len(atom_set & scaffold_atoms) / max(1, len(atom_set))) >= 0.5:
                        keep_indices.append(idx)
            row = _build_group_keep_row(rep, smiles, source_tokens, groups, keep_indices)
            if row is None:
                continue
            row.update(
                {
                    "task": "scaffold_decoration",
                    "n_kept_groups": int(len(keep_indices)),
                    "scaffold_smiles": Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True),
                }
            )
            writer.writerow(row)
            written += 1
    return {
        "output_path": str(out),
        "representation": representation,
        "task": "scaffold_decoration",
        "n_rows": written,
        "min_scaffold_heavy_atoms": int(min_scaffold_heavy_atoms),
    }


def _init_eval_worker(representation: str) -> None:
    global _EVAL_WORKER_REP
    _EVAL_WORKER_REP = build_representation(representation)


def _eval_scaffold_prediction_local(rep, pred_repr: str, scaffold: str) -> tuple[str, int, int]:
    pred_s = ""
    is_valid = False
    preserve = False
    try:
        pred_s = rep.decode(pred_repr)
        is_valid = canonicalize_smiles(pred_s, require_single_component=True) is not None
    except Exception:
        is_valid = False

    if is_valid:
        gmol = Chem.MolFromSmiles(pred_s)
        smol = Chem.MolFromSmiles(scaffold)
        if gmol is not None and smol is not None and gmol.HasSubstructMatch(smol):
            preserve = True
    return pred_s, int(is_valid), int(preserve)


def _eval_scaffold_prediction_worker(payload: tuple[str, str]) -> tuple[str, int, int]:
    rep = _EVAL_WORKER_REP
    if rep is None:
        raise RuntimeError("Eval worker is not initialized.")
    pred_repr, scaffold = payload
    return _eval_scaffold_prediction_local(rep, pred_repr=pred_repr, scaffold=scaffold)


def _eval_linker_prediction_local(
    rep,
    pred_repr: str,
    frag_a: str,
    frag_b: str,
) -> tuple[str, str, int, int, int]:
    pred_s = ""
    pred_linker = ""
    valid = False
    preserve = False
    connect_ok = False
    try:
        pred_s = rep.decode(pred_repr)
        pred_linker = pred_s
        valid = canonicalize_smiles(pred_s, require_single_component=True) is not None
    except Exception:
        valid = False
    if valid:
        gm = Chem.MolFromSmiles(pred_s)
        qa = Chem.MolFromSmiles(frag_a)
        qb = Chem.MolFromSmiles(frag_b)
        if gm is not None and qa is not None and qb is not None:
            has_a = gm.HasSubstructMatch(qa)
            has_b = gm.HasSubstructMatch(qb)
            preserve = has_a and has_b
            if preserve and len(Chem.GetMolFrags(gm)) == 1:
                connect_ok = True
    return pred_s, pred_linker, int(valid), int(preserve), int(connect_ok)


def _eval_linker_prediction_from_linker_local(
    rep,
    pred_repr: str,
    frag_a: str,
    frag_b: str,
    frag_a_ap: str,
    frag_b_ap: str,
) -> tuple[str, str, int, int, int]:
    pred_s = ""
    pred_linker = ""
    valid = False
    preserve = False
    connect_ok = False
    try:
        pred_linker = rep.decode(pred_repr)
    except Exception:
        return pred_s, pred_linker, int(valid), int(preserve), int(connect_ok)

    assembled = _assemble_labeled_linker_fragments(frag_a_ap=frag_a_ap, frag_b_ap=frag_b_ap, linker_ap=pred_linker)
    if assembled is None:
        return pred_s, pred_linker, int(valid), int(preserve), int(connect_ok)

    pred_s = assembled
    valid = canonicalize_smiles(pred_s, require_single_component=True) is not None
    if valid:
        gm = Chem.MolFromSmiles(pred_s)
        qa = Chem.MolFromSmiles(frag_a)
        qb = Chem.MolFromSmiles(frag_b)
        if gm is not None and qa is not None and qb is not None:
            has_a = gm.HasSubstructMatch(qa)
            has_b = gm.HasSubstructMatch(qb)
            preserve = has_a and has_b
            connect_ok = bool(preserve and len(Chem.GetMolFrags(gm)) == 1)
    return pred_s, pred_linker, int(valid), int(preserve), int(connect_ok)


def _eval_linker_prediction_worker(
    payload: tuple[str, str, str, str, str, str],
) -> tuple[str, str, int, int, int]:
    rep = _EVAL_WORKER_REP
    if rep is None:
        raise RuntimeError("Eval worker is not initialized.")
    mode, pred_repr, frag_a, frag_b, frag_a_ap, frag_b_ap = payload
    if mode == "linker":
        return _eval_linker_prediction_from_linker_local(
            rep,
            pred_repr=pred_repr,
            frag_a=frag_a,
            frag_b=frag_b,
            frag_a_ap=frag_a_ap,
            frag_b_ap=frag_b_ap,
        )
    return _eval_linker_prediction_local(rep, pred_repr=pred_repr, frag_a=frag_a, frag_b=frag_b)


def build_scaffold_completion_dataset(
    smiles_list: Sequence[str],
    output_path: str | Path,
    min_scaffold_heavy_atoms: int = 5,
    max_items: Optional[int] = None,
    show_progress: bool = False,
    num_workers: int = 1,
    chunksize: int = 128,
) -> dict[str, Any]:
    rows = smiles_list[: max_items if max_items is not None else len(smiles_list)]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "scaffold_smiles", "scaffold_prompt"])
        if num_workers <= 0:
            num_workers = max(1, os.cpu_count() or 1)

        if num_workers == 1:
            iterator = iterable_with_progress(
                rows,
                show_progress=show_progress,
                total=len(rows),
                desc="Build scaffold dataset",
            )
            for s in iterator:
                total += 1
                row = _scaffold_row_from_smiles(s, min_scaffold_heavy_atoms=min_scaffold_heavy_atoms)
                if row is None:
                    continue
                writer.writerow([row[0], row[1], row[2]])
                kept += 1
        else:
            payloads = [(s, int(min_scaffold_heavy_atoms)) for s in rows]
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                mapped = pool.map(
                    _build_scaffold_worker,
                    payloads,
                    chunksize=max(1, int(chunksize)),
                )
                mapped = iterable_with_progress(
                    mapped,
                    show_progress=show_progress,
                    total=len(rows),
                    desc="Build scaffold dataset",
                )
                for row in mapped:
                    total += 1
                    if row is None:
                        continue
                    writer.writerow([row[0], row[1], row[2]])
                    kept += 1

    summary = {
        "output": str(out),
        "total_input": total,
        "kept": kept,
        "keep_rate": kept / max(1, total),
    }
    (out.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _strip_dummy_atoms(mol: Chem.Mol) -> Optional[Chem.Mol]:
    rw = Chem.RWMol(mol)
    to_remove = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0]
    for idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(idx)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def build_linker_design_dataset(
    smiles_list: Sequence[str],
    output_path: str | Path,
    max_items: Optional[int] = None,
    show_progress: bool = False,
    num_workers: int = 1,
    chunksize: int = 128,
) -> dict[str, Any]:
    rows = smiles_list[: max_items if max_items is not None else len(smiles_list)]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "smiles",
                "fragment_a",
                "fragment_b",
                "linker",
                "fragment_a_ap",
                "fragment_b_ap",
                "linker_ap",
            ]
        )
        if num_workers <= 0:
            num_workers = max(1, os.cpu_count() or 1)

        if num_workers == 1:
            iterator = iterable_with_progress(
                rows,
                show_progress=show_progress,
                total=len(rows),
                desc="Build linker dataset",
            )
            for s in iterator:
                total += 1
                row = _linker_row_from_smiles(s)
                if row is None:
                    continue
                writer.writerow([row[0], row[1], row[2], row[3], row[4], row[5], row[6]])
                kept += 1
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                mapped = pool.map(
                    _build_linker_worker,
                    rows,
                    chunksize=max(1, int(chunksize)),
                )
                mapped = iterable_with_progress(
                    mapped,
                    show_progress=show_progress,
                    total=len(rows),
                    desc="Build linker dataset",
                )
                for row in mapped:
                    total += 1
                    if row is None:
                        continue
                    writer.writerow([row[0], row[1], row[2], row[3], row[4], row[5], row[6]])
                    kept += 1

    summary = {
        "output": str(out),
        "total_input": total,
        "kept": kept,
        "keep_rate": kept / max(1, total),
    }
    (out.with_suffix(".summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_scaffold_morphing_dataset(
    smiles_list: Sequence[str],
    output_path: str | Path,
    max_items: Optional[int] = None,
    show_progress: bool = False,
    num_workers: int = 1,
    chunksize: int = 128,
) -> dict[str, Any]:
    rows = smiles_list[: max_items if max_items is not None else len(smiles_list)]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "smiles",
                "fragment_a",
                "fragment_b",
                "linker",
                "fragment_a_ap",
                "fragment_b_ap",
                "linker_ap",
            ]
        )
        if num_workers <= 0:
            num_workers = max(1, os.cpu_count() or 1)
        iterator = rows
        if num_workers == 1:
            iterator = iterable_with_progress(rows, show_progress=show_progress, total=len(rows), desc="Build scaffold morphing")
            for s in iterator:
                total += 1
                row = _linker_row_from_smiles(s)
                if row is None:
                    continue
                linker = Chem.MolFromSmiles(row[3])
                if linker is None or not linker.GetRingInfo().NumRings():
                    continue
                writer.writerow(list(row))
                kept += 1
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                mapped = pool.map(_build_linker_worker, rows, chunksize=max(1, int(chunksize)))
                mapped = iterable_with_progress(mapped, show_progress=show_progress, total=len(rows), desc="Build scaffold morphing")
                for row in mapped:
                    total += 1
                    if row is None:
                        continue
                    linker = Chem.MolFromSmiles(row[3])
                    if linker is None or not linker.GetRingInfo().NumRings():
                        continue
                    writer.writerow(list(row))
                    kept += 1

    summary = {
        "output": str(out),
        "task": "scaffold_morphing",
        "total_input": total,
        "kept": kept,
        "keep_rate": kept / max(1, total),
    }
    (out.with_suffix(".summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_linker_infill_dataset(
    smiles_list: Sequence[str],
    output_path: str | Path,
    representation: str = "stamp",
    stamp_vocab: Optional[str] = None,
    max_items: Optional[int] = None,
    show_progress: bool = False,
    num_workers: int = 1,
    chunksize: int = 128,
    ring_only: bool = False,
    task_name: str = "linker_design",
) -> dict[str, Any]:
    rep = build_representation(representation, stamp_vocab=stamp_vocab)
    if getattr(rep, "name", "") != "stamp":
        raise RepresentationError("Linker infill dataset currently requires STAMP representation.")
    rows = smiles_list[: max_items if max_items is not None else len(smiles_list)]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    total = 0
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_smiles",
                "source_repr",
                "prompt_repr",
                "masked_repr",
                "target_repr",
                "task",
                "n_mask_groups",
                "fragment_a_ap",
                "fragment_b_ap",
                "linker_ap",
            ],
        )
        writer.writeheader()
        if num_workers <= 0:
            num_workers = max(1, os.cpu_count() or 1)
        if num_workers == 1:
            iterator = iterable_with_progress(rows, show_progress=show_progress, total=len(rows), desc=f"Build {task_name}")
            for s in iterator:
                total += 1
                infill_row = _build_linker_group_mask_row(rep, source_smiles=s, ring_only=ring_only, task_name=task_name)
                if infill_row is None:
                    continue
                writer.writerow(infill_row)
                kept += 1
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                mapped = pool.map(_build_linker_worker, rows, chunksize=max(1, int(chunksize)))
                mapped = iterable_with_progress(mapped, show_progress=show_progress, total=len(rows), desc=f"Build {task_name}")
                for row in mapped:
                    total += 1
                    if row is None:
                        continue
                    infill_row = _build_linker_group_mask_row(rep, source_smiles=row[0], ring_only=ring_only, task_name=task_name)
                    if infill_row is None:
                        continue
                    writer.writerow(infill_row)
                    kept += 1
    summary = {
        "output_path": str(out),
        "representation": representation,
        "task": task_name,
        "total_input": total,
        "n_rows": kept,
        "ring_only": bool(ring_only),
    }
    (out.with_suffix(".summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _load_scaffold_rows(path: str | Path, max_rows: Optional[int] = None) -> list[dict[str, str]]:
    out = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                {
                    "smiles": row["smiles"],
                    "scaffold_smiles": row["scaffold_smiles"],
                    "scaffold_prompt": row.get("scaffold_prompt", row["scaffold_smiles"]),
                }
            )
            if max_rows is not None and len(out) >= max_rows:
                break
    return out


def _load_csv_records(path: str | Path, max_rows: Optional[int] = None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append({k: (v or "") for k, v in row.items()})
            if max_rows is not None and len(out) >= max_rows:
                break
    return out


def _sample_infill_orders(labels: torch.Tensor, random_ratio: float = 1.0) -> torch.Tensor:
    """Observed tokens first, masked targets afterwards in random order."""
    bsz, seqlen = labels.shape
    device = labels.device
    orders = torch.arange(seqlen, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, seqlen).clone()
    for b in range(bsz):
        masked = torch.nonzero(labels[b] != -100, as_tuple=False).squeeze(1)
        observed = torch.nonzero(labels[b] == -100, as_tuple=False).squeeze(1)
        if int(masked.numel()) > 1 and random.random() < random_ratio:
            perm = torch.randperm(int(masked.numel()), device=device)
            masked = masked[perm]
        orders[b] = torch.cat([observed, masked], dim=0)
    return orders


def _compute_ao_infill_loss(
    model: TinyDecoderLM,
    masked_ids: torch.Tensor,
    labels: torch.Tensor,
    random_ratio: float = 1.0,
) -> torch.Tensor:
    bsz, seqlen = masked_ids.shape
    device = masked_ids.device
    orders = _sample_infill_orders(labels, random_ratio=random_ratio)
    logits = model(masked_ids, orders=orders)
    batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, seqlen)
    targets = labels[batch_idx, orders]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )


def _tokenize_repr_space(text: str) -> list[str]:
    return [tok for tok in str(text).split() if tok]


def _encode_infill_records(
    records: Sequence[dict[str, str]],
    rep,
    max_seq_len: int,
) -> tuple[list[list[int]], list[list[int]], TokenVocab]:
    masked_tok: list[list[str]] = []
    source_tok: list[list[str]] = []
    for row in records:
        masked = _tokenize_repr_space(row.get("masked_repr", ""))
        target = _tokenize_repr_space(row.get("target_repr", ""))
        if not masked or not target:
            continue
        if masked.count(TokenVocab.MASK) != len(target):
            continue
        clean: list[str] = []
        t_iter = iter(target)
        labels: list[str] = []
        for tok in masked:
            if tok == TokenVocab.MASK:
                tgt = next(t_iter, None)
                if tgt is None:
                    clean = []
                    break
                clean.append(tgt)
                labels.append(tgt)
            else:
                clean.append(tok)
                labels.append(TokenVocab.PAD)
        if not clean:
            continue
        masked_tok.append(masked)
        source_tok.append(clean)

    motif_seed = None
    if hasattr(rep, "codec") and getattr(rep.codec, "motif_tokenizer", None) is not None:
        motif_seed = list(rep.codec.motif_tokenizer.idx2smi)
    vocab = TokenVocab.build(source_tok + masked_tok, min_freq=1, add_mask_token=True, seed_tokens=motif_seed)

    masked_ids: list[list[int]] = []
    label_ids: list[list[int]] = []
    for row in records:
        masked = _tokenize_repr_space(row.get("masked_repr", ""))
        target = _tokenize_repr_space(row.get("target_repr", ""))
        if not masked or not target:
            continue
        if masked.count(TokenVocab.MASK) != len(target):
            continue
        x_ids = vocab.encode_tokens(masked)
        labels: list[int] = []
        t_iter = iter(vocab.encode_tokens(target))
        for tok in masked:
            if tok == TokenVocab.MASK:
                tgt = next(t_iter, None)
                if tgt is None:
                    labels = []
                    break
                labels.append(int(tgt))
            else:
                labels.append(-100)
        if not labels or len(labels) != len(x_ids):
            continue
        masked_ids.append(x_ids[:max_seq_len])
        label_ids.append(labels[:max_seq_len])
    return masked_ids, label_ids, vocab


def _load_linker_rows(path: str | Path, max_rows: Optional[int] = None) -> list[dict[str, str]]:
    out = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                {
                    "smiles": row["smiles"],
                    "fragment_a": row["fragment_a"],
                    "fragment_b": row["fragment_b"],
                    "linker": row["linker"],
                    "fragment_a_ap": row.get("fragment_a_ap", row["fragment_a"]),
                    "fragment_b_ap": row.get("fragment_b_ap", row["fragment_b"]),
                    "linker_ap": row.get("linker_ap", row["linker"]),
                }
            )
            if max_rows is not None and len(out) >= max_rows:
                break
    return out


def train_completion_model(
    records: Sequence[dict[str, str]],
    prompt_key: str,
    target_key: str,
    representation: str,
    output_dir: str | Path,
    prompt_template: Optional[str] = None,
    target_template: Optional[str] = None,
    cfg: Optional[LMConfig] = None,
    show_progress: bool = True,
    device: Optional[str] = None,
    encode_workers: int = 1,
    encode_chunksize: int = 128,
    stamp_vocab: Optional[str] = None,
    prompt_is_repr: bool = False,
    target_is_repr: bool = False,
) -> dict[str, Any]:
    rep = build_representation(representation, stamp_vocab=stamp_vocab)
    if isinstance(rep, UnsupportedRepresentation):
        raise RepresentationError(
            f"Cannot train completion model for unsupported placeholder: {representation}"
        )

    cfg = cfg or LMConfig()
    model_type = _normalize_model_type(cfg.model_type)
    if model_type not in {"decoder", "ao_gpt"}:
        raise ValueError(f"train_completion_model only supports decoder|ao_gpt, got {cfg.model_type}")
    cfg.model_type = model_type
    _set_seed(cfg.seed)
    dev = _resolve_torch_device(device)

    if encode_workers <= 0:
        encode_workers = max(1, os.cpu_count() or 1)
    print(
        f"[train-completion] tokenize_workers={encode_workers}, tokenize_chunksize={max(1, int(encode_chunksize))}",
        flush=True,
    )
    seqs = _tokenize_completion_records(
        records,
        representation=representation,
        stamp_vocab=stamp_vocab,
        prompt_key=prompt_key,
        target_key=target_key,
        prompt_template=prompt_template,
        target_template=target_template,
        prompt_is_repr=bool(prompt_is_repr),
        target_is_repr=bool(target_is_repr),
        show_progress=show_progress,
        desc=f"Tokenize completion[{representation}]",
        num_workers=encode_workers,
        chunksize=encode_chunksize,
    )

    if not seqs:
        raise ValueError("No valid completion records after tokenization.")

    n_valid = max(1, int(0.1 * len(seqs)))
    train_tok = seqs[:-n_valid]
    valid_tok = seqs[-n_valid:]

    vocab = TokenVocab.build(train_tok, min_freq=1)
    ds_train = NextTokenDataset(_ids_with_special_tokens(train_tok, vocab), cfg.max_seq_len, vocab.pad_id)
    ds_valid = NextTokenDataset(_ids_with_special_tokens(valid_tok, vocab), cfg.max_seq_len, vocab.pad_id)

    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    use_amp, amp_dtype, use_scaler, amp_label = _resolve_amp_mode(cfg.amp, dev)

    collate_fn = build_dynamic_lm_collate_fn(
        vocab.pad_id,
        pad_to_multiple_of=(8 if dev.type == "cuda" else None),
    )
    pin_memory = dev.type == "cuda"
    train_loader_kwargs: Dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "drop_last": False,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "num_workers": max(0, int(cfg.num_data_workers)),
    }
    valid_loader_kwargs: Dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
        "num_workers": max(0, int(cfg.num_data_workers)),
    }
    if cfg.num_data_workers > 0:
        pf = max(2, int(cfg.prefetch_factor))
        pw = bool(cfg.persistent_data_workers)
        train_loader_kwargs["prefetch_factor"] = pf
        train_loader_kwargs["persistent_workers"] = pw
        valid_loader_kwargs["prefetch_factor"] = pf
        valid_loader_kwargs["persistent_workers"] = pw
    loader_train = DataLoader(ds_train, **train_loader_kwargs)
    loader_valid = DataLoader(ds_valid, **valid_loader_kwargs)

    model = TinyDecoderLM(vocab_size=len(vocab.itos), cfg=cfg)
    if cfg.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[misc]
            print("[train-completion] torch.compile enabled", flush=True)
        except Exception as exc:
            print(f"[train-completion] torch.compile unavailable/fallback: {exc}", flush=True)
    model.to(dev)
    n_model_params = int(sum(p.numel() for p in model.parameters()))
    if dev.type == "cuda":
        gpu_name = torch.cuda.get_device_name(dev)
        print(f"[train-completion] device={dev} ({gpu_name})", flush=True)
    else:
        print(f"[train-completion] device={dev}", flush=True)
    print(f"[train-completion] n_model_params={n_model_params} (~{n_model_params/1e6:.2f}M)", flush=True)
    train_effective_max_seq_len = max((len(x) for x, _ in ds_train.samples), default=0)
    valid_effective_max_seq_len = max((len(x) for x, _ in ds_valid.samples), default=0)
    print(
        f"[train-completion] dynamic_padding=batch_max (cfg_max_seq_len={cfg.max_seq_len}, "
        f"train_effective_max_seq_len={train_effective_max_seq_len}, "
        f"valid_effective_max_seq_len={valid_effective_max_seq_len})",
        flush=True,
    )
    print(
        f"[train-completion] amp={amp_label}, tf32={bool(cfg.tf32)}, "
        f"num_data_workers={max(0, int(cfg.num_data_workers))}, prefetch_factor={max(2, int(cfg.prefetch_factor))}",
        flush=True,
    )

    adamw_kwargs: Dict[str, Any] = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    if dev.type == "cuda":
        try:
            adamw_kwargs["fused"] = True
        except Exception:
            pass
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    except TypeError:
        adamw_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)  # pragma: no cover
    non_blocking = dev.type == "cuda"

    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        train_iter = iterable_with_progress(
            loader_train,
            show_progress=show_progress,
            total=len(loader_train),
            desc=f"Train completion[{representation}] ep{epoch}",
        )
        for x, y in train_iter:
            x = x.to(dev, non_blocking=non_blocking)
            y = y.to(dev, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                if use_amp
                else nullcontext()
            )
            with amp_ctx:
                if model_type == "ao_gpt":
                    loss = _compute_ao_gpt_loss(model, x, y, cfg, random_ratio=0.9)
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for x, y in loader_valid:
                x = x.to(dev, non_blocking=non_blocking)
                y = y.to(dev, non_blocking=non_blocking)
                amp_ctx = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                    if use_amp
                    else nullcontext()
                )
                with amp_ctx:
                    if model_type == "ao_gpt":
                        loss = _compute_ao_gpt_loss(model, x, y, cfg, random_ratio=0.9)
                    else:
                        logits = model(x)
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
                valid_losses.append(float(loss.item()))

        tr = float(np.mean(train_losses)) if train_losses else float("nan")
        va = float(np.mean(valid_losses)) if valid_losses else float("nan")
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr,
                "valid_loss": va,
                "train_ppl": float(math.exp(tr)) if not math.isnan(tr) else float("nan"),
                "valid_ppl": float(math.exp(va)) if not math.isnan(va) else float("nan"),
            }
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp_vocab_rel = None
    if stamp_vocab is not None:
        import shutil
        dest = out_dir / "stamp_vocab.txt"
        shutil.copy2(stamp_vocab, dest)
        stamp_vocab_rel = "stamp_vocab.txt"

    torch.save(
        {
            "model_state": (model._orig_mod if hasattr(model, "_orig_mod") else model).state_dict(),
            "cfg": vars(cfg),
            "representation": representation,
            "model_type": model_type,
            "stamp_vocab": stamp_vocab_rel,
            "task": "completion",
        },
        out_dir / "model.pt",
    )
    (out_dir / "vocab.json").write_text(
        json.dumps(vocab.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "representation": representation,
                "prompt_key": prompt_key,
                "target_key": target_key,
                "prompt_template": prompt_template,
                "target_template": target_template,
                "prompt_is_repr": bool(prompt_is_repr),
                "target_is_repr": bool(target_is_repr),
                "n_records": len(records),
                "model_type": model_type,
                "cfg": vars(cfg),
                "device": str(dev),
                "n_model_params": n_model_params,
                "padding_mode": "dynamic_batch_max",
                "train_effective_max_seq_len": train_effective_max_seq_len,
                "valid_effective_max_seq_len": valid_effective_max_seq_len,
                "tokenize_workers": encode_workers,
                "tokenize_chunksize": max(1, int(encode_chunksize)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {"output_dir": str(out_dir), "history": history, "device": str(dev), "n_model_params": n_model_params}


def _extract_completion_suffix(tokens: Sequence[str]) -> list[str]:
    if "<SEP>" not in tokens:
        return list(tokens)
    i = tokens.index("<SEP>")
    return list(tokens[i + 1 :])


def _fill_single_mask(prompt_tokens: Sequence[str], fill_tokens: Sequence[str]) -> list[str]:
    tokens = list(prompt_tokens)
    if TokenVocab.MASK not in tokens:
        return tokens + list(fill_tokens)
    idx = tokens.index(TokenVocab.MASK)
    return tokens[:idx] + list(fill_tokens) + tokens[idx + 1 :]


def _sample_ao_infill_sequences(
    model: TinyDecoderLM,
    vocab: TokenVocab,
    prompt_tokens: Sequence[str],
    n_samples: int = 1,
    temperature: float = 1.0,
    top_k: int = 50,
    device: Optional[torch.device] = None,
) -> list[list[str]]:
    dev = device or next(model.parameters()).device
    prompt_ids = vocab.encode_tokens(list(prompt_tokens))
    x = torch.tensor(prompt_ids, dtype=torch.long, device=dev).unsqueeze(0).repeat(n_samples, 1)
    max_steps = int((x == vocab.mask_id).sum(dim=1).max().item())
    if max_steps <= 0:
        return [vocab.decode_ids(prompt_ids) for _ in range(n_samples)]

    for _ in range(max_steps):
        remaining_mask = x == vocab.mask_id
        if not bool(remaining_mask.any().item()):
            break
        orders = []
        for b in range(x.size(0)):
            observed = torch.nonzero(~remaining_mask[b], as_tuple=False).squeeze(1)
            masked = torch.nonzero(remaining_mask[b], as_tuple=False).squeeze(1)
            orders.append(torch.cat([observed, masked], dim=0))
        orders_t = torch.stack(orders, dim=0)
        with torch.no_grad():
            logits = model(x, orders=orders_t)
        logits_orig = torch.empty_like(logits)
        logits_orig.scatter_(1, orders_t.unsqueeze(-1).expand_as(logits), logits)
        for suppressed_id in (vocab.pad_id, vocab.bos_id, vocab.mask_id, vocab.eos_id):
            if 0 <= int(suppressed_id) < logits_orig.size(-1):
                logits_orig[:, :, int(suppressed_id)] = float("-inf")
        logits_orig = _apply_top_k_logits(logits_orig, top_k=top_k)
        probs = torch.softmax(logits_orig, dim=-1)
        conf = probs.max(dim=-1).values.masked_fill(~remaining_mask, float("-inf"))
        next_pos = conf.argmax(dim=-1)
        chosen_logits = logits_orig[torch.arange(x.size(0), device=dev), next_pos]
        sampled = torch.argmax(_add_gumbel_noise(chosen_logits, temperature=temperature), dim=-1).to(dtype=torch.long)
        x[torch.arange(x.size(0), device=dev), next_pos] = sampled
    return [vocab.decode_ids(row) for row in x.detach().cpu().tolist()]


def evaluate_fragment_infill(
    model_dir: str | Path,
    infill_csv: str | Path,
    output_dir: str | Path,
    max_eval: Optional[int] = 1000,
    n_samples_per_prompt: int = 1,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 50,
    show_progress: bool = True,
    device: str | None = None,
    eval_workers: int = 1,
    eval_chunksize: int = 256,
) -> dict[str, Any]:
    model, vocab, rep, dev = load_trained_model(model_dir, device=device)
    rows = _load_csv_records(infill_csv)
    if max_eval is not None:
        rows = rows[: max(1, int(max_eval))]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated_rows: list[dict[str, Any]] = []
    reconstructed_smiles: list[str] = []
    target_smiles: list[str] = []

    iterator = iterable_with_progress(rows, show_progress=show_progress, total=len(rows), desc="Eval infill")
    for row in iterator:
        prompt_raw = row.get("masked_repr", "")
        target_raw = row.get("target_repr", "")
        source_smiles = row.get("source_smiles", "")
        task_name = row.get("task", "")
        try:
            prompt_tokens = _encode_completion_prompt_tokens(rep, prompt_raw, is_repr=True)
        except Exception:
            continue
        seqs = _sample_ao_infill_sequences(
            model,
            vocab,
            prompt_tokens=prompt_tokens,
            n_samples=n_samples_per_prompt,
            temperature=temperature,
            top_k=top_k,
            device=dev,
        )
        for toks in seqs:
            full_repr = rep.detokenize(toks)
            target_tokens = _encode_completion_target_tokens(rep, target_raw, is_repr=True)
            prompt_toks = _encode_completion_prompt_tokens(rep, prompt_raw, is_repr=True)
            pred_target_tokens: list[str] = []
            for src_tok, pred_tok in zip(prompt_toks, toks):
                if src_tok == TokenVocab.MASK:
                    pred_target_tokens.append(pred_tok)
            pred_repr = rep.detokenize(pred_target_tokens)
            has_attachment_linker_fields = bool(row.get("fragment_a_ap")) and bool(row.get("fragment_b_ap")) and bool(row.get("linker_ap"))
            if task_name in {"linker_design", "scaffold_morphing"} and has_attachment_linker_fields:
                pred_linker = ""
                pred_smiles = ""
                try:
                    pred_linker = rep.decode(pred_repr)
                except Exception:
                    pred_linker = ""
                if pred_linker:
                    pred_smiles = _assemble_labeled_linker_fragments(
                        frag_a_ap=row.get("fragment_a_ap", ""),
                        frag_b_ap=row.get("fragment_b_ap", ""),
                        linker_ap=pred_linker,
                    ) or ""
            else:
                try:
                    pred_smiles = rep.decode(full_repr)
                except Exception:
                    pred_smiles = ""
            generated_rows.append(
                {
                    "source_smiles": source_smiles,
                    "task": task_name,
                    "prompt_repr": prompt_raw,
                    "target_repr": target_raw,
                    "pred_target_repr": pred_repr,
                    "pred_full_repr": full_repr,
                    "pred_smiles": pred_smiles,
                    "valid": int(bool(pred_smiles)),
                    "exact_target_match": int(pred_repr == target_raw),
                }
            )
            reconstructed_smiles.append(pred_smiles)
            target_smiles.append(source_smiles)

    canonical_pred = [canonicalize_smiles(s, require_single_component=True) if s else None for s in reconstructed_smiles]
    canonical_target = [canonicalize_smiles(s, require_single_component=True) if s else None for s in target_smiles]
    exact_full_match = sum(int(p is not None and p == t) for p, t in zip(canonical_pred, canonical_target))
    metrics = evaluate_generated_smiles(
        reconstructed_smiles,
        train_smiles=target_smiles,
        reference_smiles=target_smiles,
        generated_canonical=canonical_pred,
        assume_train_canonical=False,
        reference_max_samples=None,
        eval_workers=eval_workers,
        eval_chunksize=eval_chunksize,
    )
    metrics.update(
        {
            "model_dir": str(model_dir),
            "infill_csv": str(infill_csv),
            "n_eval": len(generated_rows),
            "n_samples_per_prompt": int(n_samples_per_prompt),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "exact_full_match_count": int(exact_full_match),
            "exact_full_match_rate": float(exact_full_match) / max(1, len(generated_rows)),
            "exact_target_match_rate": float(sum(int(r["exact_target_match"]) for r in generated_rows)) / max(1, len(generated_rows)),
        }
    )

    with (out / "fragment_infill_predictions.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_smiles",
                "task",
                "prompt_repr",
                "target_repr",
                "pred_target_repr",
                "pred_full_repr",
                "pred_smiles",
                "valid",
                "exact_target_match",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in generated_rows:
            writer.writerow(row)

    (out / "fragment_infill_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def evaluate_scaffold_completion(
    model_dir: str | Path,
    scaffold_csv: str | Path,
    output_dir: str | Path,
    max_eval: Optional[int] = 1000,
    n_samples_per_prompt: int = 1,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    show_progress: bool = True,
    device: str | None = None,
    eval_workers: int = 1,
    eval_chunksize: int = 256,
) -> dict[str, Any]:
    model, vocab, rep, dev = load_trained_model(model_dir, device=device)
    rows = _load_scaffold_rows(scaffold_csv, max_rows=max_eval)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    generated_rows = []
    recovered_smiles: list[str] = []
    if eval_workers <= 0:
        eval_workers = max(1, os.cpu_count() or 1)
    eval_chunksize = max(1, int(eval_chunksize))
    print(
        f"[eval-scaffold] eval_workers={eval_workers}, eval_chunksize={eval_chunksize}",
        flush=True,
    )

    iterator = iterable_with_progress(rows, show_progress=show_progress, total=len(rows), desc="Eval scaffold")
    for row in iterator:
        target = row["smiles"]
        scaffold = row["scaffold_smiles"]
        scaffold_prompt = row.get("scaffold_prompt", scaffold)

        try:
            prompt_tokens = _encode_completion_prompt_tokens(rep, scaffold_prompt) + ["<SEP>"]
        except Exception:
            continue

        seqs = sample_model_sequences(
            model,
            vocab,
            n_samples=n_samples_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            prompt_tokens=prompt_tokens,
            device=dev,
        )

        for toks in seqs:
            suffix = _extract_completion_suffix(toks)
            pred_repr = rep.detokenize(suffix)
            generated_rows.append(
                {
                    "target_smiles": target,
                    "scaffold_smiles": scaffold,
                    "pred_repr": pred_repr,
                    "pred_smiles": "",
                    "valid": 0,
                    "scaffold_preserved": 0,
                }
            )

    eval_payloads = [(r["pred_repr"], r["scaffold_smiles"]) for r in generated_rows]
    if eval_workers == 1:
        results = []
        for payload in iterable_with_progress(
            eval_payloads,
            show_progress=show_progress,
            total=len(eval_payloads),
            desc="Post-eval scaffold",
        ):
            results.append(_eval_scaffold_prediction_local(rep, pred_repr=payload[0], scaffold=payload[1]))
    else:
        with ProcessPoolExecutor(
            max_workers=eval_workers,
            initializer=_init_eval_worker,
            initargs=(rep.name,),
        ) as pool:
            mapped = pool.map(
                _eval_scaffold_prediction_worker,
                eval_payloads,
                chunksize=eval_chunksize,
            )
            mapped = iterable_with_progress(
                mapped,
                show_progress=show_progress,
                total=len(eval_payloads),
                desc="Post-eval scaffold",
            )
            results = list(mapped)

    valid = 0
    preserved = 0
    success = 0
    for i, (pred_s, is_valid_i, preserve_i) in enumerate(results):
        generated_rows[i]["pred_smiles"] = pred_s
        generated_rows[i]["valid"] = int(is_valid_i)
        generated_rows[i]["scaffold_preserved"] = int(preserve_i)
        if is_valid_i:
            valid += 1
            recovered_smiles.append(pred_s)
        if preserve_i:
            preserved += 1
        if is_valid_i and preserve_i:
            success += 1

    n_total = len(generated_rows)
    unique_valid = len(
        set(
            canonicalize_smiles(s, require_single_component=True)
            for s in recovered_smiles
            if canonicalize_smiles(s, require_single_component=True) is not None
        )
    )

    metrics = {
        "n_total": n_total,
        "validity": valid / max(1, n_total),
        "scaffold_preservation_rate": preserved / max(1, valid),
        "completion_success_rate": success / max(1, n_total),
        "diversity": unique_valid / max(1, valid),
    }

    with (out / "scaffold_completion_predictions.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_smiles",
                "scaffold_smiles",
                "pred_repr",
                "pred_smiles",
                "valid",
                "scaffold_preserved",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(generated_rows)

    (out / "scaffold_completion_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def evaluate_linker_design(
    model_dir: str | Path,
    linker_csv: str | Path,
    output_dir: str | Path,
    max_eval: Optional[int] = 1000,
    n_samples_per_prompt: int = 1,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    show_progress: bool = True,
    device: str | None = None,
    eval_workers: int = 1,
    eval_chunksize: int = 256,
) -> dict[str, Any]:
    model, vocab, rep, dev = load_trained_model(model_dir, device=device)
    rows = _load_linker_rows(linker_csv, max_rows=max_eval)
    meta_target_key = ""
    meta_path = Path(model_dir) / "meta.json"
    if meta_path.exists():
        try:
            meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
            meta_target_key = str(meta_obj.get("target_key", "")).strip()
        except Exception:
            meta_target_key = ""
    linker_mode = meta_target_key in {"linker_ap", "linker_with_attachments"}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    generated_rows = []
    valid_smiles: list[str] = []
    if eval_workers <= 0:
        eval_workers = max(1, os.cpu_count() or 1)
    eval_chunksize = max(1, int(eval_chunksize))
    print(
        f"[eval-linker] eval_workers={eval_workers}, eval_chunksize={eval_chunksize}, mode={'linker' if linker_mode else 'full'}",
        flush=True,
    )

    iterator = iterable_with_progress(rows, show_progress=show_progress, total=len(rows), desc="Eval linker")
    for row in iterator:
        target = row["smiles"]
        frag_a = row["fragment_a"]
        frag_b = row["fragment_b"]
        frag_a_ap = row.get("fragment_a_ap", frag_a)
        frag_b_ap = row.get("fragment_b_ap", frag_b)
        prompt_a = frag_a_ap if linker_mode else frag_a
        prompt_b = frag_b_ap if linker_mode else frag_b
        prompt = f"{prompt_a}.{prompt_b}"

        try:
            prompt_tokens = _encode_completion_prompt_tokens(rep, prompt) + ["<SEP>"]
        except Exception:
            continue

        seqs = sample_model_sequences(
            model,
            vocab,
            n_samples=n_samples_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            prompt_tokens=prompt_tokens,
            device=dev,
        )

        for toks in seqs:
            suffix = _extract_completion_suffix(toks)
            pred_repr = rep.detokenize(suffix)
            generated_rows.append(
                {
                    "target_smiles": target,
                    "fragment_a": frag_a,
                    "fragment_b": frag_b,
                    "fragment_a_ap": frag_a_ap,
                    "fragment_b_ap": frag_b_ap,
                    "pred_repr": pred_repr,
                    "pred_linker": "",
                    "pred_smiles": "",
                    "valid": 0,
                    "fragment_preserved": 0,
                    "connection_success": 0,
                }
            )

    mode_name = "linker" if linker_mode else "full"
    eval_payloads = [
        (
            mode_name,
            r["pred_repr"],
            r["fragment_a"],
            r["fragment_b"],
            r["fragment_a_ap"],
            r["fragment_b_ap"],
        )
        for r in generated_rows
    ]
    if eval_workers == 1:
        results = []
        for payload in iterable_with_progress(
            eval_payloads,
            show_progress=show_progress,
            total=len(eval_payloads),
            desc="Post-eval linker",
        ):
            results.append(
                _eval_linker_prediction_local(
                    rep,
                    pred_repr=payload[1],
                    frag_a=payload[2],
                    frag_b=payload[3],
                )
                if payload[0] == "full"
                else _eval_linker_prediction_from_linker_local(
                    rep,
                    pred_repr=payload[1],
                    frag_a=payload[2],
                    frag_b=payload[3],
                    frag_a_ap=payload[4],
                    frag_b_ap=payload[5],
                )
            )
    else:
        with ProcessPoolExecutor(
            max_workers=eval_workers,
            initializer=_init_eval_worker,
            initargs=(rep.name,),
        ) as pool:
            mapped = pool.map(
                _eval_linker_prediction_worker,
                eval_payloads,
                chunksize=eval_chunksize,
            )
            mapped = iterable_with_progress(
                mapped,
                show_progress=show_progress,
                total=len(eval_payloads),
                desc="Post-eval linker",
            )
            results = list(mapped)

    n_valid = 0
    n_preserve = 0
    n_connect = 0
    for i, (pred_s, pred_linker, valid_i, preserve_i, connect_i) in enumerate(results):
        generated_rows[i]["pred_linker"] = pred_linker
        generated_rows[i]["pred_smiles"] = pred_s
        generated_rows[i]["valid"] = int(valid_i)
        generated_rows[i]["fragment_preserved"] = int(preserve_i)
        generated_rows[i]["connection_success"] = int(connect_i)
        if valid_i:
            n_valid += 1
            valid_smiles.append(pred_s)
        if preserve_i:
            n_preserve += 1
        if connect_i:
            n_connect += 1

    n_total = len(generated_rows)
    unique_valid = len(
        set(
            canonicalize_smiles(s, require_single_component=True)
            for s in valid_smiles
            if canonicalize_smiles(s, require_single_component=True) is not None
        )
    )

    metrics = {
        "n_total": n_total,
        "validity": n_valid / max(1, n_total),
        "fragment_preservation_rate": n_preserve / max(1, n_valid),
        "connection_success_rate": n_connect / max(1, n_total),
        "diversity": unique_valid / max(1, n_valid),
    }

    with (out / "linker_design_predictions.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_smiles",
                "fragment_a",
                "fragment_b",
                "fragment_a_ap",
                "fragment_b_ap",
                "pred_repr",
                "pred_linker",
                "pred_smiles",
                "valid",
                "fragment_preserved",
                "connection_success",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(generated_rows)

    (out / "linker_design_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics
