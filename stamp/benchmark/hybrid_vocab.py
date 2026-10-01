"""Hybrid motif+char STAMP vocabulary.

Strategy:
  - High-frequency motifs (freq >= cutoff) stay as atomic tokens — one vocab id
    per motif. Rare motifs are always expanded to SMILES char tokens, letting
    the model learn them via atom-level shared structure.
  - During training, even atomic motifs are expanded to chars with a small
    probability `p(freq)` (log-interpolated: higher near the cutoff, lower for
    the very frequent motifs). This closes the train/inference gap for the
    char form, which is the only form in which OOV motifs can appear.
  - No special motif-boundary markers are added. The STAMP structural tokens
    `[J_*]`, `[B_*]`, `[S_*]`, `[END]` already delimit motifs.

Vocab layout (stoi / itos):
  0..4:   <pad> <bos> <eos> <unk> <mask>
  5..K-1: STAMP structural tokens ([J_*], [B_*], [S_*], [END], [BACK], [Ring])
  K..K+C: SMILES char tokens (regex-level SMILES tokens, 49 distinct on our
          v3 motif corpus)
  K+C..:  atomic motif SMILES (freq >= cutoff). Single-token-SMILES motifs
          like ``C`` or ``[CH2]`` are NOT duplicated here; they share the char
          id that was allocated above.

Tokens stream out of the STAMP canonical form (space-separated motif tokens
+ structural tokens) via ``encode_stream``, which expands rare motifs and
probabilistically expands atomic motifs.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Covers motif SMILES used by the merged ZINC vocabulary. Bracketed groups,
# "%NN" ring ids, and multi-char atoms must come before shorter fallbacks so
# the regex does not split tokens such as "%10", "Si", or "Zn".
SMILES_RE = re.compile(
    r"(\[[^\]]+\]|%[0-9]{2}|Br|Cl|[A-Z][a-z]?|[bcnops]|[0-9]|\(|\)|=|#|-|\+|/|\\|@|\.|:|\*|~|\$|>|<)"
)


def smiles_char_tokens(smiles: str) -> List[str]:
    """Tokenize a SMILES string into regex-level atomic tokens.

    Raises ValueError if the regex does not cover the whole string.
    """
    toks = SMILES_RE.findall(smiles)
    if "".join(toks) != smiles:
        raise ValueError(f"SMILES char tokenizer did not cover input: {smiles!r} -> {toks!r}")
    return toks


def _read_frequency_file(path: str | Path) -> Dict[str, int]:
    """Read motif frequencies from either a motif-vocab or full-frequency file."""
    out: Dict[str, int] = {}
    freq_col = 2
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            raw = line.rstrip("\n")
            if not raw:
                continue
            if raw.startswith("#"):
                if "motif_smi" in raw and "freq" in raw and "in_source_vocab" in raw:
                    freq_col = 1
                continue
            if line_no == 0:
                try:
                    json.loads(raw)
                    continue
                except json.JSONDecodeError:
                    pass
            parts = raw.split("\t")
            if len(parts) != 3:
                continue
            try:
                out[parts[0]] = int(parts[freq_col])
            except (IndexError, ValueError):
                continue
    return out


STAMP_STRUCT_RE = re.compile(r"^\[(?:J_\d+|B_\d+|S_(?:R|S|E|Z)|Ring)\]$")
CONDITION_RE = re.compile(r"^\[C_[A-Z0-9_]+\]$")
FCSTAMP_CONTROL_TOKENS = (
    "[TASK_DENO]",
    "[TASK_LINKER]",
    "[TASK_EXTEND]",
    "[TASK_DECORATE]",
    "[TASK_REMASK]",
    "[TASK_SCAFFOLD_MORPH]",
    "[TASK_SUPERSTRUCTURE]",
    "[INFILL]",
    "[INFILL_PATH]",
    "[INFILL_SUBTREE]",
    "[INFILL_TAIL]",
    "[GEN]",
    "[FRAG_SEP]",
    "[EMPTY]",
    "[EOS]",
    "[/FILL]",
    *tuple(f"[INFILL_{i}]" for i in range(16)),
    *tuple(f"[FILL_{i}]" for i in range(16)),
)
FCSTAMP_CONTROL_SET = set(FCSTAMP_CONTROL_TOKENS)
FCSTAMP_CONTROL_RE = re.compile(r"^\[(?:TASK_[A-Z0-9_]+|INFILL(?:_[A-Z0-9]+)?|FILL_\d+)\]$")
COUNTERION_TOKENS = {
    "[H+]",
    "[H-]",
    "[Li+]",
    "[Na+]",
    "[K+]",
    "[Rb+]",
    "[Cs+]",
    "[F-]",
    "[Cl-]",
    "[Br-]",
    "[I-]",
    "[O-]",
    "[OH-]",
    "[NH4+]",
    "[Mg+2]",
    "[Mg++]",
    "[Ca+2]",
    "[Ca++]",
    "[Sr+2]",
    "[Ba+2]",
    "[Al+3]",
    "[Fe+2]",
    "[Fe+3]",
    "[Cu+]",
    "[Cu+2]",
    "[Zn+2]",
    "[Ag+]",
    "[Mn+2]",
    "[Li]",
    "[Na]",
    "[K]",
    "[Mg]",
    "[Ca]",
    "[Al]",
    "[Mn]",
    "[Fe]",
    "[Co]",
    "[Ni]",
    "[Cu]",
    "[Zn]",
    "[Ag]",
    "[Pd]",
    "[Pt]",
    "[Au]",
    "[Hg]",
}
COMMON_BRACKET_ATOM_TOKENS = {
    "[2H]",
    "[3H]",
    "[11C]",
    "[13C]",
    "[13CH]",
    "[13CH2]",
    "[13CH3]",
    "[14C]",
    "[15N]",
    "[18F]",
    "[18O]",
    "[C-]",
    "[CH-]",
    "[CH2-]",
    "[N-]",
    "[NH-]",
    "[NH2-]",
    "[NH2+]",
    "[NH3+]",
    "[O-]",
    "[S-]",
    "[Se-]",
}


def is_stamp_structural(tok: str) -> bool:
    return bool(STAMP_STRUCT_RE.match(tok)) or tok in {"[END]", "[BACK]", "."}


def is_condition_token(tok: str) -> bool:
    return bool(CONDITION_RE.match(tok))


def is_fcstamp_control_token(tok: str) -> bool:
    return tok in FCSTAMP_CONTROL_SET or bool(FCSTAMP_CONTROL_RE.match(tok))


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
MASK = "<mask>"
_SPECIAL = (PAD, BOS, EOS, UNK, MASK)


@dataclass
class HybridVocab:
    """Vocabulary + atomic-motif expansion table.

    ``itos`` is the master token list (indices = token ids). ``atomic_motifs``
    maps each atomic-motif string to its id (may overlap char-token ids for
    length-1 SMILES motifs). ``motif_freq`` stores the raw motif frequency
    (for fallback-probability scheduling). ``motif_expansion`` caches the
    SMILES char-token id list for every atomic motif — needed at train time
    when we roll a fallback expansion.
    """

    itos: List[str]
    atomic_motifs: Dict[str, int] = field(default_factory=dict)
    motif_freq: Dict[str, int] = field(default_factory=dict)
    motif_expansion: Dict[str, List[int]] = field(default_factory=dict)
    freq_cutoff: int = 5000

    def __post_init__(self) -> None:
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.bos_id = self.stoi[BOS]
        self.eos_id = self.stoi[EOS]
        self.unk_id = self.stoi[UNK]
        self.mask_id = self.stoi[MASK]
        # Freq bounds for log-interpolated fallback prob
        freqs = list(self.motif_freq.values()) or [self.freq_cutoff]
        self._log_freq_max = math.log(max(freqs))
        self._log_freq_min = math.log(self.freq_cutoff)

    # ---- Building ----

    @classmethod
    def build(
        cls,
        motif_vocab_path: str | Path,
        freq_cutoff: int = 5000,
        extra_stamp_tokens: Optional[Sequence[str]] = None,
        frequency_path: Optional[str | Path] = None,
        max_atomic_motifs: int = 0,
    ) -> "HybridVocab":
        """Read a v3_cm_union-style motif vocab file and build a HybridVocab.

        The file is tab-separated with three columns (smiles, num_heavy_atoms,
        freq) and a leading JSON-config line.
        """
        path = Path(motif_vocab_path)
        motifs: List[Tuple[str, int]] = []
        with path.open("r", encoding="utf-8") as f:
            first = f.readline()
            try:
                json.loads(first)
            except json.JSONDecodeError:
                # Not a JSON header; rewind by re-reading as a data line.
                parts = first.rstrip("\n").split("\t")
                if len(parts) == 3:
                    motifs.append((parts[0], int(parts[2])))
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                motifs.append((parts[0], int(parts[2])))

        if frequency_path:
            freq_override = _read_frequency_file(frequency_path)
            motifs = [(smi, int(freq_override.get(smi, 0))) for smi, _ in motifs]

        atomic = [(smi, freq) for smi, freq in motifs if freq >= freq_cutoff]
        if max_atomic_motifs > 0 and len(atomic) > max_atomic_motifs:
            atomic = sorted(atomic, key=lambda x: (-x[1], x[0]))[:max_atomic_motifs]

        # Collect SMILES char tokens across ALL motifs (so rare motifs can be
        # expanded) plus the chars appearing inside atomic motifs (for the
        # training-time fallback expansion).
        char_set: set[str] = set()
        for smi, _ in motifs:
            for t in smiles_char_tokens(smi):
                char_set.add(t)
        # Keep rare-but-valid SMILES syntax representable even when it is not
        # present in the motif frequency file used to build the hybrid vocab.
        char_set.update(set("0123456789"))
        char_set.update({"(", ")", "=", "#", "-", "+", "/", "\\", "@", ".", ":", "*", "~", "$", ">", "<"})
        char_set.update(COUNTERION_TOKENS)
        char_set.update(COMMON_BRACKET_ATOM_TOKENS)

        itos: List[str] = list(_SPECIAL)

        # STAMP structural tokens. Default includes the commonly used joint
        # ([J_0]..[J_29]), branch ([B_1]..[B_12]), stereo and control tokens;
        # callers may pass `extra_stamp_tokens` for anything custom.
        stamp_struct = [f"[J_{i}]" for i in range(30)]
        stamp_struct += [f"[B_{i}]" for i in range(1, 13)]
        stamp_struct += ["[S_R]", "[S_S]", "[S_E]", "[S_Z]", "[Ring]", "[END]", "[BACK]", "."]
        stamp_struct += list(FCSTAMP_CONTROL_TOKENS)
        for t in stamp_struct:
            if t not in itos:
                itos.append(t)
        if extra_stamp_tokens:
            for t in extra_stamp_tokens:
                if t not in itos:
                    itos.append(t)

        # SMILES char tokens (sorted for determinism).
        for t in sorted(char_set):
            if t not in itos:
                itos.append(t)

        # Atomic motifs (freq >= cutoff). Single-token SMILES motifs (e.g., 'C',
        # '[CH2]') share ids with the SMILES-char entries; multi-char motifs
        # get fresh ids.
        # Sorted by descending freq for determinism + for convenient head/tail.
        for smi, _ in sorted(atomic, key=lambda x: (-x[1], x[0])):
            if smi not in itos:
                itos.append(smi)

        # Finalize.
        vocab = cls(itos=itos, freq_cutoff=freq_cutoff)

        atomic_ids: Dict[str, int] = {}
        freq_map: Dict[str, int] = {}
        expansion: Dict[str, List[int]] = {}
        for smi, freq in atomic:
            atomic_ids[smi] = vocab.stoi[smi]
            freq_map[smi] = int(freq)
            expansion[smi] = [vocab.stoi[t] for t in smiles_char_tokens(smi)]

        vocab.atomic_motifs = atomic_ids
        vocab.motif_freq = freq_map
        vocab.motif_expansion = expansion
        vocab.__post_init__()  # refresh freq bounds
        return vocab

    # ---- Fallback probability schedule ----

    def fallback_prob(self, freq: int, p_low: float = 0.02, p_high: float = 0.15) -> float:
        """Log-interpolate fallback prob. Most frequent motif → p_low; at the
        freq cutoff → p_high. Monotone decreasing in freq."""
        if freq <= self.freq_cutoff:
            return p_high
        f = math.log(max(1, freq))
        denom = self._log_freq_max - self._log_freq_min
        t = 0.0 if denom <= 1e-9 else (f - self._log_freq_min) / denom
        t = max(0.0, min(1.0, t))
        return p_high * (1.0 - t) + p_low * t

    # ---- Encoding ----

    def _expand_motif_to_chars(self, smi: str) -> List[int]:
        if smi in self.motif_expansion:
            return list(self.motif_expansion[smi])
        # Encode on the fly for rare motifs (always expanded).
        return [self.stoi.get(t, self.unk_id) for t in smiles_char_tokens(smi)]

    def encode_stream(
        self,
        tokens: Sequence[str],
        training: bool = False,
        rng: Optional[random.Random] = None,
        p_low: float = 0.02,
        p_high: float = 0.15,
    ) -> List[int]:
        """Encode a STAMP token stream (list of structural tokens + motif
        SMILES, as stored in `stamp_smiles`) into token ids.

        During training, atomic motifs may be expanded to SMILES chars with
        probability `fallback_prob(freq)`. Rare motifs (below cutoff) always
        expand.
        """
        rnd = rng if rng is not None else random
        out: List[int] = []
        for tok in tokens:
            if is_condition_token(tok) or is_fcstamp_control_token(tok):
                out.append(self.stoi.get(tok, self.unk_id))
                continue
            if is_stamp_structural(tok):
                out.append(self.stoi.get(tok, self.unk_id))
                continue
            if tok in self.atomic_motifs:
                if training:
                    p = self.fallback_prob(self.motif_freq[tok], p_low=p_low, p_high=p_high)
                    if rnd.random() < p:
                        out.extend(self._expand_motif_to_chars(tok))
                        continue
                out.append(self.atomic_motifs[tok])
            else:
                out.extend(self._expand_motif_to_chars(tok))
        return out

    # ---- Persistence ----

    def to_dict(self) -> Dict:
        return {
            "itos": self.itos,
            "atomic_motifs": self.atomic_motifs,
            "motif_freq": self.motif_freq,
            "motif_expansion": self.motif_expansion,
            "freq_cutoff": self.freq_cutoff,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "HybridVocab":
        return cls(
            itos=list(d["itos"]),
            atomic_motifs=dict(d.get("atomic_motifs", {})),
            motif_freq=dict(d.get("motif_freq", {})),
            motif_expansion={k: list(v) for k, v in d.get("motif_expansion", {}).items()},
            freq_cutoff=int(d.get("freq_cutoff", 5000)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path) -> "HybridVocab":
        return cls.from_dict(json.loads(Path(path).read_text()))
