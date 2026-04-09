from __future__ import annotations

import base64
import importlib
import re
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import List, Sequence

from rdkit import Chem, RDLogger

from ..core.stamp_codec import STAMPEncoder
from .common import canonicalize_smiles


class RepresentationError(RuntimeError):
    pass


class BaseRepresentation:
    name: str = "base"
    supports_decode: bool = True

    def encode(self, smiles: str) -> str:
        raise NotImplementedError

    def decode(self, text: str) -> str:
        raise NotImplementedError

    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError

    def detokenize(self, tokens: Sequence[str]) -> str:
        raise NotImplementedError

    def is_special_token(self, token: str) -> bool:
        return False


class SMILESRepresentation(BaseRepresentation):
    name = "smiles"

    def encode(self, smiles: str) -> str:
        s = canonicalize_smiles(smiles)
        if s is None:
            raise RepresentationError(f"Invalid SMILES: {smiles}")
        return s

    def decode(self, text: str) -> str:
        s = canonicalize_smiles(text)
        if s is None:
            raise RepresentationError(f"Invalid SMILES decode: {text}")
        return s

    def tokenize(self, text: str) -> List[str]:
        return list(text)

    def detokenize(self, tokens: Sequence[str]) -> str:
        return "".join(tokens)


class STAMPRepresentation(BaseRepresentation):
    name = "stamp"

    def __init__(self) -> None:
        self.codec = STAMPEncoder()

    def encode(self, smiles: str) -> str:
        return self.codec.encode_smiles(smiles)

    def decode(self, text: str) -> str:
        mol = self.codec.decode_stamp_to_mol(text)
        if len(Chem.GetMolFrags(mol)) != 1:
            raise RepresentationError("Decoded STAMP is disconnected (multiple fragments).")
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    def tokenize(self, text: str) -> List[str]:
        return [t for t in text.split() if t]

    def detokenize(self, tokens: Sequence[str]) -> str:
        return " ".join(tokens)

    def is_special_token(self, token: str) -> bool:
        return (
            token in {"[END]", "[BACK]"}
            or token.startswith("[J_")
            or token.startswith("[B_")
            or token.startswith("[S_")
        )


class SELFIESRepresentation(BaseRepresentation):
    name = "selfies"

    def __init__(self) -> None:
        try:
            import selfies as sf  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RepresentationError(
                "SELFIES representation requires `selfies` package. Install via `pip install selfies`."
            ) from exc
        self.sf = sf

    def encode(self, smiles: str) -> str:
        s = canonicalize_smiles(smiles)
        if s is None:
            raise RepresentationError(f"Invalid SMILES: {smiles}")
        return self.sf.encoder(s)

    def decode(self, text: str) -> str:
        smiles = self.sf.decoder(text)
        s = canonicalize_smiles(smiles)
        if s is None:
            raise RepresentationError(f"Invalid SELFIES decode: {text}")
        return s

    def tokenize(self, text: str) -> List[str]:
        return list(self.sf.split_selfies(text))

    def detokenize(self, tokens: Sequence[str]) -> str:
        return "".join(tokens)

    def is_special_token(self, token: str) -> bool:
        return token in {"[nop]"}


_SMILES_TOKEN_RE = re.compile(
    r"(\[[^\[\]]+\]|Br|Cl|Si|Se|Li|Na|Ca|Al|Mg|Zn|Sn|Ag|Fe|As|Pt|Au|Cu|Mn|Co|Ni|Pb|Hg|Bi|Kr|Xe|He|Ne|Ar|%\d{2}|[A-Za-z]|[0-9]|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|\*|\$)"
)


def _tokenize_smiles_lex(smiles: str) -> List[str]:
    toks: List[str] = []
    pos = 0
    for m in _SMILES_TOKEN_RE.finditer(smiles):
        if m.start() > pos:
            toks.extend(list(smiles[pos : m.start()]))
        toks.append(m.group(0))
        pos = m.end()
    if pos < len(smiles):
        toks.extend(list(smiles[pos:]))
    return toks


def _b64_pack(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def _b64_unpack(payload: str) -> str:
    pad = "=" * ((4 - len(payload) % 4) % 4)
    return base64.urlsafe_b64decode((payload + pad).encode("ascii")).decode("utf-8")


def _pack_tag(kind: str, payload: str) -> str:
    return f"[{kind}_{_b64_pack(payload)}]"


def _unpack_tag(token: str, allowed_kinds: set[str]) -> str:
    if not (token.startswith("[") and token.endswith("]")):
        raise RepresentationError(f"Malformed token: {token}")
    body = token[1:-1]
    if "_" not in body:
        raise RepresentationError(f"Malformed token body: {token}")
    kind, payload = body.split("_", 1)
    if kind not in allowed_kinds:
        raise RepresentationError(f"Unexpected token kind `{kind}` for token {token}")
    try:
        return _b64_unpack(payload)
    except Exception as exc:
        raise RepresentationError(f"Invalid token payload for {token}") from exc


def _is_atomish_token(tok: str) -> bool:
    if not tok:
        return False
    if tok.startswith("["):
        return True
    return tok[0].isalpha()


class GroupSELFIESRepresentation(BaseRepresentation):
    name = "group-selfies"

    def __init__(self) -> None:
        self._group_selfies = self._load_group_selfies()
        self._grammar = self._build_grammar()

    @staticmethod
    def _load_group_selfies():
        try:
            return importlib.import_module("group_selfies")
        except Exception as pip_exc:
            repo_root = Path(__file__).resolve().parents[2]
            vendor_root = repo_root / "third_party" / "group-selfies"
            if vendor_root.exists():
                vendor_path = str(vendor_root)
                if vendor_path not in sys.path:
                    sys.path.insert(0, vendor_path)
                try:
                    return importlib.import_module("group_selfies")
                except Exception as vendor_exc:
                    raise RepresentationError(
                        "Group SELFIES import failed for both pip-installed and vendored group-selfies. "
                        f"Tried vendored path: {vendor_root}"
                    ) from vendor_exc
            raise RepresentationError(
                "Group SELFIES representation requires `group_selfies`. "
                "Either install the package or clone "
                "`https://github.com/aspuru-guzik-group/group-selfies.git` into `third_party/group-selfies`."
            ) from pip_exc

    def _build_grammar(self):
        try:
            with self._suppress_rdkit_warnings():
                return self._group_selfies.GroupGrammar.essential_set()
        except Exception as exc:
            raise RepresentationError(
                "Failed to initialize Group SELFIES essential grammar."
            ) from exc

    @staticmethod
    @contextmanager
    def _suppress_rdkit_warnings():
        RDLogger.DisableLog("rdApp.warning")
        try:
            yield
        finally:
            RDLogger.EnableLog("rdApp.warning")

    def encode(self, smiles: str) -> str:
        s = canonicalize_smiles(smiles)
        if s is None:
            raise RepresentationError(f"Invalid SMILES: {smiles}")
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            raise RepresentationError(f"Invalid SMILES: {smiles}")
        try:
            with self._suppress_rdkit_warnings():
                text = self._grammar.full_encoder(Chem.Mol(mol))
        except Exception as exc:
            raise RepresentationError(f"Failed to encode group-selfies from SMILES: {smiles}") from exc
        if not text:
            raise RepresentationError("Failed to encode group-selfies text.")
        return str(text)

    def decode(self, text: str) -> str:
        raw = text.strip()
        if not raw:
            raise RepresentationError("Empty group-selfies text.")

        try:
            gselfies = self.detokenize(self.tokenize(raw))
            with self._suppress_rdkit_warnings():
                mol = self._grammar.decoder(gselfies)
        except Exception as exc:
            raise RepresentationError(f"Invalid group-selfies decode: {text}") from exc
        if mol is None:
            raise RepresentationError(f"Invalid group-selfies decode: {text}")
        s = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if s is None:
            raise RepresentationError(f"Invalid group-selfies decode: {text}")
        return s

    def tokenize(self, text: str) -> List[str]:
        raw = text.strip()
        if not raw:
            return []
        # Allow whitespace-separated tokens from model outputs.
        gselfies = "".join(raw.split()) if any(ch.isspace() for ch in raw) else raw
        try:
            return list(self._group_selfies.split_selfies(gselfies))
        except Exception as exc:
            raise RepresentationError(f"Failed to tokenize group-selfies text: {text}") from exc

    def detokenize(self, tokens: Sequence[str]) -> str:
        toks = [t for t in tokens if t]
        if not toks:
            return ""
        return "".join(toks)


class FragSMILESRepresentation(BaseRepresentation):
    name = "fragsmiles"

    _LINK_RE = re.compile(r"^<\d+[RSrs]?>$")

    def __init__(self) -> None:
        self._chemicalgof = self._load_chemicalgof()

    @staticmethod
    def _load_chemicalgof():
        try:
            return importlib.import_module("chemicalgof")
        except Exception as pip_exc:
            repo_root = Path(__file__).resolve().parents[2]
            vendor_root = repo_root / "third_party" / "chemicalgof"
            if vendor_root.exists():
                vendor_path = str(vendor_root)
                if vendor_path not in sys.path:
                    sys.path.insert(0, vendor_path)
                try:
                    return importlib.import_module("chemicalgof")
                except Exception as vendor_exc:
                    raise RepresentationError(
                        "FragSMILES import failed for both pip-installed and vendored chemicalgof. "
                        f"Tried vendored path: {vendor_root}"
                    ) from vendor_exc
            raise RepresentationError(
                "FragSMILES representation requires `chemicalgof`. "
                "Either install via `pip install git+https://github.com/f48r1/chemicalgof.git` "
                "or clone the repo into `third_party/chemicalgof`."
            ) from pip_exc

    def encode(self, smiles: str) -> str:
        s = canonicalize_smiles(smiles)
        if s is None:
            raise RepresentationError(f"Invalid SMILES: {smiles}")
        try:
            text = self._chemicalgof.encode(
                s,
                canonical=True,
                random=False,
                capitalize_chirality=True,
            )
        except Exception as exc:
            raise RepresentationError(f"Failed to encode fragsmiles from SMILES: {smiles}") from exc
        if not text:
            raise RepresentationError("Failed to encode fragsmiles text.")
        return str(text)

    def decode(self, text: str) -> str:
        tokens = self.tokenize(text)
        if not tokens:
            raise RepresentationError("Empty fragsmiles text.")
        try:
            raw = self._chemicalgof.decode(tokens, strict_chirality=True)
        except Exception as exc:
            raise RepresentationError(f"Invalid fragsmiles decode: {text}") from exc
        s = canonicalize_smiles(raw)
        if s is None:
            raise RepresentationError(f"Invalid fragsmiles decode: {text}")
        return s

    def tokenize(self, text: str) -> List[str]:
        raw = text.strip()
        if not raw:
            return []
        # Fast path for model-generated sequences that are already space-tokenized.
        if any(ch.isspace() for ch in raw):
            toks = [t for t in raw.split() if t]
            if len(toks) > 1:
                return toks
        try:
            return list(self._chemicalgof.split(raw))
        except Exception as exc:
            raise RepresentationError(f"Failed to tokenize fragsmiles text: {text}") from exc

    def detokenize(self, tokens: Sequence[str]) -> str:
        return " ".join(t for t in tokens if t)

    def is_special_token(self, token: str) -> bool:
        return self._LINK_RE.fullmatch(token) is not None


class SAFERepresentation(BaseRepresentation):
    name = "safe"

    _SAFE_SPLIT_RE = re.compile(
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>>?|\*|\$|%\d{2}|[0-9])"
    )

    def __init__(self) -> None:
        self._safe_converter = self._load_safe_converter()

    @staticmethod
    def _install_vendored_safe_package(vendor_pkg_root: Path) -> None:
        # Avoid importing safe.__init__ (which pulls heavy training deps) and only expose package path.
        for key in list(sys.modules.keys()):
            if key == "safe" or key.startswith("safe."):
                sys.modules.pop(key, None)
        pkg = types.ModuleType("safe")
        pkg.__path__ = [str(vendor_pkg_root)]
        sys.modules["safe"] = pkg

    @staticmethod
    def _load_safe_converter():
        # Prefer official pip package if fully importable.
        try:
            safe_mod = importlib.import_module("safe")
            if hasattr(safe_mod, "SAFEConverter"):
                return safe_mod.SAFEConverter(slicer="brics", ignore_stereo=False)
        except Exception:
            pass

        repo_root = Path(__file__).resolve().parents[2]
        vendor_pkg_root = repo_root / "third_party" / "safe" / "safe"
        if not vendor_pkg_root.exists():
            raise RepresentationError(
                "SAFE representation requires official `safe` codec. "
                "Install `safe-mol` or clone `https://github.com/datamol-io/safe.git` "
                "into `third_party/safe`."
            )

        try:
            SAFERepresentation._install_vendored_safe_package(vendor_pkg_root)
            converter_mod = importlib.import_module("safe.converter")
        except Exception as exc:
            raise RepresentationError(
                "Failed to import vendored SAFE codec from `third_party/safe`. "
                "Please ensure runtime deps `datamol` and `loguru` are available."
            ) from exc

        return converter_mod.SAFEConverter(slicer="brics", ignore_stereo=False)

    @classmethod
    def _safe_split(cls, safe_text: str) -> List[str]:
        toks = list(cls._SAFE_SPLIT_RE.findall(safe_text))
        if not toks or "".join(toks) != safe_text:
            raise RepresentationError(f"Failed to tokenize SAFE text: {safe_text}")
        return toks

    def encode(self, smiles: str) -> str:
        s = canonicalize_smiles(smiles)
        if s is None:
            raise RepresentationError(f"Invalid SMILES: {smiles}")
        try:
            out = self._safe_converter.encoder(
                s,
                canonical=True,
                randomize=False,
                allow_empty=True,
            )
        except Exception as exc:
            raise RepresentationError(f"Failed to encode SAFE from SMILES: {smiles}") from exc
        if not out:
            raise RepresentationError("Failed to encode SAFE text.")
        return str(out)

    def decode(self, text: str) -> str:
        raw = self.detokenize(self.tokenize(text))
        if not raw:
            raise RepresentationError("Empty SAFE text.")
        try:
            decoded = self._safe_converter.decoder(
                raw,
                as_mol=False,
                canonical=True,
                fix=True,
                remove_added_hs=True,
                remove_dummies=True,
            )
        except Exception as exc:
            raise RepresentationError(f"Invalid SAFE decode: {text}") from exc
        if not decoded:
            raise RepresentationError(f"Invalid SAFE decode: {text}")
        s = canonicalize_smiles(str(decoded))
        if s is None:
            raise RepresentationError(f"Invalid SAFE decode: {text}")
        return s

    def tokenize(self, text: str) -> List[str]:
        raw = text.strip()
        if not raw:
            return []
        safe_text = "".join(raw.split()) if any(ch.isspace() for ch in raw) else raw
        return self._safe_split(safe_text)

    def detokenize(self, tokens: Sequence[str]) -> str:
        return "".join(t for t in tokens if t)


class UnsupportedRepresentation(BaseRepresentation):
    supports_decode = False

    def __init__(self, name: str) -> None:
        self.name = name

    def encode(self, smiles: str) -> str:
        raise RepresentationError(
            f"Representation `{self.name}` is a placeholder. Please implement concrete encoder/decoder first."
        )

    def decode(self, text: str) -> str:
        raise RepresentationError(f"Representation `{self.name}` is a placeholder.")

    def tokenize(self, text: str) -> List[str]:
        raise RepresentationError(f"Representation `{self.name}` is a placeholder.")

    def detokenize(self, tokens: Sequence[str]) -> str:
        raise RepresentationError(f"Representation `{self.name}` is a placeholder.")


def build_representation(name: str) -> BaseRepresentation:
    key = name.strip().lower()
    if key == "smiles":
        return SMILESRepresentation()
    if key == "stamp":
        return STAMPRepresentation()
    if key == "selfies":
        return SELFIESRepresentation()
    if key in {"group-selfies", "group_selfies", "groupselfies"}:
        return GroupSELFIESRepresentation()
    if key in {"fragsmiles", "frag-smiles", "frag_smiles"}:
        return FragSMILESRepresentation()
    if key == "safe":
        return SAFERepresentation()
    if key in {
        "t-smiles",
        "t_smiles",
    }:
        return UnsupportedRepresentation(name=key)
    raise RepresentationError(f"Unknown representation: {name}")
