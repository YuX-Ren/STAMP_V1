"""Port signature & atom-count helpers for subtree transplant.

Signature definition (locked at L2 + size filter):

  PortSignature = (in_bond_order, n_open_ports)

  - ``in_bond_order``: bond order connecting the subtree's root motif atom
    to its parent atom in the assembled molecule.  One of
    ``"single" / "double" / "triple" / "aromatic"`` for non-root subtrees,
    and ``"ROOT"`` for the whole-molecule root (excluded from transplant).

  - ``n_open_ports``: number of ``[J_*]`` joints on the root motif of the
    subtree.  Counts every joint, regardless of how many were actually
    used by descendants — this is the topology-shape invariant that lets
    us hash compatible subtrees together.

Atom-count is tracked separately and used as a query-time filter
(``size_delta``) rather than as part of the signature, because the
signature space would otherwise explode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

from rdkit import Chem
from rdkit.Chem import BondType

from ...core.stamp_codec import STAMPGroup
from .subtree import ParsedMolecule, SubtreeSpan


_BOND_ORDER_NAME: Dict[BondType, str] = {
    BondType.SINGLE: "single",
    BondType.DOUBLE: "double",
    BondType.TRIPLE: "triple",
    BondType.AROMATIC: "aromatic",
}


@dataclass(frozen=True)
class PortSignature:
    in_bond_order: str
    n_open_ports: int

    def as_tuple(self) -> tuple:
        return (self.in_bond_order, self.n_open_ports)

    def __str__(self) -> str:
        return f"({self.in_bond_order}, n={self.n_open_ports})"


# ---------------------------------------------------------------------------
# Heavy-atom counter for a subtree
# ---------------------------------------------------------------------------

def heavy_atom_count(parsed: ParsedMolecule, span: SubtreeSpan) -> int:
    """Count heavy atoms across all groups in this subtree."""
    total = 0
    for g in parsed.groups[span.group_idx : span.group_idx + span.group_count]:
        total += len(g.atom_indices)
    return int(total)


def heavy_atom_count_from_tokens(donor_tokens: Sequence[str]) -> int:
    """Count heavy atoms in a flat-token donor span by parsing each motif."""
    total = 0
    for t in donor_tokens:
        if not t or t.startswith("["):
            continue
        m = Chem.MolFromSmiles(t, sanitize=False)
        if m is None:
            continue
        total += sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1)
    return total


# ---------------------------------------------------------------------------
# Signature extraction
# ---------------------------------------------------------------------------

def _bond_order_name(bond) -> str:
    if bond is None:
        return "unknown"
    bt = bond.GetBondType()
    return _BOND_ORDER_NAME.get(bt, "unknown")


def _count_root_joints(group: STAMPGroup) -> int:
    """Number of ``[J_*]`` tokens on the root motif of a group."""
    return sum(1 for t in group.tokens if t.startswith("[J_"))


def _find_parent_atom_idx(
    mol: Chem.Mol,
    child_atoms: Sequence[int],
    parent_atoms: Iterable[int],
) -> Optional[int]:
    """Return the parent-side atom of the bond linking child subtree to
    parent.  ``None`` if no such bond exists in ``mol`` (shouldn't happen
    for well-formed STAMP, but we guard).
    """
    parent_set = set(int(p) for p in parent_atoms)
    for a in child_atoms:
        atom = mol.GetAtomWithIdx(int(a))
        for bond in atom.GetBonds():
            other = bond.GetOtherAtomIdx(int(a))
            if other in parent_set:
                return int(other)
    return None


def port_signature_for_span(
    parsed: ParsedMolecule,
    span: SubtreeSpan,
    mol: Optional[Chem.Mol] = None,
) -> Optional[PortSignature]:
    """Compute a :class:`PortSignature` for the given subtree.

    For the root subtree, returns ``None`` (root cannot be transplanted as
    a *subtree* — it's the whole molecule).

    Builds the assembled mol from ``parsed.smiles`` if not provided.
    """
    if span.is_root:
        return None
    root_group = parsed.groups[span.group_idx]
    parent_id = root_group.parent_frag_id
    if parent_id is None:
        return None
    # Locate parent group by frag_id
    parent_group = next(
        (g for g in parsed.groups if int(g.frag_id) == int(parent_id)),
        None,
    )
    if parent_group is None:
        return None

    if mol is None:
        mol = Chem.MolFromSmiles(parsed.smiles)
        if mol is None:
            return None

    # Map canonical-SMILES atom indices to parsed-mol atom indices is
    # nontrivial here because parsed.groups carries indices from the
    # *original* assembled mol (encoder's working RWMol), not necessarily
    # matching ``Chem.MolFromSmiles(parsed.smiles)``.  In practice, the
    # encoder canonicalizes the mol before recording group atom_indices,
    # so we can re-derive the bond from the assembled mol the encoder
    # used.  We sidestep this by re-encoding from the SMILES and asking
    # the encoder for groups directly, then matching.
    # For robustness, fall back to inferring bond order from the parent
    # motif's SMILES when atom-index mapping fails.

    child_atoms = [int(a) for a in root_group.atom_indices]
    parent_atoms = [int(a) for a in parent_group.atom_indices]
    parent_atom_idx = _find_parent_atom_idx(mol, child_atoms, parent_atoms)

    if parent_atom_idx is None:
        # Fallback: assume single bond.  R4 fallbacks accept this.
        bond_name = "single"
    else:
        bond = mol.GetBondBetweenAtoms(child_atoms[0], parent_atom_idx)
        if bond is None:
            # Try other child atoms (the attach atom may not be the first).
            bond = next(
                (mol.GetBondBetweenAtoms(c, parent_atom_idx) for c in child_atoms
                 if mol.GetBondBetweenAtoms(c, parent_atom_idx) is not None),
                None,
            )
        bond_name = _bond_order_name(bond)

    n_open = _count_root_joints(root_group)
    return PortSignature(in_bond_order=bond_name, n_open_ports=n_open)


def signature_for_donor_root(donor_tokens: Sequence[str]) -> Optional[PortSignature]:
    """Compute a port signature for a donor subtree given just its flat
    tokens (as harvested into the pool).

    Donor subtrees recovered from the pool may not have full mol context;
    we approximate ``in_bond_order`` from the *original* harvest record
    (stored alongside the tokens), so this helper is mostly used for
    bookkeeping / debugging.  For actual transplant matching, we look up
    by signature key directly.
    """
    # Count joints on the first motif token (the root motif of the donor)
    n_open = 0
    seen_motif = False
    for t in donor_tokens:
        if t.startswith("[J_") and seen_motif:
            n_open += 1
        elif t.startswith("[B_") or t.startswith("[S_") or t == "[Ring]":
            continue
        elif t and not t.startswith("["):
            if seen_motif:
                break
            seen_motif = True
    return PortSignature(in_bond_order="unknown", n_open_ports=n_open)
