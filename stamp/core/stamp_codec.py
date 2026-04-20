from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem

if TYPE_CHECKING:
    from ..vocab.principal_subgraph import PrincipalSubgraphTokenizer


@dataclass(frozen=True)
class Fragment:
    frag_id: int
    atom_indices: Tuple[int, ...]
    motif: str
    heavy_atom_count: int
    orig_to_local: Dict[int, int]


@dataclass(frozen=True)
class FragmentEdge:
    edge_id: int
    left: int
    right: int
    left_local_atom: int
    right_local_atom: int
    left_stereo_label: Optional[str]
    right_stereo_label: Optional[str]


@dataclass(frozen=True)
class Port:
    atom_idx: int


@dataclass(frozen=True)
class PortSpec:
    atom_idx: int
    is_ring_closure: bool = False


@dataclass
class BranchFrame:
    remaining_groups: int
    saved_stack: List[Port]


@dataclass(frozen=True)
class RingBackEdge:
    edge_id: int
    ancestor: int
    descendant: int
    ancestor_local_atom: int
    descendant_local_atom: int
    ancestor_stereo_label: Optional[str]
    descendant_stereo_label: Optional[str]


@dataclass(frozen=True)
class STAMPGroup:
    frag_id: int
    atom_indices: Tuple[int, ...]
    motif: str
    tokens: Tuple[str, ...]
    parent_frag_id: Optional[int]
    subtree_group_count: int


@dataclass
class STAMPDecodeReport:
    ignored_back: int = 0
    branch_without_active_port: int = 0
    ignored_dangling_joints: int = 0
    ignored_orphan_stereo: int = 0
    out_of_range_joint_wraps: int = 0
    ignored_orphan_rings: int = 0
    failed_ring_closures: int = 0
    synthesized_ports: int = 0
    fallback_parent_ports: int = 0
    dropped_fragments: int = 0
    truncated_branches: int = 0

    def total_fallbacks(self) -> int:
        return int(
            self.ignored_back
            + self.branch_without_active_port
            + self.ignored_dangling_joints
            + self.ignored_orphan_stereo
            + self.out_of_range_joint_wraps
            + self.ignored_orphan_rings
            + self.failed_ring_closures
            + self.synthesized_ports
            + self.fallback_parent_ports
            + self.dropped_fragments
            + self.truncated_branches
        )

    def to_dict(self) -> Dict[str, int]:
        payload = {
            "ignored_back": int(self.ignored_back),
            "branch_without_active_port": int(self.branch_without_active_port),
            "ignored_dangling_joints": int(self.ignored_dangling_joints),
            "ignored_orphan_stereo": int(self.ignored_orphan_stereo),
            "out_of_range_joint_wraps": int(self.out_of_range_joint_wraps),
            "ignored_orphan_rings": int(self.ignored_orphan_rings),
            "failed_ring_closures": int(self.failed_ring_closures),
            "synthesized_ports": int(self.synthesized_ports),
            "fallback_parent_ports": int(self.fallback_parent_ports),
            "dropped_fragments": int(self.dropped_fragments),
            "truncated_branches": int(self.truncated_branches),
        }
        payload["total_fallbacks"] = int(self.total_fallbacks())
        return payload


class STAMPEncoder:
    """
    Sequential Tree-based Attachment Motif Presentation (STAMP) encoder.

    This implementation follows the core ideas in stamp.md:
    - Fragment motifs are pure canonical SMILES tokens.
    - Connections are external local joint tokens [J_i].
    - Topology is encoded with branch tokens [B_n].
    """

    def __init__(
        self,
        cut_rule: str = "exocyclic_single",
        min_branch_n: int = 1,
        max_branch_n: Optional[int] = None,
        motif_tokenizer: Optional["PrincipalSubgraphTokenizer"] = None,
    ) -> None:
        self.cut_rule = cut_rule
        self.min_branch_n = min_branch_n
        self.max_branch_n = max_branch_n
        self.motif_tokenizer = motif_tokenizer

    def encode_smiles(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return self.encode_mol(mol)

    def decode_stamp(self, stamp: str, canonical: bool = True) -> str:
        mol = self.decode_stamp_to_mol(stamp)
        return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=True)

    def decode_stamp_to_mol(
        self,
        stamp: str,
        *,
        strict: bool = False,
        return_report: bool = False,
    ) -> Chem.Mol | Tuple[Chem.Mol, STAMPDecodeReport]:
        tokens = [t for t in stamp.split() if t]
        if not tokens:
            raise ValueError("Empty STAMP sequence.")

        rw = Chem.RWMol()
        port_stack: List[Port] = []
        branch_frames: List[BranchFrame] = []
        stereo_map: Dict[int, str] = {}
        report = STAMPDecodeReport()
        is_root = True
        i = 0

        while i < len(tokens):
            tok = tokens[i]

            if tok == "[END]":
                break

            if tok == "[BACK]":
                if strict:
                    raise ValueError("Decoder does not support [BACK] trajectories in strict mode.")
                # Safety fallback: ignore backtracking controls at decode time.
                report.ignored_back += 1
                i += 1
                continue

            if self._is_branch(tok):
                n = self._parse_branch(tok)
                active = port_stack.pop() if port_stack else None
                if active is None:
                    if strict:
                        raise ValueError(f"Branch token without active port at position {i}: {tok}")
                    report.branch_without_active_port += 1
                saved = list(port_stack)
                port_stack = [active] if active is not None else []
                branch_frames.append(BranchFrame(remaining_groups=max(0, n), saved_stack=saved))
                i += 1
                continue

            if self._is_joint(tok):
                if strict:
                    raise ValueError(f"Unexpected joint token without motif at position {i}: {tok}")
                # Safety fallback: a dangling joint without a motif-group to attach
                # to carries no actionable information, so skip it. Consume an
                # optional following stereo tag as part of the same stray port.
                report.ignored_dangling_joints += 1
                i += 1
                if i < len(tokens) and self._is_stereo(tokens[i]):
                    report.ignored_orphan_stereo += 1
                i += 1
                continue

            if self._is_ring(tok):
                if strict:
                    raise ValueError(f"Unexpected [Ring] token without joint at position {i}: {tok}")
                report.ignored_orphan_rings += 1
                i += 1
                continue

            if self._is_stereo(tok):
                if strict:
                    raise ValueError(f"Unexpected stereo token without joint at position {i}: {tok}")
                # Safety fallback: ignore orphan stereo tags that are not paired
                # with a joint token.
                report.ignored_orphan_stereo += 1
                i += 1
                continue

            # Keep latent chirality from motif tokens until the whole molecule
            # is assembled; early sanitization can drop valid '@' annotations.
            frag = Chem.MolFromSmiles(tok, sanitize=False)
            if frag is None:
                raise ValueError(f"Invalid motif token at position {i}: {tok}")

            rw, frag_offset = self._append_fragment(rw, frag)
            i += 1

            ports: List[PortSpec] = []
            local_heavy = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() > 1]
            if not local_heavy:
                local_heavy = [a.GetIdx() for a in frag.GetAtoms()]
            if not local_heavy:
                raise ValueError("Motif has no atoms, cannot decode.")

            while i < len(tokens) and self._is_joint(tokens[i]):
                j = self._parse_joint(tokens[i])
                i += 1

                # R4.1: out-of-range [J_i] fallback by modulo heavy-atom count.
                if j >= len(local_heavy):
                    if strict:
                        raise ValueError(f"Out-of-range joint token for motif at position {i-1}: [J_{j}]")
                    report.out_of_range_joint_wraps += 1
                local_idx = local_heavy[j % len(local_heavy)]
                if i < len(tokens) and self._is_stereo(tokens[i]):
                    label = self._parse_stereo(tokens[i])
                    i += 1
                    if label is not None:
                        stereo_map[frag_offset + local_idx] = label
                is_ring_closure = False
                if i < len(tokens) and self._is_ring(tokens[i]):
                    is_ring_closure = True
                    i += 1

                ports.append(
                    PortSpec(
                        atom_idx=frag_offset + local_idx,
                        is_ring_closure=is_ring_closure,
                    )
                )

            if is_root:
                if any(p.is_ring_closure for p in ports):
                    if strict:
                        raise ValueError("Root motif cannot start with [Ring] closures.")
                    report.ignored_orphan_rings += sum(1 for p in ports if p.is_ring_closure)
                    ports = [p for p in ports if not p.is_ring_closure]
                for p in reversed(ports):
                    port_stack.append(Port(atom_idx=p.atom_idx))
                is_root = False
            else:
                if not ports:
                    if strict:
                        raise ValueError(f"Motif without joints at position {i-1}")
                    # R4.5a: motif has no joints — synthesize a default
                    # port on atom 0 so it can still connect.
                    report.synthesized_ports += 1
                    default_local = local_heavy[0] if local_heavy else 0
                    ports = [PortSpec(atom_idx=frag_offset + default_local)]
                if ports[0].is_ring_closure:
                    if strict:
                        raise ValueError(f"First port of a child motif cannot be [Ring] at position {i-1}")
                    report.ignored_orphan_rings += 1
                    ports[0] = PortSpec(atom_idx=ports[0].atom_idx, is_ring_closure=False)

                parent_port = port_stack.pop() if port_stack else None
                if parent_port is None:
                    if strict:
                        raise ValueError(f"Port stack exhausted before motif at position {i-1}")
                    # R4.5b: port stack exhausted — scan existing atoms
                    # for any with available valence.
                    report.fallback_parent_ports += 1
                    parent_port = self._find_fallback_port(rw, frag_offset)

                connected = False
                if parent_port is not None:
                    input_candidates = self._rotated_port_candidates(
                        Port(atom_idx=ports[0].atom_idx),
                        [Port(atom_idx=p.atom_idx) for p in ports],
                    )
                    result = self._try_add_bond_with_fallback(rw, parent_port, input_candidates)
                    connected = result is not None

                if not connected and frag_offset > 0:
                    if strict:
                        raise ValueError(f"Could not connect fragment ending at position {i-1}")
                    # R4.5c: could not connect this fragment at all —
                    # remove it to keep the molecule connected.
                    report.dropped_fragments += 1
                    atoms_to_remove = sorted(
                        range(frag_offset, rw.GetNumAtoms()), reverse=True
                    )
                    for aidx in atoms_to_remove:
                        rw.RemoveAtom(aidx)
                    continue

                outputs: List[Port] = []
                for p in ports[1:]:
                    if p.is_ring_closure:
                        ring_target = port_stack.pop() if port_stack else None
                        if ring_target is None:
                            if strict:
                                raise ValueError(
                                    f"[Ring] closure without available ancestor port after motif at position {i-1}"
                                )
                            report.failed_ring_closures += 1
                            continue
                        result = self._try_add_bond_with_fallback(
                            rw,
                            ring_target,
                            [Port(atom_idx=p.atom_idx)],
                        )
                        if result is None:
                            if strict:
                                raise ValueError(
                                    f"Could not close [Ring] bond after motif at position {i-1}"
                                )
                            report.failed_ring_closures += 1
                            continue
                    else:
                        outputs.append(Port(atom_idx=p.atom_idx))

                for p in reversed(outputs):
                    port_stack.append(p)

            if branch_frames:
                # [B_n] counts motif-groups in subtree, including nested branches.
                for frame in branch_frames:
                    frame.remaining_groups -= 1
                while branch_frames and branch_frames[-1].remaining_groups <= 0:
                    done = branch_frames.pop()
                    port_stack = done.saved_stack

        # R4.4: if branch lengths exceed remaining tokens, truncate naturally.
        if branch_frames:
            if strict:
                raise ValueError("Branch frames left open at end of sequence.")
            report.truncated_branches += len(branch_frames)
        while branch_frames:
            done = branch_frames.pop()
            port_stack = done.saved_stack

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        self._restore_chirality(mol, stereo_map)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=False)
        if return_report:
            return mol, report
        return mol

    def roundtrip_smiles(self, smiles: str) -> Tuple[str, str, bool]:
        canonical_in = self._canonicalize_smiles(smiles)
        stamp = self.encode_smiles(smiles)
        decoded = self.decode_stamp(stamp, canonical=True)
        ok = canonical_in == decoded
        return stamp, decoded, ok

    @staticmethod
    def _canonicalize_smiles(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    @staticmethod
    def _is_joint(token: str) -> bool:
        return token.startswith("[J_") and token.endswith("]")

    @staticmethod
    def _is_ring(token: str) -> bool:
        return token == "[Ring]"

    @staticmethod
    def _parse_joint(token: str) -> int:
        try:
            value = int(token[3:-1])
        except ValueError as exc:
            raise ValueError(f"Invalid joint token: {token}") from exc
        if value < 0:
            raise ValueError(f"Joint index must be >= 0: {token}")
        return value

    @staticmethod
    def _is_branch(token: str) -> bool:
        return token.startswith("[B_") and token.endswith("]")

    @staticmethod
    def _parse_branch(token: str) -> int:
        try:
            value = int(token[3:-1])
        except ValueError as exc:
            raise ValueError(f"Invalid branch token: {token}") from exc
        return value

    @staticmethod
    def _is_stereo(token: str) -> bool:
        return token.startswith("[S_") and token.endswith("]")

    @staticmethod
    def _parse_stereo(token: str) -> Optional[str]:
        value = token[3:-1].strip().upper()
        if value in {"R", "S", "E", "Z"}:
            return value
        # Safety fallback: unknown stereo token is ignored.
        return None

    @staticmethod
    def _append_fragment(rw: Chem.RWMol, frag: Chem.Mol) -> Tuple[Chem.RWMol, int]:
        offset = rw.GetNumAtoms()
        combined = Chem.CombineMols(rw.GetMol(), frag)
        return Chem.RWMol(combined), offset

    @staticmethod
    def _rotated_port_candidates(start_port: Port, candidates: Sequence[Port]) -> List[Port]:
        if not candidates:
            return []
        try:
            start_idx = list(candidates).index(start_port)
        except ValueError:
            return list(candidates)
        seq = list(candidates)
        return seq[start_idx:] + seq[:start_idx]

    @staticmethod
    def _try_add_bond_with_fallback(
        rw: Chem.RWMol,
        parent_port: Port,
        input_candidates: Sequence[Port],
    ) -> Optional[Port]:
        p = parent_port.atom_idx
        for cand in input_candidates:
            c = cand.atom_idx
            if p == c:
                continue
            if rw.GetBondBetweenAtoms(p, c) is not None:
                return cand
            test = Chem.RWMol(rw)
            try:
                test.AddBond(p, c, Chem.BondType.SINGLE)
            except Exception:
                continue
            if STAMPEncoder._has_valence_violation(test, p, c):
                continue
            rw.AddBond(p, c, Chem.BondType.SINGLE)
            return cand
        return None

    @staticmethod
    def _find_fallback_port(rw: Chem.RWMol, exclude_from: int) -> Optional[Port]:
        """Find an atom in the existing molecule (before *exclude_from*)
        that can accept another single bond without a valence violation."""
        best_idx = None
        best_degree = 999
        for idx in range(exclude_from):
            atom = rw.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() <= 1:
                continue
            try:
                rw.UpdatePropertyCache(strict=False)
            except Exception:
                pass
            if atom.HasValenceViolation():
                continue
            deg = atom.GetDegree()
            if deg < best_degree:
                best_degree = deg
                best_idx = idx
        if best_idx is not None:
            return Port(atom_idx=best_idx)
        return None

    @staticmethod
    def _has_valence_violation(rw: Chem.RWMol, a_idx: int, b_idx: int) -> bool:
        try:
            rw.UpdatePropertyCache(strict=False)
        except Exception:
            return True
        a = rw.GetAtomWithIdx(a_idx)
        b = rw.GetAtomWithIdx(b_idx)
        return a.HasValenceViolation() or b.HasValenceViolation()

    @staticmethod
    def _is_potential_stereocenter(atom: Chem.Atom) -> bool:
        return atom.GetAtomicNum() > 1 and atom.GetDegree() >= 3

    @staticmethod
    def _restore_chirality(mol: Chem.Mol, stereo_map: Dict[int, str]) -> None:
        rs_targets: List[Tuple[int, str]] = []
        ez_targets: List[Tuple[int, str]] = []

        for atom_idx, target in stereo_map.items():
            if target in {"R", "S"}:
                rs_targets.append((atom_idx, target))
            elif target in {"E", "Z"}:
                ez_targets.append((atom_idx, target))

        for atom_idx, target in rs_targets:
            if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
                continue
            atom = mol.GetAtomWithIdx(atom_idx)

            if not STAMPEncoder._is_potential_stereocenter(atom):
                continue
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            Chem.AssignStereochemistry(mol, force=True, cleanIt=False)
            actual = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else ""
            if actual != target:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                Chem.AssignStereochemistry(mol, force=True, cleanIt=False)

        Chem.AssignStereochemistry(mol, force=True, cleanIt=False)
        for atom_idx, target in ez_targets:
            STAMPEncoder._set_double_bond_stereo(mol, atom_idx, target)

    @staticmethod
    def _set_double_bond_stereo(mol: Chem.Mol, atom_idx: int, target: str) -> None:
        desired = (
            Chem.rdchem.BondStereo.STEREOE
            if target == "E"
            else Chem.rdchem.BondStereo.STEREOZ
        )
        if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
            return

        atom = mol.GetAtomWithIdx(atom_idx)
        candidates: List[Chem.Bond] = []
        for bond in atom.GetBonds():
            if bond.GetBondType() != Chem.BondType.DOUBLE:
                continue
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            a_subs = [n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() != b]
            b_subs = [n.GetIdx() for n in mol.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() != a]
            if not a_subs or not b_subs:
                continue
            candidates.append(bond)

        if not candidates:
            return

        if len(candidates) == 1:
            bond = candidates[0]
        else:
            # Safety fallback for ambiguous atom labels: pick a stable bond.
            bond = sorted(candidates, key=lambda b: (b.GetBeginAtomIdx(), b.GetEndAtomIdx()))[0]

        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        a_ref = STAMPEncoder._highest_priority_substituent(mol, a, b)
        b_ref = STAMPEncoder._highest_priority_substituent(mol, b, a)
        if a_ref is None or b_ref is None:
            return

        bond.SetStereoAtoms(a_ref, b_ref)
        bond.SetStereo(desired)

    @staticmethod
    def _highest_priority_substituent(
        mol: Chem.Mol, center_idx: int, opposite_idx: int
    ) -> Optional[int]:
        subs = [
            n.GetIdx()
            for n in mol.GetAtomWithIdx(center_idx).GetNeighbors()
            if n.GetIdx() != opposite_idx
        ]
        if not subs:
            return None

        def rank(atom_idx: int) -> Tuple[int, int, int, int]:
            atom = mol.GetAtomWithIdx(atom_idx)
            cip_rank = int(atom.GetProp("_CIPRank")) if atom.HasProp("_CIPRank") else -1
            return (
                cip_rank,
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom_idx,
            )

        return max(subs, key=rank)

    def encode_mol(self, mol: Chem.Mol) -> str:
        if mol.GetNumAtoms() == 0:
            raise ValueError("Empty molecule is not supported.")

        if len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError(
                "Disconnected molecules are not supported in one STAMP string."
            )

        fragments, edges = self._fragments_and_edges_from_mol(mol)
        if not fragments:
            raise ValueError("Failed to build fragments.")
        return " ".join(self._encode_graph_tokens(fragments, edges))

    def encode_mol_to_groups(self, mol: Chem.Mol) -> List[STAMPGroup]:
        if mol.GetNumAtoms() == 0:
            raise ValueError("Empty molecule is not supported.")
        if len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError("Disconnected molecules are not supported in one STAMP string.")

        fragments, edges = self._fragments_and_edges_from_mol(mol)
        if not fragments:
            raise ValueError("Failed to build fragments.")
        return self._encode_graph_groups(fragments, edges)

    def _fragments_and_edges_from_mol(
        self,
        mol: Chem.Mol,
    ) -> Tuple[List[Fragment], List[FragmentEdge]]:
        if self.motif_tokenizer is not None:
            # Capture CIP codes and bond E/Z on the aromatic form first.
            # Kekulization can flip CIP priorities and double-bond stereo
            # labels whose neighbors change bond order during aromatization;
            # the decoder rebuilds in aromatic form, so capture must agree
            # with the aromatic reference frame.
            aro_mol = Chem.Mol(mol)
            Chem.AssignStereochemistry(aro_mol, force=True, cleanIt=True)
            cip_by_idx: Dict[int, str] = {}
            for a in aro_mol.GetAtoms():
                if a.HasProp("_CIPCode"):
                    cip_by_idx[a.GetIdx()] = a.GetProp("_CIPCode")
            bond_stereo_by_idx: Dict[int, Tuple[int, Tuple[int, int]]] = {}
            for b in aro_mol.GetBonds():
                st = b.GetStereo()
                if st in (Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOZ):
                    atoms = tuple(b.GetStereoAtoms())
                    if len(atoms) == 2:
                        bond_stereo_by_idx[b.GetIdx()] = (int(st), atoms)

            work_mol = Chem.Mol(mol)
            if self.motif_tokenizer.kekulize:
                Chem.Kekulize(work_mol, clearAromaticFlags=True)
            Chem.AssignStereochemistry(work_mol, force=True, cleanIt=True)
            for idx, code in cip_by_idx.items():
                work_mol.GetAtomWithIdx(idx).SetProp("_CIPCode", code)
            for bond_idx, (st_int, atoms) in bond_stereo_by_idx.items():
                b = work_mol.GetBondWithIdx(bond_idx)
                b.SetStereoAtoms(atoms[0], atoms[1])
                b.SetStereo(Chem.rdchem.BondStereo(st_int))
            groups = self.motif_tokenizer.decompose(work_mol, strict_vocab=True)
            if getattr(self.motif_tokenizer, "allow_ring_single_edges", False):
                return self._build_fragments_and_edges_from_groups(work_mol, groups)
            cut_bond_ids = self._cut_bonds_from_groups(work_mol, groups)
            return self._build_fragments_and_edges(work_mol, cut_bond_ids)

        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        cut_bond_ids = self._find_cut_bonds(mol)
        return self._build_fragments_and_edges(mol, cut_bond_ids)

    def _encode_graph_tokens(
        self,
        fragments: Sequence[Fragment],
        edges: Sequence[FragmentEdge],
    ) -> List[str]:
        if len(fragments) == 1:
            return [fragments[0].motif]
        if len(edges) < len(fragments) - 1:
            raise ValueError("Reduced fragment graph is disconnected under the current cut rule.")

        adj = self._build_adjacency(edges)
        root = self._choose_root(fragments, adj)
        if len(edges) == len(fragments) - 1:
            tree_children, _tree_parent = self._build_rooted_tree(root, adj)
            subtree_sizes = self._compute_subtree_sizes(root, tree_children)
            tokens: List[str] = []
            self._emit_node(
                node=root,
                parent=None,
                tokens=tokens,
                fragments=fragments,
                adj=adj,
                tree_children=tree_children,
                subtree_sizes=subtree_sizes,
            )
            return tokens

        (
            tree_children,
            tree_parent,
            tree_parent_edge,
            back_edges,
            depth,
        ) = self._build_spanning_tree_with_back_edges(
            root=root,
            graph_adj=self._build_graph_adjacency(edges),
            edge_by_id={e.edge_id: e for e in edges},
            fragments=fragments,
        )
        subtree_sizes = self._compute_subtree_sizes(root, tree_children)
        preorder = self._compute_preorder_index(root, tree_children, subtree_sizes, fragments)
        ring_outputs, ring_closures = self._organize_ring_back_edges(
            back_edges=back_edges,
            tree_parent=tree_parent,
            depth=depth,
            preorder=preorder,
        )
        tokens: List[str] = []
        self._emit_node_with_rings(
            node=root,
            parent=None,
            tokens=tokens,
            fragments=fragments,
            tree_children=tree_children,
            subtree_sizes=subtree_sizes,
            tree_parent_edge=tree_parent_edge,
            edge_by_id={e.edge_id: e for e in edges},
            ring_outputs=ring_outputs,
            ring_closures=ring_closures,
        )
        return tokens

    def _encode_graph_groups(
        self,
        fragments: Sequence[Fragment],
        edges: Sequence[FragmentEdge],
    ) -> List[STAMPGroup]:
        if len(fragments) == 1:
            frag = fragments[0]
            return [
                STAMPGroup(
                    frag_id=int(frag.frag_id),
                    atom_indices=tuple(int(x) for x in frag.atom_indices),
                    motif=str(frag.motif),
                    tokens=(str(frag.motif),),
                    parent_frag_id=None,
                    subtree_group_count=1,
                )
            ]
        if len(edges) < len(fragments) - 1:
            raise ValueError("Reduced fragment graph is disconnected under the current cut rule.")

        adj = self._build_adjacency(edges)
        root = self._choose_root(fragments, adj)
        if len(edges) == len(fragments) - 1:
            tree_children, _tree_parent = self._build_rooted_tree(root, adj)
            subtree_sizes = self._compute_subtree_sizes(root, tree_children)
            out: List[STAMPGroup] = []
            self._emit_group_nodes(
                node=root,
                parent=None,
                prefix_tokens=[],
                out=out,
                fragments=fragments,
                adj=adj,
                tree_children=tree_children,
                subtree_sizes=subtree_sizes,
            )
            return out

        (
            tree_children,
            tree_parent,
            tree_parent_edge,
            back_edges,
            depth,
        ) = self._build_spanning_tree_with_back_edges(
            root=root,
            graph_adj=self._build_graph_adjacency(edges),
            edge_by_id={e.edge_id: e for e in edges},
            fragments=fragments,
        )
        subtree_sizes = self._compute_subtree_sizes(root, tree_children)
        preorder = self._compute_preorder_index(root, tree_children, subtree_sizes, fragments)
        ring_outputs, ring_closures = self._organize_ring_back_edges(
            back_edges=back_edges,
            tree_parent=tree_parent,
            depth=depth,
            preorder=preorder,
        )
        out: List[STAMPGroup] = []
        self._emit_group_nodes_with_rings(
            node=root,
            parent=None,
            prefix_tokens=[],
            out=out,
            fragments=fragments,
            tree_children=tree_children,
            subtree_sizes=subtree_sizes,
            tree_parent_edge=tree_parent_edge,
            edge_by_id={e.edge_id: e for e in edges},
            ring_outputs=ring_outputs,
            ring_closures=ring_closures,
        )
        return out

    @staticmethod
    def _cut_bonds_from_groups(
        mol: Chem.Mol, groups: Sequence[Sequence[int]]
    ) -> List[int]:
        """Derive cut bond IDs from a tokenizer's group decomposition.

        Only exocyclic single bonds that also cross groups become cut bonds;
        any in-ring or non-single boundary bond is kept inside a single
        fragment (groups straddling such bonds are implicitly merged via
        the connected-components build in `_build_fragments_and_edges`).
        This matches the STAMP grammar constraint that cut bonds must be
        reconstructable as plain SINGLE bonds on the decoder side.
        """
        atom_to_group: Dict[int, int] = {}
        for gid, atom_ids in enumerate(groups):
            for aid in atom_ids:
                atom_to_group[aid] = gid

        next_gid = len(groups)
        for aid in range(mol.GetNumAtoms()):
            if aid not in atom_to_group:
                atom_to_group[aid] = next_gid
                next_gid += 1

        cut_bond_ids: List[int] = []
        for bond in mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if bond.IsInRing():
                continue
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            if atom_to_group[a] != atom_to_group[b]:
                cut_bond_ids.append(bond.GetIdx())
        return cut_bond_ids

    def _find_cut_bonds(self, mol: Chem.Mol) -> List[int]:
        if self.cut_rule != "exocyclic_single":
            raise ValueError(f"Unsupported cut rule: {self.cut_rule}")

        cut_bonds: List[int] = []
        for bond in mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if bond.IsInRing():
                continue
            a = bond.GetBeginAtom()
            b = bond.GetEndAtom()
            if a.IsInRing() or b.IsInRing():
                if self._single_cut_yields_valid_fragments(mol, bond.GetIdx()):
                    cut_bonds.append(bond.GetIdx())
        return cut_bonds

    @staticmethod
    def _single_cut_yields_valid_fragments(mol: Chem.Mol, bond_idx: int) -> bool:
        components = STAMPEncoder._connected_components_without_bonds(mol, [bond_idx])
        if len(components) <= 1:
            return False
        for atom_indices in components:
            frag_smi = Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=list(atom_indices),
                canonical=True,
                isomericSmiles=True,
            )
            if Chem.MolFromSmiles(frag_smi) is None:
                return False
        return True

    def _build_fragments_and_edges(
        self, mol: Chem.Mol, cut_bond_ids: Sequence[int]
    ) -> Tuple[List[Fragment], List[FragmentEdge]]:
        cut_set = set(cut_bond_ids)
        components = self._connected_components_without_bonds(mol, cut_set)
        atom_to_comp: Dict[int, int] = {}
        for comp_id, atoms in enumerate(components):
            for a in atoms:
                atom_to_comp[a] = comp_id

        fragments: List[Fragment] = []
        for comp_id, atom_indices in enumerate(components):
            motif, orig_to_local, heavy_count = self._fragment_motif_and_mapping(
                mol, atom_indices
            )
            fragments.append(
                Fragment(
                    frag_id=comp_id,
                    atom_indices=tuple(atom_indices),
                    motif=motif,
                    heavy_atom_count=heavy_count,
                    orig_to_local=orig_to_local,
                )
            )

        edges: List[FragmentEdge] = []
        for edge_id, bond_id in enumerate(cut_bond_ids):
            bond = mol.GetBondWithIdx(bond_id)
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            ca = atom_to_comp[a]
            cb = atom_to_comp[b]
            if ca == cb:
                continue

            left_local = fragments[ca].orig_to_local[a]
            right_local = fragments[cb].orig_to_local[b]
            left_stereo = self._connection_stereo_label(mol, a)
            right_stereo = self._connection_stereo_label(mol, b)
            edges.append(
                FragmentEdge(
                    edge_id=edge_id,
                    left=ca,
                    right=cb,
                    left_local_atom=left_local,
                    right_local_atom=right_local,
                    left_stereo_label=left_stereo,
                    right_stereo_label=right_stereo,
                )
            )

        return fragments, edges

    @staticmethod
    def _normalize_groups_for_stamp(
        mol: Chem.Mol,
        groups: Sequence[Sequence[int]],
    ) -> List[List[int]]:
        atom_to_gid: Dict[int, int] = {}
        normalized_groups: List[List[int]] = []
        for atom_ids in groups:
            unique = sorted({int(a) for a in atom_ids})
            if not unique:
                continue
            gid = len(normalized_groups)
            normalized_groups.append(unique)
            for aid in unique:
                atom_to_gid[aid] = gid

        for aid in range(mol.GetNumAtoms()):
            if aid not in atom_to_gid:
                gid = len(normalized_groups)
                normalized_groups.append([aid])
                atom_to_gid[aid] = gid

        parent = list(range(len(normalized_groups)))

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
            ga = atom_to_gid[a]
            gb = atom_to_gid[b]
            if ga == gb:
                continue
            if bond.GetBondType() != Chem.BondType.SINGLE:
                union(ga, gb)

        merged: Dict[int, List[int]] = {}
        for gid, atom_ids in enumerate(normalized_groups):
            merged.setdefault(find(gid), []).extend(atom_ids)

        out = [sorted(set(atom_ids)) for atom_ids in merged.values()]
        out.sort(key=lambda atom_ids: (atom_ids[0], len(atom_ids)))
        return out

    def _build_fragments_and_edges_from_groups(
        self,
        mol: Chem.Mol,
        groups: Sequence[Sequence[int]],
    ) -> Tuple[List[Fragment], List[FragmentEdge]]:
        final_groups = self._normalize_groups_for_stamp(mol, groups)
        atom_to_group: Dict[int, int] = {}
        for gid, atom_ids in enumerate(final_groups):
            for aid in atom_ids:
                atom_to_group[aid] = gid

        fragments: List[Fragment] = []
        for gid, atom_indices in enumerate(final_groups):
            motif, orig_to_local, heavy_count = self._fragment_motif_and_mapping(
                mol, atom_indices
            )
            fragments.append(
                Fragment(
                    frag_id=gid,
                    atom_indices=tuple(atom_indices),
                    motif=motif,
                    heavy_atom_count=heavy_count,
                    orig_to_local=orig_to_local,
                )
            )

        edges: List[FragmentEdge] = []
        for bond in mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            ga = atom_to_group[a]
            gb = atom_to_group[b]
            if ga == gb:
                continue
            left_local = fragments[ga].orig_to_local[a]
            right_local = fragments[gb].orig_to_local[b]
            edges.append(
                FragmentEdge(
                    edge_id=len(edges),
                    left=ga,
                    right=gb,
                    left_local_atom=left_local,
                    right_local_atom=right_local,
                    left_stereo_label=self._connection_stereo_label(mol, a),
                    right_stereo_label=self._connection_stereo_label(mol, b),
                )
            )
        return fragments, edges

    @staticmethod
    def _connection_stereo_label(mol: Chem.Mol, center_atom_idx: int) -> Optional[str]:
        atom = mol.GetAtomWithIdx(center_atom_idx)
        if atom.HasProp("_CIPCode"):
            value = atom.GetProp("_CIPCode").upper()
            if value in {"R", "S"}:
                return value
        for bond in atom.GetBonds():
            stereo = bond.GetStereo()
            if stereo == Chem.rdchem.BondStereo.STEREOE:
                return "E"
            if stereo == Chem.rdchem.BondStereo.STEREOZ:
                return "Z"
        return None

    @staticmethod
    def _connected_components_without_bonds(
        mol: Chem.Mol, removed_bonds: Iterable[int]
    ) -> List[List[int]]:
        removed = set(removed_bonds)
        n = mol.GetNumAtoms()
        seen = [False] * n
        components: List[List[int]] = []

        for start in range(n):
            if seen[start]:
                continue
            stack = [start]
            seen[start] = True
            comp: List[int] = []
            while stack:
                u = stack.pop()
                comp.append(u)
                atom = mol.GetAtomWithIdx(u)
                for bond in atom.GetBonds():
                    if bond.GetIdx() in removed:
                        continue
                    v = bond.GetOtherAtomIdx(u)
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            components.append(sorted(comp))
        return components

    @staticmethod
    def _fragment_motif_and_mapping(
        mol: Chem.Mol, atom_indices: Sequence[int]
    ) -> Tuple[str, Dict[int, int], int]:
        params = Chem.SmilesWriteParams()
        params.canonical = True
        params.doIsomericSmiles = True
        params.ignoreAtomMapNumbers = True

        motif = Chem.MolFragmentToSmiles(
            mol,
            params,
            atomsToUse=list(atom_indices),
        )

        atom_set = set(atom_indices)
        atom_map_numbers: Dict[int, int] = {}
        for idx in atom_indices:
            atom = mol.GetAtomWithIdx(idx)
            atom_map_numbers[idx] = atom.GetAtomMapNum()
            atom.SetAtomMapNum(idx + 1)

        try:
            mapped_smiles = Chem.MolFragmentToSmiles(
                mol,
                params,
                atomsToUse=list(atom_indices),
            )
        finally:
            for idx in atom_indices:
                mol.GetAtomWithIdx(idx).SetAtomMapNum(atom_map_numbers[idx])

        mapped_frag = Chem.MolFromSmiles(mapped_smiles, sanitize=False)
        if mapped_frag is None:
            raise ValueError("RDKit failed to parse mapped fragment SMILES.")

        orig_to_local: Dict[int, int] = {}
        heavy_count = 0
        for atom in mapped_frag.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num <= 0:
                continue
            orig_idx = map_num - 1
            if orig_idx in atom_set:
                orig_to_local[orig_idx] = atom.GetIdx()
            if atom.GetAtomicNum() > 1:
                heavy_count += 1

        return motif, orig_to_local, heavy_count

    @staticmethod
    def _build_adjacency(
        edges: Sequence[FragmentEdge],
    ) -> Dict[int, List[Tuple[int, int, Optional[str]]]]:
        """
        Returns:
          adj[u] = list of (v, u_local_joint, u_stereo_label)
        """
        adj: Dict[int, List[Tuple[int, int, Optional[str]]]] = {}
        for e in edges:
            adj.setdefault(e.left, []).append((e.right, e.left_local_atom, e.left_stereo_label))
            adj.setdefault(e.right, []).append((e.left, e.right_local_atom, e.right_stereo_label))
        return adj

    @staticmethod
    def _build_graph_adjacency(
        edges: Sequence[FragmentEdge],
    ) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = {}
        for e in edges:
            adj.setdefault(e.left, []).append(e.edge_id)
            adj.setdefault(e.right, []).append(e.edge_id)
        return adj

    @staticmethod
    def _edge_other(edge: FragmentEdge, node: int) -> int:
        if edge.left == node:
            return edge.right
        if edge.right == node:
            return edge.left
        raise ValueError(f"Edge {edge.edge_id} is not incident on node {node}")

    @staticmethod
    def _edge_port_for_node(edge: FragmentEdge, node: int) -> Tuple[int, Optional[str]]:
        if edge.left == node:
            return edge.left_local_atom, edge.left_stereo_label
        if edge.right == node:
            return edge.right_local_atom, edge.right_stereo_label
        raise ValueError(f"Edge {edge.edge_id} is not incident on node {node}")

    def _sorted_incident_edges(
        self,
        node: int,
        graph_adj: Dict[int, List[int]],
        edge_by_id: Dict[int, FragmentEdge],
        fragments: Sequence[Fragment],
    ) -> List[int]:
        return sorted(
            graph_adj.get(node, []),
            key=lambda edge_id: (
                -fragments[self._edge_other(edge_by_id[edge_id], node)].heavy_atom_count,
                self._edge_other(edge_by_id[edge_id], node),
                edge_id,
            ),
        )

    def _build_spanning_tree_with_back_edges(
        self,
        root: int,
        graph_adj: Dict[int, List[int]],
        edge_by_id: Dict[int, FragmentEdge],
        fragments: Sequence[Fragment],
    ) -> Tuple[
        Dict[int, List[int]],
        Dict[int, Optional[int]],
        Dict[int, int],
        List[RingBackEdge],
        Dict[int, int],
    ]:
        children: Dict[int, List[int]] = {root: []}
        parent: Dict[int, Optional[int]] = {root: None}
        parent_edge: Dict[int, int] = {}
        depth: Dict[int, int] = {root: 0}
        back_edges: List[RingBackEdge] = []
        seen_nodes = {root}
        seen_back_edges: set[int] = set()

        stack: List[Tuple[int, int]] = [(root, 0)]
        ordered_adj: Dict[int, List[int]] = {
            root: self._sorted_incident_edges(root, graph_adj, edge_by_id, fragments)
        }

        while stack:
            u, pos = stack[-1]
            incident = ordered_adj.setdefault(
                u,
                self._sorted_incident_edges(u, graph_adj, edge_by_id, fragments),
            )
            if pos >= len(incident):
                stack.pop()
                continue
            edge_id = incident[pos]
            stack[-1] = (u, pos + 1)
            edge = edge_by_id[edge_id]
            v = self._edge_other(edge, u)
            if v == parent.get(u) and edge_id == parent_edge.get(u):
                continue
            if v not in seen_nodes:
                seen_nodes.add(v)
                parent[v] = u
                parent_edge[v] = edge_id
                depth[v] = depth[u] + 1
                children.setdefault(u, []).append(v)
                children.setdefault(v, [])
                ordered_adj.setdefault(
                    v,
                    self._sorted_incident_edges(v, graph_adj, edge_by_id, fragments),
                )
                stack.append((v, 0))
                continue
            if depth.get(v, -1) >= depth[u]:
                continue
            if edge_id in seen_back_edges:
                continue
            seen_back_edges.add(edge_id)
            anc_joint, anc_stereo = self._edge_port_for_node(edge, v)
            desc_joint, desc_stereo = self._edge_port_for_node(edge, u)
            back_edges.append(
                RingBackEdge(
                    edge_id=edge_id,
                    ancestor=v,
                    descendant=u,
                    ancestor_local_atom=anc_joint,
                    descendant_local_atom=desc_joint,
                    ancestor_stereo_label=anc_stereo,
                    descendant_stereo_label=desc_stereo,
                )
            )
        return children, parent, parent_edge, back_edges, depth

    def _compute_preorder_index(
        self,
        root: int,
        tree_children: Dict[int, List[int]],
        subtree_sizes: Dict[int, int],
        fragments: Sequence[Fragment],
    ) -> Dict[int, int]:
        order: Dict[int, int] = {}
        next_idx = 0

        def dfs(node: int) -> None:
            nonlocal next_idx
            order[node] = next_idx
            next_idx += 1
            for child in self._order_children(node, tree_children, subtree_sizes, fragments):
                dfs(child)

        dfs(root)
        return order

    @staticmethod
    def _entry_child_for_descendant(
        ancestor: int,
        descendant: int,
        parent: Dict[int, Optional[int]],
    ) -> int:
        cur = descendant
        while parent.get(cur) is not None and parent[cur] != ancestor:
            cur = int(parent[cur])
        if parent.get(cur) != ancestor:
            raise ValueError(f"Node {descendant} is not in subtree of ancestor {ancestor}")
        return cur

    @staticmethod
    def _choose_root(
        fragments: Sequence[Fragment], adj: Dict[int, List[Tuple[int, int, Optional[str]]]]
    ) -> int:
        candidates: List[Tuple[int, int, int]] = []
        for f in fragments:
            deg = len(adj.get(f.frag_id, []))
            candidates.append((deg, f.heavy_atom_count, -f.frag_id))
        best = max(candidates)
        return -best[2]

    @staticmethod
    def _build_rooted_tree(
        root: int, adj: Dict[int, List[Tuple[int, int, Optional[str]]]]
    ) -> Tuple[Dict[int, List[int]], Dict[int, Optional[int]]]:
        children: Dict[int, List[int]] = {}
        parent: Dict[int, Optional[int]] = {root: None}
        stack = [root]

        while stack:
            u = stack.pop()
            children.setdefault(u, [])
            for v, _, _ in adj.get(u, []):
                if v == parent.get(u):
                    continue
                if v in parent:
                    continue
                parent[v] = u
                children[u].append(v)
                children.setdefault(v, [])
                stack.append(v)
        return children, parent

    @staticmethod
    def _compute_subtree_sizes(
        root: int, tree_children: Dict[int, List[int]]
    ) -> Dict[int, int]:
        sizes: Dict[int, int] = {}

        def dfs(u: int) -> int:
            total = 1
            for v in tree_children.get(u, []):
                total += dfs(v)
            sizes[u] = total
            return total

        dfs(root)
        return sizes

    @staticmethod
    def _port_for_neighbor(
        adj: Dict[int, List[Tuple[int, int, Optional[str]]]], u: int, v: int
    ) -> Tuple[int, Optional[str]]:
        for nb, u_joint, u_stereo in adj.get(u, []):
            if nb == v:
                return u_joint, u_stereo
        raise ValueError(f"Missing joint mapping for {u}->{v}")

    @staticmethod
    def _append_port_tokens(
        tokens: List[str],
        joint: int,
        stereo_label: Optional[str],
        *,
        ring_closure: bool = False,
    ) -> None:
        tokens.append(f"[J_{joint}]")
        if stereo_label is not None:
            tokens.append(f"[S_{stereo_label}]")
        if ring_closure:
            tokens.append("[Ring]")

    @staticmethod
    def _order_children(
        node: int,
        tree_children: Dict[int, List[int]],
        subtree_sizes: Dict[int, int],
        fragments: Sequence[Fragment],
    ) -> List[int]:
        return sorted(
            tree_children.get(node, []),
            key=lambda c: (
                -subtree_sizes[c],
                -fragments[c].heavy_atom_count,
                c,
            ),
        )

    def _branch_token(self, group_count: int) -> str:
        n = max(self.min_branch_n, group_count)
        if self.max_branch_n is not None:
            n = min(n, self.max_branch_n)
        return f"[B_{n}]"

    def _emit_node(
        self,
        node: int,
        parent: Optional[int],
        tokens: List[str],
        fragments: Sequence[Fragment],
        adj: Dict[int, List[Tuple[int, int, Optional[str]]]],
        tree_children: Dict[int, List[int]],
        subtree_sizes: Dict[int, int],
    ) -> None:
        frag = fragments[node]
        children = self._order_children(node, tree_children, subtree_sizes, fragments)

        # Main-chain child: highest-priority child in canonical order.
        main_child = children[0] if children else None
        branch_children = children[1:] if len(children) > 1 else []

        tokens.append(frag.motif)

        if parent is not None:
            joint, stereo_label = self._port_for_neighbor(adj, node, parent)
            self._append_port_tokens(tokens, joint, stereo_label)

        # To match a stack-decoder that handles [B_n] before the main-chain child,
        # we emit branch joints first, then main-chain joint.
        for child in branch_children:
            joint, stereo_label = self._port_for_neighbor(adj, node, child)
            self._append_port_tokens(tokens, joint, stereo_label)
        if main_child is not None:
            joint, stereo_label = self._port_for_neighbor(adj, node, main_child)
            self._append_port_tokens(tokens, joint, stereo_label)

        for child in branch_children:
            tokens.append(self._branch_token(subtree_sizes[child]))
            self._emit_node(
                node=child,
                parent=node,
                tokens=tokens,
                fragments=fragments,
                adj=adj,
                tree_children=tree_children,
                subtree_sizes=subtree_sizes,
            )

        if main_child is not None:
            self._emit_node(
                node=main_child,
                parent=node,
                tokens=tokens,
                fragments=fragments,
                adj=adj,
                tree_children=tree_children,
                subtree_sizes=subtree_sizes,
            )

    def _emit_group_nodes(
        self,
        node: int,
        parent: Optional[int],
        prefix_tokens: Sequence[str],
        out: List[STAMPGroup],
        fragments: Sequence[Fragment],
        adj: Dict[int, List[Tuple[int, int, Optional[str]]]],
        tree_children: Dict[int, List[int]],
        subtree_sizes: Dict[int, int],
    ) -> None:
        frag = fragments[node]
        children = self._order_children(node, tree_children, subtree_sizes, fragments)
        main_child = children[0] if children else None
        branch_children = children[1:] if len(children) > 1 else []

        group_tokens: List[str] = list(prefix_tokens)
        group_tokens.append(frag.motif)
        if parent is not None:
            joint, stereo_label = self._port_for_neighbor(adj, node, parent)
            self._append_port_tokens(group_tokens, joint, stereo_label)
        for child in branch_children:
            joint, stereo_label = self._port_for_neighbor(adj, node, child)
            self._append_port_tokens(group_tokens, joint, stereo_label)
        if main_child is not None:
            joint, stereo_label = self._port_for_neighbor(adj, node, main_child)
            self._append_port_tokens(group_tokens, joint, stereo_label)

        out.append(
            STAMPGroup(
                frag_id=int(node),
                atom_indices=tuple(int(x) for x in frag.atom_indices),
                motif=str(frag.motif),
                tokens=tuple(group_tokens),
                parent_frag_id=(None if parent is None else int(parent)),
                subtree_group_count=int(subtree_sizes[node]),
            )
        )
        for child in branch_children:
            self._emit_group_nodes(
                node=child,
                parent=node,
                prefix_tokens=[self._branch_token(subtree_sizes[child])],
                out=out,
                fragments=fragments,
                adj=adj,
                tree_children=tree_children,
                subtree_sizes=subtree_sizes,
            )
        if main_child is not None:
            self._emit_group_nodes(
                node=main_child,
                parent=node,
                prefix_tokens=[],
                out=out,
                fragments=fragments,
                adj=adj,
                tree_children=tree_children,
                subtree_sizes=subtree_sizes,
            )

    def _organize_ring_back_edges(
        self,
        back_edges: Sequence[RingBackEdge],
        tree_parent: Dict[int, Optional[int]],
        depth: Dict[int, int],
        preorder: Dict[int, int],
    ) -> Tuple[Dict[Tuple[int, int], List[RingBackEdge]], Dict[int, List[RingBackEdge]]]:
        ring_outputs: Dict[Tuple[int, int], List[RingBackEdge]] = {}
        ring_closures: Dict[int, List[RingBackEdge]] = {}

        for spec in back_edges:
            entry_child = self._entry_child_for_descendant(
                ancestor=spec.ancestor,
                descendant=spec.descendant,
                parent=tree_parent,
            )
            ring_outputs.setdefault((spec.ancestor, entry_child), []).append(spec)
            ring_closures.setdefault(spec.descendant, []).append(spec)

        for key, items in ring_outputs.items():
            items.sort(key=lambda item: (preorder.get(item.descendant, 0), item.edge_id))
        for node, items in ring_closures.items():
            items.sort(
                key=lambda item: (
                    -depth.get(item.ancestor, 0),
                    preorder.get(item.ancestor, 0),
                    item.edge_id,
                )
            )
        return ring_outputs, ring_closures

    def _emit_node_with_rings(
        self,
        node: int,
        parent: Optional[int],
        tokens: List[str],
        fragments: Sequence[Fragment],
        tree_children: Dict[int, List[int]],
        subtree_sizes: Dict[int, int],
        tree_parent_edge: Dict[int, int],
        edge_by_id: Dict[int, FragmentEdge],
        ring_outputs: Dict[Tuple[int, int], List[RingBackEdge]],
        ring_closures: Dict[int, List[RingBackEdge]],
    ) -> None:
        frag = fragments[node]
        children = self._order_children(node, tree_children, subtree_sizes, fragments)
        main_child = children[0] if children else None
        branch_children = children[1:] if len(children) > 1 else []

        tokens.append(frag.motif)
        if parent is not None:
            parent_edge = edge_by_id[tree_parent_edge[node]]
            joint, stereo_label = self._edge_port_for_node(parent_edge, node)
            self._append_port_tokens(tokens, joint, stereo_label)
        for spec in ring_closures.get(node, []):
            self._append_port_tokens(
                tokens,
                spec.descendant_local_atom,
                spec.descendant_stereo_label,
                ring_closure=True,
            )

        for child in branch_children:
            child_edge = edge_by_id[tree_parent_edge[child]]
            joint, stereo_label = self._edge_port_for_node(child_edge, node)
            self._append_port_tokens(tokens, joint, stereo_label)
            for spec in ring_outputs.get((node, child), []):
                self._append_port_tokens(tokens, spec.ancestor_local_atom, spec.ancestor_stereo_label)
        if main_child is not None:
            child_edge = edge_by_id[tree_parent_edge[main_child]]
            joint, stereo_label = self._edge_port_for_node(child_edge, node)
            self._append_port_tokens(tokens, joint, stereo_label)
            for spec in ring_outputs.get((node, main_child), []):
                self._append_port_tokens(tokens, spec.ancestor_local_atom, spec.ancestor_stereo_label)

        for child in branch_children:
            tokens.append(self._branch_token(subtree_sizes[child]))
            self._emit_node_with_rings(
                node=child,
                parent=node,
                tokens=tokens,
                fragments=fragments,
                tree_children=tree_children,
                subtree_sizes=subtree_sizes,
                tree_parent_edge=tree_parent_edge,
                edge_by_id=edge_by_id,
                ring_outputs=ring_outputs,
                ring_closures=ring_closures,
            )
        if main_child is not None:
            self._emit_node_with_rings(
                node=main_child,
                parent=node,
                tokens=tokens,
                fragments=fragments,
                tree_children=tree_children,
                subtree_sizes=subtree_sizes,
                tree_parent_edge=tree_parent_edge,
                edge_by_id=edge_by_id,
                ring_outputs=ring_outputs,
                ring_closures=ring_closures,
            )

    def _emit_group_nodes_with_rings(
        self,
        node: int,
        parent: Optional[int],
        prefix_tokens: Sequence[str],
        out: List[STAMPGroup],
        fragments: Sequence[Fragment],
        tree_children: Dict[int, List[int]],
        subtree_sizes: Dict[int, int],
        tree_parent_edge: Dict[int, int],
        edge_by_id: Dict[int, FragmentEdge],
        ring_outputs: Dict[Tuple[int, int], List[RingBackEdge]],
        ring_closures: Dict[int, List[RingBackEdge]],
    ) -> None:
        frag = fragments[node]
        children = self._order_children(node, tree_children, subtree_sizes, fragments)
        main_child = children[0] if children else None
        branch_children = children[1:] if len(children) > 1 else []

        group_tokens: List[str] = list(prefix_tokens)
        group_tokens.append(frag.motif)
        if parent is not None:
            parent_edge = edge_by_id[tree_parent_edge[node]]
            joint, stereo_label = self._edge_port_for_node(parent_edge, node)
            self._append_port_tokens(group_tokens, joint, stereo_label)
        for spec in ring_closures.get(node, []):
            self._append_port_tokens(
                group_tokens,
                spec.descendant_local_atom,
                spec.descendant_stereo_label,
                ring_closure=True,
            )
        for child in branch_children:
            child_edge = edge_by_id[tree_parent_edge[child]]
            joint, stereo_label = self._edge_port_for_node(child_edge, node)
            self._append_port_tokens(group_tokens, joint, stereo_label)
            for spec in ring_outputs.get((node, child), []):
                self._append_port_tokens(group_tokens, spec.ancestor_local_atom, spec.ancestor_stereo_label)
        if main_child is not None:
            child_edge = edge_by_id[tree_parent_edge[main_child]]
            joint, stereo_label = self._edge_port_for_node(child_edge, node)
            self._append_port_tokens(group_tokens, joint, stereo_label)
            for spec in ring_outputs.get((node, main_child), []):
                self._append_port_tokens(group_tokens, spec.ancestor_local_atom, spec.ancestor_stereo_label)

        out.append(
            STAMPGroup(
                frag_id=int(node),
                atom_indices=tuple(int(x) for x in frag.atom_indices),
                motif=str(frag.motif),
                tokens=tuple(group_tokens),
                parent_frag_id=(None if parent is None else int(parent)),
                subtree_group_count=int(subtree_sizes[node]),
            )
        )
        for child in branch_children:
            self._emit_group_nodes_with_rings(
                node=child,
                parent=node,
                prefix_tokens=[self._branch_token(subtree_sizes[child])],
                out=out,
                fragments=fragments,
                tree_children=tree_children,
                subtree_sizes=subtree_sizes,
                tree_parent_edge=tree_parent_edge,
                edge_by_id=edge_by_id,
                ring_outputs=ring_outputs,
                ring_closures=ring_closures,
            )
        if main_child is not None:
            self._emit_group_nodes_with_rings(
                node=main_child,
                parent=node,
                prefix_tokens=[],
                out=out,
                fragments=fragments,
                tree_children=tree_children,
                subtree_sizes=subtree_sizes,
                tree_parent_edge=tree_parent_edge,
                edge_by_id=edge_by_id,
                ring_outputs=ring_outputs,
                ring_closures=ring_closures,
            )
