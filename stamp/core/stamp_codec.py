from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem


@dataclass(frozen=True)
class Fragment:
    frag_id: int
    atom_indices: Tuple[int, ...]
    motif: str
    heavy_atom_count: int
    orig_to_local: Dict[int, int]


@dataclass(frozen=True)
class FragmentEdge:
    left: int
    right: int
    left_local_atom: int
    right_local_atom: int
    left_stereo_label: Optional[str]
    right_stereo_label: Optional[str]


@dataclass(frozen=True)
class Port:
    atom_idx: int


@dataclass
class BranchFrame:
    remaining_groups: int
    saved_stack: List[Port]


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
    ) -> None:
        self.cut_rule = cut_rule
        self.min_branch_n = min_branch_n
        self.max_branch_n = max_branch_n

    def encode_smiles(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return self.encode_mol(mol)

    def decode_stamp(self, stamp: str, canonical: bool = True) -> str:
        mol = self.decode_stamp_to_mol(stamp)
        return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=True)

    def decode_stamp_to_mol(self, stamp: str) -> Chem.Mol:
        tokens = [t for t in stamp.split() if t]
        if not tokens:
            raise ValueError("Empty STAMP sequence.")

        rw = Chem.RWMol()
        port_stack: List[Port] = []
        branch_frames: List[BranchFrame] = []
        stereo_map: Dict[int, str] = {}
        is_root = True
        i = 0

        while i < len(tokens):
            tok = tokens[i]

            if tok == "[END]":
                break

            if tok == "[BACK]":
                raise ValueError("Decoder does not support [BACK] trajectories.")

            if self._is_branch(tok):
                n = self._parse_branch(tok)
                active = port_stack.pop() if port_stack else None
                saved = list(port_stack)
                port_stack = [active] if active is not None else []
                branch_frames.append(BranchFrame(remaining_groups=max(0, n), saved_stack=saved))
                i += 1
                continue

            if self._is_joint(tok):
                raise ValueError(f"Unexpected joint token without motif at position {i}: {tok}")

            # Keep latent chirality from motif tokens until the whole molecule
            # is assembled; early sanitization can drop valid '@' annotations.
            frag = Chem.MolFromSmiles(tok, sanitize=False)
            if frag is None:
                raise ValueError(f"Invalid motif token at position {i}: {tok}")

            rw, frag_offset = self._append_fragment(rw, frag)
            i += 1

            ports: List[Port] = []
            local_heavy = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() > 1]
            if not local_heavy:
                local_heavy = [a.GetIdx() for a in frag.GetAtoms()]
            if not local_heavy:
                raise ValueError("Motif has no atoms, cannot decode.")

            while i < len(tokens) and self._is_joint(tokens[i]):
                j = self._parse_joint(tokens[i])
                i += 1

                # R4.1: out-of-range [J_i] fallback by modulo heavy-atom count.
                local_idx = local_heavy[j % len(local_heavy)]
                if i < len(tokens) and self._is_stereo(tokens[i]):
                    label = self._parse_stereo(tokens[i])
                    i += 1
                    if label is not None:
                        stereo_map[frag_offset + local_idx] = label

                ports.append(
                    Port(
                        atom_idx=frag_offset + local_idx,
                    )
                )

            if is_root:
                for p in reversed(ports):
                    port_stack.append(p)
                is_root = False
            else:
                if ports:
                    parent_port = port_stack.pop() if port_stack else None
                    if parent_port is not None:
                        input_candidates = self._rotated_port_candidates(ports[0], ports)
                        self._try_add_bond_with_fallback(rw, parent_port, input_candidates)
                    for p in reversed(ports[1:]):
                        port_stack.append(p)

            if branch_frames:
                # [B_n] counts motif-groups in subtree, including nested branches.
                for frame in branch_frames:
                    frame.remaining_groups -= 1
                while branch_frames and branch_frames[-1].remaining_groups <= 0:
                    done = branch_frames.pop()
                    port_stack = done.saved_stack

        # R4.4: if branch lengths exceed remaining tokens, truncate naturally.
        while branch_frames:
            done = branch_frames.pop()
            port_stack = done.saved_stack

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        self._restore_chirality(mol, stereo_map)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=False)
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

        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        cut_bond_ids = self._find_cut_bonds(mol)
        fragments, edges = self._build_fragments_and_edges(mol, cut_bond_ids)
        if not fragments:
            raise ValueError("Failed to build fragments.")
        if len(fragments) > 1 and len(edges) != len(fragments) - 1:
            raise ValueError(
                "Reduced fragment graph is not a tree under the current cut rule."
            )

        adj = self._build_adjacency(edges)
        root = self._choose_root(fragments, adj)
        tree_children, tree_parent = self._build_rooted_tree(root, adj)
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
        return " ".join(tokens)

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
        for bond_id in cut_bond_ids:
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
    def _append_port_tokens(tokens: List[str], joint: int, stereo_label: Optional[str]) -> None:
        tokens.append(f"[J_{joint}]")
        if stereo_label is not None:
            tokens.append(f"[S_{stereo_label}]")

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
