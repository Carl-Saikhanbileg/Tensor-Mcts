from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core_algo.tensor_core import TensorData
from core_algo.tensor_network import TensorNetwork, TensorNode
from core_algo.sketch import Sketch, normalize_subset
from core_algo.scoring import RankAssignmentResult


AxisLabel = str
Subset = Tuple[int, ...]


@dataclass
class ExecutedNetwork:
    graph: TensorNetwork
    root_name: str
    original_shape: Tuple[int, ...]

    @classmethod
    def from_tensor(cls, X: TensorData) -> "ExecutedNetwork":
        G = TensorNetwork()
        axis_labels = [f"free:I{i}" for i in range(X.order)]
        root = TensorNode(
            name="X",
            data=np.asarray(X.data, dtype=float).copy(),
            axis_labels=axis_labels,
        )
        G.add_node(root)
        return cls(graph=G, root_name="X", original_shape=X.shape)

    def storage_cost(self) -> int:
        return int(self.graph.storage_cost())

    def summary(self) -> str:
        lines: List[str] = []
        lines.append("ExecutedNetwork:")
        lines.append(f"  storage_cost = {self.storage_cost()}")
        lines.append(f"  num_nodes = {len(self.graph.nodes)}")
        lines.append(f"  num_edges = {len(self.graph.edges)}")
        for name in sorted(self.graph.nodes):
            node = self.graph.nodes[name]
            rank_labels = tuple(lab for lab in node.axis_labels if lab.startswith("rank:"))
            free_labels = tuple(lab for lab in node.axis_labels if lab.startswith("free:"))
            lines.append(
                f"  {name}: shape={node.shape}, "
                f"free_labels={free_labels}, rank_labels={rank_labels}"
            )
        return "\n".join(lines)


def _copy_network(net: ExecutedNetwork) -> ExecutedNetwork:
    G_old = net.graph
    G_new = TensorNetwork()
    G_new.next_rank_id = G_old.next_rank_id

    for name, node in G_old.nodes.items():
        G_new.add_node(
            TensorNode(
                name=name,
                data=np.asarray(node.data, dtype=float).copy(),
                axis_labels=list(node.axis_labels),
            )
        )

    for e in G_old.edges:
        G_new.connect(e.node_u, e.node_v, e.label)

    return ExecutedNetwork(
        graph=G_new,
        root_name=net.root_name,
        original_shape=net.original_shape,
    )


def _neighbors_with_edge_labels(G: TensorNetwork, node_name: str) -> List[Tuple[str, AxisLabel]]:
    out: List[Tuple[str, AxisLabel]] = []
    for e in G.edges:
        if e.node_u == node_name:
            out.append((e.node_v, e.label))
        elif e.node_v == node_name:
            out.append((e.node_u, e.label))
    return out


def _free_mode_ids_of_node(node: TensorNode) -> List[int]:
    out: List[int] = []
    for lab in node.axis_labels:
        if lab.startswith("free:I"):
            out.append(int(lab.split("I")[1]))
    return out


def _subtree_free_modes(
    G: TensorNetwork,
    start: str,
    parent: str,
) -> Subset:
    seen = {parent}
    stack = [start]
    free_modes: set[int] = set()

    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)

        node = G.get_node(u)
        free_modes.update(_free_mode_ids_of_node(node))

        for v, _ in _neighbors_with_edge_labels(G, u):
            if v not in seen:
                stack.append(v)

    return tuple(sorted(free_modes))


def _incident_partitions(G: TensorNetwork, node_name: str) -> List[Tuple[AxisLabel, Subset]]:
    parts: List[Tuple[AxisLabel, Subset]] = []
    for neigh, edge_label in _neighbors_with_edge_labels(G, node_name):
        subset = _subtree_free_modes(G, neigh, node_name)
        parts.append((edge_label, subset))
    return parts


def _partition_node_axes_for_osplit(
    G: TensorNetwork,
    node_name: str,
    target_subset: Subset,
) -> Optional[Tuple[List[AxisLabel], List[AxisLabel]]]:
    target = set(target_subset)
    node = G.get_node(node_name)

    left_axes: List[AxisLabel] = []
    right_axes: List[AxisLabel] = []

    for lab in node.axis_labels:
        if lab.startswith("free:I"):
            mode = int(lab.split("I")[1])
            if mode in target:
                left_axes.append(lab)
            else:
                right_axes.append(lab)

    for edge_label, subset in _incident_partitions(G, node_name):
        S = set(subset)

        if S == target:
            return ([], [])

        if S <= target:
            left_axes.append(edge_label)
        elif S.isdisjoint(target):
            right_axes.append(edge_label)
        else:
            return None

    left_modes: set[int] = set()
    for lab in left_axes:
        if lab.startswith("free:I"):
            left_modes.add(int(lab.split("I")[1]))
        else:
            for edge_lab, subset in _incident_partitions(G, node_name):
                if edge_lab == lab:
                    left_modes.update(subset)

    if left_modes != target:
        return None

    if len(left_axes) == 0 or len(right_axes) == 0:
        return None

    return left_axes, right_axes


def _permute_node_for_split(
    node: TensorNode,
    left_axes: Sequence[AxisLabel],
) -> Tuple[np.ndarray, List[AxisLabel], List[AxisLabel]]:
    left_axes = list(left_axes)
    right_axes = [lab for lab in node.axis_labels if lab not in left_axes]

    perm = [node.axis_of(lab) for lab in left_axes + right_axes]
    Xp = np.transpose(node.data, axes=perm)

    left_dim = int(prod(node.data.shape[node.axis_of(lab)] for lab in left_axes))
    right_dim = int(prod(node.data.shape[node.axis_of(lab)] for lab in right_axes))
    M = Xp.reshape(left_dim, right_dim)

    return M, left_axes, right_axes


def _svd_split_matrix(
    M: np.ndarray,
    target_rank: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    U, s, Vt = np.linalg.svd(M, full_matrices=False)

    q = len(s)
    if target_rank < 0 or target_rank > q:
        raise ValueError(
            f"Invalid target rank {target_rank}; matrix supports ranks up to {q}."
        )

    U_r = U[:, :target_rank]
    s_r = s[:target_rank]
    Vt_r = Vt[:target_rank, :]

    return U_r, s_r, Vt_r


def _make_child_name(base: str, suffix: str, G: TensorNetwork) -> str:
    name = f"{base}_{suffix}"
    if name not in G.nodes:
        return name

    i = 0
    while f"{name}_{i}" in G.nodes:
        i += 1
    return f"{name}_{i}"


def _split_node_in_network(
    G: TensorNetwork,
    node_name: str,
    left_axes: Sequence[AxisLabel],
    target_rank: int,
) -> str:
    node = G.get_node(node_name)

    M, left_axes, right_axes = _permute_node_for_split(node, left_axes)
    U_r, s_r, Vt_r = _svd_split_matrix(M, target_rank=target_rank)

    rank_label = G.fresh_rank_label()

    left_shape = [node.data.shape[node.axis_of(lab)] for lab in left_axes] + [target_rank]
    right_shape = [target_rank] + [node.data.shape[node.axis_of(lab)] for lab in right_axes]

    left_tensor = U_r.reshape(left_shape)
    right_tensor = (np.diag(s_r) @ Vt_r).reshape(right_shape)

    left_name = _make_child_name(node_name, "L", G)
    right_name = _make_child_name(node_name, "R", G)

    left_node = TensorNode(
        name=left_name,
        data=left_tensor,
        axis_labels=list(left_axes) + [rank_label],
    )
    right_node = TensorNode(
        name=right_name,
        data=right_tensor,
        axis_labels=[rank_label] + list(right_axes),
    )

    old_neighbors = _neighbors_with_edge_labels(G, node_name)

    G.remove_node(node_name)

    G.add_node(left_node)
    G.add_node(right_node)
    G.connect(left_name, right_name, rank_label)

    for neigh, edge_lab in old_neighbors:
        if neigh not in G.nodes:
            continue

        if edge_lab in left_node.axis_labels:
            G.connect(left_name, neigh, edge_lab)
        elif edge_lab in right_node.axis_labels:
            G.connect(right_name, neigh, edge_lab)
        else:
            raise RuntimeError(
                f"Lost external edge {edge_lab} when splitting node {node_name}."
            )

    G.validate()
    return right_name


def execute_osplit(
    net: ExecutedNetwork,
    target_subset: Subset,
    target_rank: int,
) -> ExecutedNetwork:
    out = _copy_network(net)
    G = out.graph

    for node_name in list(G.nodes.keys()):
        partition = _partition_node_axes_for_osplit(G, node_name, target_subset)

        if partition is None:
            continue

        left_axes, right_axes = partition

        if len(left_axes) == 0 and len(right_axes) == 0:
            return out

        new_root = _split_node_in_network(
            G=G,
            node_name=node_name,
            left_axes=left_axes,
            target_rank=target_rank,
        )
        out.root_name = new_root
        return out

    raise ValueError(f"Could not realize OSplit for subset {target_subset}.")


def execute_scored_sketch(
    X: TensorData,
    sketch: Sketch,
    rank_assignment: RankAssignmentResult | None = None,
    eps: float | None = None,
    ranks: RankAssignmentResult | None = None,
) -> ExecutedNetwork:
    if rank_assignment is None:
        rank_assignment = ranks
    if rank_assignment is None:
        raise ValueError("Must provide rank_assignment")

    net = ExecutedNetwork.from_tensor(X)

    for subset in sketch.subsets():
        subset = normalize_subset(subset)

        if subset not in rank_assignment.ranks:
            raise KeyError(f"Missing assigned rank for subset {subset}")

        r = int(rank_assignment.ranks[subset])
        net = execute_osplit(net, target_subset=subset, target_rank=r)

    net.graph.validate()
    return net