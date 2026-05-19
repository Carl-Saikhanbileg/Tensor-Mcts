from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from core_algo.tensor_network import TensorNetwork, TensorNode

AxisLabel = str

@dataclass(frozen=True)
class EdgeReductionProposal:
    edge_label: AxisLabel
    child_name: str
    parent_name: str
    old_rank: int
    new_rank: int
    cost_delta: int
    residual_error_sq: float

def _copy_executed_network(net):
    G_old, G_new = net.graph, TensorNetwork()
    G_new.next_rank_id = G_old.next_rank_id
    for name, node in G_old.nodes.items():
        G_new.add_node(TensorNode(name, node.data.copy(), list(node.axis_labels)))
    for e in G_old.edges:
        G_new.connect(e.node_u, e.node_v, e.label)
    try:
        return net.__class__(G_new, net.root_name, net.original_shape)
    except Exception:
        class _Net:
            def __init__(self, g, r, o):
                self.graph, self.root_name, self.original_shape = g, r, o
            def storage_cost(self):
                return int(self.graph.storage_cost())
        return _Net(G_new, net.root_name, net.original_shape)


def _parent_map(G: TensorNetwork, root: str) -> Dict[str, Optional[str]]:
    parent: Dict[str, Optional[str]] = {root: None}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in G.neighbors(u):
            if v not in parent:
                parent[v] = u
                stack.append(v)
    return parent


def _bfs_order(G: TensorNetwork, root: str) -> List[str]:
    visited: set = set()
    order: List[str] = []
    q: deque = deque([root])
    while q:
        u = q.popleft()
        if u in visited:
            continue
        visited.add(u)
        order.append(u)
        for v in G.neighbors(u):
            if v not in visited:
                q.append(v)
    return order


def _bond_label(child_node: TensorNode, parent_node: TensorNode) -> Optional[AxisLabel]:
    for lab in child_node.axis_labels:
        if lab.startswith("rank:") and lab in parent_node.axis_labels:
            return lab
    return None


def _bottom_up_qr_sweep(
    G: TensorNetwork,
    bfs_order: List[str],
    parents: Dict[str, Optional[str]],
) -> None:
    for name in reversed(bfs_order):
        parent_name = parents.get(name)
        if parent_name is None:
            continue  # root — nothing to do

        child = G.get_node(name)
        parent = G.get_node(parent_name)
        lab = _bond_label(child, parent)
        if lab is None:
            continue

        ax = child.axis_of(lab)
        other = [i for i in range(child.order) if i != ax]

        # Reshape to (prod_other, bond_dim) and QR-decompose
        M = np.transpose(child.data, other + [ax]).reshape(-1, child.shape[ax])
        Q, R = np.linalg.qr(M, mode="reduced")

        # Write Q back into the child, restoring the original axis order
        new_shape = [child.data.shape[i] for i in other] + [Q.shape[1]]
        inv_perm = np.argsort(other + [ax]).tolist()
        child.data = np.transpose(Q.reshape(new_shape), inv_perm)

        # Absorb R into the parent along the shared bond axis
        p_ax = parent.axis_of(lab)
        P = np.moveaxis(parent.data, p_ax, 0)       # (bond, ...)
        new_P = np.tensordot(R, P, ([1], [0]))       # (bond', ...)
        parent.data = np.moveaxis(new_P, 0, p_ax)


def _top_down_svd_sweep(
    G: TensorNetwork,
    bfs_order: List[str],
    parents: Dict[str, Optional[str]],
    budget: float,
    min_rank: int,
) -> None:
    remaining = budget

    for name in bfs_order:
        parent_name = parents.get(name)
        if parent_name is None:
            continue  # root — no parent bond

        child = G.get_node(name)
        parent_node = G.get_node(parent_name)
        lab = _bond_label(child, parent_node)
        if lab is None:
            continue
        p_ax = parent_node.axis_of(lab)
        other_p = [i for i in range(parent_node.order) if i != p_ax]
        M = np.moveaxis(parent_node.data, p_ax, -1).reshape(-1, parent_node.shape[p_ax])

        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        r_old = len(s)
        r_new = r_old  # default: no truncation
        for r in range(min_rank, r_old):
            if float(np.sum(s[r:] ** 2)) <= remaining:
                r_new = r
                break
        new_p_shape = [parent_node.data.shape[i] for i in other_p] + [r_new]
        parent_node.data = np.moveaxis(
            U[:, :r_new].reshape(new_p_shape), -1, p_ax
        )

        c_ax = child.axis_of(lab)
        C = np.moveaxis(child.data, c_ax, 0)             # (old_bond, ...)
        SV = np.diag(s[:r_new]) @ Vt[:r_new, :]         # (r_new, old_bond)
        child.data = np.moveaxis(
            np.tensordot(SV, C, ([1], [0])), 0, c_ax     # (r_new, ...)
        )

        if r_new < r_old:
            remaining -= float(np.sum(s[r_new:] ** 2))

def round_tensor_network(network, eps: float, min_rank: int = 1, max_passes: int = 1000):
    net = _copy_executed_network(network)
    G = net.graph

    if len(G.nodes) <= 1:
        return net

    root_name = net.root_name
    parents = _parent_map(G, root_name)
    bfs = _bfs_order(G, root_name)
    _bottom_up_qr_sweep(G, bfs, parents)
    root_norm_sq = float(np.linalg.norm(G.get_node(root_name).data) ** 2)
    budget = (eps ** 2) * root_norm_sq

    _top_down_svd_sweep(G, bfs, parents, budget, min_rank)

    net.graph.validate()
    return net