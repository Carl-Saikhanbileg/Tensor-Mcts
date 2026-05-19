from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt


Array = npt.NDArray[np.floating]
AxisLabel = str


@dataclass(slots=True)
class TensorNode:
    """
    A tensor node in a tensor network.

    axis_labels must have the same length as data.ndim.
    Labels are typically of the form:
        "free:I0", "free:I1", ...
        "rank:r0", "rank:r1", ...
    """
    name: str
    data: Array
    axis_labels: List[AxisLabel]

    def __post_init__(self) -> None:
        if self.data.ndim != len(self.axis_labels):
            raise ValueError(
                f"{self.name}: len(axis_labels) must equal data.ndim"
            )
        if len(set(self.axis_labels)) != len(self.axis_labels):
            raise ValueError(
                f"{self.name}: axis_labels must be unique within a node"
            )

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def order(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return int(prod(self.shape))

    def axis_of(self, label: AxisLabel) -> int:
        return self.axis_labels.index(label)

    def has_label(self, label: AxisLabel) -> bool:
        return label in self.axis_labels

    @property
    def free_labels(self) -> Tuple[AxisLabel, ...]:
        return tuple(lab for lab in self.axis_labels if lab.startswith("free:"))

    @property
    def rank_labels(self) -> Tuple[AxisLabel, ...]:
        return tuple(lab for lab in self.axis_labels if lab.startswith("rank:"))

    def rename_axis_label(self, old: AxisLabel, new: AxisLabel) -> None:
        i = self.axis_of(old)
        self.axis_labels[i] = new

    def permute_axes(self, new_order: Tuple[int, ...]) -> None:
        self.data = np.transpose(self.data, axes=new_order)
        self.axis_labels = [self.axis_labels[i] for i in new_order]


@dataclass(frozen=True, slots=True)
class TensorEdge:
    """
    Undirected contracted edge between two tensor nodes.

    label is the shared rank label, e.g. "rank:r3".
    """
    node_u: str
    node_v: str
    label: AxisLabel

    def endpoints(self) -> Tuple[str, str]:
        return self.node_u, self.node_v


@dataclass
class TensorNetwork:
    """
    Structural tensor network container.

    Invariants:
    - each node name is unique
    - each edge label appears in exactly two nodes
    - edge dimensions must match across endpoints
    """
    nodes: Dict[str, TensorNode] = field(default_factory=dict)
    edges: List[TensorEdge] = field(default_factory=list)
    next_rank_id: int = 0

    def add_node(self, node: TensorNode) -> None:
        if node.name in self.nodes:
            raise ValueError(f"Node {node.name} already exists")
        self.nodes[node.name] = node

    def remove_node(self, name: str) -> None:
        if name not in self.nodes:
            raise KeyError(f"Node {name} not found")
        self.edges = [
            e for e in self.edges
            if e.node_u != name and e.node_v != name
        ]
        del self.nodes[name]

    def get_node(self, name: str) -> TensorNode:
        return self.nodes[name]

    def fresh_rank_label(self) -> AxisLabel:
        label = f"rank:r{self.next_rank_id}"
        self.next_rank_id += 1
        return label

    def connect(self, node_u: str, node_v: str, label: AxisLabel) -> None:
        """
        Register a contracted edge between two existing nodes that already
        contain the shared rank label.
        """
        u = self.nodes[node_u]
        v = self.nodes[node_v]

        if label not in u.axis_labels:
            raise ValueError(f"{label} not found in node {node_u}")
        if label not in v.axis_labels:
            raise ValueError(f"{label} not found in node {node_v}")

        dim_u = u.shape[u.axis_of(label)]
        dim_v = v.shape[v.axis_of(label)]

        if dim_u != dim_v:
            raise ValueError(
                f"Edge dimension mismatch for {label}: "
                f"{node_u} has {dim_u}, {node_v} has {dim_v}"
            )

        for e in self.edges:
            if e.label == label:
                raise ValueError(f"Edge label {label} already connected")

        self.edges.append(TensorEdge(node_u=node_u, node_v=node_v, label=label))

    def disconnect(self, label: AxisLabel) -> None:
        self.edges = [e for e in self.edges if e.label != label]

    def neighbors(self, node_name: str) -> List[str]:
        out = []
        for e in self.edges:
            if e.node_u == node_name:
                out.append(e.node_v)
            elif e.node_v == node_name:
                out.append(e.node_u)
        return out

    def edge_labels_of(self, node_name: str) -> Tuple[AxisLabel, ...]:
        out = []
        for e in self.edges:
            if e.node_u == node_name or e.node_v == node_name:
                out.append(e.label)
        return tuple(out)

    def free_labels(self) -> Tuple[AxisLabel, ...]:
        """
        Global free labels = labels that appear on nodes but are not edge labels.
        """
        contracted = {e.label for e in self.edges}
        out = []
        for node in self.nodes.values():
            for lab in node.axis_labels:
                if lab not in contracted and lab.startswith("free:"):
                    out.append(lab)
        return tuple(sorted(set(out)))

    def contracted_labels(self) -> Tuple[AxisLabel, ...]:
        return tuple(sorted(e.label for e in self.edges))

    def storage_cost(self) -> int:
        return int(sum(node.size for node in self.nodes.values()))

    def validate(self) -> None:
        """
        Check core structural invariants.
        """
        # every edge label appears in exactly two nodes and dimensions match
        for e in self.edges:
            if e.node_u not in self.nodes:
                raise ValueError(f"Missing endpoint node {e.node_u}")
            if e.node_v not in self.nodes:
                raise ValueError(f"Missing endpoint node {e.node_v}")

            u = self.nodes[e.node_u]
            v = self.nodes[e.node_v]

            if e.label not in u.axis_labels:
                raise ValueError(f"{e.label} missing from {e.node_u}")
            if e.label not in v.axis_labels:
                raise ValueError(f"{e.label} missing from {e.node_v}")

            dim_u = u.shape[u.axis_of(e.label)]
            dim_v = v.shape[v.axis_of(e.label)]

            if dim_u != dim_v:
                raise ValueError(
                    f"Dimension mismatch on edge {e.label}: "
                    f"{e.node_u} has {dim_u}, {e.node_v} has {dim_v}"
                )

        # each contracted label should appear exactly twice globally
        counts: Dict[AxisLabel, int] = {}
        for node in self.nodes.values():
            for lab in node.axis_labels:
                if lab.startswith("rank:"):
                    counts[lab] = counts.get(lab, 0) + 1

        connected = {e.label for e in self.edges}
        for lab in connected:
            if counts.get(lab, 0) != 2:
                raise ValueError(
                    f"Contracted label {lab} must appear exactly twice globally"
                )

    @classmethod
    def from_dense_tensor(cls, data: Array, root_name: str = "X") -> "TensorNetwork":
        """
        Build the trivial one-node tensor network from a dense tensor.
        """
        node = TensorNode(
            name=root_name,
            data=data,
            axis_labels=[f"free:I{i}" for i in range(data.ndim)],
        )
        net = cls()
        net.add_node(node)
        return net

    def summary(self) -> str:
        lines = [
            f"TensorNetwork(num_nodes={len(self.nodes)}, "
            f"num_edges={len(self.edges)}, "
            f"storage_cost={self.storage_cost()})"
        ]

        for name in sorted(self.nodes):
            node = self.nodes[name]
            lines.append(
                f"  {name}: shape={node.shape}, "
                f"free={node.free_labels}, "
                f"rank={node.rank_labels}"
            )

        if self.edges:
            lines.append("  edges:")
            for e in self.edges:
                lines.append(f"    {e.node_u} --[{e.label}]-- {e.node_v}")

        return "\n".join(lines)
    

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # trivial one-node network
    X = rng.standard_normal((4, 5, 6))
    net = TensorNetwork.from_dense_tensor(X, root_name="X")
    print(net.summary())
    print()

    # small two-node example
    r = 3
    A = TensorNode(
        name="A",
        data=rng.standard_normal((4, r)),
        axis_labels=["free:I0", "rank:r0"],
    )
    B = TensorNode(
        name="B",
        data=rng.standard_normal((r, 5, 6)),
        axis_labels=["rank:r0", "free:I1", "free:I2"],
    )

    net2 = TensorNetwork()
    net2.add_node(A)
    net2.add_node(B)
    net2.connect("A", "B", "rank:r0")
    net2.validate()

    print(net2.summary())