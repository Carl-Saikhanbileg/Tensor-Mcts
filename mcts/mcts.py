from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from mcts.mcts_env import (
    SearchState,
    Subset,
    action_mask,
    apply_action,
    compute_reward,
    encode_state,
    is_terminal,
    valid_actions,
)
from mcts.network_model import PolicyValueNet


@dataclass
class MCTSNode:
    state: SearchState
    parent: Optional[MCTSNode]
    action_from_parent: Optional[Subset]
    prior: float
    children: Dict[Subset, MCTSNode] = field(default_factory=dict)
    N: int = 0
    W: float = 0.0

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


class MCTS:
    def __init__(self, model: PolicyValueNet, c_puct: float = 1.0, n_sims: int = 50):
        self.model = model
        self.c_puct = c_puct
        self.n_sims = n_sims

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children and not is_terminal(node.state):
            sqrt_n = math.sqrt(sum(c.N for c in node.children.values()) + 1)
            best, best_score = None, -float("inf")
            for a in valid_actions(node.state):
                if a not in node.children:
                    continue
                child = node.children[a]
                score = child.Q + self.c_puct * child.prior * sqrt_n / (1 + child.N)
                if score > best_score:
                    best_score, best = score, child
            if best is None:
                break
            node = best
        return node

    def _expand(self, node: MCTSNode) -> float:
        if is_terminal(node.state):
            return compute_reward(node.state)

        enc = torch.tensor(encode_state(node.state), dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(action_mask(node.state), dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            p, v = self.model(enc, mask)

        priors = p.squeeze(0).numpy()
        value = float(v.squeeze(0))

        for a in valid_actions(node.state):
            idx = node.state.all_subsets.index(a)
            node.children[a] = MCTSNode(
                state=apply_action(node.state, a),
                parent=node,
                action_from_parent=a,
                prior=float(priors[idx]),
            )

        return value

    def _backup(self, node: MCTSNode, value: float) -> None:
        cur: Optional[MCTSNode] = node
        while cur is not None:
            cur.N += 1
            cur.W += value
            cur = cur.parent

    def run(self, root_state: SearchState) -> np.ndarray:
        root = MCTSNode(root_state, None, None, 1.0)

        for _ in range(self.n_sims):
            leaf = self._select(root)
            value = self._expand(leaf)
            self._backup(leaf, value)

        counts = np.zeros(len(root_state.all_subsets), dtype=np.float32)
        for a, child in root.children.items():
            idx = root_state.all_subsets.index(a)
            counts[idx] = child.N

        total = counts.sum()
        return counts / total if total > 0 else counts
