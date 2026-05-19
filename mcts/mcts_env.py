from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from core_algo.tensor_core import TensorData
from core_algo.sketch import all_osplit_subsets, is_laminar_family, make_sketch

Subset = Tuple[int, ...]


@dataclass
class SearchState:
    X: TensorData
    selected: List[Subset]
    all_subsets: List[Subset]
    max_splits: int
    eps: float


def initial_state(X: TensorData, eps: float, max_splits: int) -> SearchState:
    return SearchState(
        X=X,
        selected=[],
        all_subsets=all_osplit_subsets(X.order),
        max_splits=max_splits,
        eps=eps,
    )


def valid_actions(state: SearchState) -> List[Subset]:
    return [
        s for s in state.all_subsets
        if s not in state.selected and is_laminar_family(state.selected + [s])
    ]


def is_terminal(state: SearchState) -> bool:
    return len(state.selected) >= state.max_splits or len(valid_actions(state)) == 0


def apply_action(state: SearchState, action: Subset) -> SearchState:
    return SearchState(
        X=state.X,
        selected=state.selected + [action],
        all_subsets=state.all_subsets,
        max_splits=state.max_splits,
        eps=state.eps,
    )


def compute_reward(state: SearchState) -> float:
    if not state.selected:
        return 0.0
    return len(state.selected) / max(state.max_splits, 1)


def encode_state(state: SearchState) -> np.ndarray:
    n = len(state.all_subsets)
    out = np.zeros(3 * n, dtype=np.float32)
    selected = set(state.selected)

    for i, s in enumerate(state.all_subsets):
        b = 3 * i
        out[b] = float(s in selected)
        out[b + 1] = len(s) / max(state.X.order, 1)
        out[b + 2] = len(state.selected) / max(state.max_splits, 1)

    return out


def action_mask(state: SearchState) -> np.ndarray:
    valid = set(valid_actions(state))
    return np.array([float(s in valid) for s in state.all_subsets], dtype=np.float32)


def state_dim(tensor_order: int) -> int:
    return 3 * len(all_osplit_subsets(tensor_order))


def n_actions(tensor_order: int) -> int:
    return len(all_osplit_subsets(tensor_order))