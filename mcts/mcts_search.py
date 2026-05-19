from __future__ import annotations

from typing import List
import time
import numpy as np

from core_algo.tensor_core import TensorData
from core_algo.sketch import make_sketch, Sketch
from core_algo.preprocessing import preprocess_singular_values
from core_algo.scoring import assign_ranks_via_constraints, RankAssignmentResult
from core_algo.execution import ExecutedNetwork
from core_algo.search import (
    ScoredSketch,
    ExecutedCandidate,
    SearchDiagnostics,
    SearchResult,
    execute_and_round_candidate,
)

from mcts.mcts_env import initial_state, is_terminal, apply_action, valid_actions
from mcts.mcts import MCTS
from mcts.network_model import PolicyValueNet


def sample_sketch(
    X: TensorData,
    eps: float,
    max_splits: int,
    mcts: MCTS,
    tau: float = 1.0,
) -> Sketch:
    state = initial_state(X, eps, max_splits)

    while not is_terminal(state):
        pi = mcts.run(state)
        actions = valid_actions(state)

        if not actions:
            break

        idxs = np.array([state.all_subsets.index(a) for a in actions])
        probs = pi[idxs]

        if probs.sum() <= 0:
            probs = np.ones(len(actions), dtype=np.float32) / len(actions)
        else:
            probs = probs / probs.sum()

        if tau <= 0:
            action = actions[int(np.argmax(probs))]
        else:
            probs = probs ** (1.0 / tau)
            probs = probs / probs.sum()
            action = actions[int(np.random.choice(len(actions), p=probs))]

        state = apply_action(state, action)

    return make_sketch(state.selected)


def mcts_top_k_scored_sketches(
    X: TensorData,
    eps: float,
    max_splits: int,
    k: int,
    model: PolicyValueNet,
    n_samples: int = 50,
    n_sims: int = 20,
    c_puct: float = 1.0,
    tau: float = 1.0,
) -> tuple[List[ScoredSketch], int, float]:
    t0 = time.perf_counter()
    cache = preprocess_singular_values(X)
    preprocessing_time = time.perf_counter() - t0

    mcts = MCTS(model, c_puct=c_puct, n_sims=n_sims)

    seen = set()
    scored: List[ScoredSketch] = []

    for _ in range(n_samples):
        sketch = sample_sketch(X, eps, max_splits, mcts, tau)
        key = sketch.subsets()

        if not key or key in seen:
            continue

        seen.add(key)

        try:
            ra = assign_ranks_via_constraints(
                cache=cache,
                sketch=sketch,
                eps=eps,
                tensor_fro_norm_sq=cache.tensor_fro_norm_sq,
            )
            scored.append(ScoredSketch(sketch, ra, int(ra.estimated_cost)))
        except Exception:
            pass

    scored.sort(key=lambda s: (s.estimated_cost, len(s.sketch.ops)))
    return scored[:k], len(seen), preprocessing_time


def structure_search_mcts(
    X: TensorData,
    eps: float,
    model: PolicyValueNet,
    max_splits: int = 3,
    top_k: int = 5,
    n_samples: int = 50,
    n_sims: int = 20,
    c_puct: float = 1.0,
    tau: float = 1.0,
) -> SearchResult:
    diag = SearchDiagnostics()
    best_net = ExecutedNetwork.from_tensor(X)
    best_cost = int(best_net.storage_cost())
    best_scored = ScoredSketch(
        Sketch(()),
        RankAssignmentResult({}, 0.0, best_cost),
        best_cost,
    )

    t0 = time.perf_counter()
    top_scored, total, pre_time = mcts_top_k_scored_sketches(
        X=X,
        eps=eps,
        max_splits=max_splits,
        k=top_k,
        model=model,
        n_samples=n_samples,
        n_sims=n_sims,
        c_puct=c_puct,
        tau=tau,
    )
    diag.scoring_time_sec = time.perf_counter() - t0
    diag.preprocessing_time_sec = pre_time
    diag.num_total_sketches = total
    diag.num_scored_sketches = len(top_scored)

    exec_cands: List[ExecutedCandidate] = []
    t0 = time.perf_counter()

    for s in top_scored:
        c = execute_and_round_candidate(X, s, eps)
        exec_cands.append(c)

        if c.failed:
            diag.num_failed_executions += 1
            continue

        if c.rounded_cost < best_cost:
            best_cost = c.rounded_cost
            best_net = c.network_after_round
            best_scored = s

    diag.execution_time_sec = time.perf_counter() - t0
    diag.num_executed_sketches = len(exec_cands)

    return SearchResult(
        best_network=best_net,
        best_cost=best_cost,
        best_scored_sketch=best_scored,
        top_scored_sketches=top_scored,
        executed_candidates=exec_cands,
        diagnostics=diag,
    )