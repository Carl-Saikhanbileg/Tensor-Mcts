from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time
import numpy as np
from core_algo.tensor_core import TensorData
from core_algo.sketch import Sketch, enumerate_sketches
from core_algo.preprocessing import preprocess_singular_values, PrecomputedSVDCache
from core_algo.scoring import RankAssignmentResult, assign_ranks_via_constraints
from core_algo.execution import ExecutedNetwork, execute_scored_sketch
from core_algo.rounding import round_tensor_network

@dataclass
class ScoredSketch:
    sketch: Sketch
    rank_assignment: RankAssignmentResult
    estimated_cost: int
    @property
    def total_error_sq(self) -> float: return float(self.rank_assignment.total_error_sq)

@dataclass
class ExecutedCandidate:
    scored_sketch: ScoredSketch
    network_before_round: ExecutedNetwork
    network_after_round: ExecutedNetwork
    executed_cost: int
    rounded_cost: int
    execution_time_sec: float
    failed: bool = False
    fail_reason: Optional[str] = None

@dataclass
class SearchDiagnostics:
    num_total_sketches: int = 0
    num_scored_sketches: int = 0
    num_executed_sketches: int = 0
    num_failed_executions: int = 0
    preprocessing_time_sec: float = 0.0
    scoring_time_sec: float = 0.0
    execution_time_sec: float = 0.0

@dataclass
class SearchResult:
    best_network: ExecutedNetwork
    best_cost: int
    best_scored_sketch: ScoredSketch
    top_scored_sketches: List[ScoredSketch] = field(default_factory=list)
    executed_candidates: List[ExecutedCandidate] = field(default_factory=list)
    diagnostics: SearchDiagnostics = field(default_factory=SearchDiagnostics)

def top_k_scored_sketches(X: TensorData, cache: PrecomputedSVDCache, eps: float, max_splits: int, k: int) -> Tuple[List[ScoredSketch], int]:
    scored, num_total = [], 0
    for sketch in enumerate_sketches(X.order, max_splits):
        num_total += 1
        try:
            ra = assign_ranks_via_constraints(cache, sketch, eps, cache.tensor_fro_norm_sq)
            scored.append(ScoredSketch(sketch, ra, int(ra.estimated_cost)))
        except: continue
    scored.sort(key=lambda s: (s.estimated_cost, len(s.sketch.ops)))
    return scored[:k], num_total

def execute_and_round_candidate(X: TensorData, scored_sketch: ScoredSketch, eps: float) -> ExecutedCandidate:
    t0 = time.perf_counter()
    try:
        nb = execute_scored_sketch(X, scored_sketch.sketch, scored_sketch.rank_assignment, eps)
        na = round_tensor_network(nb, eps)
        return ExecutedCandidate(scored_sketch, nb, na, int(nb.storage_cost()), int(na.storage_cost()), time.perf_counter()-t0)
    except Exception as e:
        dn = ExecutedNetwork.from_tensor(X)
        return ExecutedCandidate(scored_sketch, dn, dn, int(dn.storage_cost()), int(dn.storage_cost()), time.perf_counter()-t0, True, str(e))

def structure_search(X: TensorData, eps: float, max_splits: int = 3, top_k: int = 5) -> SearchResult:
    diag = SearchDiagnostics()
    best_net = ExecutedNetwork.from_tensor(X)
    best_cost = int(best_net.storage_cost())
    
    # Phase I: Preprocess
    t0 = time.perf_counter()
    cache = preprocess_singular_values(X)
    diag.preprocessing_time_sec = time.perf_counter() - t0
    
    # Phase II+III: Enumerate & Score
    t0 = time.perf_counter()
    top_scored, total = top_k_scored_sketches(X, cache, eps, max_splits, top_k)
    diag.scoring_time_sec, diag.num_total_sketches, diag.num_scored_sketches = time.perf_counter()-t0, total, len(top_scored)
    
    best_scored = ScoredSketch(Sketch(()), RankAssignmentResult({}, 0.0, best_cost), best_cost)
    if top_scored: best_scored = top_scored[0]
    
    # Phase IV: Execute & Round Top-K
    exec_cands, t0 = [], time.perf_counter()
    for s in top_scored:
        c = execute_and_round_candidate(X, s, eps)
        exec_cands.append(c)
        if not c.failed and c.rounded_cost < best_cost:
            best_cost, best_net, best_scored = c.rounded_cost, c.network_after_round, s
        if c.failed: diag.num_failed_executions += 1
            
    diag.execution_time_sec, diag.num_executed_sketches = time.perf_counter()-t0, len(exec_cands)
    return SearchResult(best_net, best_cost, best_scored, top_scored, exec_cands, diag)

def print_search_result(res: SearchResult):
    print(f"SEARCH COMPLETE\nBest Cost: {res.best_cost}\nBest Subsets: {res.best_scored_sketch.sketch.subsets()}")
    print(f"Diagnostics: Pre={res.diagnostics.preprocessing_time_sec:.3f}s, Score={res.diagnostics.scoring_time_sec:.3f}s, Exec={res.diagnostics.execution_time_sec:.3f}s")

# Backward compatibility for tests.py
def structure_search_prototype(
    X,
    eps: float,
    max_splits: int = 3,
    top_k: int = 5,
):
    return structure_search(
        X=X,
        eps=eps,
        max_splits=max_splits,
        top_k=top_k,
    )

if __name__ == "__main__":
    data = np.einsum("i,j,k,l,m->ijklm", *[np.random.randn(10) for _ in range(5)])
    res = structure_search(TensorData(data), eps=0.1, max_splits=4)
    print_search_result(res)