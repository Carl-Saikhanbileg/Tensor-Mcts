from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from core_algo.sketch import Sketch, normalize_subset

Subset = Tuple[int, ...]


@dataclass(frozen=True)
class SplitScoreData:
    subset: Subset
    row_dim: int
    col_dim: int
    singular_values: np.ndarray

    @property
    def max_rank(self) -> int:
        return min(self.row_dim, self.col_dim)


@dataclass(frozen=True)
class RankAssignmentResult:
    ranks: Dict[Subset, int]
    total_error_sq: float
    estimated_cost: int


def _extract_split_table(cache) -> Mapping[Subset, SplitScoreData]:
    if hasattr(cache, "split_scores"):
        return cache.split_scores
    if hasattr(cache, "by_subset"):
        return cache.by_subset
    if isinstance(cache, dict):
        return cache
    raise TypeError(f"Unsupported cache type: {type(cache)}")


def _get_split_data(cache, subset: Iterable[int]) -> SplitScoreData:
    key = normalize_subset(subset)
    table = _extract_split_table(cache)
    if key not in table:
        raise KeyError(f"No precomputed SVD data for subset {key}")

    val = table[key]
    if isinstance(val, SplitScoreData):
        return val
    if isinstance(val, dict):
        return SplitScoreData(
            subset=key,
            row_dim=int(val["row_dim"]),
            col_dim=int(val["col_dim"]),
            singular_values=np.asarray(val["singular_values"], dtype=float),
        )
    return SplitScoreData(
        subset=key,
        row_dim=int(val.row_dim),
        col_dim=int(val.col_dim),
        singular_values=np.asarray(val.singular_values, dtype=float),
    )


def _tensor_order(cache) -> int:
    if hasattr(cache, "tensor_order"):
        return int(cache.tensor_order)
    if hasattr(cache, "tensor_shape"):
        return len(cache.tensor_shape)
    if hasattr(cache, "split_scores"):
        table = cache.split_scores
    elif hasattr(cache, "by_subset"):
        table = cache.by_subset
    elif isinstance(cache, dict):
        table = cache
    else:
        raise ValueError("Cache must expose tensor_order, tensor_shape, or split table.")

    if not table:
        raise ValueError("Cannot infer tensor order from empty cache.")

    max_mode = -1
    for subset in table.keys():
        if subset:
            max_mode = max(max_mode, max(subset))
    if max_mode < 0:
        raise ValueError("Cannot infer tensor order from cache keys.")
    return max_mode + 1


def _fro_norm_sq(cache, subs: List[Subset]) -> float:
    if hasattr(cache, "tensor_fro_norm_sq"):
        return float(cache.tensor_fro_norm_sq)
    svs = _get_split_data(cache, subs[0]).singular_values
    return float(np.sum(svs ** 2))


def tail_error_sq_for_rank(svs: np.ndarray, rank: int) -> float:
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    if rank >= len(svs):
        return 0.0
    return float(np.sum(svs[rank:] ** 2))


def min_rank_for_error_sq(svs: np.ndarray, max_err_sq: float) -> int:
    if max_err_sq < 0:
        raise ValueError("max_err_sq must be nonnegative")
    suffix = np.cumsum(svs[::-1] ** 2)[::-1]
    for r in range(len(svs)):
        if float(suffix[r]) <= max_err_sq:
            return r
    return len(svs)


def _immediate_parent(s: Subset, fam: Iterable[Subset]) -> Optional[Subset]:
    S = set(s)
    supersets = [normalize_subset(t) for t in fam if set(t) > S]
    return min(supersets, key=lambda t: (len(t), t)) if supersets else None


def _immediate_children(s: Subset, fam: Iterable[Subset]) -> List[Subset]:
    fam_n = [normalize_subset(t) for t in fam]
    s_n = normalize_subset(s)
    return sorted(
        [
            t for t in fam_n
            if t != s_n and set(t) < set(s_n) and _immediate_parent(t, fam_n) == s_n
        ],
        key=lambda x: (len(x), x),
    )


def _subset_dim(cache, subset: Subset) -> int:
    """
    Product of tensor mode sizes on `subset`, inferred from cached matricizations.

    If `subset` itself is present in the cache, its row_dim is the wanted product.
    Otherwise, look up its complement and use col_dim.
    """
    subset = normalize_subset(subset)
    d = _tensor_order(cache)
    comp = tuple(i for i in range(d) if i not in set(subset))

    table = _extract_split_table(cache)

    if subset in table:
        return int(_get_split_data(cache, subset).row_dim)
    if comp in table:
        return int(_get_split_data(cache, comp).col_dim)

    raise KeyError(f"Could not infer dimension for subset {subset} or complement {comp}")


def estimate_sketch_cost(cache, sketch: Sketch, ranks: Mapping[Subset, int]) -> int:
    """
    Structural cost model for a laminar sketch.

    Important fix:
    - use the true tensor order from the cache, not max mode appearing in the sketch
    - compute subset dimensions using subset/complement lookup, since the cache stores
      canonical bipartitions only
    """
    fam = [normalize_subset(s) for s in sketch.subsets()]
    if not fam:
        raise ValueError("Empty sketch")

    d = _tensor_order(cache)
    universe = tuple(range(d))
    total = 0

    for S in fam:
        kids = _immediate_children(S, fam)
        covered_by_kids = set().union(*(set(c) for c in kids)) if kids else set()
        res = tuple(i for i in S if i not in covered_by_kids)

        f_dim = _subset_dim(cache, res) if res else 1
        p_r = int(ranks[S]) if _immediate_parent(S, fam) is not None else 1
        c_r = int(np.prod([ranks[c] for c in kids], dtype=np.int64)) if kids else 1

        total += f_dim * p_r * c_r

    tops = [S for S in fam if _immediate_parent(S, fam) is None]
    covered = set().union(*(set(S) for S in tops)) if tops else set()
    r_res = tuple(i for i in universe if i not in covered)

    r_f = _subset_dim(cache, r_res) if r_res else 1
    r_p = int(np.prod([ranks[c] for c in tops], dtype=np.int64)) if tops else 1
    total += r_f * r_p

    return int(total)


def candidate_ranks_for_subset(cache, subset: Subset, max_c: int = 16) -> List[int]:
    q = _get_split_data(cache, subset).max_rank
    if q + 1 <= max_c:
        return list(range(q + 1))

    core = {0, 1, q} | set(range(2, min(6, q + 1)))
    grid = np.unique(
        np.round(np.exp(np.linspace(0, np.log(max(q, 1)), max_c))).astype(int)
    )
    return sorted({int(r) for r in grid if 0 <= r <= q} | core)


def _branch_and_bound(
    subs: List[Subset],
    cands_desc: Dict[Subset, List[int]],
    tail_errs: Dict[Subset, Dict[int, float]],
    budget: float,
    cache,
    sketch: Sketch,
) -> Optional[RankAssignmentResult]:
    n = len(subs)
    if n == 0:
        return RankAssignmentResult({}, 0.0, 0)

    min_per = [min(tail_errs[s].values()) for s in subs]
    suffix_min = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_min[i] = suffix_min[i + 1] + min_per[i]

    if suffix_min[0] > budget:
        return None

    best_cost = [int(2**62)]
    best_ranks = [None]
    best_err = [float("inf")]
    partial = [0] * n

    def _dfs(idx: int, accum: float) -> None:
        if idx == n:
            rd = {subs[i]: partial[i] for i in range(n)}
            cost = estimate_sketch_cost(cache, sketch, rd)
            if cost < best_cost[0]:
                best_cost[0] = cost
                best_ranks[0] = dict(rd)
                best_err[0] = accum
            return

        s = subs[idx]
        for r in cands_desc[s]:
            new_accum = accum + tail_errs[s][r]

            if new_accum + suffix_min[idx + 1] > budget:
                break

            partial[idx] = r
            _dfs(idx + 1, new_accum)

    _dfs(0, 0.0)

    if best_ranks[0] is None:
        return None
    return RankAssignmentResult(
        ranks=best_ranks[0],
        total_error_sq=float(best_err[0]),
        estimated_cost=int(best_cost[0]),
    )


def _milp_fallback(
    subs: List[Subset],
    cands_asc: Dict[Subset, List[int]],
    tail_errs: Dict[Subset, Dict[int, float]],
    budget: float,
    cache,
    sketch: Sketch,
) -> Optional[RankAssignmentResult]:
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        return None

    n = len(subs)
    cand_lists = [cands_asc[s] for s in subs]

    offsets = [0]
    for cl in cand_lists:
        offsets.append(offsets[-1] + len(cl))
    n_vars = offsets[-1]
    if n_vars == 0:
        return None

    baseline_ranks = {s: 1 for s in subs}
    baseline_cost = estimate_sketch_cost(cache, sketch, baseline_ranks)

    c_obj = np.zeros(n_vars)
    for i, s in enumerate(subs):
        for j, r in enumerate(cand_lists[i]):
            probe = dict(baseline_ranks)
            probe[s] = r
            c_obj[offsets[i] + j] = (
                estimate_sketch_cost(cache, sketch, probe) - baseline_cost
            )

    A_rows: List[np.ndarray] = []
    lo: List[float] = []
    hi: List[float] = []

    for i in range(n):
        row = np.zeros(n_vars)
        row[offsets[i]:offsets[i + 1]] = 1.0
        A_rows.append(row)
        lo.append(1.0)
        hi.append(1.0)

    err_row = np.zeros(n_vars)
    for i, s in enumerate(subs):
        for j, r in enumerate(cand_lists[i]):
            err_row[offsets[i] + j] = tail_errs[s][r]
    A_rows.append(err_row)
    lo.append(-np.inf)
    hi.append(float(budget))

    A = np.vstack(A_rows)

    res = milp(
        c_obj,
        constraints=LinearConstraint(A, lo, hi),
        integrality=np.ones(n_vars),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
    )

    if not res.success:
        return None

    ranks_out: Dict[Subset, int] = {}
    for i, s in enumerate(subs):
        seg = res.x[offsets[i]:offsets[i + 1]]
        ranks_out[s] = cand_lists[i][int(np.argmax(seg))]

    total_err = sum(tail_errs[s][ranks_out[s]] for s in subs)
    if total_err > budget * (1.0 + 1e-8):
        return None

    return RankAssignmentResult(
        ranks=ranks_out,
        total_error_sq=float(total_err),
        estimated_cost=int(estimate_sketch_cost(cache, sketch, ranks_out)),
    )


def _greedy_repair(
    subs: List[Subset],
    cands_asc: Dict[Subset, List[int]],
    tail_errs: Dict[Subset, Dict[int, float]],
    budget: float,
    cache,
    sketch: Sketch,
) -> RankAssignmentResult:
    curr = {s: cands_asc[s][0] for s in subs}
    idx_of = {s: {r: i for i, r in enumerate(cands_asc[s])} for s in subs}
    total_err = sum(tail_errs[s][curr[s]] for s in subs)

    while total_err > budget:
        best_move: Optional[Tuple[Subset, int, float]] = None
        best_score = -inf
        cur_cost = estimate_sketch_cost(cache, sketch, curr)

        for s in subs:
            i = idx_of[s][curr[s]]
            if i + 1 >= len(cands_asc[s]):
                continue
            r_next = cands_asc[s][i + 1]
            d_err = tail_errs[s][curr[s]] - tail_errs[s][r_next]
            if d_err <= 0:
                continue
            probe = {**curr, s: r_next}
            d_cost = estimate_sketch_cost(cache, sketch, probe) - cur_cost
            score = d_err / max(d_cost, 1)
            if score > best_score:
                best_score, best_move = score, (s, r_next, d_err)

        if best_move is None:
            raise ValueError(
                "Greedy repair failed: no feasible rank assignment exists for this sketch."
            )
        curr[best_move[0]], total_err = best_move[1], total_err - best_move[2]

    return RankAssignmentResult(
        ranks=curr,
        total_error_sq=float(total_err),
        estimated_cost=int(estimate_sketch_cost(cache, sketch, curr)),
    )


_BB_NODE_LIMIT = 30_000


def assign_ranks_via_constraints(
    cache,
    sketch: Sketch,
    eps: float,
    tensor_fro_norm_sq: Optional[float] = None,
    *,
    max_rank_candidates: int = 16,
    **kwargs,
) -> RankAssignmentResult:
    subs = [normalize_subset(s) for s in sketch.subsets()]
    if not subs:
        return RankAssignmentResult({}, 0.0, 0)

    if tensor_fro_norm_sq is None:
        tensor_fro_norm_sq = _fro_norm_sq(cache, subs)
    budget = (eps ** 2) * float(tensor_fro_norm_sq)

    cands_asc: Dict[Subset, List[int]] = {
        s: sorted(candidate_ranks_for_subset(cache, s, max_rank_candidates))
        for s in subs
    }
    cands_desc: Dict[Subset, List[int]] = {
        s: list(reversed(v)) for s, v in cands_asc.items()
    }

    tail_errs: Dict[Subset, Dict[int, float]] = {
        s: {
            r: tail_error_sq_for_rank(_get_split_data(cache, s).singular_values, r)
            for r in cands_asc[s]
        }
        for s in subs
    }

    for s in subs:
        min_e = min(tail_errs[s].values())
        if min_e > budget:
            raise ValueError(
                f"Infeasible sketch: split {s} cannot satisfy the error bound "
                f"(min tail error {min_e:.4g} > budget {budget:.4g})."
            )

    n_nodes = 1
    for s in subs:
        n_nodes *= sum(1 for r in cands_asc[s] if tail_errs[s][r] <= budget)

    if n_nodes <= _BB_NODE_LIMIT:
        result = _branch_and_bound(subs, cands_desc, tail_errs, budget, cache, sketch)
        if result is not None:
            return result

    result = _milp_fallback(subs, cands_asc, tail_errs, budget, cache, sketch)
    if result is not None:
        return result

    return _greedy_repair(subs, cands_asc, tail_errs, budget, cache, sketch)


def assign_ranks_greedily(cache, sketch: Sketch, eps: float) -> RankAssignmentResult:
    return assign_ranks_via_constraints(cache, sketch, eps)