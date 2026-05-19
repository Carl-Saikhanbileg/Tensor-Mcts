from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Tuple

import numpy as np

from core_algo.tensor_core import TensorData
from core_algo.preprocessing import preprocess_singular_values
from core_algo.sketch import make_sketch, enumerate_sketches
from core_algo.scoring import assign_ranks_via_constraints
from core_algo.execution import execute_scored_sketch
from core_algo.rounding import round_tensor_network


Subset = Tuple[int, ...]


@dataclass(frozen=True)
class Report:
    subsets: Tuple[Subset, ...]
    ranks: Dict[Subset, int]
    est: int
    exec: int
    rounded: int


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def cp_tensor(shape=(4, 5, 6, 7), rank=2, seed=0, noise=0.0) -> TensorData:
    rng = np.random.default_rng(seed)
    X = np.zeros(shape)
    for _ in range(rank):
        X += np.einsum("a,b,c,d->abcd", *[rng.standard_normal(n) for n in shape])
    if noise:
        X += noise * rng.standard_normal(shape)
    return TensorData(X)


def random_tensor(shape=(4, 5, 6, 7), seed=0) -> TensorData:
    return TensorData(np.random.default_rng(seed).standard_normal(shape))


def ranks(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def corr(x: List[float], y: List[float]) -> float:
    xm, ym = sum(x) / len(x), sum(y) / len(y)
    num = sum((a - xm) * (b - ym) for a, b in zip(x, y))
    dx = sqrt(sum((a - xm) ** 2 for a in x))
    dy = sqrt(sum((b - ym) ** 2 for b in y))
    return num / (dx * dy) if dx and dy else float("nan")


def spearman(est: List[int], actual: List[int]) -> float:
    return corr(ranks(est), ranks(actual))


def reports_for(X: TensorData, eps=0.20, max_splits=2) -> List[Report]:
    cache = preprocess_singular_values(X)
    out = []

    for sketch in enumerate_sketches(X.order, max_splits):
        try:
            ra = assign_ranks_via_constraints(cache, sketch, eps, cache.tensor_fro_norm_sq)
            before = execute_scored_sketch(X, sketch, ra, eps)
            after = round_tensor_network(before, eps)
            out.append(Report(
                subsets=sketch.subsets(),
                ranks=dict(ra.ranks),
                est=int(ra.estimated_cost),
                exec=int(before.storage_cost()),
                rounded=int(after.storage_cost()),
            ))
        except Exception:
            pass

    return out


def by_est(reports: List[Report]) -> List[Report]:
    return sorted(reports, key=lambda r: (r.est, len(r.subsets), r.subsets))


def by_actual(reports: List[Report]) -> List[Report]:
    return sorted(reports, key=lambda r: (r.rounded, r.exec, len(r.subsets), r.subsets))


def position(items: List[Report], subsets: Tuple[Subset, ...]) -> int:
    return next(i + 1 for i, r in enumerate(items) if r.subsets == subsets)


def test_order_invariance() -> None:
    X = cp_tensor(rank=2, seed=0)
    cache = preprocess_singular_values(X)

    s1 = make_sketch([(0,), (0, 1)])
    s2 = make_sketch([(0, 1), (0,)])

    assert_true(s1.subsets() == s2.subsets(), "sketch order changed canonical subsets")

    ra1 = assign_ranks_via_constraints(cache, s1, 0.20, cache.tensor_fro_norm_sq)
    ra2 = assign_ranks_via_constraints(cache, s2, 0.20, cache.tensor_fro_norm_sq)

    assert_true(ra1.ranks == ra2.ranks, "rank assignment depends on input order")
    assert_true(ra1.estimated_cost == ra2.estimated_cost, "estimated cost depends on input order")

    n1 = execute_scored_sketch(X, s1, ra1, 0.20)
    n2 = execute_scored_sketch(X, s2, ra2, 0.20)

    assert_true(n1.storage_cost() == n2.storage_cost(), "execution depends on input order")


def test_10_tensors() -> None:
    cases = []

    for i in range(5):
        rank = i % 3 + 1
        cases.append((f"cp_rank{rank}", cp_tensor(rank=rank, seed=100 + i, noise=0.02 * i)))

    for i in range(5):
        cases.append(("random", random_tensor(seed=200 + i)))

    print("case | type      | n  | actual_rank(est_top) | est_rank(actual_top) | spearman")
    print("-" * 80)

    hits = 0

    for i, (name, X) in enumerate(cases):
        rs = reports_for(X)
        assert_true(len(rs) > 0, f"case {i}: no feasible candidates")

        est_order = by_est(rs)
        actual_order = by_actual(rs)

        actual_rank_est_top = position(actual_order, est_order[0].subsets)
        est_rank_actual_top = position(est_order, actual_order[0].subsets)
        rho = spearman([r.est for r in rs], [r.rounded for r in rs])

        hits += int(actual_rank_est_top == 1)

        print(f"{i:>4} | {name:<9} | {len(rs):>2} | {actual_rank_est_top:>20} | {est_rank_actual_top:>20} | {rho:>8.3f}")
        print(f"     est #1:    est={est_order[0].est}, exec={est_order[0].exec}, round={est_order[0].rounded}, subsets={est_order[0].subsets}, ranks={est_order[0].ranks}")
        print(f"     actual #1: est={actual_order[0].est}, exec={actual_order[0].exec}, round={actual_order[0].rounded}, subsets={actual_order[0].subsets}, ranks={actual_order[0].ranks}")

    assert_true(hits > 0, "estimated top was never actual top")


if __name__ == "__main__":
    test_order_invariance()
    test_10_tensors()
    print("PASS")
