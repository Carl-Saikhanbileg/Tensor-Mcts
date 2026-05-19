from __future__ import annotations

import traceback
import numpy as np

from core_algo.tensor_core import TensorData
from core_algo.preprocessing import preprocess_singular_values
from core_algo.scoring import assign_ranks_via_constraints
from core_algo.sketch import enumerate_sketches, Sketch
from core_algo.execution import ExecutedNetwork, execute_scored_sketch
from core_algo.rounding import round_tensor_network
from core_algo.search import structure_search_prototype


# ============================================================
# Small test utilities
# ============================================================

def banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run_test(name: str, fn) -> None:
    print(f"\n[TEST] {name}")
    try:
        fn()
        print(f"[PASS] {name}")
    except Exception as e:
        print(f"[FAIL] {name}")
        print(f"Reason: {e}")
        traceback.print_exc()
        raise


# ============================================================
# Tensor factories
# ============================================================

def make_rank1_tensor(shape=(10, 10, 10, 10, 10), seed=0) -> TensorData:
    """
    Outer-product tensor, should compress very well.
    """
    rng = np.random.default_rng(seed)
    vecs = [rng.standard_normal(n) for n in shape]
    expr = ",".join(chr(ord("a") + i) for i in range(len(shape)))
    out = "".join(chr(ord("a") + i) for i in range(len(shape)))
    data = np.einsum(f"{expr}->{out}", *vecs)
    return TensorData(data)


def make_low_rank_4d(shape=(8, 9, 10, 11), rank=2, seed=0) -> TensorData:
    """
    Sum of a few separable terms.
    """
    rng = np.random.default_rng(seed)
    data = np.zeros(shape, dtype=float)
    for _ in range(rank):
        a = rng.standard_normal(shape[0])
        b = rng.standard_normal(shape[1])
        c = rng.standard_normal(shape[2])
        d = rng.standard_normal(shape[3])
        data += np.einsum("i,j,k,l->ijkl", a, b, c, d)
    return TensorData(data)


def make_random_tensor(shape=(6, 7, 8, 9), seed=0) -> TensorData:
    rng = np.random.default_rng(seed)
    return TensorData(rng.standard_normal(shape))


# ============================================================
# Unit-ish tests for each stage
# ============================================================

def test_preprocessing_basic() -> None:
    X = make_random_tensor((4, 5, 6, 7), seed=1)
    cache = preprocess_singular_values(X)

    print(cache.summary())

    check(cache.tensor_shape == X.shape, "tensor shape mismatch")
    check(cache.tensor_order == X.order, "tensor order mismatch")
    check(len(cache.split_scores) > 0, "cache should not be empty")

    for subset, info in cache.split_scores.items():
        check(info.row_dim > 0, f"bad row_dim for subset {subset}")
        check(info.col_dim > 0, f"bad col_dim for subset {subset}")
        check(len(info.singular_values) == min(info.row_dim, info.col_dim),
              f"bad singular-value count for subset {subset}")


def test_sketch_enumeration_basic() -> None:
    sketches = list(enumerate_sketches(d=4, max_splits=3))
    print(f"num sketches: {len(sketches)}")

    check(len(sketches) > 0, "expected at least one sketch")

    for s in sketches[:10]:
        print("sketch:", s.subsets())
        check(s.is_valid(), f"invalid sketch found: {s.subsets()}")


def test_scoring_basic() -> None:
    X = make_random_tensor((4, 5, 6, 7), seed=2)
    cache = preprocess_singular_values(X)
    sketches = list(enumerate_sketches(d=X.order, max_splits=2))
    check(len(sketches) > 0, "no sketches generated")

    sketch = sketches[0]
    result = assign_ranks_via_constraints(
        cache=cache,
        sketch=sketch,
        eps=0.2,
        tensor_fro_norm_sq=cache.tensor_fro_norm_sq,
    )

    print("sketch:", sketch.subsets())
    print("assigned ranks:", result.ranks)
    print("total error sq:", result.total_error_sq)
    print("estimated cost:", result.estimated_cost)

    check(isinstance(result.ranks, dict), "ranks must be a dict")
    check(result.estimated_cost >= 0, "estimated cost must be nonnegative")


def test_execution_basic() -> None:
    X = make_low_rank_4d((6, 7, 8, 9), rank=2, seed=3)
    cache = preprocess_singular_values(X)
    sketches = list(enumerate_sketches(d=X.order, max_splits=2))

    # choose the first feasible sketch
    chosen_sketch = None
    chosen_assignment = None

    for sketch in sketches:
        try:
            rank_assignment = assign_ranks_via_constraints(
                cache=cache,
                sketch=sketch,
                eps=0.2,
                tensor_fro_norm_sq=cache.tensor_fro_norm_sq,
            )
            chosen_sketch = sketch
            chosen_assignment = rank_assignment
            break
        except Exception:
            continue

    check(chosen_sketch is not None, "could not find feasible sketch to execute")

    net = execute_scored_sketch(
        X=X,
        sketch=chosen_sketch,
        rank_assignment=chosen_assignment,
        eps=0.2,
    )

    print("executed sketch:", chosen_sketch.subsets())
    print(net.summary())

    check(net.storage_cost() > 0, "executed network must have positive storage cost")
    net.graph.validate()


def test_rounding_basic() -> None:
    X = make_low_rank_4d((6, 7, 8, 9), rank=2, seed=4)
    cache = preprocess_singular_values(X)
    sketches = list(enumerate_sketches(d=X.order, max_splits=2))

    chosen_sketch = None
    chosen_assignment = None
    for sketch in sketches:
        try:
            rank_assignment = assign_ranks_via_constraints(
                cache=cache,
                sketch=sketch,
                eps=0.25,
                tensor_fro_norm_sq=cache.tensor_fro_norm_sq,
            )
            chosen_sketch = sketch
            chosen_assignment = rank_assignment
            break
        except Exception:
            continue

    check(chosen_sketch is not None, "could not find feasible sketch for rounding test")

    net_before = execute_scored_sketch(
        X=X,
        sketch=chosen_sketch,
        rank_assignment=chosen_assignment,
        eps=0.25,
    )
    cost_before = net_before.storage_cost()

    net_after = round_tensor_network(
        network=net_before,
        eps=0.25,
    )
    cost_after = net_after.storage_cost()

    print("cost before rounding:", cost_before)
    print("cost after rounding :", cost_after)
    print(net_after.summary())

    check(cost_after <= cost_before, "rounding should not increase storage cost")
    net_after.graph.validate()


# ============================================================
# End-to-end search tests
# ============================================================

def test_search_on_rank1_tensor() -> None:
    X = make_rank1_tensor((10, 10, 10, 10, 10), seed=5)

    result = structure_search_prototype(
        X=X,
        eps=0.10,
        max_splits=4,
        top_k=5,
    )
    dense_size = X.size
    best_cost = result.best_cost
    cr = dense_size / best_cost if best_cost > 0 else float("inf")

    print("dense:", dense_size)
    print("best cost:", best_cost)
    print("compression ratio (CR):", cr)
    print("topology:", result.best_scored_sketch.sketch.subsets())
    print("ranks:", result.best_scored_sketch.rank_assignment.ranks)
    print(result.best_network.summary())
    dense_size = X.size
    compressed_size = result.best_cost
    cr = dense_size / compressed_size if compressed_size > 0 else float("inf")

    print("dense:", dense_size)
    print("best cost:", compressed_size)
    print("compression ratio:", cr)
    print("topology:", result.best_scored_sketch.sketch.subsets())
    print("ranks:", result.best_scored_sketch.rank_assignment.ranks)
    print(result.best_network.summary())

    check(compressed_size > 0, "compressed size must be positive")
    check(compressed_size <= dense_size, "should not beat dense by becoming larger")


def test_search_on_low_rank_tensor() -> None:
    X = make_low_rank_4d((8, 9, 10, 11), rank=2, seed=6)

    result = structure_search_prototype(
        X=X,
        eps=0.15,
        max_splits=3,
        top_k=5,
    )

    print("dense:", X.size)
    print("best cost:", result.best_cost)
    print("topology:", result.best_scored_sketch.sketch.subsets())
    print("ranks:", result.best_scored_sketch.rank_assignment.ranks)
    print(result.best_network.summary())

    check(result.best_cost > 0, "best cost must be positive")


def test_search_on_random_tensor() -> None:
    X = make_random_tensor((6, 7, 8, 9), seed=7)

    result = structure_search_prototype(
        X=X,
        eps=0.20,
        max_splits=2,
        top_k=3,
    )

    print("dense:", X.size)
    print("best cost:", result.best_cost)
    print("topology:", result.best_scored_sketch.sketch.subsets())
    print("ranks:", result.best_scored_sketch.rank_assignment.ranks)
    print(result.best_network.summary())

    check(result.best_cost > 0, "best cost must be positive")


# ============================================================
# Stress / regression-style tests
# ============================================================

def test_empty_sketch_dense_baseline() -> None:
    X = make_random_tensor((4, 4, 4), seed=8)
    net = ExecutedNetwork.from_tensor(X)

    print(net.summary())
    check(net.storage_cost() == X.size, "dense baseline cost mismatch")


def test_multiple_runs_stability() -> None:
    for seed in range(3):
        X = make_low_rank_4d((5, 6, 7, 8), rank=2, seed=100 + seed)
        result = structure_search_prototype(
            X=X,
            eps=0.20,
            max_splits=2,
            top_k=3,
        )
        print(f"seed={seed}, best_cost={result.best_cost}")
        check(result.best_cost > 0, f"bad best_cost for seed {seed}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    banner("RUNNING TEST SUITE")

    tests = [
        ("preprocessing basic", test_preprocessing_basic),
        ("sketch enumeration basic", test_sketch_enumeration_basic),
        ("scoring basic", test_scoring_basic),
        ("execution basic", test_execution_basic),
        ("rounding basic", test_rounding_basic),
        ("search on rank-1 tensor", test_search_on_rank1_tensor),
        ("search on low-rank tensor", test_search_on_low_rank_tensor),
        ("search on random tensor", test_search_on_random_tensor),
        ("empty sketch dense baseline", test_empty_sketch_dense_baseline),
        ("multiple runs stability", test_multiple_runs_stability),
    ]

    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            run_test(name, fn)
            passed += 1
        except Exception:
            failed += 1
            break

    banner("TEST SUMMARY")
    print(f"passed: {passed}")
    print(f"failed: {failed}")


if __name__ == "__main__":
    main()