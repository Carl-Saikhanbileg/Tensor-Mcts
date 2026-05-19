from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Iterator, List, Sequence, Tuple


# ============================================================
# Basic subset utilities
# ============================================================


def normalize_subset(subset: Iterable[int]) -> Tuple[int, ...]:
    return tuple(sorted(subset))


def is_strict_subset(a: Sequence[int], b: Sequence[int]) -> bool:
    A, B = set(a), set(b)
    return A < B


def is_subset(a: Sequence[int], b: Sequence[int]) -> bool:
    A, B = set(a), set(b)
    return A <= B


def is_disjoint(a: Sequence[int], b: Sequence[int]) -> bool:
    return set(a).isdisjoint(b)


def is_crossing(a: Sequence[int], b: Sequence[int]) -> bool:
    """
    Crossing means:
    - they overlap
    - neither contains the other

    This is exactly the bad case ruled out in the paper's sketch validity check.
    """
    A, B = set(a), set(b)
    return len(A & B) > 0 and not (A <= B or B <= A)


def is_laminar_pair(a: Sequence[int], b: Sequence[int]) -> bool:
    """
    Valid pair relation for tree-structured partitions:
    - nested
    - or disjoint
    """
    return not is_crossing(a, b)


def is_laminar_family(subsets: Sequence[Sequence[int]]) -> bool:
    n = len(subsets)
    for i in range(n):
        for j in range(i + 1, n):
            if not is_laminar_pair(subsets[i], subsets[j]):
                return False
    return True


# ============================================================
# Canonical bipartition-side handling
# ============================================================


def canonical_osplit_subset(subset: Iterable[int], d: int) -> Tuple[int, ...]:
    """
    OSplit(I, r) and OSplit(I^c, r) represent the same bipartition.
    We store only one canonical side.

    Rule:
    - keep the smaller side
    - if equal size, keep lexicographically smaller
    """
    s = normalize_subset(subset)
    comp = tuple(i for i in range(d) if i not in s)

    if len(s) < len(comp):
        return s
    if len(comp) < len(s):
        return comp
    return min(s, comp)


def all_osplit_subsets(d: int) -> List[Tuple[int, ...]]:
    """
    Enumerate all unique OSplit target subsets for an order-d tensor.
    """
    out = []
    seen = set()

    for mask in range(1, 1 << d):
        subset = tuple(i for i in range(d) if (mask >> i) & 1)
        if len(subset) == d:
            continue

        c = canonical_osplit_subset(subset, d)
        if c not in seen:
            seen.add(c)
            out.append(c)

    out.sort(key=lambda s: (len(s), s))
    return out


# ============================================================
# OSplit and Sketch objects
# ============================================================


@dataclass(frozen=True)
class OSplitSketchOp:
    """
    Symbolic output-directed split:
        OSplit(I, □)

    Only the subset I is stored here.
    Rank is left unspecified and will be assigned later during scoring.
    """
    subset: Tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "subset", normalize_subset(self.subset))

    @property
    def size(self) -> int:
        return len(self.subset)

    def is_nested_with(self, other: "OSplitSketchOp") -> bool:
        return is_subset(self.subset, other.subset) or is_subset(other.subset, self.subset)

    def is_disjoint_from(self, other: "OSplitSketchOp") -> bool:
        return is_disjoint(self.subset, other.subset)

    def crosses(self, other: "OSplitSketchOp") -> bool:
        return is_crossing(self.subset, other.subset)

    def is_laminar_with(self, other: "OSplitSketchOp") -> bool:
        return is_laminar_pair(self.subset, other.subset)


@dataclass(frozen=True)
class Sketch:
    """
    Sketch program:
        OSplit(I1, □1); OSplit(I2, □2); ...

    This module stores only structure, not ranks.
    """
    ops: Tuple[OSplitSketchOp, ...]

    def __post_init__(self) -> None:
        normalized = tuple(
            sorted(
                (OSplitSketchOp(op.subset) for op in self.ops),
                key=lambda op: (len(op.subset), op.subset),
            )
        )
        object.__setattr__(self, "ops", normalized)

    def __len__(self) -> int:
        return len(self.ops)

    def subsets(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(op.subset for op in self.ops)

    def is_valid(self) -> bool:
        return is_laminar_family(self.subsets())

    def contains_subset(self, subset: Iterable[int]) -> bool:
        subset = normalize_subset(subset)
        return subset in self.subsets()

    def max_subset_size(self) -> int:
        if not self.ops:
            return 0
        return max(len(op.subset) for op in self.ops)

    def depth_proxy(self) -> int:
        """
        Simple structural proxy:
        maximum chain length under inclusion.

        This is not execution depth exactly, but useful later for analysis.
        """
        subsets = self.subsets()
        if not subsets:
            return 0

        dp = {}
        for s in subsets:
            best = 1
            for t in subsets:
                if s != t and is_strict_subset(t, s):
                    best = max(best, dp.get(t, 1) + 1)
            dp[s] = best
        return max(dp.values())

    def sorted_subsets(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(sorted(self.subsets(), key=lambda s: (len(s), s)))


# ============================================================
# Paper-style sketch validity logic
# ============================================================


def is_valid_sketch_subsets(subsets: Sequence[Tuple[int, ...]]) -> bool:
    """
    This mirrors the paper's main validity rule:

    Invalid if there exist Ii, Ij such that
    - Ii ∩ Ij != empty
    - Ii not subset of Ij
    - Ij not subset of Ii

    Equivalently: subsets must form a laminar family.
    """
    return is_laminar_family(subsets)


def make_sketch(subsets: Sequence[Iterable[int]]) -> Sketch:
    ops = tuple(OSplitSketchOp(normalize_subset(s)) for s in subsets)
    sketch = Sketch(ops)
    return sketch


# ============================================================
# Enumeration helpers
# ============================================================


def candidate_osplits(d: int) -> List[OSplitSketchOp]:
    return [OSplitSketchOp(s) for s in all_osplit_subsets(d)]


def enumerate_sketches(
    d: int,
    max_splits: int,
) -> List[Sketch]:
    """
    Full sketch enumeration up to max_splits.

    This follows the paper's high-level logic:
    1. enumerate all possible OSplit target subsets
    2. enumerate combinations up to max_splits
    3. keep only laminar families
    """
    ops = candidate_osplits(d)
    out = []

    for k in range(1, max_splits + 1):
        for chosen in combinations(ops, k):
            subsets = tuple(op.subset for op in chosen)
            if is_valid_sketch_subsets(subsets):
                out.append(Sketch(chosen))

    return out


def iter_sketches(
    d: int,
    max_splits: int,
) -> Iterator[Sketch]:
    """
    Generator version of enumerate_sketches.
    Use this when search space gets large.
    """
    ops = candidate_osplits(d)

    for k in range(1, max_splits + 1):
        for chosen in combinations(ops, k):
            subsets = tuple(op.subset for op in chosen)
            if is_valid_sketch_subsets(subsets):
                yield Sketch(chosen)


# ============================================================
# Useful higher-level structural filters
# ============================================================


def filter_max_subset_size(
    sketches: Iterable[Sketch],
    max_subset_size: int,
) -> List[Sketch]:
    return [s for s in sketches if s.max_subset_size() <= max_subset_size]


def filter_by_required_subset(
    sketches: Iterable[Sketch],
    required_subset: Iterable[int],
) -> List[Sketch]:
    required_subset = normalize_subset(required_subset)
    return [s for s in sketches if s.contains_subset(required_subset)]


def filter_by_num_splits(
    sketches: Iterable[Sketch],
    num_splits: int,
) -> List[Sketch]:
    return [s for s in sketches if len(s) == num_splits]


def group_sketches_by_num_splits(
    sketches: Iterable[Sketch],
) -> dict[int, List[Sketch]]:
    out: dict[int, List[Sketch]] = {}
    for s in sketches:
        out.setdefault(len(s), []).append(s)
    return out


# ============================================================
# Optional structural tree helpers
# ============================================================


def immediate_parent_subset(
    subset: Sequence[int],
    family: Sequence[Sequence[int]],
) -> Tuple[int, ...] | None:
    """
    Return the smallest strict superset of `subset` inside the family.
    """
    subset = normalize_subset(subset)
    supersets = [
        normalize_subset(t)
        for t in family
        if subset != normalize_subset(t) and set(subset) < set(t)
    ]
    if not supersets:
        return None
    return min(supersets, key=lambda t: len(t))


def immediate_children_subsets(
    subset: Sequence[int],
    family: Sequence[Sequence[int]],
) -> List[Tuple[int, ...]]:
    """
    Return subsets whose immediate parent is `subset`.
    """
    subset = normalize_subset(subset)
    fam = [normalize_subset(t) for t in family]
    out = []

    for t in fam:
        if t == subset:
            continue
        if set(t) < set(subset):
            parent = immediate_parent_subset(t, fam)
            if parent == subset:
                out.append(t)

    out.sort(key=lambda s: (len(s), s))
    return out


# ============================================================
# Demo
# ============================================================


if __name__ == "__main__":
    d = 4
    sketches = enumerate_sketches(d=d, max_splits=3)

    print("number of candidate OSplit subsets:", len(candidate_osplits(d)))
    print("number of valid sketches:", len(sketches))
    print()

    for s in sketches[:10]:
        print(
            "subsets =", s.subsets(),
            "| num_splits =", len(s),
            "| depth_proxy =", s.depth_proxy(),
        )