from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Tuple
import numpy as np
from core_algo.tensor_core import TensorData
from core_algo.sketch import canonical_osplit_subset, all_osplit_subsets, normalize_subset

Subset = Tuple[int, ...]

@dataclass(frozen=True)
class SplitScoreData:
    subset: Subset
    row_dim: int
    col_dim: int
    singular_values: np.ndarray
    @property
    def max_rank(self) -> int: return min(self.row_dim, self.col_dim)
    @property
    def fro_norm_sq(self) -> float: return float(np.sum(self.singular_values ** 2))
    @property
    def fro_norm(self) -> float: return float(np.sqrt(self.fro_norm_sq))

@dataclass
class PrecomputedSVDCache:
    tensor_shape: Tuple[int, ...]
    tensor_order: int
    tensor_size: int
    tensor_fro_norm_sq: float
    split_scores: Dict[Subset, SplitScoreData] = field(default_factory=dict)
    def __contains__(self, s: Iterable[int]) -> bool: return normalize_subset(s) in self.split_scores
    def __getitem__(self, s: Iterable[int]) -> SplitScoreData: return self.split_scores[normalize_subset(s)]
    def get(self, s: Iterable[int]) -> SplitScoreData: return self[s]
    @property
    def by_subset(self) -> Mapping[Subset, SplitScoreData]: return self.split_scores
    def subsets(self) -> Tuple[Subset, ...]: return tuple(sorted(self.split_scores.keys(), key=lambda s: (len(s), s)))
    def summary(self) -> str:
        res = [f"PrecomputedSVDCache\n  shape={self.tensor_shape}\n  size={self.tensor_size}\n  splits={len(self.split_scores)}"]
        for s in self.subsets():
            d = self.split_scores[s]
            res.append(f"  subset={s} | row={d.row_dim} | col={d.col_dim} | max_r={d.max_rank}")
        return "\n".join(res)

def subset_product(shape: Tuple[int, ...], subset: Iterable[int]) -> int:
    return int(np.prod([shape[i] for i in subset]))

def complement_subset(subset: Iterable[int], d: int) -> Subset:
    S = set(subset)
    return tuple(i for i in range(d) if i not in S)

def compute_singular_values_for_subset(X: TensorData, subset: Iterable[int]) -> SplitScoreData:
    subset = normalize_subset(subset)
    M, _, _ = X.matricize(subset)
    s = np.linalg.svd(M, full_matrices=False, compute_uv=False)
    return SplitScoreData(subset, int(M.shape[0]), int(M.shape[1]), np.asarray(s, dtype=float))

def preprocess_singular_values(X: TensorData) -> PrecomputedSVDCache:
    scores = {s: compute_singular_values_for_subset(X, s) for s in all_osplit_subsets(X.order)}
    return PrecomputedSVDCache(X.shape, X.order, X.size, X.fro_norm_sq, scores)

def get_split_score_data(cache: PrecomputedSVDCache, subset: Iterable[int]) -> SplitScoreData:
    return cache[subset]

def dense_tensor_cost(X: TensorData) -> int:
    return int(X.size)

def print_preprocessing_summary(cache: PrecomputedSVDCache) -> None:
    print(cache.summary())

if __name__ == "__main__":
    X = TensorData(np.random.default_rng(0).standard_normal((4, 5, 6, 7)))
    cache = preprocess_singular_values(X)
    print_preprocessing_summary(cache)