from __future__ import annotations
from dataclasses import dataclass
from math import prod
from typing import Iterable, Tuple
import numpy as np


@dataclass
class TensorData:
    data: np.ndarray

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def order(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return int(prod(self.shape))

    @property
    def fro_norm(self) -> float:
        return float(np.linalg.norm(self.data))

    @property
    def fro_norm_sq(self) -> float:
        return float(self.fro_norm ** 2)

    def matricize(self, row_modes: Iterable[int]) -> Tuple[np.ndarray, Tuple[int, ...], Tuple[int, ...]]:
        row_modes = tuple(sorted(row_modes))
        col_modes = tuple(i for i in range(self.order) if i not in row_modes)
        perm = row_modes + col_modes
        permuted = np.transpose(self.data, axes=perm)
        M = permuted.reshape(int(prod(self.shape[i] for i in row_modes)), int(prod(self.shape[j] for j in col_modes)))
        return M, row_modes, col_modes


def make_random_tensor(shape: Tuple[int, ...], seed: int | None = None) -> TensorData:
    return TensorData(np.random.default_rng(seed).standard_normal(shape))


if __name__ == "__main__":
    X = make_random_tensor((4, 5, 6), seed=0)
    M, rows, cols = X.matricize((0, 2))
    print("shape:", X.shape)
    print("matricized:", M.shape)