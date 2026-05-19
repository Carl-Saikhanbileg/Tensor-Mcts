import time
import numpy as np

from core_algo.tensor_core import TensorData
from core_algo.search import structure_search
from mcts.network_model import make_model
from mcts.mcts_search import structure_search_mcts
from mcts.mcts_env import state_dim, n_actions


def make_low_rank_4d(shape=(8, 9, 10, 11), rank=2, seed=0):
    rng = np.random.default_rng(seed)
    data = np.zeros(shape)

    for _ in range(rank):
        a = rng.standard_normal(shape[0])
        b = rng.standard_normal(shape[1])
        c = rng.standard_normal(shape[2])
        d = rng.standard_normal(shape[3])
        data += np.einsum("i,j,k,l->ijkl", a, b, c, d)

    return TensorData(data)


def run():
    X = make_low_rank_4d()
    eps = 0.15
    max_splits = 3
    top_k = 50

    t0 = time.perf_counter()
    enum_res = structure_search(X, eps=eps, max_splits=max_splits, top_k=top_k)
    enum_time = time.perf_counter() - t0
    for n_samples in [5, 10, 25, 50, 100, 200]:
        model = make_model(state_dim(X.order), n_actions(X.order))

        t0 = time.perf_counter()
        mcts_res = structure_search_mcts(
            X=X,
            eps=eps,
            model=model,
            max_splits=max_splits,
            top_k=top_k,
            n_samples=n_samples,
            n_sims=5,
            tau=1.0,
        )
        mcts_time = time.perf_counter() - t0

        print()
        print("n_samples:", n_samples)
        print("mcts cost:", mcts_res.best_cost)
        print("mcts CR:", X.size / mcts_res.best_cost)
        print("mcts time:", mcts_time)
        print("mcts topology:", mcts_res.best_scored_sketch.sketch.subsets())


if __name__ == "__main__":
    run()