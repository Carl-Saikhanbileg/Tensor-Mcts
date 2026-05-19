import time, numpy as np, matplotlib.pyplot as plt
from core_algo.tensor_core import TensorData
from core_algo.search import structure_search
from mcts.network_model import make_model
from mcts.mcts_search import structure_search_mcts
from mcts.mcts_env import state_dim, n_actions

plt.rcParams.update({
    "figure.figsize": (9, 5.4),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.frameon": False,
})

def cp_tensor(shape, rank=2, seed=0, noise=0.01):
    rng = np.random.default_rng(seed)
    inds = "abcdefghijklmnopqrstuvwxyz"[:len(shape)]
    expr = ",".join(inds) + "->" + inds
    X = sum(np.einsum(expr, *[rng.standard_normal(n) for n in shape]) for _ in range(rank))
    return TensorData(X + noise * rng.standard_normal(shape))

def contract(net):
    nodes = {k: (v.data.copy(), list(v.axis_labels)) for k, v in net.graph.nodes.items()}
    edges = list(net.graph.edges)
    while edges:
        e = edges.pop(0)
        A, la = nodes.pop(e.node_u)
        B, lb = nodes.pop(e.node_v)
        ia, ib = la.index(e.label), lb.index(e.label)
        C = np.tensordot(A, B, axes=([ia], [ib]))
        labs = la[:ia] + la[ia + 1:] + lb[:ib] + lb[ib + 1:]
        name = e.node_u + "_" + e.node_v
        edges = [type(old)(
            name if old.node_u in (e.node_u, e.node_v) else old.node_u,
            name if old.node_v in (e.node_u, e.node_v) else old.node_v,
            old.label
        ) for old in edges]
        nodes[name] = (C, labs)
    Y, labs = next(iter(nodes.values()))
    return np.transpose(Y, [labs.index(f"free:I{i}") for i in range(len(net.original_shape))])

def relerr(X, net):
    return np.linalg.norm(contract(net) - X.data) / np.linalg.norm(X.data)

def trial(shape, seed, eps=0.15, max_splits=3, top_k=15, n_samples=50):
    X = cp_tensor(shape, seed=seed)
    model = make_model(state_dim(X.order), n_actions(X.order))

    t = time.perf_counter()
    er = structure_search(X, eps=eps, max_splits=max_splits, top_k=top_k)
    et = time.perf_counter() - t

    t = time.perf_counter()
    mr = structure_search_mcts(X, eps, model, max_splits, top_k, n_samples, 5, 1.0, 1.0)
    mt = time.perf_counter() - t

    ee, me = relerr(X, er.best_network), relerr(X, mr.best_network)
    return dict(
        order=X.order, size=X.size,
        enum_time=et, mcts_time=mt,
        enum_cr=X.size / er.best_cost, mcts_cr=X.size / mr.best_cost,
        enum_err=ee, mcts_err=me,
        enum_ok=ee <= eps * 1.000001, mcts_ok=me <= eps * 1.000001,
    )

def make_shapes():
    out = []
    for n in range(4, 17):
        out.append((n, n + 1, n + 2, n + 3))
    for n in range(3, 13):
        out.append((n, n + 1, n + 2, n + 3, n + 4))
    for n in range(3, 10):
        out.append((n, n, n + 1, n + 1, n + 2))
    for n in range(4, 12):
        out.append((n, n + 2, n + 3, n + 5))
    for n in range(3, 8):
        out.append((n, n + 1, n + 3, n + 4, n + 6))
    return sorted(out, key=np.prod)

def plot_line(x, y1, y2, title, ylabel, fname, logy=False, eps=None):
    plt.figure()
    plt.plot(x, y1, "-o", lw=2, ms=4, label="Enumeration")
    plt.plot(x, y2, "-s", lw=2, ms=4, label="MCTS")
    if eps is not None:
        plt.axhline(eps, ls="--", lw=1.8, label="epsilon")
    if logy:
        plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Dense tensor size, log scale")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=240)
    plt.show()

def main():
    eps = 0.15
    rows = []

    for i, shape in enumerate(make_shapes()):
        print(f"[{i+1:02d}] shape={shape}, size={np.prod(shape)}")
        try:
            r = trial(shape, 1000 + i, eps=eps)
            rows.append(r)
            print(" enum:", round(r["enum_time"], 4), "s CR", round(r["enum_cr"], 2), "err", round(r["enum_err"], 4), r["enum_ok"])
            print(" mcts :", round(r["mcts_time"], 4), "s CR", round(r["mcts_cr"], 2), "err", round(r["mcts_err"], 4), r["mcts_ok"])
        except Exception as e:
            print(" failed:", e)

    rows = sorted(rows, key=lambda r: r["size"])
    size = np.array([r["size"] for r in rows])
    order = np.array([r["order"] for r in rows])
    et = np.array([r["enum_time"] for r in rows])
    mt = np.array([r["mcts_time"] for r in rows])
    ecr = np.array([r["enum_cr"] for r in rows])
    mcr = np.array([r["mcts_cr"] for r in rows])
    ee = np.array([r["enum_err"] for r in rows])
    me = np.array([r["mcts_err"] for r in rows])

    plot_line(size, et, mt, f"Runtime to Find Decomposition, eps={eps}", "Runtime seconds", "runtime_vs_size.png")
    plot_line(size, et, mt, f"Runtime Scaling, eps={eps}", "Runtime seconds, log scale", "runtime_vs_size_log.png", logy=True)
    plot_line(size, ee, me, "Correctness: Relative Error vs Tensor Size", "Relative reconstruction error", "error_vs_size.png", eps=eps)
    plot_line(size, ecr, mcr, "Compression Ratio vs Tensor Size", "Compression ratio", "cr_vs_size.png")

    plt.figure()
    plt.scatter(size[order == 4], et[order == 4], s=45, marker="o", label="Enum 4D")
    plt.scatter(size[order == 5], et[order == 5], s=45, marker="o", label="Enum 5D")
    plt.scatter(size[order == 4], mt[order == 4], s=45, marker="s", label="MCTS 4D")
    plt.scatter(size[order == 5], mt[order == 5], s=45, marker="s", label="MCTS 5D")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Dense tensor size, log scale")
    plt.ylabel("Runtime seconds, log scale")
    plt.title("Runtime by Tensor Order")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("runtime_by_order.png", dpi=240)
    plt.show()

    print("\nsaved: runtime_vs_size.png, runtime_vs_size_log.png, error_vs_size.png, cr_vs_size.png, runtime_by_order.png")

if __name__ == "__main__":
    main()