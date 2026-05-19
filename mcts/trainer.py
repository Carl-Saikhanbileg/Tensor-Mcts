from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core_algo.tensor_core import TensorData
from mcts.mcts_env import (
    SearchState,
    Subset,
    action_mask,
    apply_action,
    compute_reward,
    encode_state,
    initial_state,
    is_terminal,
    n_actions,
    state_dim,
    valid_actions,
)
from mcts.network_model import PolicyValueNet, make_model
from mcts.mcts import MCTS

Sample = Tuple[np.ndarray, np.ndarray, np.ndarray, float]


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 10_000
    n_sims: int = 50
    c_puct: float = 1.0
    tau: float = 1.0
    epochs_per_iter: int = 5
    n_iterations: int = 100
    hidden: int = 256


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: List[Sample] = []

    def push(self, samples: List[Sample]) -> None:
        self.data.extend(samples)
        if len(self.data) > self.capacity:
            self.data = self.data[-self.capacity :]

    def sample(self, n: int) -> List[Sample]:
        return random.sample(self.data, min(n, len(self.data)))

    def __len__(self) -> int:
        return len(self.data)


def run_episode(
    X: TensorData,
    eps: float,
    max_splits: int,
    mcts: MCTS,
    tau: float,
) -> List[Sample]:
    state = initial_state(X, eps, max_splits)
    trajectory: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    while not is_terminal(state):
        pi = mcts.run(state)
        enc = encode_state(state)
        mask = action_mask(state)
        trajectory.append((enc, mask, pi))

        if tau > 0:
            probs = pi ** (1.0 / tau)
            probs /= probs.sum() + 1e-8
            idx = int(np.random.choice(len(probs), p=probs))
        else:
            idx = int(np.argmax(pi))

        state = apply_action(state, state.all_subsets[idx])

    z = compute_reward(state)
    return [(enc, mask, pi, z) for enc, mask, pi in trajectory]


def compute_loss(batch: List[Sample], model: PolicyValueNet) -> torch.Tensor:
    encs = torch.tensor(np.stack([s[0] for s in batch]), dtype=torch.float32)
    masks = torch.tensor(np.stack([s[1] for s in batch]), dtype=torch.float32)
    pi_targets = torch.tensor(np.stack([s[2] for s in batch]), dtype=torch.float32)
    z_targets = torch.tensor([s[3] for s in batch], dtype=torch.float32)

    p_pred, v_pred = model(encs, masks)
    policy_loss = -(pi_targets * torch.log(p_pred + 1e-8)).sum(dim=-1).mean()
    value_loss = nn.functional.mse_loss(v_pred, z_targets)
    return policy_loss + value_loss


def train(
    tensors: List[TensorData],
    tensor_order: int,
    eps: float,
    max_splits: int,
    cfg: TrainConfig = TrainConfig(),
) -> PolicyValueNet:
    model = make_model(state_dim(tensor_order), n_actions(tensor_order), cfg.hidden)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_size)
    mcts = MCTS(model, c_puct=cfg.c_puct, n_sims=cfg.n_sims)

    for it in range(cfg.n_iterations):
        X = random.choice(tensors)
        model.eval()
        samples = run_episode(X, eps, max_splits, mcts, cfg.tau)
        buffer.push(samples)

        if len(buffer) < cfg.batch_size:
            continue

        model.train()
        total_loss = 0.0
        for _ in range(cfg.epochs_per_iter):
            batch = buffer.sample(cfg.batch_size)
            loss = compute_loss(batch, model)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if (it + 1) % 10 == 0:
            print(
                f"iter {it + 1}/{cfg.n_iterations}  "
                f"loss={total_loss / cfg.epochs_per_iter:.4f}  "
                f"buffer={len(buffer)}"
            )

    model.eval()
    return model


def predict(
    X: TensorData,
    model: PolicyValueNet,
    eps: float,
    max_splits: int,
) -> List[Subset]:
    state = initial_state(X, eps, max_splits)
    selected: List[Subset] = []
    model.eval()
    with torch.no_grad():
        while not is_terminal(state):
            enc = torch.tensor(encode_state(state), dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(action_mask(state), dtype=torch.float32).unsqueeze(0)
            p, _ = model(enc, mask)
            priors = p.squeeze(0).numpy()
            actions = valid_actions(state)
            if not actions:
                break
            best = max(actions, key=lambda a: priors[state.all_subsets.index(a)])
            state = apply_action(state, best)
            selected.append(best)
    return selected
