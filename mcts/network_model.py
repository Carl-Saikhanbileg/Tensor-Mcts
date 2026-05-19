from __future__ import annotations

import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(self, in_dim: int, n_act: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, n_act)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        logits = self.policy_head(h)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        p = torch.softmax(logits, dim=-1)
        v = self.value_head(h).squeeze(-1)
        return p, v


def make_model(in_dim: int, n_act: int, hidden: int = 256) -> PolicyValueNet:
    return PolicyValueNet(in_dim, n_act, hidden)
