"""Modular DQN / Double DQN agent with hard target-network updates."""
from __future__ import annotations
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DQNConfig:
    hidden_sizes: tuple[int, ...] = (128, 128)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    replay_capacity: int = 100_000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 30_000
    target_update_interval: int = 500  # gradient updates, not env transitions
    double_dqn: bool = True
    gradient_clip_norm: float = 10.0

    def __post_init__(self):
        if not (0 <= self.gamma <= 1):
            raise ValueError("gamma must be between 0 and 1")
        if not (0 <= self.epsilon_final <= self.epsilon_start <= 1):
            raise ValueError("require 0 <= epsilon_final <= epsilon_start <= 1")
        if min(self.learning_rate, self.gradient_clip_norm) <= 0:
            raise ValueError("learning_rate and gradient_clip_norm must be positive")
        if min(self.replay_capacity, self.epsilon_decay_steps, self.target_update_interval) < 1:
            raise ValueError("replay_capacity, epsilon_decay_steps and target_update_interval must be positive")
        if any(int(n) <= 0 for n in self.hidden_sizes):
            raise ValueError("hidden layer widths must be positive")


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for width in hidden_sizes:
            layers.extend((nn.Linear(last, width), nn.ReLU()))
            last = width
        layers.append(nn.Linear(last, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: DQNConfig, device: str = "cpu", seed: int = 0):
        self.config = config
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.rng = np.random.default_rng(seed)
        self.online = QNetwork(obs_dim, action_dim, config.hidden_sizes).to(self.device)
        self.target = QNetwork(obs_dim, action_dim, config.hidden_sizes).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.updates = 0

    def epsilon(self, env_step: int) -> float:
        if env_step <= 0:
            return self.config.epsilon_start
        if env_step >= self.config.epsilon_decay_steps:
            return self.config.epsilon_final
        fraction = env_step / self.config.epsilon_decay_steps
        return self.config.epsilon_start + fraction * (self.config.epsilon_final - self.config.epsilon_start)

    @torch.no_grad()
    def act(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_dim))
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(obs).argmax(dim=1).item())

    def update(self, batch: tuple[np.ndarray, ...]) -> float:
        obs, actions, rewards, next_obs, terminated = (
            torch.as_tensor(x, device=self.device) for x in batch
        )
        q = self.online(obs).gather(1, actions.long().unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = self.online(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target(next_obs).max(dim=1).values
            target = rewards + self.config.gamma * (1.0 - terminated) * next_q
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path, global_step: int, *, metadata: dict | None = None):
        torch.save({
            "online": self.online.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(), "updates": self.updates,
            "global_step": global_step, "obs_dim": self.online.net[0].in_features,
            "action_dim": self.action_dim, "config": asdict(self.config),
            "metadata": metadata or {},
        }, path)

    @classmethod
    def load(cls, path, device: str = "cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        config = DQNConfig(**checkpoint["config"])
        agent = cls(checkpoint["obs_dim"], checkpoint["action_dim"], config, device=device)
        agent.online.load_state_dict(checkpoint["online"])
        agent.target.load_state_dict(checkpoint["target"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.updates = checkpoint["updates"]
        return agent, checkpoint
