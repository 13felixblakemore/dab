"""Preallocated NumPy replay buffer (low-overhead random minibatch sampling)."""
from __future__ import annotations
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        if capacity < 1 or obs_dim < 1:
            raise ValueError("capacity and obs_dim must be positive")
        self.capacity = capacity
        self.observations = np.empty((capacity, obs_dim), dtype=np.float32)
        self.next_observations = np.empty((capacity, obs_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.float32)
        self.size = 0
        self.index = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.index
        self.observations[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_observations[i] = next_obs
        self.terminated[i] = terminated
        self.index = (i + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size: int):
        if self.size == 0:
            raise ValueError("Can't sample an empty replay buffer")
        idx = self.rng.integers(self.size, size=batch_size)
        return (self.observations[idx], self.actions[idx], self.rewards[idx],
                self.next_observations[idx], self.terminated[idx])

    def __len__(self):
        return self.size
