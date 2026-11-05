"""Preallocated CPU replay, random minibatches transferred together to JAX."""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, n_actions, seed=0):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.observations = np.empty((capacity,obs_dim), dtype=np.float32)
        self.next_observations = np.empty((capacity,obs_dim), dtype=np.float32)
        self.actions = np.empty((capacity,),dtype=np.int32)
        self.terminals = np.zeros((capacity,),dtype=np.float32)
        self.next_action_mask = np.ones((capacity,n_actions),dtype=np.bool_)
        self.rng = np.random.default_rng(seed)
        self.ptr = self.size = 0
        self.n_actions = n_actions

    def add(self, obs, action, next_obs, terminal=False, next_actions=None):
        i = self.ptr
        self.observations[i] = obs
        self.actions[i] = action
        self.next_observations[i] = next_obs
        self.terminals[i] = terminal
        if next_actions is None:
            self.next_action_mask[i] = True
        else:
            mask = np.zeros(self.n_actions,dtype=np.bool_)
            mask[np.asarray(next_actions,dtype=np.int32)] = True
            if not mask.any():
                raise ValueError("next_actions must contain at least one valid action")
            self.next_action_mask[i] = mask
        self.ptr = (self.ptr+1) % self.capacity
        self.size = min(self.size+1,self.capacity)

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        idx = self.rng.integers(0,self.size,size=batch_size)
        return {"obs":self.observations[idx],"actions":self.actions[idx],
                "next_obs":self.next_observations[idx],"terminals":self.terminals[idx],
                "next_action_mask":self.next_action_mask[idx]}

    def sample_many(self, updates, batch_size, with_goals=False, with_other_actions=False):
        """Vectorised independent sampling of source/other minibatches.

        Returns arrays shaped (updates, batch_size, ...). Random draw order
        differs from repeated sample(), but the training distribution is equal.
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        n=(updates,batch_size)
        ids=self.rng.integers(0,self.size,size=n)
        other=self.rng.integers(0,self.size,size=n)
        batches = {"obs":self.observations[ids],"actions":self.actions[ids],
                   "next_obs":self.next_observations[ids],"terminals":self.terminals[ids],
                   "next_action_mask":self.next_action_mask[ids]}
        results = [batches, self.observations[other]]
        if with_goals:
            goal_ids=self.rng.integers(0,self.size,size=n)
            results.append(self.observations[goal_ids])
        if with_other_actions:
            results.append(self.actions[other])
        return tuple(results)
