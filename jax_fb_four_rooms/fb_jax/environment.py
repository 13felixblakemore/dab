"""Reward-free/goal-conditioned continuous Four Rooms.

Geometry and four discrete moves follow the supplied continuous_four_rooms.py.
Uses a *local* NumPy RNG (rather than global randomness) for reproducibility.
Compatible with the reference reset/step signatures, without Gymnasium dependency.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WallBlock:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def contains(self, p, margin=0.0):
        return (self.xmin - margin <= p[0] <= self.xmax + margin
                and self.ymin - margin <= p[1] <= self.ymax + margin)

    def intersects_segment(self, p1, p2):
        """Closed-box slab intersection, rejecting tangencies with zero overlap."""
        if self.contains(p2, margin=-1e-6):
            return True
        tmin, tmax = 0.0, 1.0
        for axis, lo, hi in ((0, self.xmin, self.xmax), (1, self.ymin, self.ymax)):
            delta = float(p2[axis] - p1[axis])
            if abs(delta) < 1e-9:
                if p1[axis] < lo or p1[axis] > hi:
                    return False
            else:
                a, b = (lo-p1[axis])/delta, (hi-p1[axis])/delta
                tmin, tmax = max(tmin, min(a,b)), min(tmax, max(a,b))
                if tmin > tmax:
                    return False
        return tmax > 1e-6 and (tmax-tmin) > 1e-6


def classic_walls(size=1.0):
    blocks = [(6,0,7,1), (6,2,7,8), (6,9,7,11),
              (0,4,2,5), (3,4,6,5), (7,4,9,5), (10,4,11,5)]
    scale = size/11
    return [WallBlock(*(v*scale for v in block)) for block in blocks]


class ContinuousFourRooms:
    ACTIONS = ("LEFT", "RIGHT", "UP", "DOWN", "STAY")

    def __init__(self, size=1.0, step_size=0.05, movement_noise=0.005,
                 seed=1, explorable=True, include_stay_action=False,
                 threshold_distance=0.1):
        if size <= 0 or step_size <= 0 or movement_noise < 0:
            raise ValueError("size and step_size must be positive; noise nonnegative")
        self.size = float(size)
        self.step_size = float(step_size)
        self.movement_noise = float(movement_noise)
        self.threshold_distance = float(threshold_distance)
        self.explorable = bool(explorable)
        self.include_stay_action = bool(include_stay_action)
        self.n_actions = 5 if include_stay_action else 4
        self.wall_blocks = classic_walls(self.size)
        self._rng = np.random.default_rng(seed)
        self._position = np.array([0.5,0.5], dtype=np.float32)*self.size
        self._goal = None
        self._moves = np.array([[-1,0],[1,0],[0,1],[0,-1],[0,0]], dtype=np.float32)*step_size

    def seed(self, seed):
        self._rng = np.random.default_rng(seed)

    def is_free(self, point):
        p = np.asarray(point)
        return (p.shape == (2,) and np.all(np.isfinite(p))
                and np.all(p > 0) and np.all(p < self.size)
                and not any(w.contains(p) for w in self.wall_blocks))

    def sample_point(self):
        for _ in range(100000):
            p = self._rng.uniform(1e-6,self.size-1e-6,size=2)
            if self.is_free(p):
                return p.astype(np.float32)
        raise RuntimeError("Could not sample a free location")

    def reset(self, state=None, goal=None):
        p = self.sample_point() if state is None else np.asarray(state, dtype=np.float32)
        if not self.is_free(p):
            raise ValueError("Starting position is outside navigable space")
        self._position = p.copy()
        if self.explorable:
            self._goal = None
        else:
            g = self.sample_point() if goal is None else np.asarray(goal, dtype=np.float32)
            if not self.is_free(g):
                raise ValueError("Goal outside navigable space")
            self._goal = g.copy()
        return self.current_position

    def step(self, action):
        if not isinstance(action, (int,np.integer)) or not 0 <= action < self.n_actions:
            raise ValueError(f"Action must be an integer in [0, {self.n_actions})")
        movement = self._moves[action] + self._rng.normal(0,self.movement_noise,2)
        proposed = np.clip(self._position.astype(np.float64)+movement,0,self.size)
        collides = any(w.intersects_segment(self._position,proposed) for w in self.wall_blocks)
        if not collides:
            self._position = proposed.astype(np.float32)
        reward = 0.0
        done = False
        if self._goal is not None:
            done = bool(np.max(np.abs(self._position-self._goal)) <= self.threshold_distance)
            reward = float(done)
        return (self.current_position,reward,done,
                {"goes_through_wall":collides,"proposed_position":proposed.copy()})

    @property
    def current_position(self):
        return self._position.copy()

    @property
    def goal(self):
        return None if self._goal is None else self._goal.copy()

    def get_available_actions(self, state=None):
        # Like the source: wall collisions are handled by the environment.
        return list(range(self.n_actions))

    def get_action_space(self):
        return list(range(self.n_actions))
