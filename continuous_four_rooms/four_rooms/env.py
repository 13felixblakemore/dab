"""Continuous-position, discrete-action Four Rooms navigation environment.

Coordinates are in [0, 1]^2. The central horizontal and vertical walls
split the square into four rooms; each half-wall has one doorway.

Implements the Gymnasium reset/step API, with optional Gymnasium dependency.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _Base = gym.Env
except ImportError:  # The project runs without gymnasium installed.
    _Base = object

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
            self.shape = tuple(shape)
            self.dtype = np.dtype(dtype)
            self.np_random = np.random.default_rng()

        def seed(self, seed=None):
            self.np_random = np.random.default_rng(seed)

        def sample(self):
            return self.np_random.uniform(self.low, self.high).astype(self.dtype)

        def contains(self, x):
            x = np.asarray(x)
            return x.shape == self.shape and np.all(x >= self.low) and np.all(x <= self.high)

    class _Discrete:
        def __init__(self, n):
            self.n = n
            self.np_random = np.random.default_rng()

        def seed(self, seed=None):
            self.np_random = np.random.default_rng(seed)

        def sample(self):
            return int(self.np_random.integers(self.n))

        def contains(self, x):
            return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    class _Spaces:
        Box = _Box
        Discrete = _Discrete

    spaces = _Spaces()


class ContinuousFourRoomsEnv(_Base):
    """A point agent navigates around walls in a continuous 2-D domain.

    Actions: 0=right, 1=left, 2=up, 3=down.
    Observation: float32 [agent_x, agent_y, goal_x, goal_y].

    A move is swept in small substeps, making tunnelling through walls impossible.
    Collision blocks the remaining motion (no artificial teleportation).
    Collision treats agent as a small axis-aligned square for cheap clearance.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 12}
    ACTIONS = np.asarray([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float32)

    def __init__(
        self,
        doorway_width: float = 0.18,
        wall_thickness: float = 0.03,
        agent_radius: float = 0.012,
        step_size: float = 0.045,
        collision_substep: float = 0.008,
        motion_noise: float = 0.0,
        goal_radius: float = 0.055,
        max_episode_steps: int = 200,
        start_pos: tuple[float, float] = (0.15, 0.15),
        goal_pos: tuple[float, float] = (0.85, 0.85),
        start_noise: float = 0.04,
        step_penalty: float = -0.01,
        success_reward: float = 1.0,
        shaping_scale: float = 0.0,
        shaping_gamma: float = 0.99,
    ):
        if not (0.06 <= doorway_width < 0.45):
            raise ValueError("doorway_width must be in [0.06, 0.45)")
        if wall_thickness <= 0 or wall_thickness >= 0.1:
            raise ValueError("wall_thickness must be in (0, 0.1)")
        if agent_radius <= 0 or agent_radius >= doorway_width / 2:
            raise ValueError("agent_radius must be positive and smaller than half a doorway")
        if not (0 < step_size <= 0.5) or collision_substep <= 0:
            raise ValueError("step_size must be in (0, 0.5], collision_substep positive")
        if motion_noise < 0 or start_noise < 0 or goal_radius <= 0 or max_episode_steps < 1 or not (0 <= shaping_gamma <= 1):
            raise ValueError("Noise/radius/episode length outside allowable range")

        self.render_mode = "rgb_array"
        self.doorway_width = doorway_width
        self.wall_thickness = wall_thickness
        self.agent_radius = agent_radius
        self.step_size = step_size
        self.collision_substep = collision_substep
        self.motion_noise = motion_noise
        self.goal_radius = goal_radius
        self.max_episode_steps = int(max_episode_steps)
        self.start_pos = np.asarray(start_pos, dtype=np.float32)
        self.goal = np.asarray(goal_pos, dtype=np.float32)
        self.start_noise = start_noise
        self.step_penalty = step_penalty
        self.success_reward = success_reward
        self.shaping_scale = shaping_scale
        self.shaping_gamma = shaping_gamma

        # (xmin, ymin, xmax, ymax), excluding the doorway in each half-wall.
        half = wall_thickness / 2
        door = doorway_width / 2
        self.walls: list[tuple[float, float, float, float]] = [
            (0.5-half, 0.0, 0.5+half, 0.25-door),
            (0.5-half, 0.25+door, 0.5+half, 0.75-door),
            (0.5-half, 0.75+door, 0.5+half, 1.0),
            (0.0, 0.5-half, 0.25-door, 0.5+half),
            (0.25+door, 0.5-half, 0.75-door, 0.5+half),
            (0.75+door, 0.5-half, 1.0, 0.5+half),
        ]
        # Gymnasium-compatible spaces, available even if gymnasium isn't installed.
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.rng = np.random.default_rng()
        self.position = self.start_pos.copy()
        self.steps = 0
        for name, p in [("start_pos", self.start_pos), ("goal_pos", self.goal)]:
            if not self._is_free(p):
                raise ValueError(f"{name} intersects a wall or environment boundary")

    def _is_free(self, p: np.ndarray) -> bool:
        r = self.agent_radius
        x, y = float(p[0]), float(p[1])
        if x < r or x > 1-r or y < r or y > 1-r:
            return False
        for x0, y0, x1, y1 in self.walls:
            if x0-r <= x <= x1+r and y0-r <= y <= y1+r:
                return False
        return True

    def _observation(self) -> np.ndarray:
        return np.concatenate((self.position, self.goal)).astype(np.float32)

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(self.position-self.goal))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            if hasattr(super(), "reset"):
                super().reset(seed=seed)
        self.steps = 0
        # Optional deterministic position override is useful for controlled evaluation.
        pos = None if options is None else options.get("start_pos")
        if pos is not None:
            chosen = np.asarray(pos, dtype=np.float32)
            if not self._is_free(chosen):
                raise ValueError("Requested start position is blocked")
        elif self.start_noise:
            chosen = self.start_pos.copy()
            for _ in range(1000):
                trial = self.start_pos + self.rng.uniform(-self.start_noise, self.start_noise, 2)
                if self._is_free(trial):
                    chosen = trial.astype(np.float32)
                    break
            else:
                raise RuntimeError("Could not sample a valid initial position")
        else:
            chosen = self.start_pos.copy()
        self.position = chosen
        return self._observation(), {"distance_to_goal": self._goal_distance()}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action!r}")
        old_distance = self._goal_distance()
        displacement = self.ACTIONS[int(action)] * self.step_size
        if self.motion_noise:
            displacement += self.rng.normal(0.0, self.motion_noise, 2).astype(np.float32)
        # Splitting a move into substeps avoids 'jumping over' walls.
        n = max(1, math.ceil(float(np.linalg.norm(displacement)) / self.collision_substep))
        delta = displacement / n
        collision = False
        for _ in range(n):
            candidate = self.position + delta
            if not self._is_free(candidate):
                collision = True
                break
            self.position = candidate
        self.steps += 1
        new_distance = self._goal_distance()
        terminated = bool(new_distance <= self.goal_radius)
        truncated = bool(self.steps >= self.max_episode_steps and not terminated)
        # Potential-based shaping: gamma * phi(s') - phi(s), phi=-distance.
        # Terminal potential is zero. Use the agent's discount as shaping_gamma.
        reward = self.step_penalty + self.shaping_scale * (
            old_distance - (0.0 if terminated else self.shaping_gamma * new_distance)
        )
        if terminated:
            reward += self.success_reward
        info: dict[str, Any] = {
            "is_success": terminated,
            "distance_to_goal": new_distance,
            "collision": collision,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def render(self):
        """Return a lightweight 256 x 256 RGB view (no rendering dependency)."""
        scale = 256
        image = np.full((scale, scale, 3), 250, dtype=np.uint8)
        for x0, y0, x1, y1 in self.walls:
            px0, px1 = int(x0*scale), int(x1*scale)
            py0, py1 = int((1-y1)*scale), int((1-y0)*scale)
            image[py0:py1+1, px0:px1+1] = (55, 60, 70)
        for point, color, radius in ((self.goal, (50, 170, 90), self.goal_radius),
                                     (self.position, (230, 90, 60), 0.017)):
            cx, cy = int(point[0]*(scale-1)), int((1-point[1])*(scale-1))
            r = max(2, int(radius*scale))
            ys, xs = np.ogrid[max(0,cy-r):min(scale,cy+r+1), max(0,cx-r):min(scale,cx+r+1)]
            mask = (xs-cx)**2 + (ys-cy)**2 <= r*r
            region = image[max(0,cy-r):min(scale,cy+r+1), max(0,cx-r):min(scale,cx+r+1)]
            region[mask] = color
        return image

    def close(self):
        pass
