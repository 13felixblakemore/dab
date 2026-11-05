import numpy as np
import pytest

from four_rooms import ContinuousFourRoomsEnv


def test_observations_and_step_api():
    env = ContinuousFourRoomsEnv(start_noise=0)
    obs, info = env.reset(seed=123)
    assert obs.shape == (4,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info["distance_to_goal"] > 0
    for action in range(4):
        obs, rew, done, trunc, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isfinite(rew)
        assert isinstance(done, bool) and isinstance(trunc, bool)
    assert env.render().shape == (256, 256, 3)
    with pytest.raises(ValueError):
        env.step(9)


def test_seeding_reproducibility():
    env = ContinuousFourRoomsEnv(start_noise=0.06, motion_noise=0.004)
    trajectories = []
    for _ in range(2):
        obs, _ = env.reset(seed=99)
        states = [obs]
        for action in (0, 2, 2, 1, 3, 0):
            obs, *_ = env.step(action)
            states.append(obs)
        trajectories.append(np.array(states))
    np.testing.assert_array_equal(trajectories[0], trajectories[1])


def test_wall_blocks_and_no_tunnelling():
    env = ContinuousFourRoomsEnv(start_noise=0, step_size=0.3)
    env.reset(options={"start_pos": [0.39, 0.15]})
    obs, _, _, _, info = env.step(0)  # horizontal right into vertical wall
    assert info["collision"]
    assert obs[0] < 0.5-env.wall_thickness/2-env.agent_radius+1e-6
    assert obs[1] == pytest.approx(0.15, abs=1e-6)


def test_doorways_passage_horizontal_and_vertical():
    env = ContinuousFourRoomsEnv(start_noise=0, step_size=0.045)
    env.reset(options={"start_pos": [0.40, 0.25]})
    for _ in range(6):
        obs, *_ = env.step(0)
    assert obs[0] > 0.5  # right into lower-right room through vertical door

    env.reset(options={"start_pos": [0.25, 0.40]})
    for _ in range(6):
        obs, *_ = env.step(2)
    assert obs[1] > 0.5  # up into top-left room through horizontal door


def test_termination_truncation_and_goal_reward():
    env = ContinuousFourRoomsEnv(start_noise=0, max_episode_steps=2,
                                 start_pos=(0.79, 0.85), step_penalty=-0.01)
    env.reset()
    _, reward, terminal, truncated, info = env.step(0)
    assert terminal and not truncated and info["is_success"]
    assert reward == pytest.approx(0.99)

    env = ContinuousFourRoomsEnv(start_noise=0, max_episode_steps=2)
    env.reset()
    _, _, terminated, truncated, _ = env.step(3)
    assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(3)
    assert not terminated and truncated


def test_blocked_start_and_geometry():
    with pytest.raises(ValueError):
        ContinuousFourRoomsEnv(start_pos=(0.5, 0.1))
    env = ContinuousFourRoomsEnv()
    with pytest.raises(ValueError):
        env.reset(options={"start_pos": [0.5, 0.5]})
    assert len(env.walls) == 6


def test_complete_door_to_door_route_reaches_goal():
    env = ContinuousFourRoomsEnv(start_noise=0)
    env.reset(seed=0)
    # Lower-left -> upper-left (through x=0.25 gap) -> upper-right (y=0.75 gap).
    actions = [0]*2 + [2]*14 + [0]*14 + [2]*2
    terminal = False
    for action in actions:
        _, _, terminal, truncated, info = env.step(action)
        assert not truncated
        assert not info["collision"]
        if terminal:
            break
    assert terminal
    assert info["is_success"]
