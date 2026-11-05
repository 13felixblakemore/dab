from pathlib import Path

import numpy as np
import torch

from four_rooms import DQNAgent, DQNConfig, ReplayBuffer


def test_replay_buffer_wrap_and_sample():
    buffer = ReplayBuffer(capacity=3, obs_dim=4, seed=12)
    obs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    for action in range(5):
        buffer.add(obs * action, action % 4, action * .1, obs, action == 4)
    assert len(buffer) == 3
    result = buffer.sample(20)
    assert len(result) == 5
    assert result[0].shape == (20, 4)
    assert result[1].dtype == np.int64
    assert result[-1].dtype == np.float32


def test_dqn_update_increases_q_for_positive_terminal_reward(tmp_path: Path):
    torch.manual_seed(1)
    config = DQNConfig(hidden_sizes=(16,), target_update_interval=2)
    agent = DQNAgent(4, 4, config, seed=1)
    for p in agent.online.parameters():
        torch.nn.init.zeros_(p)
    agent.target.load_state_dict(agent.online.state_dict())
    obs = np.zeros((10, 4), dtype=np.float32)
    batch = (obs, np.zeros(10, dtype=np.int64), np.ones(10, dtype=np.float32),
             obs, np.ones(10, dtype=np.float32))
    before = agent.online(torch.zeros(1, 4))[0, 0].item()
    assert agent.update(batch) > 0
    after = agent.online(torch.zeros(1, 4))[0, 0].item()
    assert after > before
    assert agent.updates == 1
    agent.update(batch)
    assert all(torch.equal(x, y) for x, y in zip(agent.online.parameters(), agent.target.parameters()))

    path = tmp_path / "agent.pt"
    agent.save(path, global_step=123, metadata={"environment": {}, "seed": 1})
    reloaded, checkpoint = DQNAgent.load(path)
    assert checkpoint["global_step"] == 123
    assert reloaded.updates == 2
    assert reloaded.act(np.zeros(4, dtype=np.float32)) == agent.act(np.zeros(4, dtype=np.float32))


def test_epsilon_anneals():
    config = DQNConfig(epsilon_start=1.0, epsilon_final=.1, epsilon_decay_steps=10)
    agent = DQNAgent(4, 4, config)
    assert agent.epsilon(0) == 1.0
    assert np.isclose(agent.epsilon(5), .55)
    assert agent.epsilon(100) == .1
