import csv
import json
from pathlib import Path

import pytest

from four_rooms import DQNAgent
from four_rooms.experiment import evaluate_policy, load_config, make_env, run_training
from plot_results import plot_training


def test_yaml_merges_and_rejects_typos(tmp_path):
    p = tmp_path / "change.yaml"
    p.write_text("training:\n  total_steps: 30\n", encoding="utf8")
    cfg = load_config(p)
    assert cfg["training"]["total_steps"] == 30
    assert cfg["agent"]["double_dqn"] is True
    p.write_text("training:\n  total_stpes: 30\n", encoding="utf8")
    with pytest.raises(ValueError, match="Unknown configuration key"):
        load_config(p)


def test_end_to_end_smoke(tmp_path):
    cfg = load_config()
    cfg["training"].update(total_steps=40, eval_interval=20, eval_episodes=2,
                           learning_starts=8, batch_size=8, train_frequency=1,
                           log_interval=10)
    cfg["agent"].update(hidden_sizes=[16, 16], replay_capacity=80,
                        target_update_interval=5)
    cfg["environment"].update(max_episode_steps=10)
    cfg["output"].update(plot_curve=True, rolling_window=2)
    run_dir = tmp_path / "integration"
    summary = run_training(cfg, output_override=run_dir)
    assert summary["gradient_updates"] > 0
    assert summary["total_steps"] == 40
    for name in ("best.pt", "last.pt", "episodes.csv", "losses.csv",
                 "evaluations.csv", "summary.json", "training_curve.png", "resolved_config.yaml"):
        assert (run_dir / name).stat().st_size > 0
    with (run_dir / "evaluations.csv").open(newline="") as f:
        evaluations = list(csv.DictReader(f))
    assert len(evaluations) == 2
    assert (run_dir / "training_curve2.png") == plot_training(run_dir, run_dir / "training_curve2.png")
    agent, ckpt = DQNAgent.load(run_dir / "best.pt")
    assert ckpt["metadata"]["seed"] == cfg["training"]["seed"]
    evaluated = evaluate_policy(agent, make_env(cfg), n_episodes=2, seed=11, capture_trajectory=True)
    assert 0 <= evaluated["success_rate"] <= 1
    assert evaluated["trajectory"].shape[1] == 2
    assert json.loads((run_dir / "summary.json").read_text())["total_steps"] == 40
