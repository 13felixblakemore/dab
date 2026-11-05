"""YAML loading, training loop, evaluation, CSV logs and checkpoints."""
from __future__ import annotations

import csv
import json
import random
import time
from copy import deepcopy
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from .dqn import DQNAgent, DQNConfig
from .env import ContinuousFourRoomsEnv
from .replay import ReplayBuffer

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _merge_strict(base: dict, changes: dict, prefix: str = "") -> dict:
    for key, value in changes.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected a mapping for {prefix}{key}")
            _merge_strict(base[key], value, prefix=f"{prefix}{key}.")
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict:
    with DEFAULT_CONFIG.open(encoding="utf8") as f:
        base = yaml.safe_load(f)
    if path is None:
        return base
    with Path(path).open(encoding="utf8") as f:
        changes = yaml.safe_load(f) or {}
    if not isinstance(changes, dict):
        raise ValueError("Top-level YAML must be a mapping")
    return _merge_strict(base, changes)


def _valid_config(config: dict) -> None:
    tr = config["training"]
    for k in ("total_steps", "batch_size", "train_frequency", "eval_interval",
              "eval_episodes", "log_interval", "torch_threads"):
        if not isinstance(tr[k], int) or tr[k] < 1:
            raise ValueError(f"training.{k} must be a positive integer")
    if tr["learning_starts"] < 0:
        raise ValueError("training.learning_starts must be nonnegative")
    if tr["device"] not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("training.device must be auto, cpu, cuda or mps")
    if config["environment"]["shaping_scale"] and abs(
        config["environment"]["shaping_gamma"] - config["agent"]["gamma"]
    ) > 1e-9:
        raise ValueError("For potential-based shaping, environment.shaping_gamma must equal agent.gamma")
    DQNConfig(**config["agent"])
    allowed = set(ContinuousFourRoomsEnv.__init__.__code__.co_varnames)
    extra = set(config["environment"]) - allowed
    if extra:
        raise ValueError(f"Unknown environment keys: {sorted(extra)}")


def make_env(config: dict):
    return ContinuousFourRoomsEnv(**config["environment"])


def evaluate_policy(agent: DQNAgent, env: ContinuousFourRoomsEnv, n_episodes: int,
                    seed: int, capture_trajectory: bool = False) -> dict:
    returns, lengths, successes = [], [], []
    trajectory = None
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        states = [obs[:2].copy()] if capture_trajectory and episode == 0 else None
        done = False
        episode_return = 0.0
        episode_length = 0
        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            if states is not None:
                states.append(obs[:2].copy())
            episode_return += reward
            episode_length += 1
            done = terminated or truncated
        returns.append(episode_return)
        lengths.append(episode_length)
        successes.append(float(info["is_success"]))
        if states is not None:
            trajectory = np.array(states)
    result = {"mean_return": float(np.mean(returns)),
              "success_rate": float(np.mean(successes)),
              "mean_length": float(np.mean(lengths))}
    if capture_trajectory:
        result["trajectory"] = trajectory
    return result


def _csv_open(path, columns):
    handle = open(path, "w", encoding="utf8", newline="")
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    return handle, writer


def run_training(config: dict, output_override: str | Path | None = None,
                 plot_override: bool | None = None) -> dict:
    """Train a DQN, checkpoint best/last, evaluate and write reproducible logs."""
    from plot_results import plot_training  # imported lazily; plotting is optional

    config = deepcopy(config)
    _valid_config(config)
    tr = config["training"]
    seed = int(tr["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(tr["torch_threads"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = tr["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if output_override is not None:
        outdir = Path(output_override)
    elif config["output"]["directory"]:
        outdir = Path(config["output"]["directory"])
    else:
        name = f"dqn_{datetime.now():%Y%m%d_%H%M%S}_seed{seed}"
        outdir = Path("results") / name
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf8")

    env = make_env(config)
    eval_env = make_env(config)
    agent_config = DQNConfig(**config["agent"])
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, agent_config, device, seed=seed)
    buffer = ReplayBuffer(agent_config.replay_capacity, env.observation_space.shape[0], seed=seed + 1)

    episode_file, episode_writer = _csv_open(
        outdir / "episodes.csv", ["step", "episode", "return", "length", "success", "epsilon"])
    eval_file, eval_writer = _csv_open(
        outdir / "evaluations.csv", ["step", "mean_return", "success_rate", "mean_length"])
    loss_file, loss_writer = _csv_open(outdir / "losses.csv", ["step", "mean_loss"])
    obs, _ = env.reset(seed=seed)
    episode_return, episode_length, episodes, successes = 0.0, 0, 0, 0
    loss_sum, loss_count = 0.0, 0
    best_eval = (-float("inf"), -float("inf"))
    last_eval = {}
    started = time.perf_counter()
    try:
        for step in range(1, tr["total_steps"] + 1):
            epsilon = agent.epsilon(step - 1)
            action = agent.act(obs, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            buffer.add(obs, action, reward, next_obs, terminated)
            obs = next_obs
            episode_return += reward
            episode_length += 1

            if (step >= tr["learning_starts"] and len(buffer) >= tr["batch_size"]
                    and step % tr["train_frequency"] == 0):
                loss_sum += agent.update(buffer.sample(tr["batch_size"]))
                loss_count += 1

            if terminated or truncated:
                episodes += 1
                successes += int(terminated)
                episode_writer.writerow({"step": step, "episode": episodes,
                    "return": episode_return, "length": episode_length,
                    "success": int(terminated), "epsilon": epsilon})
                obs, _ = env.reset()
                episode_return, episode_length = 0.0, 0

            if step % tr["log_interval"] == 0 or step == tr["total_steps"]:
                if loss_count:
                    loss_writer.writerow({"step": step, "mean_loss": loss_sum / loss_count})
                loss_sum, loss_count = 0.0, 0
                episode_file.flush()
                loss_file.flush()
                print(f"step {step:>7}/{tr['total_steps']} | episodes {episodes:>4} | "
                      f"train_success {successes / max(1, episodes):.1%} | epsilon {epsilon:.3f}", flush=True)

            if step % tr["eval_interval"] == 0 or step == tr["total_steps"]:
                last_eval = evaluate_policy(agent, eval_env, tr["eval_episodes"], seed + 10000)
                eval_writer.writerow({"step": step, **last_eval})
                eval_file.flush()
                score = (last_eval["success_rate"], last_eval["mean_return"])
                if score > best_eval:
                    best_eval = score
                    agent.save(outdir / "best.pt", step, metadata={
                        "environment": config["environment"], "seed": seed})
                print(f"  eval success {last_eval['success_rate']:.1%} | "
                      f"return {last_eval['mean_return']:.3f}", flush=True)

        agent.save(outdir / "last.pt", tr["total_steps"], metadata={
            "environment": config["environment"], "seed": seed})
    finally:
        episode_file.close()
        eval_file.close()
        loss_file.close()
        env.close()
        eval_env.close()

    elapsed = time.perf_counter() - started
    summary = {"total_steps": tr["total_steps"], "episodes": episodes,
        "training_success_rate": successes / max(1, episodes),
        "gradient_updates": agent.updates, "device": str(device),
        "seconds": round(elapsed, 2), "steps_per_second": round(tr["total_steps"] / max(elapsed, 1e-9), 2),
        "final_evaluation": last_eval, "best_evaluation_success_rate": best_eval[0],
        "output_directory": str(outdir)}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    plot = config["output"]["plot_curve"] if plot_override is None else plot_override
    if plot:
        plot_training(outdir, outdir / "training_curve.png", rolling_window=config["output"]["rolling_window"])
    return summary
