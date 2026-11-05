"""Plot DQN training CSV files without rerunning experiments."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path):
    with Path(path).open(newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


def plot_training(run_dir: str | Path, save_path: str | Path | None = None,
                  rolling_window: int = 20) -> Path:
    run_dir = Path(run_dir)
    save_path = Path(save_path) if save_path else run_dir / "training_curve.png"
    episodes = _read_csv(run_dir / "episodes.csv")
    evaluations = _read_csv(run_dir / "evaluations.csv")
    losses = _read_csv(run_dir / "losses.csv")
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, layout="constrained")

    if episodes:
        steps = np.array([float(x["step"]) for x in episodes])
        returns = np.array([float(x["return"]) for x in episodes])
        axes[0].plot(steps, returns, alpha=0.22, lw=0.7, label="Episode return")
        if len(returns) >= rolling_window:
            kernel = np.ones(rolling_window) / rolling_window
            smoothed = np.convolve(returns, kernel, mode="valid")
            axes[0].plot(steps[rolling_window-1:], smoothed, lw=2, label=f"{rolling_window}-episode mean")
        axes[0].legend(loc="best")
    axes[0].set_ylabel("Training return")

    if evaluations:
        steps = [int(x["step"]) for x in evaluations]
        successes = [float(x["success_rate"]) for x in evaluations]
        axes[1].plot(steps, successes, "o-", lw=1.6)
    axes[1].set_ylabel("Eval success rate")
    axes[1].set_ylim(-0.03, 1.03)

    if losses:
        axes[2].plot([int(x["step"]) for x in losses],
                     [float(x["mean_loss"]) for x in losses])
    axes[2].set_xlabel("Environment steps")
    axes[2].set_ylabel("Mean TD loss")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Four Rooms — DQN training", fontsize=14)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_trajectory(env, trajectory: np.ndarray, output: str | Path) -> Path:
    """Save map with doors, wall segments, goal and agent evaluation trajectory."""
    from matplotlib.patches import Rectangle, Circle
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    for i, (x0, y0, x1, y1) in enumerate(env.walls):
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, color="#313945",
                               label="Walls" if i == 0 else None))
    ax.add_patch(Circle(env.goal, env.goal_radius, color="#58b578", alpha=.5, label="Goal"))
    ax.plot(trajectory[:, 0], trajectory[:, 1], linewidth=1.4, color="#227dc5", label="Path")
    ax.scatter(*trajectory[0], marker="o", c="orange", s=50, label="Start", zorder=5)
    ax.scatter(*trajectory[-1], marker="x", c="red", s=70, label="Final", zorder=5)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="x", ylabel="y", title="Greedy evaluation trajectory")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot CSV training results")
    parser.add_argument("run_dir", help="Directory created by train.py")
    parser.add_argument("--output", help="Save path (default: run_dir/training_curve.png)")
    parser.add_argument("--window", type=int, default=20, help="Training-return rolling window")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be >= 1")
    print(plot_training(args.run_dir, args.output, args.window))


if __name__ == "__main__":
    main()
