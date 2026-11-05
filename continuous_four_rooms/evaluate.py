"""Evaluate a saved DQN checkpoint; optionally save its first trajectory."""
import argparse
import json
from pathlib import Path

from four_rooms import ContinuousFourRoomsEnv, DQNAgent
from four_rooms.experiment import evaluate_policy
from plot_results import plot_trajectory


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved Four Rooms DQN")
    parser.add_argument("--checkpoint", required=True, help="best.pt or last.pt")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=23456)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trajectory", help="Optional path to save example trajectory PNG")
    parser.add_argument("--json-out", help="Optional path to save evaluation metrics as JSON")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    agent, checkpoint = DQNAgent.load(args.checkpoint, device=args.device)
    env = ContinuousFourRoomsEnv(**checkpoint["metadata"]["environment"])
    results = evaluate_policy(agent, env, args.episodes, args.seed, bool(args.trajectory))
    if args.trajectory:
        plot_trajectory(env, results.pop("trajectory"), args.trajectory)
    print(json.dumps(results, indent=2))
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"checkpoint": args.checkpoint, "episodes": args.episodes,
                                      "seed": args.seed, **results}, indent=2), encoding="utf8")
    env.close()


if __name__ == "__main__":
    main()
