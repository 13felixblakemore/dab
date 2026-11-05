"""Train with: python train.py --config configs/default.yaml [--plot]"""
import argparse
import json

from four_rooms.experiment import DEFAULT_CONFIG, load_config, run_training


def main():
    parser = argparse.ArgumentParser(description="Train DQN on Continuous Four Rooms")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config (may override a subset of defaults)")
    parser.add_argument("--output", help="Override output directory")
    plot = parser.add_mutually_exclusive_group()
    plot.add_argument("--plot", action="store_true", help="Plot learning curve at end")
    plot.add_argument("--no-plot", action="store_true", help="Do not plot learning curve")
    args = parser.parse_args()
    plot_override = True if args.plot else False if args.no_plot else None
    summary = run_training(load_config(args.config), output_override=args.output, plot_override=plot_override)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
