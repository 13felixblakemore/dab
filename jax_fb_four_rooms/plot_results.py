"""Standalone training loss plotter."""
import argparse
from pathlib import Path
from fb_jax.plots import plot_training

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("results_dir",help="Directory containing training_metrics.csv")
    args=p.parse_args()
    out=Path(args.results_dir)
    plot_training(out/"training_metrics.csv",out/"training_curve.png")
    print(f"Saved {out/'training_curve.png'}")
