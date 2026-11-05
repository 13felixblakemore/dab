"""Train with `python train.py --config configs/default.yaml`."""
import argparse
from fb_jax.experiment import train


def main():
    parser=argparse.ArgumentParser(description="Reward-free JAX FB Four Rooms trainer")
    parser.add_argument("--config",default="configs/default.yaml")
    parser.add_argument("--output",default=None,help="Override training.output_dir")
    args=parser.parse_args()
    out,summary=train(args.config,args.output)
    print(f"Saved outputs to {out}; updates={summary['updates']}; "
          f"time={summary['seconds']:.1f}s")


if __name__=="__main__":
    main()
