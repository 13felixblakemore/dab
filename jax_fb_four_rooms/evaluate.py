"""Evaluate zero-shot goal navigation from a trained FB checkpoint."""
import argparse
import json
from pathlib import Path
from fb_jax.agent import FBAgent
from fb_jax.navigation import evaluate_navigation,evaluate_random_navigation
from fb_jax.plots import plot_trajectory
import yaml


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--config",default=None,help="Optional YAML with matching environment geometry")
    parser.add_argument("--episodes",type=int,default=30)
    parser.add_argument("--max-steps",type=int,default=120)
    parser.add_argument("--goal-radius",type=float,default=0.07)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output",default=None)
    parser.add_argument("--random-baseline",action="store_true",
                        help="Also evaluate random actions on identical start/goal tasks")
    args=parser.parse_args()
    agent=FBAgent.load(args.checkpoint)
    if args.config:
        with open(args.config,encoding="utf-8") as f:
            env_kwargs=yaml.safe_load(f).get("environment",{})
    else:
        env_kwargs={"size":agent.features.size,"include_stay_action":agent.config.n_actions==5}
    result,example,env=evaluate_navigation(agent,env_kwargs,args.episodes,
                                           args.max_steps,args.seed,goal_radius=args.goal_radius)
    out=Path(args.output) if args.output else Path(args.checkpoint).parent
    out.mkdir(parents=True,exist_ok=True)
    if args.random_baseline:
        result["random_baseline"] = evaluate_random_navigation(
            env_kwargs,args.episodes,args.max_steps,args.seed,args.goal_radius)
    (out/"evaluation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    plot_trajectory(env,example["states"],example["goal"],out/"trajectory.png")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
