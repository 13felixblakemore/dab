"""Configuration, data collection, and reward-free FB training."""
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import time

import numpy as np
import yaml

from .agent import FBAgent, FBConfig
from .environment import ContinuousFourRooms
from .features import Features
from .replay import ReplayBuffer


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    transitions_per_epoch: int = 1500
    rollout_horizon: int = 30
    updates_per_epoch: int = 40
    min_buffer_size: int = 128
    buffer_size: int = 200_000
    seed: int = 1
    output_dir: str = "results/default"
    checkpoint_every: int = 20
    plot: bool = True
    use_terminal_bootstrap_mask: bool = False  # reference passes terminal=False
    backend: str = "jax"  # jax, numpy_scan, or numpy_legacy
    num_envs: int = 25   # JAX vectorised independent environments

    def __post_init__(self):
        for name in ("epochs","transitions_per_epoch","rollout_horizon","updates_per_epoch",
                     "min_buffer_size","buffer_size"):
            if getattr(self,name)<1:
                raise ValueError(f"{name} must be positive")
        if self.backend not in ("jax","numpy_scan","numpy_legacy"):
            raise ValueError("backend must be jax, numpy_scan or numpy_legacy")
        if self.num_envs < 1 or (self.backend=="jax" and
                                  self.transitions_per_epoch % self.num_envs != 0):
            raise ValueError("jax backend needs num_envs dividing transitions_per_epoch")
        if self.backend=="jax" and self.buffer_size < self.transitions_per_epoch:
            raise ValueError("jax backend needs buffer_size >= transitions_per_epoch")
        if self.checkpoint_every<0:
            raise ValueError("checkpoint_every cannot be negative")


def load_configuration(path):
    with open(path,"r",encoding="utf-8") as file:
        config=yaml.safe_load(file)
    if not isinstance(config,dict):
        raise ValueError("YAML root must be a dictionary")
    env_args=config.get("environment",{})
    features=Features(**config.get("features",{}))
    agent_cfg=FBConfig(**config.get("agent",{}))
    trainer=TrainingConfig(**config.get("training",{}))
    if agent_cfg.n_actions != (5 if env_args.get("include_stay_action",False) else 4):
        raise ValueError("agent.n_actions must match environment.include_stay_action")
    if agent_cfg.obs_dim != 2:
        raise ValueError("Four Rooms observations are 2-dimensional")
    if abs(features.size-env_args.get("size",1.0))>1e-6:
        raise ValueError("features.size must match environment.size")
    if trainer.min_buffer_size < agent_cfg.batch_size:
        raise ValueError("training.min_buffer_size must be >= agent.batch_size")
    return env_args,features,agent_cfg,trainer


def train(config_path,override_output=None,verbose=True):
    env_args,features,agent_cfg,training=load_configuration(config_path)
    # Reference run_fb.py uses explorable=False (random goals) and resets on
    # goal hits, while explicitly storing terminal=False. This is configurable.
    env_args={**env_args,"seed":training.seed}
    agent=FBAgent(agent_cfg,features,seed=training.seed)
    if training.backend=="jax":
        import jax
        from .jax_environment import JAXFourRoomsConfig,reset_all
        from .jax_replay import init_device_replay
        from .agent import device_epoch
        cfg=JAXFourRoomsConfig(**{k:v for k,v in env_args.items() if k!="seed"})
        rollout_state=reset_all(jax.random.PRNGKey(training.seed+101),
                                training.num_envs,cfg)
        replay=init_device_replay(training.buffer_size,2,cfg.n_actions)
        run_epoch=jax.jit(lambda p,t,opt,rep,state,key:
            device_epoch(p,t,opt,rep,state,key,cfg,agent_cfg,features,
                         training.transitions_per_epoch//training.num_envs,
                         training.updates_per_epoch,training.rollout_horizon,
                         training.use_terminal_bootstrap_mask,training.min_buffer_size))
    else:
        env=ContinuousFourRooms(**env_args)
        rng=np.random.default_rng(training.seed+101)
        replay=ReplayBuffer(training.buffer_size,2,env.n_actions,seed=training.seed+202)
        obs=env.reset()
        episode_t=0
    out=Path(override_output or training.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    with (out/"resolved_config.yaml").open("w",encoding="utf-8") as f:
        yaml.safe_dump({"environment":env_args,"features":asdict(features),
                        "agent":asdict(agent_cfg),"training":asdict(training)},f,sort_keys=False)
    metrics=[]
    started=time.perf_counter()
    for epoch in range(1,training.epochs+1):
        if training.backend=="jax":
            (agent.params,agent.target_params,agent.opt_state,replay,
             rollout_state,agent.key,epoch_metrics,collision_rate)=run_epoch(
                agent.params,agent.target_params,agent.opt_state,replay,
                rollout_state,agent.key)
            # Synchronise once per epoch, not once per gradient update.
            epoch_metrics={k:float(v) for k,v in epoch_metrics.items()}
            collisions=float(collision_rate)
            replay_size=int(replay.size)
            if replay_size>=training.min_buffer_size:
                agent.updates+=training.updates_per_epoch
        else:
            collisions=0
            for _ in range(training.transitions_per_epoch):
                action=int(rng.integers(env.n_actions))
                next_obs,_,done,info=env.step(action)
                replay.add(obs,action,next_obs,
                           terminal=(done and training.use_terminal_bootstrap_mask),
                           next_actions=env.get_available_actions(next_obs))
                collisions+=int(info["goes_through_wall"])
                obs=next_obs
                episode_t+=1
                if done or episode_t>=training.rollout_horizon:
                    obs=env.reset()
                    episode_t=0
            collisions/=training.transitions_per_epoch
            replay_size=replay.size
            epoch_metrics={}
            if replay.size>=training.min_buffer_size:
                if training.backend=="numpy_scan":
                    epoch_metrics=agent.update_many(replay,training.updates_per_epoch)
                else:
                    history=[agent.update(replay) for _ in range(training.updates_per_epoch)]
                    epoch_metrics={k:float(np.mean([row[k] for row in history]))
                                   for k in history[0]}
                agent.soft_update()
        record={"epoch":epoch,"steps":epoch*training.transitions_per_epoch,
                "updates":agent.updates,"buffer_size":replay_size,
                "collision_rate":collisions}
        if replay_size>=training.min_buffer_size:
            record.update(epoch_metrics)
        metrics.append(record)
        if verbose and (epoch==1 or epoch%max(1,training.epochs//20)==0):
            print(f"epoch={epoch:4d}/{training.epochs} steps={record['steps']:7d} "
                  f"loss={record.get('loss',float('nan')):10.4f} "
                  f"buffer={replay_size:7d}",flush=True)
        if training.checkpoint_every and epoch%training.checkpoint_every==0:
            agent.save(out/f"fb_epoch_{epoch:04d}.npz")
    # JAX dispatch is asynchronous; saving copies parameters back to the host.
    agent.save(out/"fb_final.npz")
    keys=list(metrics[0].keys())
    with (out/"training_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)
    summary={"epochs":training.epochs,"total_steps":training.epochs*training.transitions_per_epoch,
             "updates":agent.updates,"seconds":time.perf_counter()-started,
             "steps_per_second":training.epochs*training.transitions_per_epoch/(time.perf_counter()-started),
             "final_loss":metrics[-1].get("loss"),"checkpoint":str(out/"fb_final.npz")}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    if training.plot:
        from .plots import plot_training
        plot_training(out/"training_metrics.csv",out/"training_curve.png")
    return out,summary
