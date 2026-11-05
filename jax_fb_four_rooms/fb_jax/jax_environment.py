"""Pure JAX, vectorised continuous Four Rooms dynamics.

Coordinates, noisy actions, wall slab intersection, goals and reset semantics
mirror `environment.py`; RNG streams differ (JAX vs NumPy). No Python loops per step.
"""
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .environment import classic_walls


@dataclass(frozen=True)
class JAXFourRoomsConfig:
    size: float = 1.0
    step_size: float = 0.05
    movement_noise: float = 0.005
    explorable: bool = False
    include_stay_action: bool = False
    threshold_distance: float = 0.1

    def __post_init__(self):
        if self.size <= 0 or self.step_size <= 0 or self.movement_noise < 0:
            raise ValueError("Invalid environment dimensions/noise")

    @property
    def n_actions(self):
        return 5 if self.include_stay_action else 4


class RolloutState(NamedTuple):
    positions: jax.Array  # (num_envs, 2)
    goals: jax.Array      # (num_envs, 2), ignored if explorable
    episode_steps: jax.Array  # (num_envs,)
    key: jax.Array


def _walls(cfg):
    return jnp.asarray([[b.xmin,b.ymin,b.xmax,b.ymax] for b in classic_walls(cfg.size)],
                       dtype=jnp.float32)


def is_free_batch(positions, walls, size):
    """Check a batch of positions against wall rectangles and strict edges."""
    p = jnp.asarray(positions)
    within = jnp.all((p > 0) & (p < size) & jnp.isfinite(p),axis=-1)
    inside = jnp.all((p[:,None,:] >= walls[None,:,:2]) &
                     (p[:,None,:] <= walls[None,:,2:]),axis=-1)
    return within & ~jnp.any(inside,axis=1)


def sample_free_points(key, num, walls, size):
    """Uniform rejection sampling of free points; vectorised over environments."""
    candidates=jnp.zeros((num,2),dtype=jnp.float32)
    pending=jnp.ones((num,),dtype=jnp.bool_)

    def cond(carry):
        return jnp.any(carry[2])

    def body(carry):
        key,points,pending=carry
        key,sub=jax.random.split(key)
        proposal=jax.random.uniform(sub,(num,2),minval=1e-6,maxval=size-1e-6)
        accept=pending & is_free_batch(proposal,walls,size)
        points=jnp.where(accept[:,None],proposal,points)
        return key,points,pending & ~accept

    key,points,_=jax.lax.while_loop(cond,body,(key,candidates,pending))
    return key,points


def reset_all(key, num_envs, cfg):
    walls=_walls(cfg)
    key,positions=sample_free_points(key,num_envs,walls,cfg.size)
    if cfg.explorable:
        goals=jnp.zeros_like(positions)
    else:
        key,goals=sample_free_points(key,num_envs,walls,cfg.size)
    return RolloutState(positions,goals,jnp.zeros(num_envs,dtype=jnp.int32),key)


def segment_collisions(start, end, walls):
    """Vectorised closed-box segment/wall collision, matching WallBlock's slab test.

    start/end: (N,2); walls: (W,4). Touching a wall face without
    nonzero segment overlap isn't counted as collision.
    """
    delta=end-start
    low=walls[:,:2]
    high=walls[:,2:]
    d=delta[:,None,:]  # (N,1,2)
    p=start[:,None,:]
    almost_zero=jnp.abs(d)<1e-9
    outside=almost_zero & ((p<low[None,:,:]) | (p>high[None,:,:]))
    safe_d=jnp.where(almost_zero,1.0,d)
    t1=(low[None,:,:]-p)/safe_d
    t2=(high[None,:,:]-p)/safe_d
    t_enter=jnp.where(almost_zero,-jnp.inf,jnp.minimum(t1,t2))
    t_leave=jnp.where(almost_zero,jnp.inf,jnp.maximum(t1,t2))
    tmin=jnp.maximum(0.0,jnp.max(t_enter,axis=-1))
    tmax=jnp.minimum(1.0,jnp.min(t_leave,axis=-1))
    endpoint_inside=jnp.all((end[:,None,:]>=low[None,:,:]+1e-6) &
                            (end[:,None,:]<=high[None,:,:]-1e-6),axis=-1)
    intersects=(~jnp.any(outside,axis=-1)) & ((tmax>1e-6)&((tmax-tmin)>1e-6))
    return jnp.any(intersects|endpoint_inside,axis=1)


def step_batch(state, actions, cfg, horizon):
    """One simultaneous environment step, with done/horizon auto-resets.

    Returns the physical transition end before auto-reset, so replay never
    contains a transition to the reset state.
    """
    n=state.positions.shape[0]
    walls=_walls(cfg)
    key,noise_key=jax.random.split(state.key)
    moves=jnp.array([[-1,0],[1,0],[0,1],[0,-1],[0,0]],dtype=jnp.float32)*cfg.step_size
    noise=jax.random.normal(noise_key,(n,2))*cfg.movement_noise
    proposed=jnp.clip(state.positions+moves[actions]+noise,0.0,cfg.size)
    collisions=segment_collisions(state.positions,proposed,walls)
    after=jnp.where(collisions[:,None],state.positions,proposed)
    done=(jnp.max(jnp.abs(after-state.goals),axis=-1)<=cfg.threshold_distance)
    if cfg.explorable:
        done=jnp.zeros_like(done)
    steps=state.episode_steps+1
    resets=done | (steps>=horizon)

    # Sampling every step is inexpensive after JIT, and keeps reset sampling
    # fully vectorised and independent of Python control flow.
    key,new_pos=sample_free_points(key,n,walls,cfg.size)
    if cfg.explorable:
        new_goals=state.goals
    else:
        key,new_goals=sample_free_points(key,n,walls,cfg.size)
    new_state=RolloutState(jnp.where(resets[:,None],new_pos,after),
                           jnp.where(resets[:,None],new_goals,state.goals),
                           jnp.where(resets,0,steps),key)
    return new_state, {"obs":state.positions,"actions":actions.astype(jnp.int32),
                       "next_obs":after,"done":done,"collisions":collisions}


def rollout(state, steps_per_env, cfg, horizon, terminal_mask=False):
    """Compiled-scan-friendly uniform random data collection."""
    n=state.positions.shape[0]
    def body(state,_):
        key,act_key=jax.random.split(state.key)
        state=state._replace(key=key)
        actions=jax.random.randint(act_key,(n,),0,cfg.n_actions)
        state,step=step_batch(state,actions,cfg,horizon)
        return state,step
    state,steps=jax.lax.scan(body,state,None,length=steps_per_env)
    flat=lambda a:jnp.reshape(a,(-1,)+a.shape[2:])
    data={"obs":flat(steps["obs"]),"actions":flat(steps["actions"]),
          "next_obs":flat(steps["next_obs"]),
          "terminals":flat(steps["done"]).astype(jnp.float32)*float(terminal_mask),
          "next_action_mask":jnp.ones((steps_per_env*n,cfg.n_actions),dtype=jnp.bool_)}
    return state,data,jnp.mean(steps["collisions"].astype(jnp.float32))
