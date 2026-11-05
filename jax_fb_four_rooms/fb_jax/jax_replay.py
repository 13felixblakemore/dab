"""Device-resident replay buffer for fully JIT-compiled JAX training."""
from typing import NamedTuple
import jax
import jax.numpy as jnp


class DeviceReplay(NamedTuple):
    obs: jax.Array
    next_obs: jax.Array
    actions: jax.Array
    terminals: jax.Array
    next_action_mask: jax.Array
    size: jax.Array
    ptr: jax.Array


def init_device_replay(capacity, obs_dim, n_actions):
    return DeviceReplay(jnp.zeros((capacity,obs_dim),jnp.float32),
                        jnp.zeros((capacity,obs_dim),jnp.float32),
                        jnp.zeros((capacity,),jnp.int32),
                        jnp.zeros((capacity,),jnp.float32),
                        jnp.ones((capacity,n_actions),jnp.bool_),
                        jnp.array(0,jnp.int32),jnp.array(0,jnp.int32))


def add_many(replay, data):
    count=data["actions"].shape[0]
    capacity=replay.obs.shape[0]
    if count>capacity:
        raise ValueError("Device replay capacity must fit at least one epoch")
    ids=(replay.ptr+jnp.arange(count,dtype=jnp.int32))%capacity
    return DeviceReplay(replay.obs.at[ids].set(data["obs"]),
                        replay.next_obs.at[ids].set(data["next_obs"]),
                        replay.actions.at[ids].set(data["actions"]),
                        replay.terminals.at[ids].set(data["terminals"]),
                        replay.next_action_mask.at[ids].set(data["next_action_mask"]),
                        jnp.minimum(capacity,replay.size+count),
                        (replay.ptr+count)%capacity)


def sample(replay, indices):
    return {"obs":replay.obs[indices],"next_obs":replay.next_obs[indices],
            "actions":replay.actions[indices],"terminals":replay.terminals[indices],
            "next_action_mask":replay.next_action_mask[indices]}
