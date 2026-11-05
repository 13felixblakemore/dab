"""Functional MLPs; no Flax or Haiku required."""
import jax
import jax.numpy as jnp


def init_mlp(key, dims):
    keys = jax.random.split(key, len(dims)-1)
    params = []
    for k, din, dout in zip(keys,dims[:-1],dims[1:]):
        kw,kb = jax.random.split(k)
        bound = din ** -0.5  # PyTorch nn.Linear default uniform init.
        params.append({"w":jax.random.uniform(kw,(din,dout),minval=-bound,maxval=bound),
                       "b":jax.random.uniform(kb,(dout,),minval=-bound,maxval=bound)})
    return tuple(params)


def apply_mlp(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    last = params[-1]
    return x @ last["w"] + last["b"]


def forward(params, obs_features, z, embed_dim, n_actions):
    z = z / jnp.sqrt(1+jnp.sum(z*z,axis=-1,keepdims=True)/embed_dim)
    raw = apply_mlp(params,jnp.concatenate([obs_features,z],axis=-1))
    # Match PyTorch: raw.reshape(batch, embed_dim, n_actions), then
    # present action-major axes to callers as (batch, n_actions, embed_dim).
    return jnp.transpose(raw.reshape((-1,embed_dim,n_actions)), (0,2,1))


def backward(params, obs_features):
    return apply_mlp(params,obs_features)
