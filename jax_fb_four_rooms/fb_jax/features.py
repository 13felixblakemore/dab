"""Differentiable state featurisation, matching the original 21x21 RBF grid."""
from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Features:
    mode: str = "rbf"
    dim: int = 21
    sigma: float = 0.05
    size: float = 1.0

    @property
    def output_dim(self):
        return self.dim**2 if self.mode == "rbf" else 2

    def __post_init__(self):
        if self.mode not in ("rbf", "identity"):
            raise ValueError("mode must be rbf or identity")
        if self.dim < 2 or self.sigma <= 0 or self.size <= 0:
            raise ValueError("Invalid feature grid parameters")


def transform(obs, features:Features):
    obs = jnp.asarray(obs,dtype=jnp.float32)
    if features.mode == "identity":
        return obs
    xs = jnp.linspace(0.0,features.size,features.dim)
    xx, yy = jnp.meshgrid(xs,xs)
    centers = jnp.stack([xx.reshape(-1),yy.reshape(-1)],axis=-1)
    distances = jnp.sum((obs[:,None,:]-centers[None,:,:])**2,axis=-1)
    # logsumexp stabilisation prevents NaNs at small sigma / far points.
    logits = -distances/(2*features.sigma**2)
    weights = jnp.exp(logits-jnp.max(logits,axis=1,keepdims=True))
    return weights/jnp.sum(weights,axis=1,keepdims=True)
