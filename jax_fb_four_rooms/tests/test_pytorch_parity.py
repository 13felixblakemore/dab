"""Optional cross-framework numerical regression against the supplied torch equations.

`pytest -q` skips this test when PyTorch is absent: production JAX has no
PyTorch dependency. We use identical weights, inputs and z draws to compare
outputs, all loss terms and the online parameter gradients (not RNG streams).
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

torch = pytest.importorskip("torch")
import torch.nn.functional as tf

from fb_jax.agent import FBConfig, FBAgent, fb_losses, init_adam, adam_update
from fb_jax.features import Features
from fb_jax.networks import forward as jax_forward, backward as jax_backward


def torch_tree(tree, require_grad=False):
    return {name: tuple({k: torch.tensor(np.asarray(layer[k]),
                    dtype=torch.float32, requires_grad=require_grad)
                    for k in ("w", "b")} for layer in layers)
            for name, layers in tree.items()}


def reference_mlp(params, x):
    for layer in params[:-1]:
        x = tf.relu(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


def reference_forward(params, obs, z, d, a):
    z = z / torch.sqrt(1 + torch.linalg.vector_norm(z,dim=-1,keepdim=True)**2 / d)
    raw = reference_mlp(params, torch.cat([obs,z],dim=-1))
    return raw.reshape(-1,d,a)  # critical PyTorch reference ordering


def reference_loss(params, targets, batch, other, z, cfg):
    obs, nxt, action, mask = (batch["obs"],batch["next_obs"],
                              batch["actions"],batch["next_action_mask"])
    with torch.no_grad():
        f_next = reference_forward(targets["forward"],nxt,z,cfg.embed_dim,cfg.n_actions)
        q = torch.einsum('sda,sd->sa',f_next,z)
        q[z.eq(0).all(dim=-1)] = 0
        pi = tf.softmax((q/cfg.temperature).masked_fill(~mask,-float('inf')),dim=-1)
        f_pi = torch.einsum('sda,sa->sd',f_next,pi)
        b_tar = reference_mlp(targets['backward'],other)
        z_next = torch.einsum('sd,td->st',f_pi,b_tar)
    f = reference_forward(params['forward'],obs,z,cfg.embed_dim,cfg.n_actions)
    f = f[torch.arange(len(obs)),:,action]
    b = reference_mlp(params['backward'],obs)
    b_other = reference_mlp(params['backward'],other)
    diag = torch.einsum('sd,sd->s',f,b)
    mat = torch.einsum('sd,td->st',f,b_other)
    fb = 0.5*(mat-cfg.discount_factor*z_next).square().mean()-diag.mean()
    cross = torch.einsum('sd,td->st',b,b_other)
    cross_detach = torch.einsum('sd,td->st',b,b_other.detach())
    b_diag_detach = torch.einsum('sd,sd->s',b,b.detach())
    reg = (cross_detach*cross.detach()).mean()-b_diag_detach.mean()
    return fb + cfg.reg_coef * reg, fb, reg


def test_jax_matches_torch_forward_loss_and_gradients():
    cfg = FBConfig(embed_dim=5,hidden_dim=16,num_layers=1,n_actions=4,
                   obs_dim=2,batch_size=7,max_grad_norm=None)
    agent = FBAgent(cfg,Features(mode='identity'),seed=77)
    rng = np.random.default_rng(101)
    n=cfg.batch_size
    obs = rng.uniform(0,1,(n,2)).astype('float32')
    nxt = rng.uniform(0,1,(n,2)).astype('float32')
    other = rng.uniform(0,1,(n,2)).astype('float32')
    z = rng.normal(0,1,(n,cfg.embed_dim)).astype('float32'); z[1] = 0
    actions = rng.integers(cfg.n_actions,size=n).astype('int32')
    mask = np.ones((n,cfg.n_actions),dtype=bool)
    mask[1,2] = False; mask[4,1] = False
    batch = {'obs':obs,'next_obs':nxt,'actions':actions,
             'next_action_mask':mask,'terminals':np.zeros(n,dtype='float32')}
    tp = torch_tree(agent.params,require_grad=True)
    tt = torch_tree(agent.target_params)
    tb={k:torch.as_tensor(v) for k,v in batch.items()}
    tn=torch.as_tensor(other); tz=torch.as_tensor(z)
    f_t = reference_forward(tp['forward'],tb['obs'],tz,cfg.embed_dim,cfg.n_actions)
    f_j = jax_forward(agent.params['forward'],jnp.array(obs),jnp.array(z),cfg.embed_dim,cfg.n_actions)
    np.testing.assert_allclose(np.asarray(f_j),f_t.detach().numpy().transpose(0,2,1),rtol=3e-6,atol=3e-6)
    loss_t,fb_t,reg_t=reference_loss(tp,tt,tb,tn,tz,cfg)
    loss_t.backward()
    (loss_j,metrics_j),grad_j = jax.value_and_grad(fb_losses,has_aux=True)(
        agent.params,agent.target_params,{k:jnp.asarray(v) for k,v in batch.items()},
        jnp.asarray(other),jax.random.PRNGKey(0),cfg,agent.features,jnp.asarray(z))
    np.testing.assert_allclose(np.asarray(loss_j),loss_t.detach().numpy(),rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(np.asarray(metrics_j['fb_loss']),fb_t.detach().numpy(),rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(np.asarray(metrics_j['reg_loss']),reg_t.detach().numpy(),rtol=2e-5,atol=2e-5)
    for name in ('forward','backward'):
        for jlayer,tlayer in zip(grad_j[name],tp[name]):
            for key in ('w','b'):
                np.testing.assert_allclose(np.asarray(jlayer[key]),tlayer[key].grad.numpy(),rtol=4e-4,atol=4e-5)
    # Validate the exact Adam parameter update against torch.optim.Adam.
    torch_optim = torch.optim.Adam([v for layers in tp.values() for layer in layers
                                    for v in (layer['w'],layer['b'])],lr=cfg.learning_rate)
    torch_optim.step()
    updated_j, _ = adam_update(agent.params,grad_j,init_adam(agent.params),cfg.learning_rate)
    for name in ('forward','backward'):
        for jlayer,tlayer in zip(updated_j[name],tp[name]):
            for key in ('w','b'):
                np.testing.assert_allclose(np.asarray(jlayer[key]),tlayer[key].detach().numpy(),rtol=1e-5,atol=1e-6)


def test_jax_rbf_matches_reference_torch_rbf():
    from fb_jax.features import Features,transform
    feat=Features(mode='rbf',dim=21,sigma=0.05)
    obs=torch.tensor([[0.13,0.57],[0.93,0.25],[0.51,0.52]],dtype=torch.float32)
    xx,yy=torch.meshgrid(torch.linspace(0,1,21),torch.linspace(0,1,21),indexing='xy')
    xm,ym=xx.reshape(-1),yy.reshape(-1)
    dist=(obs[:,0:1]-xm)**2+(obs[:,1:2]-ym)**2
    orig=torch.exp(-dist/(2*feat.sigma**2))
    orig=orig/orig.sum(dim=-1,keepdim=True)
    np.testing.assert_allclose(np.asarray(transform(jnp.asarray(obs.numpy()),feat)),orig.numpy(),rtol=2e-5,atol=1e-7)
