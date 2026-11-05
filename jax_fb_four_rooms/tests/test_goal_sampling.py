"""Goal-directed training mode: z = stop_gradient(B(s_goal)) from replay."""
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import pytest
import yaml

from fb_jax.agent import FBAgent, FBConfig, fb_losses, scanned_train_steps
from fb_jax.features import Features, transform
from fb_jax.networks import backward
from fb_jax.replay import ReplayBuffer
from fb_jax.experiment import train


def make_agent(mode='replay_goal'):
    return FBAgent(FBConfig(batch_size=8, hidden_dim=16, embed_dim=5, num_layers=1,
                            max_grad_norm=10., z_sampling=mode),
                   Features(mode='identity'),seed=13)


def make_replay():
    r = ReplayBuffer(100,2,4,seed=27)
    rng = np.random.default_rng(31)
    for i in range(60):
        r.add(rng.random(2),int(rng.integers(4)),rng.random(2))
    return r


def test_config_rejects_invalid_mode():
    with pytest.raises(ValueError,match='z_sampling'):
        FBConfig(z_sampling='wrong')


def test_goal_loss_equals_explicit_backward_embedding_and_is_detached():
    agent=make_agent()
    rb=make_replay()
    batch=rb.sample(8)
    other=rb.sample(8)['obs']
    goals=rb.sample(8)['obs']
    z=jax.lax.stop_gradient(backward(agent.params['backward'],transform(goals,agent.features)))
    key=jax.random.PRNGKey(11)
    loss_goal,metrics_goal=fb_losses(agent.params,agent.target_params,batch,other,key,
                                    agent.config,agent.features,goal_obs=goals)
    loss_fixed,metrics_fixed=fb_losses(agent.params,agent.target_params,batch,other,key,
                                      agent.config,agent.features,fixed_z=z)
    np.testing.assert_allclose(np.asarray(loss_goal),np.asarray(loss_fixed),rtol=1e-6)
    for k in metrics_goal:
        np.testing.assert_allclose(np.asarray(metrics_goal[k]),np.asarray(metrics_fixed[k]),rtol=1e-6)
    # A fixed z should yield identical gradients if B(goal) is detached.
    grad_goal=jax.grad(lambda p:fb_losses(p,agent.target_params,batch,other,key,
                                          agent.config,agent.features,goal_obs=goals)[0])(agent.params)
    grad_fixed=jax.grad(lambda p:fb_losses(p,agent.target_params,batch,other,key,
                                           agent.config,agent.features,fixed_z=z)[0])(agent.params)
    for x,y in zip(jax.tree.leaves(grad_goal),jax.tree.leaves(grad_fixed)):
        np.testing.assert_allclose(np.asarray(x),np.asarray(y),rtol=1e-5,atol=1e-5)
    with pytest.raises(ValueError,match='goal_obs'):
        fb_losses(agent.params,agent.target_params,batch,other,key,agent.config,agent.features)


def test_cpu_buffer_independent_goal_samples_and_single_update(tmp_path):
    rb=make_replay()
    b,o,g=rb.sample_many(4,8,with_goals=True)
    assert b['obs'].shape==(4,8,2) and o.shape==(4,8,2) and g.shape==(4,8,2)
    assert not np.array_equal(o,g)
    a=make_agent()
    result=a.update(rb)
    assert all(np.isfinite(x) for x in result.values())
    result=a.update_many(rb,3)
    assert all(np.isfinite(x) for x in result.values())
    assert a.updates==4
    a.save(tmp_path/'goal.npz')
    loaded=FBAgent.load(tmp_path/'goal.npz')
    assert loaded.config.z_sampling=='replay_goal'
    assert np.isfinite(loaded.update(rb)['loss'])


@pytest.mark.parametrize('backend',['jax','numpy_scan','numpy_legacy'])
def test_goal_training_end_to_end(tmp_path,backend):
    from pathlib import Path
    source=Path(__file__).resolve().parents[1]/'configs'/'smoke.yaml'
    config=yaml.safe_load(source.read_text())
    config['agent']['z_sampling']='replay_goal'
    config['training'].update({'backend':backend,'plot':False,'checkpoint_every':0})
    p=tmp_path/f'{backend}.yaml'
    p.write_text(yaml.safe_dump(config))
    out,summary=train(p,tmp_path/backend,verbose=False)
    assert summary['updates']==4
    assert np.isfinite(summary['final_loss'])
    assert (out/'fb_final.npz').exists()
    assert FBAgent.load(out/'fb_final.npz').config.z_sampling=='replay_goal'
