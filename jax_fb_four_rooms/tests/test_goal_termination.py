"""Goal-dependent bootstrapping and optional state-action successor measures."""
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from fb_jax.agent import FBAgent, FBConfig, fb_losses, goal_bootstrap_mask, _b_embedding
from fb_jax.features import Features, transform
from fb_jax.experiment import train
from fb_jax.networks import backward
from fb_jax.replay import ReplayBuffer


def test_goal_mask_euclidean_boundary_and_independent_rows():
    next_states = jnp.array([[0.2, 0.3], [0.5, 0.5], [0.9, 0.5], [0.1, 0.1]])
    goals = jnp.array([[0.2, 0.3], [0.53, 0.54], [0.5, 0.5], [0.1, 0.1]])
    mask = goal_bootstrap_mask(next_states, goals, 0.1)
    np.testing.assert_array_equal(mask, [0, 0, 1, 0])
    # Physical terminal signals should ALSO stop the bootstrap.
    mask = goal_bootstrap_mask(next_states, goals, 0.1, jnp.array([0., 0., 1., 0.]))
    np.testing.assert_array_equal(mask, [0, 0, 0, 0])
    jit = jax.jit(lambda x, y: goal_bootstrap_mask(x, y, 0.1))
    np.testing.assert_array_equal(jit(next_states, goals), [0, 0, 1, 0])


def make_batch(batch_size=4):
    return {'obs': jnp.array([[0.1,0.2],[0.3,0.1],[0.4,0.7],[0.8,0.7]],jnp.float32),
            'actions':jnp.array([0,1,2,3],jnp.int32),
            'next_obs':jnp.array([[0.2,0.3],[0.5,0.1],[0.5,0.7],[0.8,0.8]],jnp.float32),
            'terminals':jnp.zeros(batch_size,jnp.float32),
            'next_action_mask':jnp.ones((batch_size,4),jnp.bool_)}


def test_goal_td_mask_applies_to_rows_not_columns():
    cfg=FBConfig(batch_size=4,embed_dim=5,hidden_dim=12,num_layers=0,
                 z_sampling='replay_goal',terminate_on_goal=True,goal_tolerance=0.05)
    agent=FBAgent(cfg,Features(mode='identity'),seed=5)
    batch=make_batch()
    other=jnp.array([[.2,.2],[.7,.8],[.1,.8],[.2,.7]],jnp.float32)
    goals=jnp.array([[.2,.3],[.8,.5],[.5,.7],[.2,.2]],jnp.float32)
    key=jax.random.PRNGKey(11)
    loss_masked,metrics=fb_losses(agent.params,agent.target_params,batch,other,key,cfg,
                                 agent.features,goal_obs=goals)
    assert np.isfinite(loss_masked)
    np.testing.assert_allclose(float(metrics['goal_hit_rate']), 0.5)

    cfg_without_mask=replace(cfg,terminate_on_goal=False)
    loss_unmasked,_=fb_losses(agent.params,agent.target_params,batch,other,key,
                              cfg_without_mask,agent.features,goal_obs=goals)
    assert abs(float(loss_masked - loss_unmasked)) > 1e-8
    # If all next states are goals, changing the target networks cannot change
    # the masked TD loss, proving their contribution is genuinely zero.
    all_goals=batch['next_obs']
    loss1,_=fb_losses(agent.params,agent.target_params,batch,other,key,cfg,
                      agent.features,goal_obs=all_goals)
    alternate_targets=jax.tree.map(lambda x:x*500,agent.target_params)
    loss2,_=fb_losses(agent.params,alternate_targets,batch,other,key,cfg,
                      agent.features,goal_obs=all_goals)
    np.testing.assert_allclose(loss1,loss2,atol=1e-6)
    # But the non-bootstrapped diagonal term is still present.
    assert np.isfinite(loss1)


def test_action_conditioned_backward_matches_explicit_and_action_average():
    cfg=FBConfig(batch_size=4,embed_dim=5,hidden_dim=12,num_layers=0,
                 backward_input='state_action',z_sampling='replay_goal',
                 terminate_on_goal=True)
    agent=FBAgent(cfg,Features(mode='identity'),seed=10)
    states=jnp.array([[.1,.2],[.3,.4]],jnp.float32)
    acts=jnp.array([1,3],jnp.int32)
    expected=backward(agent.params['backward'],jnp.concatenate(
        [states,jax.nn.one_hot(acts,4)],axis=-1))
    np.testing.assert_allclose(_b_embedding(agent.params['backward'],states,cfg,acts),expected)
    avg=np.mean([np.asarray(_b_embedding(agent.params['backward'],states,cfg,
                                        jnp.full((2,),a,dtype=jnp.int32)))
                 for a in range(4)],axis=0)
    np.testing.assert_allclose(agent.embed_B(states),avg,rtol=1e-5,atol=1e-6)
    assert np.linalg.norm(agent.embed_B(states,[0,0])-agent.embed_B(states,[1,1]))>1e-6
    batch=make_batch()
    other=jnp.array([[.2,.2],[.7,.8],[.1,.8],[.2,.7]],jnp.float32)
    goals=jnp.array([[.2,.3],[.8,.5],[.5,.7],[.2,.2]],jnp.float32)
    other_acts=jnp.array([3,0,2,1])
    _,metrics=fb_losses(agent.params,agent.target_params,batch,other,
                        jax.random.PRNGKey(5),cfg,agent.features,
                        goal_obs=goals,other_actions=other_acts)
    assert np.isfinite(metrics['loss'])
    with pytest.raises(ValueError,match='other_actions'):
        fb_losses(agent.params,agent.target_params,batch,other,
                  jax.random.PRNGKey(5),cfg,agent.features,goal_obs=goals)


def test_configuration_rejects_non_goal_sampling_and_invalid_radius():
    with pytest.raises(ValueError,match='replay_goal'):
        FBConfig(z_sampling='cauchy',terminate_on_goal=True)
    with pytest.raises(ValueError,match='tolerance'):
        FBConfig(goal_tolerance=0.)
    with pytest.raises(ValueError,match='backward_input'):
        FBConfig(backward_input='unsupported')


@pytest.mark.parametrize('backend',['jax','numpy_scan','numpy_legacy'])
@pytest.mark.parametrize('backward_mode',['state','state_action'])
def test_all_training_backends_with_goal_termination(tmp_path,backend,backward_mode):
    original=Path(__file__).resolve().parents[1]/'configs'/'smoke.yaml'
    cfg=yaml.safe_load(original.read_text())
    cfg['agent'].update(z_sampling='replay_goal',terminate_on_goal=True,
                        goal_tolerance=0.1,backward_input=backward_mode)
    cfg['training'].update(backend=backend,plot=False,checkpoint_every=0)
    config_path=tmp_path/'test.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    out,summary=train(config_path,tmp_path/f'{backend}_{backward_mode}',verbose=False)
    assert summary['updates']==4
    assert np.isfinite(summary['final_loss'])
    agent=FBAgent.load(out/'fb_final.npz')
    assert agent.config.terminate_on_goal is True
    assert agent.config.backward_input==backward_mode
    assert agent.embed_B([[.2,.3]]).shape==(1,agent.config.embed_dim)
    assert agent.embed_F([[.2,.3]]).shape==(1,4,agent.config.embed_dim)
