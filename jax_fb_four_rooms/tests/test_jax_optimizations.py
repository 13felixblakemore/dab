"""Regression tests for the faster JAX implementation."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import pytest
import yaml

from fb_jax.environment import ContinuousFourRooms, classic_walls
from fb_jax.jax_environment import (JAXFourRoomsConfig, RolloutState, _walls,
                                     is_free_batch, sample_free_points, reset_all,
                                     segment_collisions, step_batch, rollout)
from fb_jax.jax_replay import init_device_replay, add_many, sample
from fb_jax.agent import FBAgent, FBConfig, train_step
from fb_jax.features import Features
from fb_jax.experiment import TrainingConfig, train


def check_trees_close(a, b, atol=2e-6):
    a_leaves, _=jax.tree.flatten(a)
    b_leaves, _=jax.tree.flatten(b)
    assert len(a_leaves)==len(b_leaves)
    for x,y in zip(a_leaves,b_leaves):
        np.testing.assert_allclose(np.asarray(x),np.asarray(y),atol=atol,rtol=atol)


def test_segment_collision_and_is_free_match_numpy():
    rng=np.random.default_rng(11)
    env=ContinuousFourRooms(movement_noise=0)
    cfg=JAXFourRoomsConfig(movement_noise=0)
    start=rng.uniform(.001,.999,(2000,2)).astype(np.float32)
    end=rng.uniform(.001,.999,(2000,2)).astype(np.float32)
    walls=_walls(cfg)
    jax_collision=np.asarray(jax.jit(segment_collisions)(jnp.asarray(start),jnp.asarray(end),walls))
    ref=np.array([any(w.intersects_segment(a,b) for w in env.wall_blocks)
                  for a,b in zip(start,end)])
    np.testing.assert_array_equal(jax_collision,ref)
    points=rng.uniform(0,1,(1000,2)).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(is_free_batch(jnp.asarray(points),walls,1.0)),
                                  [env.is_free(p) for p in points])


def test_vectorised_step_matches_numpy_noiseless_and_resets():
    cfg=JAXFourRoomsConfig(step_size=0.2,movement_noise=0,explorable=False,
                           threshold_distance=0.1)
    state=RolloutState(jnp.array([[5.5/11,5.5/11],[5.5/11,8.5/11],[.2,.2]],jnp.float32),
                       jnp.array([[.8,.8],[.8,.8],[.4,.2]],jnp.float32),
                       jnp.zeros(3,dtype=jnp.int32),jax.random.PRNGKey(4))
    actions=jnp.array([1,1,1],jnp.int32)
    next_state,step=jax.jit(lambda s,a:step_batch(s,a,cfg,30))(state,actions)
    assert bool(step['collisions'][0]) and not bool(step['collisions'][1])
    for i in range(3):
        env=ContinuousFourRooms(step_size=0.2,movement_noise=0,explorable=False)
        env.reset(state=np.asarray(state.positions[i]),goal=np.asarray(state.goals[i]))
        expected,_,done,info=env.step(1)
        np.testing.assert_allclose(np.asarray(step['next_obs'][i]),expected,atol=1e-6)
        assert bool(step['done'][i])==done
        assert bool(step['collisions'][i])==info['goes_through_wall']
    # Third action reaches its goal; replay stores the terminal location, not reset location.
    assert bool(step['done'][2]); assert not np.array_equal(np.asarray(next_state.positions[2]),
                                                               np.asarray(step['next_obs'][2]))


def test_rollout_reproducible_finite_and_on_device():
    cfg=JAXFourRoomsConfig()
    fn=jax.jit(lambda key:rollout(reset_all(key,7,cfg),4,cfg,3,False))
    state,a,rate=fn(jax.random.PRNGKey(7))
    state2,b,rate2=fn(jax.random.PRNGKey(7))
    check_trees_close(a,b)
    assert a['obs'].shape==(28,2)
    assert a['next_action_mask'].all()
    assert np.all(np.asarray(a['terminals'])==0)
    assert np.all(np.asarray(is_free_batch(state.positions,_walls(cfg),cfg.size)))
    assert float(rate)>=0
    assert np.max(np.asarray(state.episode_steps))<3


def test_device_replay_wraps_and_samples():
    rep=init_device_replay(4,2,4)
    def insert(x):
        n=len(x)
        return {'obs':jnp.asarray(x,dtype=jnp.float32),
                'next_obs':jnp.asarray(x,dtype=jnp.float32)+1,
                'actions':jnp.arange(n,dtype=jnp.int32),
                'terminals':jnp.zeros(n,jnp.float32),
                'next_action_mask':jnp.ones((n,4),jnp.bool_)}
    rep=add_many(rep,insert([[1,1],[2,2],[3,3]]))
    rep=add_many(rep,insert([[4,4],[5,5],[6,6]]))
    assert int(rep.size)==4 and int(rep.ptr)==2
    obs=np.asarray(sample(rep,jnp.arange(4))['obs'])
    np.testing.assert_array_equal(obs,[[5,5],[6,6],[3,3],[4,4]])
    with pytest.raises(ValueError):
        add_many(rep,insert([[1,1]]*5))


def test_scanned_steps_equal_individual_updates():
    cfg=FBConfig(batch_size=6,hidden_dim=24,num_layers=1,embed_dim=7)
    agent=FBAgent(cfg,Features(mode='identity'),seed=9)
    rng=np.random.default_rng(43)
    n=3
    batches={'obs':rng.random((n,6,2)).astype('float32'),
             'next_obs':rng.random((n,6,2)).astype('float32'),
             'actions':rng.integers(0,4,(n,6)).astype('int32'),
             'terminals':np.zeros((n,6),dtype='float32'),
             'next_action_mask':np.ones((n,6,4),dtype=bool)}
    other=rng.random((n,6,2)).astype('float32')
    jparams,jopt,jkey,mean=agent._compiled_many(agent.params,agent.target_params,
                                                agent.opt_state,batches,other,agent.key)
    params,opt,key=agent.params,agent.opt_state,agent.key
    manual=[]
    for i in range(n):
        key,sub=jax.random.split(key)
        params,opt,metrics=train_step(params,agent.target_params,opt,
                                       {k:v[i] for k,v in batches.items()},other[i],sub,cfg,agent.features)
        manual.append(metrics)
    check_trees_close(jparams,params)
    check_trees_close(jopt,opt)
    check_trees_close(jkey,key)
    for k in mean:
        np.testing.assert_allclose(np.asarray(mean[k]),
                                   np.mean([float(m[k]) for m in manual]),rtol=3e-5,atol=1e-5)


@pytest.mark.parametrize('backend', ['numpy_scan','numpy_legacy'])
def test_alternate_backends_smoke(tmp_path,backend):
    source=Path(__file__).resolve().parents[1]/'configs'/'smoke.yaml'
    cfg=yaml.safe_load(source.read_text())
    cfg['training']['backend']=backend
    cfg['training']['plot']=False
    cfg['training']['checkpoint_every']=0
    config_path=tmp_path/f'{backend}.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    out,summary=train(config_path,tmp_path/backend,verbose=False)
    assert summary['updates']==4 and (out/'fb_final.npz').is_file()
    assert np.isfinite(summary['final_loss'])


def test_bad_jax_buffer_or_env_count():
    with pytest.raises(ValueError):
        TrainingConfig(backend='jax',num_envs=7,transitions_per_epoch=50)
    with pytest.raises(ValueError):
        TrainingConfig(backend='jax',transitions_per_epoch=50,buffer_size=20)
