from dataclasses import replace
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from fb_jax.agent import FBAgent,FBConfig,sample_cauchy_ball
from fb_jax.features import Features,transform
from fb_jax.replay import ReplayBuffer


def agent_small(seed=3):
    return FBAgent(FBConfig(embed_dim=8,hidden_dim=32,num_layers=0,batch_size=8,
                            max_grad_norm=10,temperature=10),
                   Features(mode="rbf",dim=5,sigma=0.1),seed=seed)


def make_buffer():
    rng=np.random.default_rng(3)
    rb=ReplayBuffer(30,2,4,seed=4)
    for i in range(25):
        obs=rng.uniform(0.05,0.95,2).astype(np.float32)
        rb.add(obs,int(rng.integers(4)),obs+0.02,False,[0,1,2,3])
    return rb


def test_feature_grid_normalised_and_identity():
    observations=jnp.array([[0.2,0.8],[0.7,0.3]])
    rbf=np.asarray(transform(observations,Features(dim=5,sigma=0.1)))
    assert rbf.shape==(2,25)
    np.testing.assert_allclose(rbf.sum(axis=1),1,atol=1e-6)
    raw=np.asarray(transform(observations,Features(mode="identity")))
    np.testing.assert_allclose(raw,np.asarray(observations))


def test_embedding_policy_and_sr_shapes():
    agent=agent_small()
    obs=np.array([[0.1,0.2],[0.3,0.4],[0.5,0.6]],dtype=np.float32)
    assert agent.embed_B(obs).shape==(3,8)
    assert agent.embed_F(obs).shape==(3,4,8)
    assert agent.embed_F(obs,np.ones(8)).shape==(3,4,8)
    assert agent.action_values(obs,np.ones(8)).shape==(3,4)
    assert agent.export_sr(obs).shape==(3,3)
    assert agent.act(obs[0],np.zeros(8),available_actions=[2])==2
    with pytest.raises(ValueError):
        agent.export_sr(obs,available_actions=np.zeros((3,4),dtype=bool))


def test_cauchy_draws_include_zero_vectors():
    z,zero=sample_cauchy_ball(jax.random.PRNGKey(10),200,8,0.3)
    assert z.shape==(200,8)
    assert 30<int(np.sum(np.asarray(zero)))<90
    assert np.all(np.isfinite(np.asarray(z)))
    assert np.all(np.asarray(z)[np.asarray(zero)]==0)


def test_update_finite_and_changes_parameters():
    agent=agent_small()
    old=np.asarray(agent.params["forward"][0]["w"]).copy()
    metrics=agent.update(make_buffer())
    assert set(("loss","fb_loss","reg_loss","td_mse","mean_q")).issubset(metrics)
    assert all(np.isfinite(v) for v in metrics.values())
    assert not np.array_equal(old,np.asarray(agent.params["forward"][0]["w"]))
    assert agent.updates==1


def test_polyak_updates_in_correct_direction():
    agent=agent_small()
    before=np.asarray(agent.target_params["forward"][0]["w"]).copy()
    agent.update(make_buffer())
    online=np.asarray(agent.params["forward"][0]["w"])
    agent.soft_update()
    target=np.asarray(agent.target_params["forward"][0]["w"])
    np.testing.assert_allclose(target,0.95*before+0.05*online,atol=1e-6)


def test_safe_checkpoint_roundtrip_and_update(tmp_path):
    agent=agent_small()
    rb=make_buffer()
    agent.update(rb)
    path=tmp_path/"model.npz"
    agent.save(path)
    loaded=FBAgent.load(path)
    assert loaded.updates==1
    for x,y in zip(jax.tree.leaves(agent.params),jax.tree.leaves(loaded.params)):
        np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
    np.testing.assert_allclose(agent.embed_B([[0.3,0.7]]),loaded.embed_B([[0.3,0.7]]))
    assert np.isfinite(loaded.update(rb)["loss"])


def test_zero_temperature_and_bad_config_rejected():
    with pytest.raises(ValueError):
        FBConfig(temperature=0)
    with pytest.raises(ValueError):
        Features(mode="foobar")


def test_legacy_checkpoint_preserves_forward_predictions(tmp_path):
    """Loading version-1 action-major output weights must preserve predictions."""
    import json
    agent = agent_small()
    path = tmp_path / 'v2.npz'
    agent.save(path)
    before = agent.embed_F([[.21,.19],[.81,.68]], np.ones(8,dtype=np.float32))
    d,a = agent.config.embed_dim,agent.config.n_actions
    # Inverse of version 1 -> 2 permutation in the loader.
    perm = np.array([action*d+embed for embed in range(d) for action in range(a)])
    inverse = np.argsort(perm)
    with np.load(path,allow_pickle=False) as npz:
        arrays = {k:npz[k] for k in npz.files}
    for tree in ('params','targets','m','v'):
        i = len(agent.params['forward'])-1
        arrays[f'{tree}.forward.{i}.w'] = arrays[f'{tree}.forward.{i}.w'][:,inverse]
        arrays[f'{tree}.forward.{i}.b'] = arrays[f'{tree}.forward.{i}.b'][inverse]
    metadata=json.loads(str(arrays['metadata']))
    metadata.pop('format_version')
    arrays['metadata']=np.array(json.dumps(metadata))
    old=tmp_path/'old_v1.npz'
    np.savez_compressed(old,**arrays)
    loaded=FBAgent.load(old)
    np.testing.assert_allclose(loaded.embed_F([[.21,.19],[.81,.68]], np.ones(8,dtype=np.float32)),before,
                               atol=1e-7,rtol=1e-7)
