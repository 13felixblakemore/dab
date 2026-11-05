import numpy as np
from fb_jax.analysis import sr_distance,kmedoids,fb_features,assign_clusters
from test_agent import agent_small


def test_symmetric_distances_and_clusters():
    agent=agent_small()
    samples=np.array([[.2,.2],[.8,.2],[.2,.8],[.8,.8]],dtype=np.float32)
    f,b=fb_features(agent,samples)
    d=sr_distance(f,b)
    assert d.shape==(4,4)
    assert np.isfinite(d).all() and (d>=0).all()
    np.testing.assert_allclose(d,d.T,atol=1e-5)
    med,label,cost=kmedoids(d,2)
    assert len(med)==2 and len(label)==4 and np.isfinite(cost)
    assert assign_clusters(agent,samples,samples[med]).shape==(4,)
