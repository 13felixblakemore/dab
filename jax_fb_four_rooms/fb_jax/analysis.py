"""Successor-density geometry and k-medoids, ported from the supplied notebook.

Clamped log ratio is a heuristic pseudo-distance, NOT a proven metric.
The cluster graph is built from *observed transitions* only.
"""
from collections import Counter
import numpy as np


def fb_features(agent,observations,chunk=2048):
    """Return z=0 F averaged over actions, and B (both shape N×D)."""
    obs=np.asarray(observations,dtype=np.float32).reshape(-1,agent.config.obs_dim)
    f_list,b_list=[],[]
    for start in range(0,len(obs),chunk):
        part=obs[start:start+chunk]
        f_list.append(agent.embed_F(part).mean(axis=1))
        b_list.append(agent.embed_B(part))
    return np.concatenate(f_list),np.concatenate(b_list)


def sr_distance(f_a,b_a,f_b=None,b_b=None,eps=1e-6):
    """Symmetrised clipped log successor-density ratio from source notebook.

    d(A_i,B_j) = 0.5*[log(M(B_j,B_j)/M(A_i,B_j)) +
                        log(M(A_i,A_i)/M(B_j,A_i))].
    """
    if f_b is None:
        f_b,b_b=f_a,b_a
    f_a,b_a,f_b,b_b=map(np.asarray,(f_a,b_a,f_b,b_b))
    ab=np.maximum(f_a@b_b.T,eps)
    ba=np.maximum(f_b@b_a.T,eps)
    aa=np.maximum(np.sum(f_a*b_a,axis=-1),eps)
    bb=np.maximum(np.sum(f_b*b_b,axis=-1),eps)
    result=0.5*(np.log(bb[None,:]/ab)+np.log(aa[:,None]/ba.T))
    return np.maximum(result,0.0)


def kmedoids(distances,k,max_iters=50):
    """Deterministic farthest-first init + medoid reassignment; O(N²) memory."""
    d=np.asarray(distances,dtype=np.float64)
    n=len(d)
    if d.shape!=(n,n) or not (1<=k<=n) or not np.isfinite(d).all():
        raise ValueError("Distance matrix must be finite N×N; 1 <= k <= N")
    med=[int(d.sum(axis=1).argmin())]
    while len(med)<k:
        candidates=d[:,med].min(axis=1).copy()
        candidates[med]=-np.inf
        med.append(int(candidates.argmax()))
    for _ in range(max_iters):
        labels=d[:,med].argmin(axis=1)
        new=[]
        for c in range(k):
            members=np.flatnonzero(labels==c)
            if len(members)==0:
                new.append(med[c]);continue
            sub=d[np.ix_(members,members)]
            new.append(int(members[sub.sum(axis=1).argmin()]))
        if new==med:
            break
        med=new
    labels=d[:,med].argmin(axis=1)
    cost=float(d[np.arange(n),np.asarray(med)[labels]].sum())
    return np.asarray(med,dtype=int),labels,cost


def assign_clusters(agent,points,medoids,chunk=2048):
    fmed,bmed=fb_features(agent,medoids)
    output=[]
    for start in range(0,len(points),chunk):
        f,b=fb_features(agent,points[start:start+chunk])
        output.append(sr_distance(f,b,fmed,bmed).argmin(axis=1))
    return np.concatenate(output)


def observed_cluster_edges(agent,env,medoids,rng,n_steps=3000,horizon=30,
                           min_crossings=5):
    """Random walk, count ONLY within-episode directed cluster transitions."""
    transitions=[]
    obs=env.reset();step=0
    for _ in range(n_steps):
        action=int(rng.integers(env.n_actions))
        next_obs,_,_,_=env.step(action)
        transitions.append((obs,next_obs))
        obs=next_obs
        step+=1
        if step>=horizon:
            step=0;obs=env.reset()
    states=np.asarray(transitions,dtype=np.float32).reshape(-1,2)
    cluster_ids=assign_clusters(agent,states,medoids).reshape(-1,2)
    counts=Counter((int(a),int(b)) for a,b in cluster_ids if a!=b)
    return {edge:count for edge,count in sorted(counts.items()) if count>=min_crossings}
