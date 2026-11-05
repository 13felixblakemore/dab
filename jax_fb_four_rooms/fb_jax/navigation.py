"""Zero-shot goal reaching via z≈E[B(s) | s near goal].

This follows the notebook's cluster-goal idea; success is evaluated using
Euclidean proximity, which is NOT the FB training objective.
"""
import numpy as np
from .environment import ContinuousFourRooms


def goal_direction(agent,env,goal,rng,n_samples=128,radius=0.07):
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    samples=[]
    for _ in range(n_samples*100):
        point=goal+rng.uniform(-radius,radius,2)
        if env.is_free(point):
            samples.append(point)
        if len(samples)>=n_samples:
            break
    if not samples:
        raise ValueError("Could not sample navigable points near goal")
    return agent.embed_B(np.asarray(samples,dtype=np.float32)).mean(axis=0)


def sample_eval_tasks(env_kwargs, episodes, seed, goal_radius):
    """Sample one fixed start/goal suite, independently of policy movement RNG."""
    task_env = ContinuousFourRooms(**{**(env_kwargs or {}),
                                     "explorable": True,"seed":seed+100_003})
    tasks=[]
    for ep in range(episodes):
        start=task_env.sample_point()
        for _ in range(10000):
            goal=task_env.sample_point()
            if np.linalg.norm(goal-start)>2*goal_radius+0.1:
                break
        else:
            raise RuntimeError("Could not sample a sufficiently distant goal")
        tasks.append((start,goal))
    return tasks


def evaluate_navigation(agent,env_kwargs=None,episodes=20,max_steps=120,
                        seed=123,epsilon=0.0,goal_radius=0.07):
    if episodes<1 or max_steps<1:
        raise ValueError("episodes and max_steps must be positive")
    env=ContinuousFourRooms(**{**(env_kwargs or {}),"explorable":True,"seed":seed})
    rng=np.random.default_rng(seed+888)
    successes=0;lengths=[];example=None
    # Approximate room labels split at the centres of the two thick walls.
    def region(point):
        return (int(point[0] > env.size * 6.5/11),
                int(point[1] > env.size * 4.5/11))
    counts={"same_room":0,"different_rooms":0}
    hits={"same_room":0,"different_rooms":0}
    tasks=sample_eval_tasks(env_kwargs,episodes,seed,goal_radius)
    for start,goal in tasks:
        grouping="same_room" if region(start)==region(goal) else "different_rooms"
        counts[grouping]+=1
        obs=env.reset(state=start)
        z=goal_direction(agent,env,goal,rng,radius=goal_radius)
        states=[obs.copy()]
        successful=False
        for t in range(max_steps):
            act=agent.act(obs,z,epsilon=epsilon,rng=rng)
            obs,_,_,_=env.step(act)
            states.append(obs.copy())
            if np.linalg.norm(obs-goal)<=goal_radius:
                successful=True
                break
        successes+=int(successful)
        hits[grouping]+=int(successful)
        lengths.append(t+1)
        if example is None or (successful and not example["success"]):
            example={"states":np.asarray(states),"goal":goal,"success":successful}
    result={"episodes":episodes,"success_rate":successes/episodes,
            "successes":successes,"mean_episode_length":float(np.mean(lengths)),
            "max_steps":max_steps,"goal_radius":goal_radius,"seed":seed,
            "room_breakdown":{k:{"trials":counts[k],"successes":hits[k],
                              "success_rate":hits[k]/counts[k] if counts[k] else None}
                              for k in counts}}
    return result,example,env


def evaluate_random_navigation(env_kwargs=None,episodes=20,max_steps=120,
                               seed=123,goal_radius=0.07):
    """Paired-task random-action baseline (same starts/goals as FB evaluation)."""
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    env=ContinuousFourRooms(**{**(env_kwargs or {}),"explorable":True,"seed":seed})
    rng=np.random.default_rng(seed+888)
    successes=0
    for start,goal in sample_eval_tasks(env_kwargs,episodes,seed,goal_radius):
        env.reset(state=start)
        for _ in range(max_steps):
            position,_,_,_=env.step(int(rng.integers(env.n_actions)))
            if np.linalg.norm(position-goal)<=goal_radius:
                successes += 1
                break
    return {"episodes":episodes,"successes":successes,
            "success_rate":successes/episodes,"max_steps":max_steps,
            "goal_radius":goal_radius,"seed":seed}
