import numpy as np
import pytest
from fb_jax.environment import ContinuousFourRooms


def test_reproducible_random_trajectories():
    a=ContinuousFourRooms(seed=7)
    b=ContinuousFourRooms(seed=7)
    np.testing.assert_array_equal(a.reset(),b.reset())
    for action in (0,1,2,3,1,0,3):
        np.testing.assert_array_equal(a.step(action)[0],b.step(action)[0])


def test_thick_wall_cannot_be_tunnelled_through():
    env=ContinuousFourRooms(step_size=0.2,movement_noise=0)
    start=np.array([5.5/11,5.5/11])
    env.reset(state=start)
    obs,_,_,info=env.step(1) # right across closed central divider
    assert info["goes_through_wall"]
    np.testing.assert_allclose(obs,start,atol=1e-7)


def test_door_allows_crossing():
    env=ContinuousFourRooms(step_size=0.2,movement_noise=0)
    start=np.array([5.5/11,8.5/11])
    env.reset(state=start)
    obs,_,_,info=env.step(1)
    assert not info["goes_through_wall"]
    assert obs[0]>7/11


def test_valid_sampling_and_actions():
    env=ContinuousFourRooms(seed=18)
    for _ in range(200):
        assert env.is_free(env.sample_point())
    assert env.n_actions==4
    assert env.get_available_actions()==[0,1,2,3]
    with pytest.raises(ValueError):
        env.step(5)
    with pytest.raises(ValueError):
        env.reset(state=[6.5/11,5.5/11])


def test_goal_mode_and_stay_action():
    env=ContinuousFourRooms(explorable=False,include_stay_action=True,
                            movement_noise=0,threshold_distance=0.1)
    env.reset(state=[0.2,0.2],goal=[0.2,0.2])
    _,reward,done,_=env.step(4)
    assert done and reward==1


def test_navigation_tasks_are_fixed_and_random_baseline_reproducible():
    from fb_jax.navigation import sample_eval_tasks, evaluate_random_navigation
    import numpy as np
    first = sample_eval_tasks({}, 12, 44, 0.07)
    second = sample_eval_tasks({}, 12, 44, 0.07)
    for (s1,g1),(s2,g2) in zip(first,second):
        np.testing.assert_array_equal(s1,s2)
        np.testing.assert_array_equal(g1,g2)
        assert np.linalg.norm(s1-g1)>0.24
    assert evaluate_random_navigation(episodes=12,max_steps=20,seed=44) == \
           evaluate_random_navigation(episodes=12,max_steps=20,seed=44)
