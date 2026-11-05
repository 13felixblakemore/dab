# Goal-conditioned episodic successor updates

## Objective

With a separate replay-sampled state `goal_obs[i]`, build

- `z[i] = stop_gradient(B(goal_obs[i]))` in state-only B mode.
- `z[i] = stop_gradient(mean_a B(goal_obs[i], a))` in state-action B mode.
- `m[i] = 1[||next_obs[i] - goal_obs[i]||_2 > epsilon] * (1 - terminal[i])`.

The row-wise TD residual is

`delta[i,j] = F(obs[i], action[i], z[i]) · B(other_obs[j], other_action[j])
              - gamma * m[i] * E_{a~pi_z_i}[F_target(next_obs[i], a, z[i])]
                · B_target(other_obs[j], other_action[j])`.

`other_action[j]` is omitted for the original state-only backward function.
The diagonal `F(obs[i],action[i],z[i]) · B(obs[i],action[i])` remains.
The original asymmetric backward regulariser is preserved.

## Configurations

- `configs/goal_terminating.yaml` — original B(s), goal termination only.
- `configs/goal_terminating_state_action.yaml` — action-conditioned B(s,a)
  and action-averaged state goals.
- `configs/goal_directed.yaml` — unchanged previous goal-conditioned baseline.

## Implementation

- `fb_jax/agent.py`: config flags, shape-safe mask, optional B(s,a), target.
- `fb_jax/replay.py`: sampled independent state/action comparison pairs on CPU.
- `tests/test_goal_termination.py`: new tests including all three backends.

### Scope

This alters the learned target, not the replay data-collection policy. It does
not introduce a terminal state in the environment for each sampled goal. `z`
remains non-differentiable to its source B embedding by design. The new
state-action version is a separate model architecture and cannot directly use
previous state-only backward weights.

## Validation

- 10 new goal-termination tests passed.
- 32 previous tests passed, split into 3 groups (21+8+1+2).
- CPU 12-epoch short run: 6,000 transitions, 144 gradient updates, finite metrics.
- A combined all-tests single process exceeded the execution timeout; the
  individually run groups all passed.
