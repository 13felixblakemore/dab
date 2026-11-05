# Continuous Four Rooms + DQN

A self-contained PyTorch implementation of **DQN / Double DQN** for a **continuous-position Four Rooms** navigation environment. The state is continuous, but the *action space is discrete* (up/down/left/right), which is required by conventional DQN. Uses NumPy for physics/replay, YAML for experiment configuration, and Matplotlib for plots.

## Quick start

From this folder, with Python 3.10+:

```bash
pip install -r requirements.txt
python train.py --config configs/default.yaml
python plot_results.py results/YOUR_RUN_DIRECTORY
python evaluate.py --checkpoint results/YOUR_RUN_DIRECTORY/best.pt --episodes 100 --trajectory results/YOUR_RUN_DIRECTORY/trajectory.png --json-out results/YOUR_RUN_DIRECTORY/eval.json
```

If desired, `pip install gymnasium` enables native Gymnasium types; **the code also runs without it** using small compatible space classes. If using Gymnasium, the env can be used with standard `reset`/`step` interfaces.

Short, functional smoke run (does **not** establish that learning converges):

```bash
python train.py --config configs/smoke.yaml
pytest -q
```

Override output or plotting using `--output results/my_run`, `--plot`, or `--no-plot`. Config files may specify only the keys they override; defaults are loaded first. Unknown config keys raise errors rather than being silently ignored.

## Environment

- Continuous agent coordinates `(x, y) ∈ [0, 1]²`, goal in upper-right room; starts in lower-left room.
- Four discrete cardinal actions. Movement uses short collision-checked substeps to prevent crossing walls at speed.
- Cross-shaped central walls and **four doorways** (one in each half of the cross); doorway width and wall thickness are configurable.
- Observation: `[x, y, goal_x, goal_y]`, `float32`, always between 0 and 1.
- Reward: step cost (default `−0.01`) and success reward (`+1`); configurable potential-based distance shaping is **on in `configs/default.yaml`** (`shaping_scale: 0.5`), but off when constructing the Python environment directly. If enabled, align `shaping_gamma` with DQN `gamma`.
- Episodes end on reaching the goal or truncate at the episode step limit. Replay masks bootstrap values for `terminated` only; time-limit truncation does not incorrectly become a terminal state.
- `reset(seed=...)`, `reset(options={"start_pos": [x, y]})`, `step(action)`, and `render()` producing an RGB NumPy array.
- Start-position jitter and action noise configurable.

**Note:** Four Rooms with a sparse terminal reward can be challenging for epsilon-greedy DQN. The default 100k steps is a starting point, **not a claim of reliable convergence**. To compare, try `shaping_scale: 0.0` for the sparse-reward variant; compare sparse and shaped experiments using evaluation success rate, not their differently shaped returns. Random walk and success probabilities can vary with seed. The potential-based shaping formulation assumes a consistent discount and zero terminal potential; it is not a shortest-path oracle through walls.

## DQN

- MLP Q-network with configurable hidden widths, Huber TD loss, Adam, gradient clipping.
- Preallocated NumPy replay buffer; batched sampling.
- Epsilon-greedy linear annealing; hard target updates; optional Double DQN (`double_dqn: true`).
- Separately seeded evaluation with `epsilon=0`, `torch.no_grad()`; same evaluation seeds at each checkpoint.
- CPU and CUDA supported (`device: auto`); one CPU thread by default to avoid expensive oversubscription for small MLPs.

## Project layout

```text
four_rooms/
  env.py            # Geometry, reset, step, rendering
  replay.py         # Circular experience replay
  dqn.py            # Networks, DQN agent, checkpoints
  experiment.py     # YAML configuration and training/evaluation routines
configs/
  default.yaml      # Full experiment configuration
  smoke.yaml        # Fast pipeline check
  sparse.yaml       # Sparse-reward comparison
  learning_probe.yaml # Config used for bundled 100k-step example
train.py            # Training CLI
plot_results.py     # Training curves and trajectory visualization
evaluate.py         # Evaluate saved checkpoints
requirements.txt
tests/               # Geometry, algorithm and integration tests
```

## Outputs

Each run creates:

| File | Meaning |
|---|---|
| `resolved_config.yaml` | Full reproducible config |
| `episodes.csv` | Training episode returns, success, length, epsilon and step |
| `evaluations.csv` | Periodic greedy evaluation mean return, success rate and length |
| `losses.csv` | Mean TD loss per logging interval |
| `best.pt`, `last.pt` | Checkpoint(s), including config/optimizer weights |
| `summary.json` | Training speed, eval and counts |
| `training_curve.png` | Train return (raw/rolling), evaluation success, TD loss |

`evaluate.py --trajectory ...` also renders a path overlaid on the rooms.

## Important implementation choices

- This is a **continuous-state discrete-action** environment, not continuous-action control. For continuous-force actions, use SAC/TD3 instead of standard DQN.
- Using `terminated` rather than `terminated or truncated` for the replay target preserves bootstrapping at artificial time limits.
- The best checkpoint is selected by greedy success rate and then mean evaluation return, not by training return.
- The default fixed goal is kept in the observation vector, making extension to random goals straightforward.
- Current code is **single-environment**; using several actors/vectorized rollout collection would be an extension for higher throughput.

## Reference run (included in `results/learning_probe`)

With `configs/learning_probe.yaml` and seed 42, 100,000 transitions ran on CPU at about **2,244 environment steps/sec** (including gradient updates and periodic evaluations). The best checkpoint (at 70k steps) reached **95% success on 20 evaluation episodes**; a separately seeded 100-episode test reached **90%**. **The final checkpoint scored 0%, illustrating DQN instability**. Preserve `best.pt` and consider more seeds, longer evals, or algorithmic improvements before interpreting performance. These are measured outcomes of one seed, not guarantees.
