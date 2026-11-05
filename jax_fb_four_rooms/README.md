# JAX Forward–Backward (FB) on Continuous Four Rooms

This is a functional, pure-JAX adaptation of the supplied PyTorch FB learner and
continuous Four Rooms environment. **This is reward-free representation learning**;
it does **not** train on goal reward or directly optimise goal-reaching success.

## Run (Ubuntu/WSL)

```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install -r requirements.txt
python train.py --config configs/smoke.yaml     # quick functionality check
python train.py --config configs/default.yaml   # moderate training
python train.py --config configs/reference.yaml # original settings, NumPy+scan
python train.py --config configs/reference_jax.yaml # same FB settings, JAX rollouts
python train.py --config configs/learning_probe.yaml  # 37.5K-step demo (~24s CPU here)
python evaluate.py --checkpoint results/default/fb_final.npz --episodes 100 --random-baseline
python plot_results.py results/default
python analyze.py --checkpoint results/default/fb_final.npz --samples 400 --clusters 12
pytest -q
```

### Modules

- `fb_jax/environment.py`: continuous `x,y` Four Rooms, seven rectangular wall segments,
  continuous noisy moves with **four discrete actions**, exact segment-wall collision tests,
  exploration with random goal resets (as in the reference) or goal-free mode;
  reproducible local RNG.
- `fb_jax/features.py`: original RBF grid `21×21` (or raw positions).
- `fb_jax/networks.py`: JAX MLP forward `F(s,a,z)` and backward `B(s)`.
- `fb_jax/agent.py`: jitted FB TD objective, Cauchy-ball policy vectors, target softmax,
  backward regularisation, Adam and target networks, embedding and successor-density API,
  safe NPZ checkpoints including Adam state.
- `fb_jax/replay.py`: preallocated host replay buffer and batch sampling.
- `fb_jax/jax_environment.py`: JAX-native, vectorised Four Rooms dynamics and reset/rollout scans.
- `fb_jax/jax_replay.py`: device-resident replay ring buffer, compiled minibatch sampling.
- `fb_jax/experiment.py`: YAML reading, three interchangeable training backends; emits metrics CSV,
  summary JSON, resolved YAML, checkpoints, and optional loss plots.
- `fb_jax/analysis.py` and `analyze.py`: the supplied notebook’s symmetric
  clipped SR log-distance, k-medoids state clustering, and observed directed
  cluster-transition graph. Produces `clusters.npz`, `clusters.png`,
  `cluster_edges.csv`; straight graph arrows do **not** describe physical paths.
- `fb_jax/navigation.py` and `evaluate.py`: zero-shot goal navigation with
  `z≈mean(B(states near goal))`, success statistics and trajectory image.

### What is actually learned?

FB factorises the discounted successor measure's **density relative to the
replay state distribution**, approximately as

`m^{π_z}(s,a,s') ≈ F(s,a,z) · B(s')`.

Policy `π_z` is induced from `Q_z(s,a)=F(s,a,z)·z`. Training uses a Boltzmann
version of this policy for the bootstrap targets; greedy action selection is
used when evaluating (`epsilon=0`). The environment's *training actions* are
**uniform random**, just like the reference (`epsilon=1.0`). FB still learns
about a family of `z`-conditioned policies by off-policy bootstrapping.

### Matching the original precisely (and deliberate differences)

- Cross-framework checked: exact forward-network output ordering
  (`raw.reshape(batch, embedding, actions)` in the reference), F/B network
  outputs, all scalar loss components, online parameter gradients, first Adam
  update, and 21×21 RBF feature encoding, using identical test tensors.
- Matches: F/B MLP sizes (including the number of extra hidden layers),
  input RBF basis, `w / sqrt(1+||w||²/d)` forward conditioning, Cauchy(0,0.5)
  directions, `zero_z_p`, 128×128 pairwise TD matrix objective and the **exact
  asymmetric detached** backward regularisation expression, Adam hyperparameters,
  target Boltzmann logits `Q/temperature`, Polyak convention
  `target ← 0.95 target + 0.05 online` **once each epoch**, and random
  data collection in fixed-length rollouts.
- Improvements: local seeded environment RNG, available-action masking in
  greedy inference, **optional** terminal bootstrap masking (off by default,
  matching the reference's `terminal=False`), stable RBF normalisation,
  small defensive parameter checks, pure-JAX
  JIT-compiled updates and no GPU-only assumptions.
- The original constructs `ContinuousFourRooms(explorable=False)`, which
  samples random goals and resets whenever a goal is reached, but calls
  `learner.update(..., terminal=False)` regardless. All supplied training YAML
  files now reproduce these reset and bootstrap semantics with
  `environment.explorable: false` and `training.use_terminal_bootstrap_mask:
  false` (default). Set `explorable: true` to collect goal-free rollouts.
  In either case, trajectories are also restarted every 30 steps.
- `reference.yaml` matches the original major training counts/parameters;
  `default.yaml` is **shorter** and enables gradient clipping. PyTorch and JAX
  random streams and environment reset samples will not be byte-identical.
  No trained
  PyTorch weights are converted. NN initialisation uses PyTorch-style uniform
  bounds, but PRNGs and optimisers will not produce numerically identical runs.
- Target F/B and the TD bootstrap are detached correctly; source's target
  update schedule is unchanged. Note that `temperature=200` makes target
  policies close to uniform; lowering it changes which successor policy is
  represented, rather than merely changing an exploration rate.

### Files and analysis

Each output folder contains:

```
resolved_config.yaml
training_metrics.csv       # epoch averages: FB loss, regularizer, TD MSE, gradients
summary.json
fb_final.npz              # parameters, targets, optimiser state and PRNG key
fb_epoch_XXXX.npz         # optional periodic checkpoints
training_curve.png        # if plotting enabled
```

`evaluate.py` additionally produces `evaluation.json` and `trajectory.png`.
Goal navigation tests **generalisation after training**, not training success.
Evaluation now pre-samples **identical starts and goals** independently of the
trajectory RNG. With `--random-baseline`, the same start/goal suite is also
run using a uniform random policy, providing a useful comparison.
The learned representation may fail to navigate even if its training loss has
fallen. An earlier 25-epoch smoke run produced decreasing loss but no successful
zero-shot navigation (0/15); this was insufficient to assess learning.
A later **full reference-sized experiment** (800 epochs, 2.4 million
transitions, 32,000 updates) completed in **603.9 seconds on CPU**, with
all 800 logged epoch metrics finite. Success on the same fixed 100-task suite
improved from **10% (epoch 200)** to **16% (epoch 400)**, **21% (epoch 600)**,
and **24% (epoch 800)**, compared with a **3% random-action baseline**.
The final model scored **21%** on a second 100-task suite (random: **10%**).
The final model was excellent within a room (14/14 and 13/15) but had
low **cross-room** success (10/86 and 8/85). These are modest overall
success rates, not solved navigation. Importantly, **TD MSE
increased from 0.67 to 91.55**, and forward/backward gradient norms grew
substantially, despite the aggregate FB loss becoming more negative. This may
indicate scale growth/optimisation instability; decreasing aggregate loss is
not evidence of a calibrated successor model. See `VALIDATION.md`. The notebook SR log distance
is a **heuristic** (it can be zero or saturated early in training). No
particular success rate is guaranteed.

### Python API

```python
import numpy as np
from fb_jax.agent import FBAgent
agent = FBAgent.load("results/default/fb_final.npz")
points = np.array([[0.2, 0.2], [0.8, 0.8]], dtype=np.float32)
B = agent.embed_B(points)   # (2, 50)
F = agent.embed_F(points)   # (2, 4, 50) for z=0
sr = agent.export_sr(points)  # pairwise successor-density estimates
z_goal = np.mean(B, axis=0)
action = agent.act([0.2, 0.2], z_goal)
```

**Performance note:** JIT compilation can take a few seconds on the first epoch.
With `backend: jax`, the vectorised environment, replay buffer, minibatch
sampling, gradient scan and once-per-epoch Polyak target update run in one
compiled function. With `backend: numpy_scan`, only the gradient scan is
compiled; `numpy_legacy` retains the original Python loop. For GPU support,
follow the official JAX accelerator installation guide for your CUDA version.
See `OPTIMIZATION_VALIDATION.md` for a measured CPU comparison and caveats.

### Optional strict PyTorch parity test

If PyTorch is installed, `pytest -q tests/test_pytorch_parity.py` checks the
loss and gradients against the reference Torch equations using matched weights,
inputs and policy vectors. The core project needs only JAX, not PyTorch.
The tests are skipped when Torch is not installed.

## Faster training paths (JAX-native, no Flax needed)

The trainer now supports three YAML values under `training.backend`:

- **`jax` (default):** vectors of independent Four Rooms environments, JAX PRNG,
  `lax.scan` rollouts, device-resident ring replay, sampled gradient updates in
  another scan, and Polyak update **once per epoch**. All work is compiled as one
  JAX function per epoch. Metrics transfer to the CPU once per epoch, not once
  per update. Set `training.num_envs: 25`; it must divide
  `transitions_per_epoch`. `buffer_size >= transitions_per_epoch` is required.
- **`numpy_scan`:** uses the original NumPy environment and CPU replay, sampling
  a whole epoch's minibatches at once; a compiled JAX scan then runs the
  updates without per-update metric synchronisation. This preserves the
  original environment semantics and data distribution (but not the exact RNG
  draw sequence), and is useful for comparisons.
- **`numpy_legacy`:** original Python/NumPy environment and per-update JAX
  dispatch for baseline benchmarking and debugging.

The `reference.yaml` remains available for checking against the source algorithm;
`reference_jax.yaml` uses the accelerated JAX backend with the same FB objective
and hyperparameters. JAX and NumPy use different PRNGs, and multiple independent
rollouts change the sample order, so their losses/trajectories are not expected
be identical even though the transition law and FB update objective are preserved.

The JAX environment mirrors the NumPy wall intersection test (including tunnelling
prevention), resets on goals or horizon, and does not feed the reset state into
replay as an episode transition. When the terminal bootstrap mask is disabled,
goal hits are stored as nonterminal, like the original PyTorch trainer.

Performance on your own accelerator may differ. Benchmark with equivalent YAML
configs and fixed update counts, remembering that the first run includes JAX
compilation. Using Flax wouldn't automatically improve these compiled kernels.

To reproduce the three-backend timing comparison using separate processes:

```bash
python benchmark.py --config configs/default.yaml --epochs 50
```

This produces `results/backend_benchmark/benchmark.json` and one checkpoint
and metrics CSV per backend. Benchmarks are sensitive to CPU/GPU hardware and
JAX compilation/warm-up; compare equal rollout and update counts.
