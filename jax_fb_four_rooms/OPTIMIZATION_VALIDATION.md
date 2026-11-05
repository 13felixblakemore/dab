# JAX FB optimisation: implementation and validation

## What changed

1. Added `scanned_train_steps` in `fb_jax/agent.py` to perform all epoch
   gradient updates in `jax.lax.scan`. The reference Polyak averaging schedule
   remains **once per epoch**, not per gradient update. Only the epoch-mean
   metrics are brought back to Python.
2. Added `ReplayBuffer.sample_many` to sample all NumPy-host minibatches in
   bulk for `numpy_scan`.
3. Added `fb_jax/jax_environment.py` with batch-based pure JAX Four Rooms
   physics, collision intersection, uniform valid-state sampling, goal/horizon
   resets and random-policy `lax.scan` collection.
4. Added `fb_jax/jax_replay.py` with a device-resident replay ring and
   JAX-based minibatch selection. The `jax` backend performs collection,
   replay insertion, minibatch sampling, updates and Polyak averaging in **one
   JIT-compiled function per epoch**.
5. Added `training.backend` (`jax`, `numpy_scan`, `numpy_legacy`) and
   `training.num_envs` YAML fields, retaining old code for regression checks.
   The `reference.yaml` keeps the original environment/rollout path but uses
   scanned SGD; `reference_jax.yaml` runs the original FB hyperparameters with
   fully JAX-native rollouts.

The FB loss, detached asymmetric backward regulariser, F/B network shapes,
Adam formula, Cauchy sampling, softmax temperature, and target network
averaging schedule are unchanged. Importantly, JAX and NumPy PRNG streams
and the scheduling of independently collected trajectories differ; **loss
curves are not expected to match exactly**.

## Tests

26 automated tests passed, run in three independent groups with CPU thread
limits to avoid framework contention:

- 16 original unit and pipeline tests (environment, FB agent, checkpoint,
  clustering and YAML training): **16 passed**.
- 2 original JAX-versus-PyTorch numerical parity tests (F/B predictions,
  full loss, gradients, Adam step and RBF features): **2 passed**.
- 8 new optimisation tests (2,000 random collision geometries compared with
  NumPy, noiseless movement/goal reset parity, JIT rollout reproducibility,
  on-device replay wrap-around, scanned-update numeric parity, and alternate
  backend smoke runs): **8 passed**.

Run independently if your machine has trouble running JAX and PyTorch in the
same test worker:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q tests/test_agent.py tests/test_analysis.py tests/test_environment.py tests/test_pipeline.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q tests/test_pytorch_parity.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q tests/test_jax_optimizations.py
```

The PyTorch parity tests require PyTorch installed; the project itself doesn't.

## CPU runtime comparison

Three benchmark runs on the **same machine**, with the same network and
FB hyperparameters (21×21 RBF inputs, 256-wide network, 50-dimensional FB
embedding, batch 128), **50 epochs, 75,000 total transitions and 2,000
Adam updates**. Gradient clipping and replay sizes matched, checkpoint
interval disabled; timing includes JAX compilation and training and the
final model copy/save, but excludes Matplotlib plotting. Each result is one
run, not a statistically robust multi-seed speed comparison.

| Backend | Time (seconds) | Environment steps/s | Relative speed |
|---|---:|---:|---:|
| `jax` | 23.70 | 3,165 | **1.46×** vs legacy |
| `numpy_scan` | 26.32 | 2,849 | **1.31×** vs legacy |
| `numpy_legacy` | 34.54 | 2,172 | 1.00× |

The JAX backend was ~11% faster than `numpy_scan` on this CPU benchmark.
Short 10-epoch benchmarks were less conclusive due to compilation overhead.
Performance depends on network size, rollout/update ratio, accelerator, and
JAX version: expect different numbers on your WSL setup. The throughput
metric counts environment transitions divided by wall-clock training time;
updates are held equal across comparisons.

## Longer smoke run and downstream evaluation

The `configs/optimized_probe.yaml` 50-epoch JAX run completed:

- **75,000** transitions and **2,000** gradient updates, finite metrics.
- Final epoch-mean FB objective **-4.1425** and TD MSE **3.932**.
- Wrote full model checkpoint, training metrics CSV and learning-curve plot.
- On a limited 15-task, 60-step zero-shot navigation evaluation, **0/15**
  goals reached; random baseline also **0/15**. This small run **does not
  demonstrate learned navigation**. It primarily validates throughput and
  correct execution; for meaningful navigation comparisons use longer runs
  and identical fixed start/goal suites.

**Note:** In the provided original reference-sized training experiment, TD
MSE increased dramatically as training progressed even though the aggregate
FB objective decreased. The optimisations do not fix that algorithmic issue.

## Reproducing

```bash
python train.py --config configs/smoke.yaml
python train.py --config configs/default.yaml
python train.py --config configs/optimized_probe.yaml
python train.py --config configs/reference_jax.yaml
python plot_results.py results/optimized_probe
```
