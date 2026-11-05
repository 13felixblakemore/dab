# Validation report: JAX FB vs the supplied PyTorch implementation

## Reproduction review

**Verified numerically against PyTorch** (matched initial tensors, inputs, and fixed policy samples):

- Forward MLP layer counts: first hidden layer + `num_layers` additional hidden layers; final outputs stored in PyTorch `(embedding, action)` order.
- Input policy-vector scaling: `z / sqrt(1 + ||z||² / embed_dim)`.
- Cauchy(0, 0.5) radial sampling, unit-normal Gaussian directions, `sqrt(embed_dim)` scale, and 25% `z=0` training samples (distributional equivalence; independent RNG streams).
- Softmax bootstrap target logits `Q_z(s',a)/200`, with available-action masking and uniform target policy for zero vectors.
- FB pairwise TD loss, diagonal term, **asymmetrically detached** backward regulariser, and detached target-network outputs.
- Forward-network outputs, total loss, FB loss, backward regularisation loss, and all online parameter gradients: JAX vs PyTorch within float32 tolerance on a matched test batch.
- The first Adam parameter update and 21×21 Gaussian RBF input features: matched to PyTorch within float32 tolerance.
- Per-epoch target averaging: `target = .95*target + .05*online` after every 40 updates, **not** after every individual update.

**Training semantics corrected:** the original creates a Four Rooms environment with `explorable=False`, samples a random goal on reset, and resets the environment when the goal is reached. Even then it stores `terminal=False`, so TD training bootstraps through every stored transition. The new `reference.yaml` matches this by setting `environment.explorable: false` and leaving `training.use_terminal_bootstrap_mask: false` (the default). Data-collection actions remain uniformly random and the rollout horizon is 30.

**Remaining intentional differences:** JAX vs PyTorch PRNG sequences, CPU NumPy replay vs Torch replay, fixed local RNG in the environment vs original global NumPy random calls, floating-point kernel/Adam rounding, faster stable RBF normalisation, checkpoint format, and optional other configurations. These prevent identical training traces but preserve the mathematical objective and source hyperparameters. Existing v1 JAX checkpoints are migrated on load to preserve predictions after correcting the F output layout.

## Complete large-run benchmark

Used `python train.py --config configs/reference.yaml --output results/verified_reference` on CPU, with no gradient clipping.

| Metric | Result |
|---|---:|
| Epochs | 800 / 800 |
| Transitions | 2,400,000 |
| SGD updates | 32,000 |
| Replay capacity | 1,000,000 |
| Elapsed (CPU) | 603.9 seconds |
| Throughput | 3,974 transitions/second |
| Epochs with non-finite logged metrics | 0 |
| Total FB objective, epoch 1 → 800 | −0.371 → −387.117 |
| TD residual MSE, epoch 1 → 800 | 0.666 → 91.552 |
| Forward gradient norm, epoch 1 → 800 | 0.301 → 142.263 |
| Backward gradient norm, epoch 1 → 800 | 0.584 → 203.387 |

Navigation test: greedy `a=argmax F(s,a,z_goal)·z_goal`; `z_goal` is mean B for points in a small goal neighbourhood. Up to 120 steps per episode, success at Euclidean distance ≤ 0.07. Fixed starts/goals independent of rollout RNG, so checkpoint comparisons are paired at the task level.

| Checkpoint | Fixed suite seed 42, 100 trials |
|---|---:|
| Epoch 200 / 600k transitions | 10% |
| Epoch 400 / 1.2m transitions | 16% |
| Epoch 600 / 1.8m transitions | 21% |
| Epoch 800 / 2.4m transitions | **24%** |
| Uniform-random action baseline, same tasks | 3% |

On a separate seed-123 suite (100 trials), the final policy succeeded **21%** versus **10%** for random actions.

**Within-room versus cross-room:** On seed 42, the final model solved 14/14 goals whose starts were in the same approximate room, but only **10/86 cross-room goals**. On seed 123 it solved 13/15 same-room and **8/85 cross-room goals**. Room labels are estimated using the centre lines of the rectangular walls. The learned model is much stronger at local goal reaching than navigation through doorways.

**Interpretation:** The implementation executes fully, mirrors the supplied loss and update schedule, and the trained policy beats random baselines on these two samples. It does **not** solve Four Rooms. Furthermore, increasing TD MSE, growing gradient norms, and large fluctuations in mean predicted Q suggest that the negative overall objective can improve even while value-estimation quality worsens. This is a research/algorithm validation concern, not a confirmed implementation bug. Investigate representation normalisation, backward regularisation, Q/value scales, and temperature *as separately labelled variants*, rather than silently changing the faithful baseline.

## Additional checks

- 16 JAX/environment/pipeline/analysis tests pass.
- 2 optional PyTorch parity tests pass.
- Cross-version checkpoint migration, seed reproducibility, wall collision/doorway traversal, replay updates, target averaging, end-to-end training and plotting were tested.
- The full model generated 12 nontrivial successor-distance clusters and 22 observed inter-cluster directed edges; these are exploratory diagnostics, not evidence of validated hierarchical options.

Run:

```bash
python -m pytest -q tests/test_agent.py tests/test_environment.py tests/test_analysis.py tests/test_pipeline.py
python -m pytest -q tests/test_pytorch_parity.py   # requires PyTorch
python evaluate.py --checkpoint results/verified_reference/fb_final.npz --episodes 100 --seed 42 --random-baseline
```
