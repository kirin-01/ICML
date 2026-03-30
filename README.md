
# Anonymous Rebuttal Artifact

This repository contains the three PINN experiments referenced in the rebuttal. The goal of this artifact is not to present the full project, but to let the AC and reviewers inspect the exact scripts, settings, figures, and reproduced numbers used in our response.

## Scope

This artifact focuses on one question raised in review:

- whether the reported behavior is caused by optimizer special-tuning
- whether extremely small MLPs can outperform larger ones under the same PINN protocol
- whether reviewer-proposed MLP settings can be reproduced under the stated optimizer and scheduler choices

All scripts here solve the same 2D Poisson PINN problem on `[-1, 1]^2` with analytical solution

```text
u(x, y) = 1 / (1 + x^2 + y^2)
```

using a loss consisting of:

- PDE residual loss on interior collocation points
- boundary condition loss on sampled boundary points

## Files Included In The Rebuttal

| File | Role in rebuttal |
|------|------------------|
| [`pinn.py`](./pinn.py) | Unified optimizer-protocol comparison across `MLP`, `KAN`, and `RationalANOVA` |
| [`pinn_mlp_param_sweep.py`](./pinn_mlp_param_sweep.py) | MLP parameter-count ablation under a fixed hybrid optimizer protocol |
| [`pinn_mlp_reviewer_protocol.py`](./pinn_mlp_reviewer_protocol.py) | Reproduction script for reviewer-style MLP settings with cosine scheduling |
| [`benchmark_comparison_three_protocols.png`](./benchmark_comparison_three_protocols.png) | Figure for the unified optimizer comparison |
| [`mlp_param_sweep_adam_lbfgs.png`](./mlp_param_sweep_adam_lbfgs.png) | Figure for the MLP size ablation |
| [`reviewer_mlp_protocol_adam.png`](./reviewer_mlp_protocol_adam.png) | Figure for the reviewer-style `Adam + cosine` runs |

## Environment

Recommended dependencies:

```bash
pip install torch torchvision timm accelerate numpy matplotlib opencv-python
```

On this machine, the experiments were run in a WSL conda environment with `torch` available.

## Experiment A: Unified Optimizer Protocols

Command:

```bash
python pinn.py
```

Purpose:

- remove the concern that only one method is benefiting from a hand-tuned optimizer
- apply the same optimizer protocol to all compared models

Protocols:

1. `Adam only`
2. `LBFGS only`
3. `Adam -> LBFGS`

Default setting:

- `2000` epochs
- switch at epoch `500` for the hybrid case

Output:

- `benchmark_comparison_three_protocols.png`

Status:

- script and figure are included
- the current README does not yet tabulate final numeric values for all nine runs

## Experiment B: MLP Parameter Sweep Under A Fixed Hybrid Protocol

Command:

```bash
python pinn_mlp_param_sweep.py
```

Protocol:

- `2000` epochs total
- `Adam` for the first `500` epochs
- `LBFGS` for the remaining `1500` epochs

Purpose:

- test whether a very small MLP can outperform larger MLPs under the same training rule
- remove optimizer asymmetry from the comparison

Reproduced results from the current run:

| MLP Config | Hidden Dims | Params | Best Train Loss | Time (s) |
|------------|-------------|--------|-----------------|----------|
| 20 params | `[3, 2]` | 20 | `1.265505e-02` | `474.86` |
| 70 params | `[3, 12]` | 70 | `3.676699e-06` | `402.02` |
| 100 params | `[5, 12]` | 100 | `1.167131e-06` | `246.40` |
| 1000 params | `[13, 64]` | 1000 | `5.900200e-07` | `152.28` |

Observation:

- in this run, the `20`-parameter MLP is clearly worse than the larger MLPs under the same `Adam -> LBFGS` protocol
- larger models improve monotonically in best training loss within this sweep

Output:

- `mlp_param_sweep_adam_lbfgs.png`

## Experiment C: Reviewer-Style MLP Protocol

Command:

```bash
python pinn_mlp_reviewer_protocol.py
```

Purpose:

- reproduce the reviewer-proposed MLP settings as closely as possible
- match the optimizer description `Adam + cosine scheduling`, with optional `Adam -> LBFGS` switching after epoch `500`

Models:

- `MLP [2,5,1]` with `Tanh`
- `MLP [2,20,1]` with `Tanh`
- `MLP [2,50,50,1]` with `Tanh`
- `MLP [2,50,50,1]` with `GELU`

Current script default:

```python
protocols_to_run = ["adam"]
```

If the hybrid variant is also desired:

```python
protocols_to_run = ["adam", "hybrid"]
```

Currently confirmed numbers from our run log:

| Model | Activation | Params | Optimizer | Scheduler | Best Train Loss | Time (s) | Status |
|-------|------------|--------|-----------|-----------|-----------------|----------|--------|
| `[2,5,1]` | Tanh | 21 | Adam | Cosine | `6.45585e-03` | `592.53` | complete |
| `[2,20,1]` | Tanh | 81 | Adam | Cosine | `1.519e-05` | `750.70` | complete |
| `[2,50,50,1]` | Tanh | 2751 | Adam | Cosine | `3.9e-07` | `142.19` | complete |
| `[2,50,50,1]` | GELU | 2751 | Adam | Cosine | pending | pending | not yet finalized in log |

Comparison to the reviewer claim:

| Setting | Reviewer-reported MSE | Our current reproduced metric | Comment |
|---------|------------------------|-------------------------------|---------|
| `[2,5,1]`, Tanh, Adam | `6.6e-3` | best train loss `6.46e-3` | numerically very close |
| `[2,20,1]`, Tanh, Adam | `6.6e-4` | best train loss `1.52e-5` | not directly comparable unless the same metric is used |
| `[2,50,50,1]`, Tanh, Adam | `6.3e-6` | best train loss `3.9e-7` | metric mismatch likely matters |
| `[2,50,50,1]`, GELU, Adam | `3.1e-5` | pending | run not yet completed in the saved log |

Important note:

- the reviewer numbers are stated as `MSE`
- some of the currently logged values above are `best train loss`, not yet a separately reported grid-evaluated solution MSE
- therefore, exact numeric disagreement should not be over-interpreted until the same evaluation metric is used on both sides

Output:

- `reviewer_mlp_protocol_adam.png`
- `reviewer_mlp_protocol_hybrid.png` if hybrid is enabled and run

## Main Takeaway For The Rebuttal

The artifact is intended to support two rebuttal points:

1. We do not rely on optimizer special treatment for only one method; instead, we provide a unified optimizer-protocol comparison across all methods.
2. Under a fixed hybrid protocol, extremely small MLPs do not automatically outperform larger MLPs; in our current sweep, the smallest MLP is substantially worse.

## Figures

The generated figures included in this repository are:

- `benchmark_comparison_three_protocols.png`
- `mlp_param_sweep_adam_lbfgs.png`
- `reviewer_mlp_protocol_adam.png`

## Reproducibility Note

Wall-clock times depend on hardware, CUDA initialization, and environment. The times shown above should be treated as run logs from this machine rather than universal speed claims.
