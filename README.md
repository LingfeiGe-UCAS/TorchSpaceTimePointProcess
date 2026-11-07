# TorchSTPP

TorchSTPP is an extensible PyTorch framework for scalable spatiotemporal
point-process modelling, with an initial focus on earthquake catalogues and
ETAS models.

## Current capabilities

- Exact ETAS2D, depth-aware ETAS3D, and three-dimensional finite-source ETAS likelihoods.
- Finite-source patch updates with blocked or FFT-based smoothing.
- Patch-level activation checkpointing and reusable background calculations.
- Differentiable temporal, spatial, depth, and background components.
- EM and L-BFGS inference compatible with the legacy Fortran workflow.
- Chunked and checkpointed event-pair computation.
- Multi-GPU exact-likelihood execution with causal-work-balanced pair budgets.
- Configurable catalogue preprocessing, domains, projections, extraction, and
  reproducible TOML-based runs.

## Installation

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## Running

Run a configured single-GPU experiment:

```bash
torchstpp path/to/config.toml
```

Run the distributed exact likelihood with an optional event-pair budget:

```bash
torchrun --standalone --nproc-per-node=4 \
  examples/distributed_exact_likelihood.py \
  path/to/config.toml \
  --pair-budget 16000000 \
  --source-alignment 256
```

The event-pair budget controls the maximum target-by-source tile size; it does
not approximate or sparsify the exact likelihood. See
`docs/指定ST事件对预算说明.md` for details.

## Repository scope

Earthquake catalogues, generated outputs, checkpoints, manuscript working
files, and archived development snapshots are intentionally excluded from the
repository.
