# Distributed ExactLikelihood

The distributed implementation uses one process per GPU. Every process keeps
the complete catalog and KDE state, while target-event blocks and triggering
compensator source blocks are assigned block-cyclically. Parameter gradients
are summed across ranks, preserving the total negative log-likelihood used by
the single-device implementation.

Launch on one node with four GPUs:

```bash
torchrun --standalone --nproc-per-node=4 \
  examples/distributed_exact_likelihood.py \
  configs/qtm_mc2_etas2d_float64_checkpoint.toml \
  --write-output
```

The TOML file is unchanged. `LOCAL_RANK` selects `cuda:0`, `cuda:1`, and so
on inside each process. Only rank 0 writes the configured output directory.
Use a new, empty output directory for every run.

The current implementation parallelizes the exact M-step. Catalog loading,
KNN bandwidth construction, EM probability updates, and final event-state
construction are still repeated on every rank. Per-GPU peak memory therefore
remains close to a single-device run; the primary benefit is reduced wall time
for exact likelihood evaluation.

Before a production benchmark, run:

```bash
python -m pytest -q tests/likelihood/test_distributed_exact.py
```

The test covers single-process fallback, two-process Gloo loss and gradient
agreement, ETAS2D Fortran EM/LBFGS agreement, checkpointing, and ETAS3D's ten
parameter gradients. NCCL performance and scaling must be measured on the
target multi-GPU server.
