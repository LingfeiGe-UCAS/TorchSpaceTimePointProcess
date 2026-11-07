# TorchSTPP

## Overview

TorchSTPP is an extensible PyTorch framework for fitting and analyzing earthquake
point-process models. It separates model-specific definitions from shared catalog
preparation, likelihood evaluation, parameter estimation, and post-fit analysis.
Compatible model components can reuse this workflow and obtain parameter gradients
through automatic differentiation.

The current implementation includes three epidemic-type aftershock sequence (ETAS)
models, covering two-dimensional point sources, three-dimensional point sources,
and three-dimensional finite sources. Fitted results support analysis of background
seismicity, event-to-event triggering probabilities, and spatial variations in
aftershock productivity within large-earthquake source regions.

## Main features

- **Extensible model interfaces** for parameters, background rates, pairwise
  triggering rates, and integrated intensities.
- **Full-history likelihood evaluation** with blockwise target-source calculations.
- **Memory-aware computation** through event-block and finite-source patch
  activation checkpointing, configurable block sizes, and pair-element budgets.
- **Finite-source updates** with blocked direct smoothing or FFT-based horizontal
  smoothing combined with conditional depth kernels.
- **Shared fitting procedures** that alternate between state updates and L-BFGS
  parameter optimization, with configurable iteration schedules and stopping criteria.
- **Single- and multi-GPU execution**, including workload-based target partitioning
  and global objective and gradient reductions.
- **Configurable data preparation** for observation domains, projections, magnitude
  and depth selection, duplicate handling, and kernel bandwidth estimation.
- **Post-fit outputs** for parameters, optimization histories, event intensities,
  background probabilities, and finite-source patch states.

## Installation

Python 3.10 or later is required. Install a PyTorch build appropriate for your CPU
or CUDA environment using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

Clone the repository and install the package from its root directory.

```bash
git clone https://github.com/LingfeiGe-UCAS/TorchSpaceTimePointProcess.git
cd TorchSpaceTimePointProcess
python -m pip install -e .
```

Package dependencies include PyTorch, Polars, and pyproj. To install the test
dependencies and run the existing tests, use

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

## Quick start

### Run the included example

The repository includes a small eight-event catalog and an externally defined
three-dimensional ETAS model. Run the following command from the repository root.

```bash
python examples/custom_etas3d.py examples/custom_etas3d.toml
```

This CPU example runs one outer iteration and prints the model name, negative
log-likelihood, and fitted parameters. Output-file writing is disabled in this
small example. A successful run prints a model name of `UserETAS3D` and an NLL
of approximately `41.6789`.

The model implementation is in [examples/custom_etas3d.py](examples/custom_etas3d.py),
with settings in [examples/custom_etas3d.toml](examples/custom_etas3d.toml).
The example registers a custom model factory and reuses the shared fitting workflow.

### Fit a built-in model

Prepare a TOML configuration for your catalog and run

```bash
torchstpp path/to/your_config.toml
```

The equivalent Python module entry point is

```bash
python -m torchstpp path/to/your_config.toml
```

Built-in model identifiers are `etas2d`, `etas3d`, and `finite_source_etas3d`.
For single-GPU execution, set `runtime.device = "cuda:0"` in your configuration.
The custom example above uses its own Python runner to register `custom_etas3d`.

## Inputs and outputs

### Inputs

An analysis is defined by a TOML configuration and an earthquake catalog.
The main configuration sections are `runtime`, `catalog`, `projection`, `domain`,
`model`, `background`, `likelihood`, `inference`, and `output`.

The catalog provides event times, locations, magnitudes, and depths for
three-dimensional models. Column mappings, units, and coordinate conventions
are specified in the configuration. Event identifiers can be supplied to retain
links to the original catalog. The included CSV example uses

```text
time,longitude,latitude,depth,mag
```

Finite-source models additionally require a geometry file containing source-group
identifiers, patch locations, depths, and productivity weights. Catalog
`finite_source_id` values associate each finite source with its parent earthquake.

Large research catalogs and experiment outputs are maintained separately.
The small catalog under `examples/data` is included for the quick-start example.

### Outputs

Enable `[output]` and specify its `directory` in the configuration to save fitted
results. Relative paths are resolved from the configuration file's directory.

| File | Contents |
| --- | --- |
| `summary.json` | Final NLL, optimization status, runtime, and recorded memory information |
| `parameters.csv` | Fitted parameters and parameter configuration |
| `optimization_history.csv` | Outer-iteration objectives, parameter trajectories, and optimizer diagnostics |
| `events.parquet` | Event identifiers, locations, background and total intensities, and background probabilities |
| `subsources.parquet` | Finite-source patch locations, productivity weights, and normalization information, when applicable |
| `checkpoint.pt` | Fitted model and analysis state for subsequent loading and evaluation |

Direct parent-offspring probabilities can be reconstructed in blocks from the
fitted state. Optional relation extraction and writing are controlled through
`[relations]`, including the probability threshold and block sizes.

## Multi-GPU computation

Distributed fitting uses one process per GPU and the NCCL communication backend.
Use a CUDA-enabled PyTorch installation and a Linux environment with the requested
number of GPUs available. Run the following command from the repository root,
using a configuration for a built-in model and your own input data.

```bash
torchrun --standalone --nproc-per-node=4 \
  examples/distributed_exact_likelihood.py \
  path/to/your_config.toml \
  --pair-budget 16000000 \
  --source-alignment 256 \
  --write-output
```

Change `--nproc-per-node` to the number of GPUs to use. `--write-output` enables
result writing by rank 0 and requires an empty output directory. Each process
receives a local CUDA device through `torchrun`.

The pair budget sets the maximum number of target-source elements in each
likelihood tile. The example value of 16 million is a starting point for your
hardware and model. Total memory use also depends on the model, finite-source
geometry, and checkpoint settings.

Event-block checkpointing is controlled by `likelihood.checkpoint_blocks`.
Finite-source patch checkpointing is controlled separately by
`finite_source.checkpoint_patches`. These options can be enabled together.
Finite-source smoothing is selected through `finite_source.smoothing_method`,
using `"blocked"` or `"fft"` for compatible regular-lattice geometries.

Additional distributed execution details are provided in
[the distributed likelihood guide](docs/distributed_exact_likelihood.md).

## Citation and license

When using TorchSTPP in research, please acknowledge the software and provide the
repository URL. The associated manuscript is titled *TorchSTPP: An extensible
PyTorch framework for scalable earthquake point-process modelling*.

```text
TorchSTPP
https://github.com/LingfeiGe-UCAS/TorchSpaceTimePointProcess
```

TorchSTPP is distributed under the [Apache License 2.0](LICENSE).
