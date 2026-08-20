"""Command-line interface for configuration-driven runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .application import run_etas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="torchstpp",
        description="Fit an ETAS2D or point-source ETAS3D model from TOML.",
    )
    parser.add_argument("config", type=Path, help="path to the ETAS TOML file")
    parser.add_argument(
        "--no-output", action="store_true", help="fit without writing result files"
    )
    args = parser.parse_args(argv)
    result = run_etas(args.config, write_outputs=not args.no_output)
    run = result.run
    print(f"converged: {run.fit_result.converged}")
    print(f"negative log-likelihood: {run.fit_result.negative_log_likelihood:.8g}")
    print(f"outer iterations: {run.fit_result.outer_iterations}")
    print(f"elapsed seconds: {run.elapsed_seconds:.6f}")
    if result.output_directory is not None:
        print(f"output: {result.output_directory}")
    return 0
