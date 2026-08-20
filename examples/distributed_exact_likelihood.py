"""Run ETAS2D or ETAS3D with a distributed exact M step.

Example::

    torchrun --standalone --nproc-per-node=4 \
        examples/distributed_exact_likelihood.py configs/run.toml --write-output
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from torchstpp.config import TorchSTPPConfig, build_project, load_config
from torchstpp.io import RelationWriter, ResultWriter
from torchstpp.likelihood import DistributedExactLikelihood
from torchstpp.relations import RelationExtractor


def _local_config(config: TorchSTPPConfig, local_rank: int) -> TorchSTPPConfig:
    data = copy.deepcopy(dict(config.data))
    runtime = dict(data.get("runtime", {}))
    runtime["device"] = f"cuda:{local_rank}"
    data["runtime"] = runtime
    return TorchSTPPConfig(data=data, path=config.path)


def _rank_zero_operation(operation, *, stage: str) -> object | None:
    result = None
    error: Exception | None = None
    if dist.get_rank() == 0:
        try:
            result = operation()
        except Exception as caught:
            error = caught
    failed = torch.tensor(
        int(error is not None),
        dtype=torch.int32,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if int(failed.item()) != 0:
        if error is None:
            raise RuntimeError(f"rank 0 failed during {stage}")
        raise RuntimeError(f"rank 0 failed during {stage}: {error}") from error
    return result


def _all_rank_operation(operation, *, stage: str):
    result = None
    error: Exception | None = None
    try:
        result = operation()
    except Exception as caught:
        error = caught
    failed = torch.tensor(
        int(error is not None),
        dtype=torch.int32,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if int(failed.item()) != 0:
        if error is None:
            raise RuntimeError(f"another rank failed during {stage}")
        raise RuntimeError(f"rank {dist.get_rank()} failed during {stage}: {error}") from error
    return result


def _output_directory(config: TorchSTPPConfig) -> Path:
    output = config.section("output")
    return config.resolve_path(output.get("directory", "../output/distributed_run"))


def _require_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")


def _write_outputs(run, config: TorchSTPPConfig) -> dict[str, object]:
    directory = _output_directory(config)
    writer = ResultWriter(directory)
    paths = writer.write_all(run)
    relation_manifest = None
    relations = config.section("relations")
    if bool(relations.get("enabled", False)):
        extractor = RelationExtractor(
            probability_threshold=float(
                relations.get("probability_threshold", 1e-4)
            ),
            target_chunk_size=int(relations.get("target_chunk_size", 2048)),
            source_chunk_size=int(relations.get("source_chunk_size", 4096)),
        )
        relation_manifest = RelationWriter(
            directory / relations.get("directory", "relations")
        ).write(run, extractor)
    return {
        "output_directory": str(directory),
        "paths": {name: str(path) for name, path in paths.items()},
        "relations": relation_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--write-output",
        action="store_true",
        help="write normal result products on rank 0",
    )
    parser.add_argument(
        "--pair-budget",
        type=int,
        default=None,
        help=(
            "optional maximum s*t pair elements per likelihood compute tile; "
            "enables causal-work-balanced target partitioning"
        ),
    )
    parser.add_argument(
        "--source-alignment",
        type=int,
        default=256,
        help="alignment used when deriving source chunk size from pair budget",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("distributed GPU execution requires CUDA")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch this program with torchrun")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        original = load_config(args.config)
        if args.write_output:
            _rank_zero_operation(
                lambda: _require_empty(_output_directory(original)),
                stage="output preflight",
            )
        project = _all_rank_operation(
            lambda: build_project(_local_config(original, local_rank)),
            stage="project construction",
        )
        assert project is not None
        project.likelihood = DistributedExactLikelihood(
            project.likelihood,
            pair_budget=args.pair_budget,
            source_alignment=args.source_alignment,
        )
        run = _all_rank_operation(project.fit, stage="distributed fit")
        assert run is not None

        output = None
        if args.write_output:
            output = _rank_zero_operation(
                lambda: _write_outputs(run, original),
                stage="result writing",
            )
        if dist.get_rank() == 0:
            pair_plan = project.likelihood.last_pair_budget_plan
            print(
                json.dumps(
                    {
                        "world_size": dist.get_world_size(),
                        "negative_log_likelihood": run.fit_result.negative_log_likelihood,
                        "outer_iterations": run.fit_result.outer_iterations,
                        "converged": run.fit_result.converged,
                        "stop_reason": run.fit_result.stop_reason,
                        "fit_elapsed_seconds": run.fit_elapsed_seconds,
                        "peak_memory_bytes_per_rank": run.peak_memory_bytes,
                        "pair_budget_plan": (
                            None
                            if pair_plan is None
                            else {
                                "pair_budget": pair_plan.pair_budget,
                                "target_ranges": pair_plan.target_ranges,
                                "source_ranges": pair_plan.source_ranges,
                                "target_chunk_size": pair_plan.target_chunk_size,
                                "source_chunk_size": pair_plan.source_chunk_size,
                            }
                        ),
                        "parameters": {
                            name: float(value.detach().item())
                            for name, value in run.model.parameter_values().items()
                        },
                        "output": output,
                    },
                    indent=2,
                ),
                flush=True,
            )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
