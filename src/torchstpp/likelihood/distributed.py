"""Distributed exact likelihood with replicated source history."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import hashlib
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from .exact import ExactLikelihood
from .partition import PairBudgetPlan, build_pair_budget_plan
from .terms import LikelihoodTerms


def block_cyclic_ranges(
    total: int,
    block_size: int,
    *,
    rank: int,
    world_size: int,
) -> Iterator[tuple[int, int]]:
    """Yield disjoint ranges assigned to one rank."""

    if total < 0:
        raise ValueError("total must be non-negative")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    for block_index, begin in enumerate(range(0, total, block_size)):
        if block_index % world_size == rank:
            yield begin, min(begin + block_size, total)


@dataclass(frozen=True)
class DistributedLikelihoodTerms:
    """Local graph and globally reduced values for one likelihood call."""

    backward_loss: torch.Tensor
    local_terms: LikelihoodTerms
    global_terms: LikelihoodTerms
    rank: int
    world_size: int


class DistributedExactLikelihood(nn.Module):
    """Parallelize exact likelihood blocks across one process per device.

    Every rank keeps the full catalog and KDE state because each target
    intensity depends on the complete earlier source history. Target blocks
    By default target blocks and triggering-compensator source blocks are assigned
    block-cyclically. Setting ``pair_budget`` instead creates time-contiguous,
    causal-work-balanced target ranges and common compute tiles satisfying
    ``target_chunk_size * source_chunk_size <= pair_budget``. Parameter
    gradients are summed explicitly so the optimized objective is
    the total NLL, not a per-rank average.
    """

    def __init__(
        self,
        exact: ExactLikelihood | None = None,
        *,
        process_group: Any | None = None,
        check_finite_gradients: bool = True,
        pair_budget: int | None = None,
        source_alignment: int = 256,
    ) -> None:
        super().__init__()
        if process_group is not None and not (
            dist.is_available() and dist.is_initialized()
        ):
            raise RuntimeError(
                "an explicit process_group requires torch.distributed initialization"
            )
        if pair_budget is not None and pair_budget < 1:
            raise ValueError("pair_budget must be positive")
        if source_alignment < 1:
            raise ValueError("source_alignment must be positive")
        self.exact = exact or ExactLikelihood()
        self.process_group = process_group
        self.check_finite_gradients = check_finite_gradients
        self.pair_budget = pair_budget
        self.source_alignment = source_alignment
        self._last_pair_budget_plan: PairBudgetPlan | None = None
        self._replicas_validated = False
        self._parameter_structure_validated = False
        self._state_structure_validated = False

    @property
    def target_chunk_size(self) -> int:
        return self.exact.target_chunk_size

    @property
    def source_chunk_size(self) -> int:
        return self.exact.source_chunk_size

    @property
    def checkpoint_blocks(self) -> bool:
        return self.exact.checkpoint_blocks

    @property
    def last_pair_budget_plan(self) -> PairBudgetPlan | None:
        """Return the most recent deterministic pair-budget plan, if enabled."""

        return self._last_pair_budget_plan

    @property
    def rank(self) -> int:
        if not self._distributed_available:
            return 0
        return dist.get_rank(group=self.process_group)

    @property
    def world_size(self) -> int:
        if not self._distributed_available:
            return 1
        return dist.get_world_size(group=self.process_group)

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def _distributed_available(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    @property
    def _distributed(self) -> bool:
        return self.world_size > 1

    def _global_source_rank(self, group_rank: int = 0) -> int:
        if self.process_group is None:
            return group_rank
        get_global_rank = getattr(dist, "get_global_rank", None)
        if get_global_rank is None:
            raise RuntimeError(
                "explicit process groups require torch.distributed.get_global_rank"
            )
        return int(get_global_rank(self.process_group, group_rank))

    @staticmethod
    def _detached_terms(terms: LikelihoodTerms) -> LikelihoodTerms:
        return LikelihoodTerms(
            negative_log_likelihood=terms.negative_log_likelihood.detach().clone(),
            log_intensity=terms.log_intensity.detach().clone(),
            background_compensator=terms.background_compensator.detach().clone(),
            triggering_compensator=terms.triggering_compensator.detach().clone(),
            target_count=terms.target_count,
        )

    @staticmethod
    def _zero_parameter_anchor(
        model: nn.Module, reference: torch.Tensor
    ) -> torch.Tensor:
        anchor = reference.new_zeros(())
        for parameter in model.parameters():
            if parameter.requires_grad:
                anchor = anchor + parameter.sum() * 0.0
        return anchor

    def synchronize_failure(
        self,
        error: Exception | None,
        reference: torch.Tensor,
        *,
        stage: str,
    ) -> None:
        """Raise on every rank if an operation failed on any rank."""

        if not self._distributed:
            if error is not None:
                raise error
            return
        failed = torch.tensor(
            int(error is not None), dtype=torch.int32, device=reference.device
        )
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self.process_group)
        if int(failed.item()) != 0:
            message = f"distributed operation failed during {stage}"
            if error is None:
                raise RuntimeError(f"{message} on another rank")
            raise RuntimeError(f"{message}: {error}") from error

    @staticmethod
    def _digest_value(hasher: Any, name: str, value: Any) -> None:
        hasher.update(name.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous().cpu()
            hasher.update(str(tensor.dtype).encode("ascii"))
            hasher.update(repr(tuple(tensor.shape)).encode("ascii"))
            hasher.update(tensor.numpy().tobytes())
        else:
            hasher.update(repr(value).encode("utf-8"))

    def validate_replicas_(
        self,
        model: nn.Module,
        catalog: Any,
        domain: Any,
        background_state: Any,
    ) -> None:
        """Validate replicated values once, outside repeated LBFGS closures."""

        if not self._distributed or self._replicas_validated:
            return
        digest: str | None = None
        error: Exception | None = None
        try:
            hasher = hashlib.sha256()
            self._digest_value(
                hasher,
                "likelihood_config",
                (
                    self.target_chunk_size,
                    self.source_chunk_size,
                    self.checkpoint_blocks,
                    self.check_finite_gradients,
                    self.pair_budget,
                    self.source_alignment,
                ),
            )
            self._digest_value(hasher, "model_type", type(model).__qualname__)
            for name, value in sorted(model.state_dict().items()):
                self._digest_value(hasher, f"model.{name}", value)
            for owner_name, owner, names in (
                (
                    "catalog",
                    catalog,
                    (
                        "event_id", "time", "position", "magnitude",
                        "completeness_magnitude", "depth", "inside",
                    ),
                ),
                (
                    "domain",
                    domain,
                    (
                        "time_zero", "time_start", "time_end", "border",
                        "spatial_kind", "include_start", "include_end",
                    ),
                ),
                (
                    "background_state",
                    background_state,
                    ("position", "bandwidth", "weights", "duration", "depth"),
                ),
            ):
                for name in names:
                    self._digest_value(
                        hasher, f"{owner_name}.{name}", getattr(owner, name, None)
                    )
            digest = hasher.hexdigest()
        except Exception as caught:
            error = caught
        self.synchronize_failure(
            error, catalog.time, stage="replicated-data validation"
        )
        assert digest is not None
        digests: list[Any] = [None] * self.world_size
        dist.all_gather_object(digests, digest, group=self.process_group)
        if any(item != digests[0] for item in digests[1:]):
            raise ValueError(
                "catalog, domain, model, and initial KDEState must match on every rank"
            )
        self._replicas_validated = True

    def _select_replicated_data(
        self, model: nn.Module, catalog: Any, domain: Any
    ) -> tuple[Any, Any]:
        targets = None
        sources = None
        error: Exception | None = None
        try:
            self.exact._validate_inputs(catalog, domain)
            targets = self.exact.select_targets(catalog, domain)
            sources = self.exact.select_sources(catalog, domain)
        except Exception as caught:
            error = caught
        self.synchronize_failure(error, catalog.time, stage="input selection")
        assert targets is not None and sources is not None
        if self._distributed:
            counts = torch.tensor(
                [len(catalog.time), len(targets.time), len(sources.time)],
                dtype=torch.int64,
                device=catalog.time.device,
            )
            minimum = counts.clone()
            maximum = counts.clone()
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN, group=self.process_group)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=self.process_group)
            if not torch.equal(minimum, maximum):
                raise ValueError("catalog, target, and source counts differ across ranks")
        return targets, sources

    def local_terms(
        self,
        model: nn.Module,
        catalog: Any,
        domain: Any,
        background_state: Any,
    ) -> LikelihoodTerms:
        """Compute this rank's disjoint contribution to the global NLL."""

        targets, sources = self._select_replicated_data(model, catalog, domain)
        result: LikelihoodTerms | None = None
        error: Exception | None = None
        try:
            log_intensity = catalog.time.new_zeros(())
            local_target_count = 0
            triggering = catalog.time.new_zeros(())
            compute_exact = self.exact

            if self.pair_budget is None:
                for begin, end in block_cyclic_ranges(
                    targets.time.numel(),
                    self.target_chunk_size,
                    rank=self.rank,
                    world_size=self.world_size,
                ):
                    target_block = targets.take(slice(begin, end))
                    log_intensity = log_intensity + compute_exact.event_log_intensity(
                        model, target_block, sources, background_state
                    )
                    local_target_count += end - begin

                for begin, end in block_cyclic_ranges(
                    sources.time.numel(),
                    self.source_chunk_size,
                    rank=self.rank,
                    world_size=self.world_size,
                ):
                    source_block = sources.take(slice(begin, end))
                    triggering = triggering + compute_exact.triggering_compensator(
                        model, source_block, domain
                    )
            else:
                background_source_count = int(background_state.position.shape[0])
                plan = build_pair_budget_plan(
                    targets.time,
                    sources.time,
                    world_size=self.world_size,
                    pair_budget=self.pair_budget,
                    target_chunk_cap=self.target_chunk_size,
                    background_source_count=background_source_count,
                    source_alignment=self.source_alignment,
                )
                self._last_pair_budget_plan = plan
                compute_exact = ExactLikelihood(
                    target_chunk_size=plan.target_chunk_size,
                    source_chunk_size=plan.source_chunk_size,
                    checkpoint_blocks=self.checkpoint_blocks,
                )

                target_begin, target_end = plan.target_range(self.rank)
                if target_end > target_begin:
                    target_block = targets.take(slice(target_begin, target_end))
                    log_intensity = compute_exact.event_log_intensity(
                        model, target_block, sources, background_state
                    )
                    local_target_count = target_end - target_begin

                source_begin, source_end = plan.source_range(self.rank)
                if source_end > source_begin:
                    source_block = sources.take(slice(source_begin, source_end))
                    triggering = compute_exact.triggering_compensator(
                        model, source_block, domain
                    )

            background = compute_exact.background_compensator(
                model, domain, background_state
            ) / self.world_size
            nll = (
                background
                + triggering
                - log_intensity
                + self._zero_parameter_anchor(model, catalog.time)
            )
            result = LikelihoodTerms(
                negative_log_likelihood=nll,
                log_intensity=log_intensity,
                background_compensator=background,
                triggering_compensator=triggering,
                target_count=local_target_count,
            )
        except Exception as caught:
            error = caught
        self.synchronize_failure(error, catalog.time, stage="local likelihood")
        assert result is not None
        return result

    def _reduce_terms(
        self, local: LikelihoodTerms
    ) -> DistributedLikelihoodTerms:
        if not self._distributed:
            return DistributedLikelihoodTerms(
                backward_loss=local.negative_log_likelihood,
                local_terms=local,
                global_terms=self._detached_terms(local),
                rank=0,
                world_size=1,
            )
        components = torch.stack(
            (
                local.log_intensity.detach(),
                local.background_compensator.detach(),
                local.triggering_compensator.detach(),
            )
        )
        global_components = components.clone()
        dist.all_reduce(
            global_components, op=dist.ReduceOp.SUM, group=self.process_group
        )
        global_log, global_background, global_triggering = global_components.unbind()
        global_nll = global_background + global_triggering - global_log
        count = torch.tensor(
            local.target_count,
            dtype=torch.int64,
            device=local.negative_log_likelihood.device,
        )
        dist.all_reduce(count, op=dist.ReduceOp.SUM, group=self.process_group)
        global_terms = LikelihoodTerms(
            negative_log_likelihood=global_nll,
            log_intensity=global_log,
            background_compensator=global_background,
            triggering_compensator=global_triggering,
            target_count=int(count.item()),
        )
        backward_loss = local.negative_log_likelihood + (
            global_nll - local.negative_log_likelihood.detach()
        )
        return DistributedLikelihoodTerms(
            backward_loss=backward_loss,
            local_terms=local,
            global_terms=global_terms,
            rank=self.rank,
            world_size=self.world_size,
        )

    def terms(
        self,
        model: nn.Module,
        catalog: Any,
        domain: Any,
        background_state: Any,
    ) -> DistributedLikelihoodTerms:
        if not self._distributed and self.pair_budget is None:
            single = self.exact.terms(model, catalog, domain, background_state)
            return DistributedLikelihoodTerms(
                backward_loss=single.negative_log_likelihood,
                local_terms=single,
                global_terms=self._detached_terms(single),
                rank=0,
                world_size=1,
            )
        return self._reduce_terms(
            self.local_terms(model, catalog, domain, background_state)
        )

    def forward(
        self,
        model: nn.Module,
        catalog: Any,
        domain: Any,
        background_state: Any,
    ) -> torch.Tensor:
        return self.terms(model, catalog, domain, background_state).backward_loss

    @torch.no_grad()
    def sum_gradients_(
        self, model_or_parameters: nn.Module | Iterable[torch.Tensor]
    ) -> None:
        """Sum dense trainable gradients across ranks in place."""

        if not self._distributed:
            return
        parameters = (
            list(model_or_parameters.parameters())
            if isinstance(model_or_parameters, nn.Module)
            else list(model_or_parameters)
        )
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        error: Exception | None = None
        try:
            for parameter in trainable:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                if parameter.grad.is_sparse or parameter.grad.shape != parameter.shape:
                    raise TypeError("distributed gradients must be dense and shape-matched")
        except Exception as caught:
            error = caught
        reference = trainable[0] if trainable else torch.zeros(())
        self.synchronize_failure(error, reference, stage="gradient validation")
        for parameter in trainable:
            dist.all_reduce(
                parameter.grad, op=dist.ReduceOp.SUM, group=self.process_group
            )
        if self.check_finite_gradients and any(
            not bool(torch.all(torch.isfinite(parameter.grad)))
            for parameter in trainable
        ):
            raise FloatingPointError("distributed gradients contain non-finite values")

    @torch.no_grad()
    def broadcast_parameters_(
        self, model: nn.Module, *, source_rank: int = 0
    ) -> None:
        if not self._distributed:
            return
        named_parameters = list(model.named_parameters())
        if not self._parameter_structure_validated:
            signature = [
                (
                    name,
                    tuple(parameter.shape),
                    str(parameter.dtype),
                    parameter.device.type,
                    bool(parameter.requires_grad),
                )
                for name, parameter in named_parameters
            ]
            signatures: list[Any] = [None] * self.world_size
            dist.all_gather_object(
                signatures, signature, group=self.process_group
            )
            if any(item != signatures[0] for item in signatures[1:]):
                raise ValueError("model parameter structure differs across ranks")
            self._parameter_structure_validated = True
        source = self._global_source_rank(source_rank)
        for _, parameter in named_parameters:
            dist.broadcast(parameter, src=source, group=self.process_group)

    @torch.no_grad()
    def broadcast_background_state_(
        self, state: Any, reference: torch.Tensor, *, source_rank: int = 0
    ) -> None:
        if not self._distributed:
            return
        if not self._state_structure_validated:
            signature = []
            for name in ("position", "bandwidth", "weights", "depth"):
                value = getattr(state, name, None)
                signature.append(
                    None
                    if value is None
                    else (name, tuple(value.shape), str(value.dtype), value.device.type)
                )
            signatures: list[Any] = [None] * self.world_size
            dist.all_gather_object(
                signatures, signature, group=self.process_group
            )
            if any(item != signatures[0] for item in signatures[1:]):
                raise ValueError("KDEState tensor structure differs across ranks")
            self._state_structure_validated = True
        source = self._global_source_rank(source_rank)
        for name in ("position", "bandwidth", "weights", "depth"):
            value = getattr(state, name, None)
            if value is not None:
                dist.broadcast(value, src=source, group=self.process_group)
        duration = torch.as_tensor(
            state.duration, dtype=reference.dtype, device=reference.device
        ).clone()
        dist.broadcast(duration, src=source, group=self.process_group)
        if isinstance(state.duration, torch.Tensor):
            state.duration.copy_(duration)
        else:
            state.duration = float(duration.item())

    def global_max(self, value: float, reference: torch.Tensor) -> float:
        if not self._distributed:
            return value
        result = torch.tensor(value, dtype=reference.dtype, device=reference.device)
        dist.all_reduce(result, op=dist.ReduceOp.MAX, group=self.process_group)
        return float(result.item())

    @torch.no_grad()
    def global_sum_(self, value: torch.Tensor) -> None:
        """Sum a replicated-shape tensor's disjoint local contributions in place."""

        if self._distributed:
            dist.all_reduce(value, op=dist.ReduceOp.SUM, group=self.process_group)

    def local_target_ranges(
        self,
        target_time: torch.Tensor,
        source_time: torch.Tensor,
        *,
        background_source_count: int,
    ) -> tuple[tuple[int, int], ...]:
        """Return this rank's target ranges using the active parallel schedule."""

        if self.pair_budget is None:
            return tuple(
                block_cyclic_ranges(
                    int(target_time.numel()),
                    self.target_chunk_size,
                    rank=self.rank,
                    world_size=self.world_size,
                )
            )
        plan = build_pair_budget_plan(
            target_time,
            source_time,
            world_size=self.world_size,
            pair_budget=self.pair_budget,
            target_chunk_cap=self.target_chunk_size,
            background_source_count=background_source_count,
            source_alignment=self.source_alignment,
        )
        begin, end = plan.target_range(self.rank)
        return tuple(
            (block_begin, min(block_begin + plan.target_chunk_size, end))
            for block_begin in range(begin, end, plan.target_chunk_size)
        )

    def barrier(self) -> None:
        if self._distributed:
            dist.barrier(group=self.process_group)

    def backward_and_sum_gradients_(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        *,
        retain_graph: bool = False,
    ) -> None:
        error: Exception | None = None
        try:
            loss.backward(retain_graph=retain_graph)
        except Exception as caught:
            error = caught
        self.synchronize_failure(error, loss, stage="backward")
        self.sum_gradients_(model)
