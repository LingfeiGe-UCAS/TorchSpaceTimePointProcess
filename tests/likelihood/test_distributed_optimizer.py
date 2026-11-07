"""Regression tests for single-leader distributed L-BFGS control flow.

Use a tiny nonlinear objective so these tests need neither CUDA nor a catalog.
Both collective and parent-process timeouts bound failures that otherwise hang.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from torchstpp.likelihood import DistributedExactLikelihood
from torchstpp.likelihood.terms import LikelihoodTerms


_SCHEDULE = (2, 8, 2)
_WORLD_SIZE = 2


class _RosenbrockModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.position = nn.Parameter(torch.tensor([-1.2, 1.0], dtype=torch.float64))

    def forward(self) -> torch.Tensor:
        x, y = self.position.unbind()
        return (1.0 - x).square() + 100.0 * (y - x.square()).square()


def _optimizer(model: nn.Module) -> torch.optim.LBFGS:
    return torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=_SCHEDULE[0],
        max_eval=_SCHEDULE[0] * 5 // 4,
        history_size=7,
        tolerance_grad=0.0,
        tolerance_change=0.0,
        line_search_fn="strong_wolfe",
    )


def _terms(local_loss: torch.Tensor) -> LikelihoodTerms:
    zero = local_loss.new_zeros(())
    return LikelihoodTerms(
        negative_log_likelihood=local_loss,
        log_intensity=zero,
        background_compensator=local_loss,
        triggering_compensator=zero,
        target_count=1,
    )


def _run_trajectory(*, coordinated: bool) -> dict:
    """Keep the same optimizer, including curvature history, across M steps."""
    model = _RosenbrockModel()
    optimizer = _optimizer(model)
    likelihood = DistributedExactLikelihood() if coordinated else None
    rank = likelihood.rank if likelihood is not None else 0
    world_size = likelihood.world_size if likelihood is not None else 1
    if rank != 0:
        # The initial EVAL must send actual leader parameters, too.
        with torch.no_grad():
            model.position.copy_(model.position.new_tensor([4.0, -3.0]))

    optimizer_step_calls = 0
    real_step = optimizer.step

    def counted_step(closure):
        nonlocal optimizer_step_calls
        optimizer_step_calls += 1
        if rank != 0:
            raise AssertionError("a worker must never call optimizer.step")
        return real_step(closure)

    optimizer.step = counted_step
    accepted = []
    trials = []
    gradients = []
    returned_losses = []
    stats = []
    closure_counts = []
    history_sizes = []
    state_totals = []
    for max_iter in _SCHEDULE:
        optimizer.param_groups[0]["max_iter"] = max_iter
        optimizer.param_groups[0]["max_eval"] = max_iter * 5 // 4
        evaluation_count = 0

        def closure():
            nonlocal evaluation_count
            evaluation_count += 1
            optimizer.zero_grad(set_to_none=True)
            trials.append(model.position.detach().clone())
            objective = model()
            if likelihood is None:
                loss = objective
                loss.backward()
            else:
                reduced = likelihood._reduce_terms(_terms(objective / world_size))
                loss = reduced.backward_loss
                likelihood.backward_and_sum_gradients_(loss, model)
            gradients.append(model.position.grad.detach().clone())
            # Deliberately unsuitable for an independent worker line search:
            # parameter-dependent values but precisely unchanged gradients.
            if rank != 0:
                loss = loss + (1.0e6 * model.position.square().sum()).detach()
            returned_losses.append(loss.detach().clone())
            return loss

        if likelihood is None:
            state = optimizer.state.get(model.position, {})
            before = (state.get("n_iter", 0), state.get("func_evals", 0))
            optimizer.step(closure)
            state = optimizer.state[model.position]
            step_stats = (
                state["n_iter"] - before[0],
                state["func_evals"] - before[1],
            )
        else:
            step_stats = likelihood.run_lbfgs_step_(optimizer, model, closure)
        state = optimizer.state.get(model.position, {})
        stats.append(step_stats)
        closure_counts.append(evaluation_count)
        accepted.append(model.position.detach().clone())
        history_sizes.append(len(state.get("old_dirs", [])))
        state_totals.append((state.get("n_iter", 0), state.get("func_evals", 0)))

    return {
        "accepted": torch.stack(accepted),
        "trials": torch.stack(trials),
        "gradients": torch.stack(gradients),
        "returned_losses": torch.stack(returned_losses),
        "stats": stats,
        "closure_counts": closure_counts,
        "history_sizes": history_sizes,
        "state_totals": state_totals,
        "optimizer_step_calls": optimizer_step_calls,
    }


def _trajectory_worker(rank: int) -> dict:
    return _run_trajectory(coordinated=True)


def _failure_worker(rank: int, phase: str) -> dict:
    model = _RosenbrockModel()
    likelihood = DistributedExactLikelihood()
    optimizer = _optimizer(model)
    closure_calls = 0
    step_calls = 0

    def closure():
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad(set_to_none=True)
        error = None
        if phase == "local_failure" and rank == 1:
            error = ValueError("intentional local likelihood failure")
        likelihood.synchronize_failure(
            error, model.position, stage="optimizer-test local likelihood"
        )
        reduced = likelihood._reduce_terms(_terms(model() / _WORLD_SIZE))
        likelihood.backward_and_sum_gradients_(reduced.backward_loss, model)
        return reduced.backward_loss

    real_step = optimizer.step

    def failing_step(closure):
        nonlocal step_calls
        step_calls += 1
        if rank != 0:
            raise AssertionError("a worker must never call optimizer.step")
        if phase == "before_closure":
            raise RuntimeError("intentional leader failure before closure")
        if phase == "after_closure":
            closure()
            raise RuntimeError("intentional leader failure after closure")
        return real_step(closure)

    optimizer.step = failing_step
    message = None
    try:
        likelihood.run_lbfgs_step_(optimizer, model, closure)
    except RuntimeError as error:
        message = str(error)

    # Not merely "both processes exited": after handling the error, ranks must
    # agree on the next collective, as the fitter performs further error sync.
    probe = torch.tensor(rank + 1, dtype=torch.int64)
    dist.all_reduce(probe)
    return {
        "message": message,
        "closure_calls": closure_calls,
        "step_calls": step_calls,
        "probe": int(probe.item()),
    }


def _reduce_worker(rank: int) -> dict:
    likelihood = DistributedExactLikelihood()
    model = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
    parameter = model.weight.squeeze()
    # Positive compensators and ordinary log-intensity values chosen so that
    # local + (global - local) differs by one ULP on rank zero, even in float64.
    background_values = (1.3013328971009641, 8.609822704622246)
    log_values = (4.9166847140006, 3.8315482800871434)
    coefficient = rank + 2
    background = parameter * coefficient + background_values[rank]
    log_intensity = parameter.new_tensor(log_values[rank])
    local_nll = background - log_intensity
    local = LikelihoodTerms(
        negative_log_likelihood=local_nll,
        log_intensity=log_intensity,
        background_compensator=background,
        triggering_compensator=parameter.new_zeros(()),
        target_count=rank + 1,
    )
    reduced = likelihood._reduce_terms(local)
    global_nll = reduced.global_terms.negative_log_likelihood
    old_order = local_nll + (global_nll - local_nll.detach())
    reduced.backward_loss.backward()
    local_gradient = model.weight.grad.detach().clone()
    likelihood.sum_gradients_(model)
    return {
        "backward_bits": reduced.backward_loss.detach().view(torch.int64),
        "global_bits": global_nll.view(torch.int64),
        "old_order_bits": old_order.detach().view(torch.int64),
        "local_gradient": local_gradient,
        "summed_gradient": model.weight.grad.detach().clone(),
        "target_count": reduced.global_terms.target_count,
        "global_requires_grad": global_nll.requires_grad,
    }


def _accepted_point_worker(rank: int) -> dict:
    """The accepted parameters need not equal the last evaluated trial."""
    model = _RosenbrockModel()
    likelihood = DistributedExactLikelihood()
    optimizer = _optimizer(model)
    trial = None

    def closure():
        nonlocal trial
        optimizer.zero_grad(set_to_none=True)
        trial = model.position.detach().clone()
        reduced = likelihood._reduce_terms(_terms(model() / _WORLD_SIZE))
        likelihood.backward_and_sum_gradients_(reduced.backward_loss, model)
        return reduced.backward_loss

    def accept_other_point(closure):
        assert rank == 0, "a worker must never call optimizer.step"
        closure()
        with torch.no_grad():
            model.position.copy_(model.position.new_tensor([0.25, 0.5]))
        optimizer.state[model.position]["n_iter"] = 1
        optimizer.state[model.position]["func_evals"] = 1

    optimizer.step = accept_other_point
    stats = likelihood.run_lbfgs_step_(optimizer, model, closure)
    return {"accepted": model.position.detach().clone(), "trial": trial, "stats": stats}


def _gloo_entry(rank: int, init_method: str, result_directory: str, case: str) -> None:
    torch.set_num_threads(1)
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(seconds=25),
        )
        if case == "trajectory":
            result = _trajectory_worker(rank)
        elif case == "reduce":
            result = _reduce_worker(rank)
        elif case == "accepted_point":
            result = _accepted_point_worker(rank)
        else:
            result = _failure_worker(rank, case)
        torch.save(result, Path(result_directory) / f"rank-{rank}.pt")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_gloo(tmp_path: Path, case: str) -> list[dict]:
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    context = mp.spawn(
        _gloo_entry,
        args=((tmp_path / "rendezvous").resolve().as_uri(), str(result_directory), case),
        nprocs=_WORLD_SIZE,
        join=False,
    )
    try:
        deadline = time.monotonic() + 60.0
        while not context.join(timeout=0.5):
            if time.monotonic() >= deadline:
                pytest.fail(f"distributed optimizer case {case!r} did not finish in 60s")
    finally:
        # A protocol regression must not leave live children after pytest fails.
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
    return [
        torch.load(result_directory / f"rank-{rank}.pt", weights_only=True)
        for rank in range(_WORLD_SIZE)
    ]


def test_single_process_lbfgs_helper_matches_direct_step():
    expected = _run_trajectory(coordinated=False)
    actual = _run_trajectory(coordinated=True)
    for key in ("accepted", "trials", "gradients", "returned_losses"):
        assert torch.equal(actual[key], expected[key]), key
    for key in ("stats", "closure_counts", "history_sizes", "state_totals"):
        assert actual[key] == expected[key], key
    assert actual["optimizer_step_calls"] == len(_SCHEDULE)


def test_two_process_lbfgs_has_one_leader_and_retains_history(tmp_path: Path):
    expected = _run_trajectory(coordinated=False)
    leader, worker = _run_gloo(tmp_path, "trajectory")
    assert leader["optimizer_step_calls"] == len(_SCHEDULE)
    assert worker["optimizer_step_calls"] == 0
    assert worker["history_sizes"] == [0] * len(_SCHEDULE)
    assert worker["state_totals"] == [(0, 0)] * len(_SCHEDULE)
    assert leader["history_sizes"] == expected["history_sizes"]
    # A first M step can exhaust max_eval in its initial strong-Wolfe search
    # before a curvature pair is stored; subsequent M steps must retain pairs.
    assert all(size > 0 for size in leader["history_sizes"][1:])
    assert leader["state_totals"] == expected["state_totals"]
    assert leader["state_totals"][-1][0] == sum(item[0] for item in leader["stats"])
    for result in (leader, worker):
        assert result["stats"] == expected["stats"]
        assert result["closure_counts"] == expected["closure_counts"]
        assert result["closure_counts"] == [item[1] for item in result["stats"]]
        assert all(0 < item[0] <= limit for item, limit in zip(result["stats"], _SCHEDULE))
        torch.testing.assert_close(result["accepted"], expected["accepted"], rtol=1e-12, atol=1e-12)
    for key in ("accepted", "trials", "gradients"):
        assert torch.equal(leader[key], worker[key]), key
        torch.testing.assert_close(leader[key], expected[key], rtol=1e-12, atol=1e-12)
    assert bool(torch.all(worker["returned_losses"] > leader["returned_losses"] + 1.0))


@pytest.mark.parametrize("phase", ("before_closure", "after_closure", "local_failure"))
def test_two_process_lbfgs_errors_reach_all_ranks_without_hanging(tmp_path: Path, phase: str):
    leader, worker = _run_gloo(tmp_path, phase)
    expected_calls = 0 if phase == "before_closure" else 1
    for result in (leader, worker):
        assert result["message"] is not None
        assert "LBFGS failed" in result["message"]
        assert result["closure_calls"] == expected_calls
        assert result["probe"] == 3
    assert leader["step_calls"] == 1
    assert worker["step_calls"] == 0
    if phase == "local_failure":
        assert "optimizer-test local likelihood" in leader["message"]
    else:
        assert f"intentional leader failure {phase.replace('_', ' ')}" in leader["message"]


def test_two_process_lbfgs_broadcasts_accepted_not_last_trial_parameters(tmp_path: Path):
    leader, worker = _run_gloo(tmp_path, "accepted_point")
    for result in (leader, worker):
        assert result["stats"] == (1, 1)
        assert torch.equal(result["accepted"], torch.tensor([0.25, 0.5], dtype=torch.float64))
        assert not torch.equal(result["accepted"], result["trial"])
    assert torch.equal(leader["trial"], worker["trial"])


def test_reduced_loss_is_bit_identical_and_retains_local_gradient(tmp_path: Path):
    results = _run_gloo(tmp_path, "reduce")
    for rank, result in enumerate(results):
        assert torch.equal(result["backward_bits"], result["global_bits"])
        assert torch.equal(result["backward_bits"], results[0]["backward_bits"])
        assert torch.equal(result["local_gradient"], torch.tensor([[rank + 2.0]], dtype=torch.float64))
        assert torch.equal(result["summed_gradient"], torch.tensor([[5.0]], dtype=torch.float64))
        assert result["target_count"] == 3
        assert not result["global_requires_grad"]
    # Prove this fixture exercises the old one-ULP discrepancy, rather than
    # allowing the regression to pass by chance with exactly representable data.
    assert not torch.equal(results[0]["old_order_bits"], results[0]["global_bits"])
    assert not torch.equal(results[0]["old_order_bits"], results[1]["old_order_bits"])
