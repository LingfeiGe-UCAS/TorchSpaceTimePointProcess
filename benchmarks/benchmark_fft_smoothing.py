"""One real-catalog responsibility update, followed by matched smoothing runs.

No M step or complete optimization is performed. Warm repeats reuse the SAME
responsibility vector to isolate smoothing cost, rather than running more EM steps.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from torchstpp.config import TorchSTPPConfig, build_project, load_config
from torchstpp.finite_source.em import (
    finite_source_patch_responsibilities, smooth_patch_responsibilities,
    _event_intensity,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = load_config(args.config)
    data = copy.deepcopy(dict(config.data))
    data["catalog"]["path"] = str(args.catalog.resolve())
    data["finite_source"]["path"] = str(args.patches.resolve())
    data["finite_source"]["smoothing_method"] = "fft"
    data["runtime"].update(device="cuda:0", dtype=args.dtype)
    print("Building actual Noto catalog, projected geometry, KDE and model", flush=True)
    started = time.perf_counter()
    project = build_project(TorchSTPPConfig(data, config.path))
    model, catalog, state = project.model, project.catalog, project.background_state
    device = catalog.device
    def sync():
        torch.cuda.synchronize(device)
    sync()
    report = dict(
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name(device),
        dtype=args.dtype, catalog_count=len(catalog), patch_count=model.patch_count,
        build_seconds=time.perf_counter() - started, config=data,
        inputs={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (args.config, args.catalog, args.patches)},
        scope="One initial-state E responsibility update over the loaded catalog; no M step",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    # Tiny FFT warm-up removes first-ever CUDA FFT initialization from timings.
    torch.fft.irfft2(torch.fft.rfft2(torch.ones((8, 8), device=device, dtype=catalog.dtype)))
    print(f"Computing real responsibilities: N={len(catalog)}, P={model.patch_count}", flush=True)
    sync()
    started = time.perf_counter()
    expected = finite_source_patch_responsibilities(
        model, catalog, state, target_chunk_size=512, source_chunk_size=2048,
        patch_chunk_size=512,
    )
    sync()
    report["responsibilities_seconds"] = time.perf_counter() - started
    assert expected is not None
    initial_tau = model.patch_tau.clone()
    arrays = {"patch_id": model.patch_id.cpu().numpy(),
              "expected": expected.cpu().numpy()}
    report["comparisons"] = []
    # All target intensities after smoothing are compared on 128 actual events.
    indices = torch.linspace(0, len(catalog) - 1, 128, device=device).long().unique()
    targets = catalog.take(indices)
    for normalization in ("fortran", "corrected"):
        row = {"normalization": normalization, "methods": {}}
        results = {}
        intensities = {}
        for method in ("blocked", "fft"):
            print(f"Smoothing {normalization}/{method}", flush=True)
            model.set_patch_tau_(initial_tau)
            model._finite_source_normalizer_signature = None
            model._fft_smoothing_cache = None
            model.tau_normalization = normalization
            model.smoothing_method = method
            sync()
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            result = smooth_patch_responsibilities(model, catalog, expected)
            sync()
            cold_seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated(device) - baseline
            warm = []
            for _ in range(args.repeats):
                sync()
                started = time.perf_counter()
                result = smooth_patch_responsibilities(model, catalog, expected)
                sync()
                warm.append(time.perf_counter() - started)
            results[method] = result
            with torch.no_grad():
                intensities[method] = _event_intensity(
                    model, catalog, targets, state, source_chunk_size=2048,
                )
            row["methods"][method] = dict(
                cold_seconds=cold_seconds, warm_seconds=warm,
                warm_median_seconds=float(np.median(warm)),
                cold_peak_extra_bytes=peak,
            )
            if method == "fft":
                row["fft_grids"] = [dict(shape=p.shape, padded_shape=p.fft_shape)
                                       for p in model._fft_smoothing_cache[1]]
            arrays[f"tau_{normalization}_{method}"] = result.patch_tau.cpu().numpy()
        a, b = results["blocked"], results["fft"]
        error = (a.patch_tau - b.patch_tau).abs()
        row["tau_max_absolute_error"] = float(error.max())
        row["tau_relative_l1_error"] = float(error.sum() / a.patch_tau.abs().sum())
        row["normalizer_max_relative_error"] = float(
            ((a.smoothing_normalizer - b.smoothing_normalizer).abs() / a.smoothing_normalizer).max()
        )
        row["sample_intensity_max_relative_error"] = float(
            ((intensities["blocked"] - intensities["fft"]).abs() / intensities["blocked"]).max()
        )
        row["groups"] = []
        for group in model.finite_source_group_ids.tolist():
            sl = model.patch_slice(group)
            wa = a.patch_tau[sl] / a.patch_tau[sl].sum()
            wb = b.patch_tau[sl] / b.patch_tau[sl].sum()
            row["groups"].append(dict(
                group=group, patches=len(wa), weight_l1_error=float((wa-wb).abs().sum()),
                fft_mass_relative_error=float((b.patch_tau[sl].sum()-expected[sl].sum()).abs()/expected[sl].sum()),
            ))
        tolerance = 5e-10 if args.dtype == "float64" else 2e-3
        row["passed"] = bool(
            row["tau_relative_l1_error"] < tolerance
            and row["normalizer_max_relative_error"] < tolerance
            and row["sample_intensity_max_relative_error"] < tolerance
            and (normalization != "corrected" or max(g["fft_mass_relative_error"] for g in row["groups"]) < tolerance)
        )
        report["comparisons"].append(row)
        print(json.dumps(row, indent=2), flush=True)
    np.savez_compressed(args.output / "smoothing_vectors.npz", **arrays)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all(r["passed"] for r in report["comparisons"]):
        raise AssertionError("Smoothing comparison exceeded error tolerance; see summary.json")
    print(f"PASS: {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
