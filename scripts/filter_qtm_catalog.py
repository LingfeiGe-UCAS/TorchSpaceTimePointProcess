"""Convert and magnitude-filter the SCEDC QTM whitespace catalog."""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
from datetime import date
from pathlib import Path
import time

import polars as pl


OUTPUT_COLUMNS = (
    "event_id",
    "qtm_event_id",
    "time",
    "longitude",
    "latitude",
    "depth",
    "mag",
    "mag_p5",
    "mag_p95",
    "template_id",
    "stack_cc",
    "mad",
    "relocated",
)


def convert_qtm_catalog(
    source: Path,
    outputs: list[tuple[float, Path]],
) -> dict[float, int]:
    """Stream one QTM file into one or more framework-ready CSV catalogs."""

    if not outputs:
        raise ValueError("at least one threshold/output pair is required")
    thresholds = [threshold for threshold, _ in outputs]
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("magnitude thresholds must be unique")
    for _, output in outputs:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    origin = date(2008, 1, 1)
    day_cache: dict[tuple[int, int, int], int] = {}
    counts = {threshold: 0 for threshold in thresholds}

    with ExitStack() as stack:
        source_file = stack.enter_context(source.open("r", encoding="utf-8"))
        writers: list[tuple[float, csv.writer]] = []
        for threshold, output in outputs:
            handle = stack.enter_context(
                output.open("x", newline="", encoding="utf-8")
            )
            writer = csv.writer(handle)
            writer.writerow(OUTPUT_COLUMNS)
            writers.append((threshold, writer))

        header = source_file.readline().split()
        if len(header) != 17 or header[10] != "MAGNITUDE":
            raise ValueError("input does not have the expected 17-column QTM header")

        for source_row, line in enumerate(source_file, start=1):
            fields = line.split()
            if len(fields) != 17:
                raise ValueError(
                    f"source data row {source_row} has {len(fields)} fields; expected 17"
                )
            magnitude = float(fields[10])
            matching = [item for item in writers if magnitude >= item[0]]
            if not matching:
                continue

            date_key = (int(fields[0]), int(fields[1]), int(fields[2]))
            day_offset = day_cache.get(date_key)
            if day_offset is None:
                day_offset = (date(*date_key) - origin).days
                day_cache[date_key] = day_offset
            seconds = (
                int(fields[3]) * 3600.0
                + int(fields[4]) * 60.0
                + float(fields[5])
            )
            elapsed_days = day_offset + seconds / 86400.0
            row = (
                source_row,
                fields[6],
                f"{elapsed_days:.12f}",
                fields[8],
                fields[7],
                fields[9],
                fields[10],
                fields[11],
                fields[12],
                fields[13],
                fields[14],
                fields[15],
                fields[16],
            )
            for threshold, writer in matching:
                writer.writerow(row)
                counts[threshold] += 1
    for _, output in outputs:
        temporary = output.with_name(f"{output.stem}.sorting.tmp{output.suffix}")
        try:
            pl.scan_csv(output).sort(["time", "event_id"]).sink_csv(temporary)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output",
        action="append",
        nargs=2,
        metavar=("MIN_MAGNITUDE", "CSV_PATH"),
        required=True,
        help="repeat for every desired magnitude threshold",
    )
    args = parser.parse_args()
    outputs = [(float(value), Path(path)) for value, path in args.output]
    started = time.perf_counter()
    counts = convert_qtm_catalog(args.source, outputs)
    for threshold, output in outputs:
        print(f"M >= {threshold:g}: {counts[threshold]} events -> {output.resolve()}")
    print(f"elapsed seconds: {time.perf_counter() - started:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
