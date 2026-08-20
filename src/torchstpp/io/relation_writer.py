import json
from pathlib import Path

import polars as pl

from ..relations import RelationExtractor
from ..results import FittedRun
from .result_writer import catalog_event_ids


class RelationWriter:
    """Write thresholded relation blocks as independent Parquet parts."""

    def __init__(self, output_directory: str | Path) -> None:
        self.output_directory = Path(output_directory)

    def write(self, run: FittedRun, extractor: RelationExtractor) -> dict:
        if self.output_directory.exists() and any(self.output_directory.iterdir()):
            raise FileExistsError(
                f"relation output directory is not empty: {self.output_directory}"
            )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        event_ids = catalog_event_ids(run.catalog)
        part_count = 0
        relation_count = 0

        for block in extractor.iter_relations(run):
            source_index = block.source_index.numpy()
            target_index = block.target_index.numpy()
            frame = pl.DataFrame({
                "source_index": source_index,
                "target_index": target_index,
                "source_event_id": event_ids[source_index],
                "target_event_id": event_ids[target_index],
                "trigger_rate": block.trigger_rate.numpy(),
                "trigger_probability": block.trigger_probability.numpy(),
            })
            path = self.output_directory / f"part-{part_count:05d}.parquet"
            frame.write_parquet(path, compression="zstd")
            part_count += 1
            relation_count += block.row_count

        manifest = {
            "probability_threshold": extractor.probability_threshold,
            "target_chunk_size": extractor.target_chunk_size,
            "source_chunk_size": extractor.source_chunk_size,
            "part_count": part_count,
            "relation_count": relation_count,
        }
        (self.output_directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest
