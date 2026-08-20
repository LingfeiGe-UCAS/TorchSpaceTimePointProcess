from collections.abc import Iterator

import torch

from ..results import FittedRun
from .result import RelationBlock


class RelationExtractor:
    """Extract exact causal relations blockwise above a probability threshold."""

    def __init__(
        self,
        *,
        probability_threshold: float = 1e-4,
        target_chunk_size: int = 2048,
        source_chunk_size: int = 4096,
    ) -> None:
        if not 0.0 < probability_threshold <= 1.0:
            raise ValueError("probability_threshold must be in (0, 1]")
        if target_chunk_size < 1 or source_chunk_size < 1:
            raise ValueError("chunk sizes must be positive")
        self.probability_threshold = probability_threshold
        self.target_chunk_size = target_chunk_size
        self.source_chunk_size = source_chunk_size

    @torch.no_grad()
    def iter_relations(self, run: FittedRun) -> Iterator[RelationBlock]:
        catalog = run.catalog
        model = run.model
        event_count = catalog.time.shape[0]
        run.event_state.validate(event_count)

        for target_begin in range(0, event_count, self.target_chunk_size):
            target_end = min(target_begin + self.target_chunk_size, event_count)
            target = catalog.take(slice(target_begin, target_end))
            target_intensity = run.event_state.total_intensity[
                target_begin:target_end
            ]

            for source_begin in range(0, event_count, self.source_chunk_size):
                source_end = min(source_begin + self.source_chunk_size, event_count)
                source = catalog.take(slice(source_begin, source_end))
                if source.time[0] >= target.time[-1]:
                    break

                trigger_rate = model.pairwise_rate(target, source)
                probability = trigger_rate / target_intensity[:, None]
                selected = probability > self.probability_threshold
                if not torch.any(selected):
                    continue

                target_local, source_local = torch.nonzero(
                    selected, as_tuple=True
                )
                yield RelationBlock(
                    source_index=(source_local + source_begin).detach().cpu(),
                    target_index=(target_local + target_begin).detach().cpu(),
                    trigger_rate=trigger_rate[selected].detach().cpu(),
                    trigger_probability=probability[selected].detach().cpu(),
                )
