"""Source-controlled batch sampling for partially labelled MTL datasets."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence

import torch
from torch.utils.data import Sampler


class SourceBalancedBatchSampler(Sampler[list[int]]):
    """Create batches with an exact, reproducible source composition.

    Parameters
    ----------
    sources:
        Source label for every row in the dataset, in dataset index order.
    samples_per_source:
        Exact number of samples drawn from each source in every batch.
        For this thesis, ``{"nih": 8, "chexpert": 8, "tbx11k": 16}``
        gives 16 cardiomegaly-labelled and 16 TB-labelled images per batch.
    seed:
        Base seed used for reproducible shuffling.
    steps_per_epoch:
        Optional fixed number of batches. If omitted, the epoch length is
        anchored to the source that is exhausted first relative to its batch
        allocation. Smaller exhausted pools are reshuffled and cycled.
    """

    def __init__(
        self,
        sources: Sequence[str],
        samples_per_source: Mapping[str, int],
        seed: int = 42,
        steps_per_epoch: int | None = None,
    ) -> None:
        super().__init__()

        if not samples_per_source:
            raise ValueError("samples_per_source must not be empty")

        self.samples_per_source = {
            str(source).strip().lower(): int(count)
            for source, count in samples_per_source.items()
        }

        if any(count <= 0 for count in self.samples_per_source.values()):
            raise ValueError("All samples_per_source counts must be positive")

        self.indices_by_source: dict[str, list[int]] = {
            source: [] for source in self.samples_per_source
        }

        for index, source in enumerate(sources):
            normalized_source = str(source).strip().lower()
            if normalized_source in self.indices_by_source:
                self.indices_by_source[normalized_source].append(index)

        for source, indices in self.indices_by_source.items():
            if not indices:
                raise ValueError(f"No dataset rows found for source: {source}")

        self.batch_size = sum(self.samples_per_source.values())
        self.seed = int(seed)
        self.epoch = 0

        if steps_per_epoch is None:
            self.steps_per_epoch = min(
                math.ceil(
                    len(self.indices_by_source[source])
                    / self.samples_per_source[source]
                )
                for source in self.samples_per_source
            )
        else:
            if steps_per_epoch <= 0:
                raise ValueError("steps_per_epoch must be positive")
            self.steps_per_epoch = int(steps_per_epoch)

    def set_epoch(self, epoch: int) -> None:
        """Select a new deterministic shuffle for a training epoch."""

        self.epoch = int(epoch)

    @staticmethod
    def _new_permutation(
        indices: list[int],
        generator: torch.Generator,
    ) -> list[int]:
        order = torch.randperm(
            len(indices),
            generator=generator,
        ).tolist()
        return [indices[position] for position in order]

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        pools = {
            source: self._new_permutation(indices, generator)
            for source, indices in self.indices_by_source.items()
        }
        positions = {source: 0 for source in pools}

        for _ in range(self.steps_per_epoch):
            batch: list[int] = []

            for source, required_count in self.samples_per_source.items():
                selected: list[int] = []

                while len(selected) < required_count:
                    pool = pools[source]
                    position = positions[source]
                    remaining = len(pool) - position
                    needed = required_count - len(selected)
                    take = min(remaining, needed)

                    selected.extend(pool[position : position + take])
                    positions[source] += take

                    if positions[source] >= len(pool):
                        pools[source] = self._new_permutation(
                            self.indices_by_source[source],
                            generator,
                        )
                        positions[source] = 0

                batch.extend(selected)

            batch_order = torch.randperm(
                len(batch),
                generator=generator,
            ).tolist()
            yield [batch[position] for position in batch_order]

    def __len__(self) -> int:
        return self.steps_per_epoch
