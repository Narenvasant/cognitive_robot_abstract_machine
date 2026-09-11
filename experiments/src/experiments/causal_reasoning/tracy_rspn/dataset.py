"""
Storing recorded attempts on disk and splitting them for fitting and evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from krrood.adapters.json_serializer import from_json, to_json
from typing_extensions import Callable, Dict, List, Self, Tuple, TypeVar

from experiments.causal_reasoning.tracy_rspn.domain import ClutterPickScene

T = TypeVar("T")


@dataclass(frozen=True)
class SuccessRate:
    """
    How often a group of attempts lifted its target.
    """

    attempt_count: int
    """
    How many attempts the group holds.
    """

    lifted_count: int
    """
    How many of them lifted the target.
    """

    @property
    def rate(self) -> float:
        """
        The lifted share.
        """
        return self.lifted_count / self.attempt_count


@dataclass
class ClutterPickDataset:
    """
    A set of recorded attempts, as written by data collection and read by the pipelines.
    """

    scenes: List[ClutterPickScene] = field(default_factory=list)
    """
    The recorded attempts.
    """

    def save(self, path: Path) -> None:
        """
        Write the attempts to a JSON file.

        :param path: Where to write; parent directories are created.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(to_json(self.scenes), indent=2))

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        Read attempts written by :meth:`save`.

        :param path: The file to read.
        :return: The dataset.
        """
        return cls(scenes=from_json(json.loads(path.read_text())))

    @property
    def success_rate(self) -> float:
        """
        Share of attempts whose target was lifted.
        """
        return sum(scene.lifted for scene in self.scenes) / len(self.scenes)

    def success_rate_by(
        self, key: Callable[[ClutterPickScene], T]
    ) -> Dict[T, SuccessRate]:
        """
        The share of lifted targets among the attempts sharing a value.

        :param key: What to group the attempts by.
        :return: Each value's success rate, by value.
        """
        by_value: Dict[T, List[ClutterPickScene]] = {}
        for scene in self.scenes:
            by_value.setdefault(key(scene), []).append(scene)
        return {
            value: SuccessRate(
                attempt_count=len(scenes),
                lifted_count=sum(scene.lifted for scene in scenes),
            )
            for value, scenes in sorted(by_value.items())
        }

    def split(
        self, train_fraction: float, random_state: np.random.Generator
    ) -> Tuple[Self, Self]:
        """
        Shuffle the attempts and split them in two.

        :param train_fraction: Share of attempts that go into the first part.
        :param random_state: Source of randomness for the shuffle.
        :return: The first and second part.
        """
        order = random_state.permutation(len(self.scenes))
        split_index = int(train_fraction * len(self.scenes))
        first = [self.scenes[index] for index in order[:split_index]]
        second = [self.scenes[index] for index in order[split_index:]]
        return type(self)(first), type(self)(second)
