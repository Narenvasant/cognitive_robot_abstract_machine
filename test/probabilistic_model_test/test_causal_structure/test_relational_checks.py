import numpy as np
import pandas as pd

from probabilistic_model.causal_structure.relational_checks import ExampleToPartCheck
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    TrainingTableColumn,
)

# %% tables of instances with parts


def tables(
    attribute_drives_parts: bool, instance_count: int = 150, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    :param attribute_drives_parts: Whether the instance's ``warm`` attribute sets its
        parts' ``wet`` attribute directly, not through the count.
    :return: One row per instance, with ``warm`` and the count of its parts, and one row
        per part, with the count, ``wet`` and its instance's position.
    """
    random_state = np.random.default_rng(seed)
    warm = random_state.integers(0, 2, instance_count)
    part_count = random_state.integers(2, 5, instance_count)
    part_rows = []
    for position in range(instance_count):
        wet_rate = 0.8 if attribute_drives_parts and warm[position] else 0.3
        for _ in range(part_count[position]):
            part_rows.append(
                {
                    "Room.count()": part_count[position],
                    "wet": int(random_state.random() < wet_rate),
                    TrainingTableColumn.EXAMPLE_POSITION: position,
                }
            )
    return (
        pd.DataFrame({"Room.warm": warm, "Room.count()": part_count}),
        pd.DataFrame(part_rows),
    )


def test_attribute_driving_parts_directly_is_rejected():
    example_rows, part_rows = tables(attribute_drives_parts=True)

    [dependence] = ExampleToPartCheck().rejected_pairs(example_rows, part_rows, "parts")

    assert (dependence.example_attribute, dependence.part_attribute) == (
        "Room.warm",
        "wet",
    )


def test_attribute_independent_of_parts_is_kept():
    example_rows, part_rows = tables(attribute_drives_parts=False)

    assert ExampleToPartCheck().rejected_pairs(example_rows, part_rows, "parts") == []
