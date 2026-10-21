from dataclasses import dataclass

import numpy as np
import pandas as pd

from probabilistic_model.causal_structure.relational_checks import ExampleToPartCheck

# %% tables of instances with parts


@dataclass
class RoomTables:
    """
    The tables of rooms and of their parts.
    """

    example_rows: pd.DataFrame
    """
    One row per room.
    """

    part_rows: pd.DataFrame
    """
    One row per part.
    """

    example_positions: list[int]
    """
    Per part row, the position of its room.
    """


def tables(
    attribute_drives_parts: bool, instance_count: int = 150, seed: int = 0
) -> RoomTables:
    """
    :param attribute_drives_parts: Whether the instance's ``warm`` attribute sets its
        parts' ``wet`` attribute directly, not through the count.
    :return: One row per instance, with ``warm`` and the count of its parts, and one row
        per part, with the count and ``wet``.
    """
    random_state = np.random.default_rng(seed)
    warm = random_state.integers(0, 2, instance_count)
    part_count = random_state.integers(2, 5, instance_count)
    part_rows, example_positions = [], []
    for position in range(instance_count):
        wet_rate = 0.8 if attribute_drives_parts and warm[position] else 0.3
        for _ in range(part_count[position]):
            part_rows.append(
                {
                    "Room.count()": part_count[position],
                    "wet": int(random_state.random() < wet_rate),
                }
            )
            example_positions.append(position)
    return RoomTables(
        pd.DataFrame({"Room.warm": warm, "Room.count()": part_count}),
        pd.DataFrame(part_rows),
        example_positions,
    )


def test_attribute_driving_parts_directly_is_rejected():
    room_tables = tables(attribute_drives_parts=True)

    [dependence] = ExampleToPartCheck().rejected_pairs(
        room_tables.example_rows,
        room_tables.part_rows,
        room_tables.example_positions,
        "parts",
    )

    assert (dependence.example_attribute, dependence.part_attribute) == (
        "Room.warm",
        "wet",
    )


def test_attribute_independent_of_parts_is_kept():
    room_tables = tables(attribute_drives_parts=False)

    assert (
        ExampleToPartCheck().rejected_pairs(
            room_tables.example_rows,
            room_tables.part_rows,
            room_tables.example_positions,
            "parts",
        )
        == []
    )
