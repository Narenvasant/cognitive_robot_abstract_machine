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
    room_tables = parts_of(warm, attribute_drives_parts, random_state)
    room_tables.example_rows = room_tables.example_rows.drop(columns=["Room.sunny"])
    return room_tables


def parts_of(
    warm: np.ndarray, attribute_drives_parts: bool, random_state: np.random.Generator
) -> RoomTables:
    """
    :param warm: Each instance's ``warm`` attribute.
    :param attribute_drives_parts: Whether ``warm`` sets the parts' ``wet`` attribute
        directly, not through the count.
    :param random_state: Source of randomness.
    :return: One row per instance, with ``warm``, ``sunny``, which reads ``warm`` wrong
        a quarter of the time, and the count of its parts; and one row per part, with
        the count and ``wet``.
    """
    instance_count = len(warm)
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
    sunny = warm ^ (random_state.random(instance_count) < 0.25)
    return RoomTables(
        pd.DataFrame(
            {"Room.warm": warm, "Room.sunny": sunny, "Room.count()": part_count}
        ),
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


# %% the dependences that remain given another attribute


def test_proxy_of_the_driving_attribute_is_dropped_given_it():
    random_state = np.random.default_rng(0)
    room_tables = parts_of(random_state.integers(0, 2, 300), True, random_state)
    check = ExampleToPartCheck()

    assert {
        dependence.example_attribute
        for dependence in check.rejected_pairs(
            room_tables.example_rows,
            room_tables.part_rows,
            room_tables.example_positions,
            "parts",
        )
    } == {"Room.warm", "Room.sunny"}
    assert [
        dependence.example_attribute
        for dependence in check.dependences_given_another_attribute(
            room_tables.example_rows,
            room_tables.part_rows,
            room_tables.example_positions,
            "parts",
            {},
        )
    ] == ["Room.warm"]


def test_part_attribute_reached_through_another_is_dropped_given_it():
    random_state = np.random.default_rng(0)
    room_tables = parts_of(random_state.integers(0, 2, 300), True, random_state)
    part_rows = room_tables.part_rows
    part_rows["dripping"] = part_rows["wet"] ^ (
        random_state.random(len(part_rows)) < 0.1
    )
    check = ExampleToPartCheck()

    assert ("Room.warm", "dripping") in {
        (dependence.example_attribute, dependence.part_attribute)
        for dependence in check.rejected_pairs(
            room_tables.example_rows,
            part_rows,
            room_tables.example_positions,
            "parts",
        )
    }
    assert [
        (dependence.example_attribute, dependence.part_attribute)
        for dependence in check.dependences_given_another_attribute(
            room_tables.example_rows,
            part_rows,
            room_tables.example_positions,
            "parts",
            {"wet": {"dripping"}, "dripping": {"wet"}},
        )
    ] == [("Room.warm", "wet")]
