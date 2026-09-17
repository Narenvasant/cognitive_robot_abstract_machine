"""
What each flat-table layout holds of a scene, and how it names it.
"""

from __future__ import annotations

from dataclasses import fields, replace

import pytest

from experiments.causal_reasoning.graspclutter6d.domain import (
    GraspClutterObject,
    GraspClutterScene,
    GraspClutterViewpoint,
)
from experiments.causal_reasoning.graspclutter6d.exceptions import (
    FlatTableSchemaMismatchError,
)
from experiments.causal_reasoning.graspclutter6d.flat_table import (
    AbsentPart,
    FlatTable,
    PartAttribute,
    SceneSchema,
    SceneView,
    TableLayout,
)


@pytest.fixture
def schema() -> SceneSchema:
    return SceneSchema()


def test_the_scalar_fields_are_the_scene_fields_that_are_not_parts(schema):
    assert set(schema.scalar_fields) == {
        scene_field.name for scene_field in fields(GraspClutterScene)
    } - set(schema.part_fields)


def test_a_part_column_is_named_by_its_field_position_and_attribute(schema):
    assert (
        schema.part_column(PartAttribute("objects", 2, "size"))
        == "GraspClutterScene.objects[2].size"
    )


def test_a_part_column_name_is_read_back_into_its_parts(schema):
    part = PartAttribute("viewpoints", 7, "clarity")
    assert schema.part_attribute(schema.part_column(part)) == part


def test_a_scene_level_name_is_not_a_part_attribute(schema):
    assert schema.part_attribute(schema.scalar_column("extent")) is None


def test_the_scalars_only_layout_holds_no_counts(schema):
    assert FlatTable(TableLayout.SCALARS).columns == list(schema.scalar_columns)


def test_the_propositional_layout_holds_the_scalars_and_the_counts(schema):
    assert FlatTable(TableLayout.PROPOSITIONAL).columns == list(
        schema.scalar_columns
    ) + list(schema.aggregation_columns)


def test_the_unrolled_layout_is_as_wide_as_the_largest_scene(synthetic_scenes):
    table = FlatTable.unrolled_for(synthetic_scenes)
    assert table.part_widths == {
        "objects": max(len(scene.objects) for scene in synthetic_scenes),
        "viewpoints": max(len(scene.viewpoints) for scene in synthetic_scenes),
    }


def test_a_position_without_a_part_holds_the_absent_symbol(synthetic_scenes, schema):
    table = FlatTable.unrolled_for(synthetic_scenes)
    smaller = replace(synthetic_scenes[0], objects=synthetic_scenes[0].objects[:1])
    row = table.row(smaller)
    column = schema.part_column(PartAttribute("objects", 1, "size"))
    assert row[column] is AbsentPart.ABSENT


def test_a_scene_with_more_parts_than_positions_is_refused(synthetic_scenes):
    table = FlatTable.unrolled_for(synthetic_scenes)
    scene = synthetic_scenes[0]
    larger = replace(scene, objects=scene.objects + [scene.objects[0]])
    assert not table.fits(larger)
    with pytest.raises(FlatTableSchemaMismatchError):
        table.row(larger)


def test_a_row_carries_every_attribute_of_every_part(synthetic_scenes, schema):
    table = FlatTable.unrolled_for(synthetic_scenes)
    scene = synthetic_scenes[0]
    row = table.row(scene)
    for index, one in enumerate(scene.objects):
        for attribute in (field.name for field in fields(GraspClutterObject)):
            column = schema.part_column(PartAttribute("objects", index, attribute))
            assert row[column] == vars(one)[attribute]
    for index, one in enumerate(scene.viewpoints):
        for attribute in (field.name for field in fields(GraspClutterViewpoint)):
            column = schema.part_column(PartAttribute("viewpoints", index, attribute))
            assert row[column] == vars(one)[attribute]


def test_only_the_unrolled_layout_can_be_scored_on_a_whole_scene():
    assert (
        FlatTable(TableLayout.PROPOSITIONAL).columns_of(SceneView.WHOLE_SCENE) is None
    )
    assert (
        FlatTable(
            TableLayout.UNROLLED, part_widths={"objects": 1, "viewpoints": 1}
        ).columns_of(SceneView.WHOLE_SCENE)
        is not None
    )


def test_only_a_layout_with_counts_can_be_scored_on_them():
    assert (
        FlatTable(TableLayout.SCALARS).columns_of(SceneView.SCALARS_AND_COUNTS) is None
    )


def test_a_dataframe_has_one_row_per_scene(synthetic_scenes):
    frame = FlatTable(TableLayout.PROPOSITIONAL).dataframe(synthetic_scenes)
    assert len(frame) == len(synthetic_scenes)
    assert list(frame.columns) == FlatTable(TableLayout.PROPOSITIONAL).columns
