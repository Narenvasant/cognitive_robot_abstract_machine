import copy

import pytest

pytest.importorskip("robocasa", reason="robocasa is not installed")
pytest.importorskip("robosuite", reason="robosuite is not installed")

from semantic_digital_twin.adapters.robocasa_dataset.exceptions import (
    RoboCasaTaskNotFoundError,
)
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.robocasa_dataset.loader import (
    RoboCasaDatasetLoader,
    _parse_robosuite_mjcf,
)
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaKitchenApplianceCategory,
    RoboCasaObjectCategory,
)
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Dishwasher
from semantic_digital_twin.world import World


@pytest.fixture(scope="session")
def robocasa_loader() -> RoboCasaDatasetLoader:
    loader = RoboCasaDatasetLoader()
    if not loader.directory.exists():
        pytest.skip(
            "RoboCasa assets not downloaded. Run "
            "'python -m robocasa.scripts.download_kitchen_assets' first."
        )
    return loader


def test_load_kitchen(robocasa_loader):
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    world = robocasa_loader.load_kitchen(
        layout_id=next(iter(LayoutType)), style_id=next(iter(StyleType))
    )
    assert len(world.bodies) > 0
    assert len(world.semantic_annotations) > 0


def test_load_kitchen_appliance(robocasa_loader):
    world = robocasa_loader.load_kitchen_appliance(
        RoboCasaKitchenApplianceCategory.CABINET
    )
    assert len(world.bodies) > 0
    assert len(world.semantic_annotations) >= 1


def test_load_kitchen_appliance_attaches_matching_annotation(robocasa_loader):
    """
    The loaded appliance is annotated with the semantic type of the requested category.
    """
    world = robocasa_loader.load_kitchen_appliance(
        RoboCasaKitchenApplianceCategory.DISHWASHER
    )
    assert any(
        isinstance(annotation, Dishwasher) for annotation in world.semantic_annotations
    )


def test_load_object(robocasa_loader):
    world = robocasa_loader.load_object(RoboCasaObjectCategory.APPLE)
    assert len(world.bodies) > 0
    annotations = world.semantic_annotations
    assert len(annotations) == 1
    assert not isinstance(annotations[0], NaturalLanguageWithTypeDescription)


def test_load_object_is_scaled_as_robocasa_scales_its_category(robocasa_loader):
    """
    RoboCasa scales every object of a category by that category's scale, so a loaded
    object has the size RoboCasa gives it, not the size of its model file.
    """
    from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES

    model_file = (
        robocasa_loader.directory / "objects" / "objaverse" / "pan" / "pan_0"
    ) / "model.xml"
    unscaled = _parse_robosuite_mjcf(MJCFParser.from_file(str(model_file)))
    scale = OBJ_CATEGORIES[RoboCasaObjectCategory.PAN]["objaverse"].scale

    loaded = robocasa_loader.load_object(RoboCasaObjectCategory.PAN)

    assert extent_of(loaded) == pytest.approx(
        [length * scale for length in extent_of(unscaled)], abs=1e-6
    )


def extent_of(world: World) -> list[float]:
    """
    :return: The length, width and height of the collision geometry of the only body
        of ``world`` that has any, in that body's own frame.
    """
    [body] = world.bodies_with_collision
    box = body.collision.as_bounding_box_collection_in_frame(body).bounding_box()
    return [box.max_x - box.min_x, box.max_y - box.min_y, box.max_z - box.min_z]


def test_load_object_from_group_without_objaverse_assets(robocasa_loader):
    """
    A category with no objaverse assets still loads from another self-contained group.
    """
    world = robocasa_loader.load_object(RoboCasaObjectCategory.POT)
    assert len(world.bodies) > 0


def test_load_task_binds_instruction_and_world(robocasa_loader):
    """
    The loaded task carries its natural-language instruction bound to the scene it plays
    out in.
    """
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    task = robocasa_loader.load_task(
        "TurnOnMicrowave",
        layout_id=LayoutType.LAYOUT001,
        style_id=StyleType.STYLE001,
    )
    assert "microwave" in task.instruction.lower()
    assert len(task.world.bodies) > 0
    assert len(task.world.semantic_annotations) > 0


def test_load_task_omits_robot(robocasa_loader):
    """
    The task world contains no robot, since semantic_digital_twin owns the robot.
    """
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    task = robocasa_loader.load_task(
        "TurnOnMicrowave",
        layout_id=LayoutType.LAYOUT001,
        style_id=StyleType.STYLE001,
    )
    assert not any(
        body.name.name.startswith(RoboCasaDatasetLoader.robot_body_name_prefixes)
        for body in task.world.bodies
    )


def test_load_task_manipulated_objects_are_in_world(robocasa_loader):
    """
    The task's manipulated objects are resolved to bodies present in the task world.
    """
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    task = robocasa_loader.load_task(
        "TurnOnMicrowave",
        layout_id=LayoutType.LAYOUT001,
        style_id=StyleType.STYLE001,
    )
    assert len(task.manipulated_objects) >= 1
    for manipulated_object in task.manipulated_objects:
        assert manipulated_object in task.world.bodies


def test_load_task_with_a_seed_places_the_same_objects_every_time(robocasa_loader):
    """
    RoboCasa samples a task's objects and where they stand; seeded, it samples the same
    scene on every load.
    """
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    first, second = (
        robocasa_loader.load_task(
            "PickPlaceCounterToStove",
            layout_id=LayoutType.LAYOUT001,
            style_id=StyleType.STYLE001,
            seed=0,
        )
        for _ in range(2)
    )

    assert first.instruction == second.instruction
    assert [body.global_pose.to_np() for body in first.manipulated_objects] == [
        pytest.approx(body.global_pose.to_np()) for body in second.manipulated_objects
    ]


def test_load_task_builds_the_kitchen_a_layout_configuration_describes(
    robocasa_loader,
):
    """
    A layout given as its configuration rather than by its id builds the kitchen that
    configuration describes: with every counter lowered, the plate the task sets on a
    counter stands at the lowered height.
    """
    import yaml
    from robocasa.models.scenes.scene_registry import (
        LayoutType,
        StyleType,
        get_layout_path,
    )

    with open(get_layout_path(LayoutType.LAYOUT001)) as layout_file:
        layout = yaml.safe_load(layout_file)
    counters = [
        fixture
        for fixture in layout["main_group"]["bottom_row"]
        if fixture["type"] == "counter"
    ]
    for counter in counters:
        counter["size"][2] = LOWERED_COUNTER_HEIGHT
        if "pos" in counter:
            counter["pos"][2] = LOWERED_COUNTER_HEIGHT / 2

    task = robocasa_loader.load_task(
        "PickPlaceCounterToStove",
        layout_id=layout,
        style_id=StyleType.STYLE001,
        seed=1,
    )

    [plate] = [
        body
        for body in task.manipulated_objects
        if body.name.name == "obj_container_main"
    ]
    plate_extent = plate.collision.as_bounding_box_collection_in_frame(
        task.world.root
    ).bounding_box()
    assert plate_extent.min_z == pytest.approx(LOWERED_COUNTER_HEIGHT, abs=0.01)


def test_load_task_leaves_the_layout_configuration_it_is_given_unchanged(
    robocasa_loader,
):
    """
    RoboCasa edits the layout it composes a kitchen from; the loader keeps the caller's
    configuration as it was, so it builds the same kitchen again.
    """
    import yaml
    from robocasa.models.scenes.scene_registry import (
        LayoutType,
        StyleType,
        get_layout_path,
    )

    with open(get_layout_path(LayoutType.LAYOUT001)) as layout_file:
        layout = yaml.safe_load(layout_file)
    as_given = copy.deepcopy(layout)

    robocasa_loader.load_task(
        "PickPlaceCounterToStove",
        layout_id=layout,
        style_id=StyleType.STYLE001,
        seed=1,
    )

    assert layout == as_given


LOWERED_COUNTER_HEIGHT = 0.75
"""
The height, in metres, a test lowers every counter of a layout to.
"""


def test_load_task_unknown_name_raises(robocasa_loader):
    """
    Requesting a task that is not registered raises a descriptive error.
    """
    from robocasa.models.scenes.scene_registry import LayoutType, StyleType

    with pytest.raises(RoboCasaTaskNotFoundError):
        robocasa_loader.load_task(
            "ThisTaskDoesNotExist",
            layout_id=LayoutType.LAYOUT001,
            style_id=StyleType.STYLE001,
        )
