from __future__ import annotations
import math
import os
import numpy as np

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import (
    Header,
    PoseStamped,
    PyCramPose,
    PyCramQuaternion,
    PyCramVector3,
)
from pycram.language import SequentialPlan
from pycram.robot_plans import (
    NavigateAction,
    NavigateActionDescription,
    PickUpAction,
    PickUpActionDescription,
    PlaceAction,
    PlaceActionDescription,
)

from krrood.probabilistic_knowledge.parameterizer import Parameterizer

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import FileMesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def simple_plan():
    """
    Create and parameterize a pick-and-place plan in a world.

    :return: A probabilistic circuit and prints of few samples
    """
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    resource_path = os.path.join(base_path, "resources")

    apartment_urdf = os.path.join(resource_path, "worlds", "apartment.urdf")
    robot_urdf = os.path.join(resource_path, "robots", "pr2.urdf")
    milk_stl = os.path.join(resource_path, "objects", "milk.stl")

    package_resolver = {
        "iai_apartment": os.path.join(resource_path, "worlds"),
        "iai_pr2_description": os.path.join(resource_path, "robots"),
        "iai_kitchen": os.path.join(resource_path, "objects"),
    }

    with open(apartment_urdf, "r") as f:
        apartment_urdf_str = f.read()
    world = URDFParser(urdf=apartment_urdf_str, package_resolver=package_resolver).parse()

    with open(robot_urdf, "r") as f:
        robot_urdf_str = f.read()
    robot_world = URDFParser(urdf=robot_urdf_str, package_resolver=package_resolver).parse()

    robot_pose = np.eye(4)
    robot_pose[:3, 3] = [0.5, 0.5, 0.0]
    world.merge_world_at_pose(robot_world, robot_pose)

    robot = PR2.from_world(world)

    milk_mesh = FileMesh.from_file(milk_stl)
    milk_body = Body(
        name=PrefixedName("milk"),
        visual=ShapeCollection([milk_mesh]),
        collision=ShapeCollection([milk_mesh]),
    )

    milk_pose = HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 1.5, 0.8, 0, 0, 0)

    with world.modify_world():
        world.add_body(milk_body)
        milk_connection = Connection6DoF.create_with_dofs(parent=world.root, child=milk_body, world=world)
        world.add_connection(milk_connection)
        milk_connection.origin = milk_pose



    context = Context(world, robot, None)

    nav_pose_pickup = PoseStamped(
        PyCramPose(PyCramVector3(1.3, 1.3, 0.0), PyCramQuaternion(0, 0, 0, 1)),
        Header("map"),
    )
    nav_pose_place = PoseStamped(
        PyCramPose(PyCramVector3(2.5, 1.5, 0.0), PyCramQuaternion(0, 0, 0, 1)),
        Header("map"),
    )
    place_pose = PoseStamped(
        PyCramPose(PyCramVector3(2.8, 1.5, 0.8), PyCramQuaternion(0, 0, 0, 1)),
        Header("map"),
    )

    plan = SequentialPlan(
        context,
        NavigateActionDescription(target_location=nav_pose_pickup),
        PickUpActionDescription(
            object_designator=milk_body,
            arm=Arms.RIGHT,
            grasp_description=GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.NoAlignment,
                robot.right_arm.manipulator,
            ),
        ),
        NavigateActionDescription(target_location=nav_pose_place),
        PlaceActionDescription(
            object_designator=milk_body,
            target_location=place_pose,
            arm=Arms.RIGHT,
        ),
    )

    plan_classes = [
        NavigateAction,
        PickUpAction,
        PlaceAction,
        GraspDescription,
        PoseStamped,
        PyCramPose,
        PyCramVector3,
        PyCramQuaternion,
        Header,
    ]

    variables = plan.parameterize_plan(classes=plan_classes)
    parameterizer = Parameterizer()
    probabilistic_circuit = parameterizer.create_fully_factorized_distribution(variables)

    def print_sample(prob_circuit, sample_values, fmt: str = "{:.2f}") -> None:
        variables_local = prob_circuit.variables
        sample = dict(zip(variables_local, sample_values))

        groups: dict[str, dict] = {}
        others: list[tuple[str, object]] = []

        for var in variables_local:
            name = var.name
            if ".pose." in name:
                base, tail = name.split(".pose.", 1)
                grp = groups.setdefault(base + ".pose", {"position": {}, "orientation": {}})
                if tail.startswith("position."):
                    comp = tail.split(".", 1)[1]
                    grp["position"][comp] = sample[var]
                if tail.startswith("orientation."):
                    comp = tail.split(".", 1)[1]
                    grp["orientation"][comp] = sample[var]
            else:
                others.append((name, sample[var]))

        for name, val in sorted(others, key=lambda x: x[0]):
            if isinstance(val, (bool, str)) or getattr(val, "__class__", None).__name__.startswith(
                ("ApproachDirection", "VerticalAlignment", "Arms")
            ):
                print(f"  {name}: {val}")
            else:
                print(f"  {name}: {fmt.format(float(val))}")

        for base in sorted(groups.keys()):
            pos = groups[base]["position"]
            ori = groups[base]["orientation"]

            px = float(pos.get("x", 0.0))
            py = float(pos.get("y", 0.0))
            pz = float(pos.get("z", 0.0))

            ox = float(ori.get("x", 0.0))
            oy = float(ori.get("y", 0.0))
            oz = float(ori.get("z", 0.0))
            ow = float(ori.get("w", 1.0))

            norm = math.sqrt(ox * ox + oy * oy + oz * oz + ow * ow) or 1.0
            oxn, oyn, ozn, own = ox / norm, oy / norm, oz / norm, ow / norm

            print(f"  {base}:")
            print(f"    position: x={fmt.format(px)}, y={fmt.format(py)}, z={fmt.format(pz)}")
            print(
                "    orientation (normalized): "
                f"x={fmt.format(oxn)}, y={fmt.format(oyn)}, z={fmt.format(ozn)}, w={fmt.format(own)}"
            )

    print(f"Plan parameterized into {len(variables)} variables.")
    print("Variable names in the probabilistic circuit:")
    for variable in probabilistic_circuit.variables:
        print(f" - {variable.name}")

    samples = probabilistic_circuit.sample(5)
    for i, sample_values in enumerate(samples):
        print(f"\nSample {i + 1}:")
        print_sample(probabilistic_circuit, sample_values)

    return plan, probabilistic_circuit

if __name__ == "__main__":
    simple_plan()
