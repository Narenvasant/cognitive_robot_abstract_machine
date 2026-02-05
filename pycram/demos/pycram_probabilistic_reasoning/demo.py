import os
import math
import time
import threading

import rclpy

from pycram.datastructures.grasp import GraspDescription
from pycram.process_module import simulated_robot
from pycram.robot_plans import ParkArmsActionDescription, NavigateActionDescription, PickUpActionDescription, \
    PlaceActionDescription, NavigateAction, PickUpAction, PlaceAction, ParkArmsAction

from krrood.probabilistic_knowledge.parameterizer import Parameterizer

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.pose import PoseStamped, PyCramPose, PyCramVector3, PyCramQuaternion, Header
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.language import SequentialPlan
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection, Connection6DoF, OmniDrive
from semantic_digital_twin.world_description.geometry import FileMesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def simple_plan():
    """
    Create and parameterize a pick-and-place plan in a world.

    :return: A plan and a probabilistic circuit; also prints a few samples.
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

    robot_pose = HomogeneousTransformationMatrix.from_xyz_rpy(1.4, 1.5, 1.0, 0, 0, 0)
    with world.modify_world():
        world.merge_world_at_pose(robot_world, robot_pose)

    robot = PR2.from_world(world)

    with world.modify_world():
        map_body = Body(name=PrefixedName("map"))
        localization_body = Body(name=PrefixedName("odom_combined"))
        world.add_body(map_body)
        world.add_body(localization_body)

        world.add_connection(FixedConnection(parent=world.root, child=map_body))
        world.add_connection(Connection6DoF.create_with_dofs(world, map_body, localization_body))

        old_root_connection = robot.root.parent_connection
        if old_root_connection is not None:
            world.remove_connection(old_root_connection)

        omni_connection = OmniDrive.create_with_dofs(
            parent=localization_body,
            child=robot.root,
            world=world,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(1.2, 1.5, 0.0, 0, 0, 0),
        )
        world.add_connection(omni_connection)

    milk_mesh = FileMesh.from_file(milk_stl)
    milk_body = Body(
        name=PrefixedName("milk"),
        visual=ShapeCollection([milk_mesh]),
        collision=ShapeCollection([milk_mesh]),
    )
    milk_pose = HomogeneousTransformationMatrix.from_xyz_rpy(2.4, 2.5, 1.01, 0, 0, 0)

    with world.modify_world():
        world.add_body(milk_body)
        milk_connection = Connection6DoF.create_with_dofs(parent=world.root, child=milk_body, world=world)
        world.add_connection(milk_connection)
        milk_connection.origin = milk_pose

    rclpy.init()
    node = rclpy.create_node("pycram_demo")
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    try:
        _tf_publisher = TFPublisher(world=world, node=node)
        _viz_publisher = VizMarkerPublisher(world=world, node=node)

        context = Context(world, robot, None)

        nav_pose_pickup = PoseStamped(
            PyCramPose(PyCramVector3(1.6, 2.5, 0.0), PyCramQuaternion(0, 0, 0, 1)),
            Header(world.get_body_by_name("map")),
        )
        nav_pose_placing = PoseStamped(
            PyCramPose(PyCramVector3(1.6, 1.5, 0.0), PyCramQuaternion(0, 0, 0, 1)),
            Header(world.get_body_by_name("map")),
        )
        placing_pose = PoseStamped(
            PyCramPose(PyCramVector3(2.4, 1.5, 1.0), PyCramQuaternion(0, 0, 0, 1)),
            Header(world.get_body_by_name("map")),
        )

        plan = SequentialPlan(
            context,
            ParkArmsActionDescription(Arms.BOTH),
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
            NavigateActionDescription(target_location=nav_pose_placing),
            PlaceActionDescription(
                object_designator=milk_body,
                target_location=placing_pose,
                arm=Arms.RIGHT,
            ),
            ParkArmsActionDescription(Arms.BOTH),
        )

        plan_classes = [
            ParkArmsAction,
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
        probabilistic_circuit = Parameterizer().create_fully_factorized_distribution(variables)

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

        with simulated_robot:
            plan.perform()

        return plan, probabilistic_circuit

    finally:
        time.sleep(0.1)
        node.destroy_node()
        rclpy.shutdown()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    simple_plan()
