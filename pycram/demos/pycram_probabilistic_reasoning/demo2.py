import os
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


def sample_to_dict(probabilistic_circuit, sample):
    """
    Convert a sample array to a dictionary mapping variable names to values.
    """
    result = {}
    for idx, var in enumerate(probabilistic_circuit.variables):
        result[var.name] = sample[idx]
    return result


def update_pose_from_sample(sample_dict, prefix, position_list, orientation_list):
    """
    Update position and orientation lists from sampled values.

    :param sample_dict: Dictionary of sampled values
    :param prefix: Variable name prefix (e.g., "NavigateAction_1.target_location")
    :param pos_list: List [x, y, z] to update
    :param ori_list: List [qx, qy, qz, qw] to update
    """
    #  position
    position_list[0] = sample_dict.get(f"{prefix}.pose.position.x", position_list[0])
    position_list[1] = sample_dict.get(f"{prefix}.pose.position.y", position_list[1])
    position_list[2] = sample_dict.get(f"{prefix}.pose.position.z", position_list[2])

    #  orientation
    orientation_list[0] = sample_dict.get(f"{prefix}.pose.orientation.x", orientation_list[0])
    orientation_list[1] = sample_dict.get(f"{prefix}.pose.orientation.y", orientation_list[1])
    orientation_list[2] = sample_dict.get(f"{prefix}.pose.orientation.z", orientation_list[2])
    orientation_list[3] = sample_dict.get(f"{prefix}.pose.orientation.w", orientation_list[3])


def update_arm_from_sample(sample_dict, pickup_description, grasp_description, place_description):
    """
    Dynamically update plan action descriptions with sampled values.

    :param plan: The SequentialPlan to update
    :param sample_dict: Dictionary of sampled values
    :param pickup_description: PickUpActionDescription object to update
    :param grasp_description: GraspDescription object to update
    :param place_description: PlaceActionDescription object to update
    """
    pickup_description.arm = int(sample_dict["PickUpAction_2.arm"])

    grasp_description.approach_direction = int(sample_dict["PickUpAction_2.grasp_description.approach_direction"])

    grasp_description.manipulation_offset = sample_dict["PickUpAction_2.grasp_description.manipulation_offset"]

    grasp_description.rotate_gripper = bool(sample_dict["PickUpAction_2.grasp_description.rotate_gripper"])

    grasp_description.vertical_alignment = int(sample_dict["PickUpAction_2.grasp_description.vertical_alignment"])

    place_description.arm = int(sample_dict["PlaceAction_4.arm"])


def set_pose_from_lists(pose_stamped, position, orientation):
    pose_stamped.pose.position.x = position[0]
    pose_stamped.pose.position.y = position[1]
    pose_stamped.pose.position.z = position[2]
    pose_stamped.pose.orientation.x = orientation[0]
    pose_stamped.pose.orientation.y = orientation[1]
    pose_stamped.pose.orientation.z = orientation[2]
    pose_stamped.pose.orientation.w = orientation[3]


def reset_action_iters(plan):
    for action_node in plan.actions:
        action_node.action_iter = None


def add_milk_body(world, milk_body, milk_pose):
    with world.modify_world():
        world.add_body(milk_body)
        milk_connection = Connection6DoF.create_with_dofs(parent=world.root, child=milk_body, world=world)
        world.add_connection(milk_connection)
        milk_connection.origin = milk_pose


def make_pose_stamped(world, pos, ori):
    return PoseStamped(
        PyCramPose(
            PyCramVector3(pos[0], pos[1], pos[2]),
            PyCramQuaternion(ori[0], ori[1], ori[2], ori[3]),
        ),
        Header(world.get_body_by_name("map")),
    )


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

        # dummy working values
        nav_pickup_position = [1.6, 2.5, 0.0]
        nav_pickup_orientation = [0.0, 0.0, 0.0, 1.0]

        nav_placing_position = [1.6, 1.5, 0.0]
        nav_placing_orientation = [0.0, 0.0, 0.0, 1.0]

        placing_position = [2.4, 1.5, 1.01]
        placing_orientation = [0.0, 0.0, 0.0, 1.0]

        nav_pose_pickup = make_pose_stamped(world, nav_pickup_position, nav_pickup_orientation)
        nav_pose_placing = make_pose_stamped(world, nav_placing_position, nav_placing_orientation)
        placing_pose = make_pose_stamped(world, placing_position, placing_orientation)

        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot.right_arm.manipulator,
        )

        place_description = PlaceActionDescription(
            object_designator=milk_body,
            target_location=placing_pose,
            arm=Arms.RIGHT,
        )

        pickup_description = PickUpActionDescription(
            object_designator=milk_body,
            arm=Arms.RIGHT,
            grasp_description=grasp_description,
        )

        plan = SequentialPlan(
            context,
            ParkArmsActionDescription(Arms.BOTH),
            NavigateActionDescription(target_location=nav_pose_pickup),
            pickup_description,
            NavigateActionDescription(target_location=nav_pose_placing),
            place_description,
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

        print("First execution (with initial values):")
        with simulated_robot:
            plan.perform()

        print(f"\n{'='*60}")
        print("Starting sample iterations...")
        print(f"{'='*60}\n")

        sample_count = 10000
        samples = probabilistic_circuit.sample(sample_count)
        results = []

        for sample_idx in range(sample_count):
            print(f"Testing Sample {sample_idx + 1}/{sample_count}")

            try:
                sample_dict = sample_to_dict(probabilistic_circuit, samples[sample_idx])

                update_pose_from_sample(
                    sample_dict,
                    "NavigateAction_1.target_location",
                    nav_pickup_position,
                    nav_pickup_orientation,
                )

                update_pose_from_sample(
                    sample_dict,
                    "NavigateAction_3.target_location",
                    nav_placing_position,
                    nav_placing_orientation,
                )

                update_pose_from_sample(
                    sample_dict,
                    "PlaceAction_4.target_location",
                    placing_position,
                    placing_orientation,
                )

                update_arm_from_sample(sample_dict, pickup_description, grasp_description, place_description)

                milk_pose = HomogeneousTransformationMatrix.from_xyz_rpy(2.4, 2.5, 1.01, 0, 0, 0)
                add_milk_body(world, milk_body, milk_pose)

                set_pose_from_lists(nav_pose_pickup, nav_pickup_position, nav_pickup_orientation)
                set_pose_from_lists(nav_pose_placing, nav_placing_position, nav_placing_orientation)
                set_pose_from_lists(placing_pose, placing_position, placing_orientation)
                reset_action_iters(plan)

                with simulated_robot:
                    plan.perform()

                print(f"✓ Sample {sample_idx + 1} SUCCESS")
                results.append((sample_idx + 1, "SUCCESS", None))

            except Exception as e:
                print(f"✗ Sample {sample_idx + 1} FAILED")
                print(f"Error: {type(e).__name__}: {str(e)}")
                results.append((sample_idx + 1, "FAILED", str(e)))


        success_count = sum(1 for _, status, _ in results if status == "SUCCESS")
        failure_count = len(results) - success_count

        print("EXECUTION SUMMARY")
        print(f"Total samples tested: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {failure_count}")
        print(f"Success rate: {success_count / len(results) * 100:.1f}%")

        return plan, probabilistic_circuit

    finally:
        time.sleep(0.1)
        node.destroy_node()
        rclpy.shutdown()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    simple_plan()
