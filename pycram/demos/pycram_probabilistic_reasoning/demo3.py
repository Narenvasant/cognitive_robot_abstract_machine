import os
import time
import threading

import rclpy

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.datastructures.pose import PoseStamped, PyCramPose, PyCramVector3, PyCramQuaternion, Header
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import MoveTorsoActionDescription, NavigateActionDescription, ParkArmsActionDescription, \
    PickUpActionDescription, PlaceActionDescription
from pycram.orm.ormatic_interface import *  # This imports DAO mappings
from pycram.view_manager import ViewManager

from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection, Connection6DoF, OmniDrive
from semantic_digital_twin.world_description.geometry import FileMesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


def sequential_plan_with_apartment():
    """
    Parameterize a SequentialPlan using krrood parameterizer in an apartment world,
    create a fully-factorized distribution and assert the correctness of sampled values
    after conditioning and truncation.
    """
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    resource_path = os.path.join(base_path, "resources")

    apartment_urdf = os.path.join(resource_path, "worlds", "apartment.urdf")
    robot_urdf = os.path.join(resource_path, "robots", "pr2.urdf")
    milk_stl = os.path.join(resource_path, "objects", "milk.stl")

    # Load world and robot
    world = URDFParser.from_file(apartment_urdf).parse()
    robot_world = URDFParser.from_file(robot_urdf).parse()

    robot_pose = HomogeneousTransformationMatrix.from_xyz_rpy(1.4, 1.5, 1.0, 0, 0, 0)
    with world.modify_world():
        world.merge_world_at_pose(robot_world, robot_pose)

    robot = PR2.from_world(world)

    # Setup map and localization bodies
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
    milk_body_1 = Body(
        name=PrefixedName("milk_1"),
        visual=ShapeCollection([milk_mesh]),
        collision=ShapeCollection([milk_mesh]),
    )
    milk_pose_1 = HomogeneousTransformationMatrix.from_xyz_rpy(2.4, 2.5, 1.01, 0, 0, 0)

    with world.modify_world():
        world.add_body(milk_body_1)
        milk_connection_1 = Connection6DoF.create_with_dofs(parent=world.root, child=milk_body_1, world=world)
        world.add_connection(milk_connection_1)
        milk_connection_1.origin = milk_pose_1

    # Initialize ROS
    rclpy.init()
    node = rclpy.create_node("sequential_plan")
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    try:
        _tf_publisher = TFPublisher(world=world, node=node)
        _viz_publisher = VizMarkerPublisher(world=world, node=node)

        context = Context(world, robot, None)

        target_location = PoseStamped.from_list([..., ..., 0], [0, 0, 0, 1], frame=None)
        global_pose = HomogeneousTransformationMatrix()
        # grasp_description = GraspDescription(
        #     approach_direction=ApproachDirection.FRONT,
        #     vertical_alignment=VerticalAlignment.NoAlignment,
        #     manipulator=ViewManager.get_arm_view(Arms.RIGHT, robot).manipulator,
        #     rotate_gripper=False,
        #     manipulation_offset=0.05,
        # )

        sp = SequentialPlan(
            context,
            ParkArmsActionDescription(arm=Arms.BOTH),
            NavigateActionDescription(
                target_location=target_location,
                keep_joint_states=...,
            ),
            PickUpActionDescription(
                object_designator=milk_pose_1,
                arm=...,
            ),
            NavigateActionDescription(
                target_location=target_location,
                keep_joint_states=...,
            ),
            PlaceActionDescription(
                object_designator=milk_pose_1,
                target_location=target_location,
                arm=...,
            ),
            ParkArmsActionDescription(arm=...),
        )


        # PickUpActionDescription(
        #     object_designator=milk_body_1,
        #     arm=...,
        #     grasp_description=GraspDescription(
        #         approach_direction=ApproachDirection.FRONT,
        #         vertical_alignment=VerticalAlignment.NoAlignment,
        #         manipulator=ViewManager.get_arm_view(Arms.BOTH, robot).manipulator,
        #     ),
        # ),


        parameterization = sp.generate_parameterizations()
        target_location.frame_id = world.root
        new_actions = []

        for i, (action, parameters) in enumerate(parameterization):
            print(f"\n--- Action {i}: {action.__class__.__name__} ---")
            print(f"Variables: {[str(v) for v in parameters.random_events_variables]}")
            print(f"Assignments: {parameters.assignments_for_conditioning}")

            distribution = parameters.create_fully_factorized_distribution()
            print(f"Distribution variables: {distribution.variables}")

            distribution, event = distribution.conditional(
                parameters.assignments_for_conditioning
            )
            print(f"Conditioned event: {event}")

            sample = distribution.sample(1)[0]
            print(f"Sample: {sample}")

            sample_dict = parameters.create_assignment_from_variables_and_sample(
                distribution.variables, sample
            )
            print(f"Sample dict keys: {list(sample_dict.keys())}")
            print(f"Sample dict values: {list(sample_dict.values())}")

            parameterized = parameters.parameterize_object_with_sample(action, sample_dict)
            print(f"Parameterized action: {parameterized}")
            new_actions.append(parameterized)

        new_plan = SequentialPlan(context, *new_actions)

        with simulated_robot:
            new_plan.perform()

        print(f"\nFinal robot pose: {robot.root.global_pose}")

    finally:
        time.sleep(0.1)
        node.destroy_node()
        rclpy.shutdown()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    sequential_plan_with_apartment()