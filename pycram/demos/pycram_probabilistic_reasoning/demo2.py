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

        import math

        # ------------------------------------------------------------
        # 1. Store numeric pose values OUTSIDE (mutable containers)
        # ------------------------------------------------------------

        nav_pickup_pos = [1.6, 2.5, 0.0]
        nav_pickup_ori = [0.0, 0.0, 0.0, 1.0]

        nav_placing_pos = [1.6, 1.5, 0.0]
        nav_placing_ori = [0.0, 0.0, 0.0, 1.0]

        placing_pos = [2.4, 1.5, 1.01]
        placing_ori = [0.0, 0.0, 0.0, 1.0]

        # ------------------------------------------------------------
        # 2. Helper to construct PoseStamped (no logic change)
        # ------------------------------------------------------------

        def make_pose_stamped(pos, ori):
            return PoseStamped(
                PyCramPose(
                    PyCramVector3(pos[0], pos[1], pos[2]),
                    PyCramQuaternion(ori[0], ori[1], ori[2], ori[3]),
                ),
                Header(world.get_body_by_name("map")),
            )

        # ------------------------------------------------------------
        # 3. Build poses (FIRST RUN)
        # ------------------------------------------------------------

        nav_pose_pickup = make_pose_stamped(nav_pickup_pos, nav_pickup_ori)
        nav_pose_placing = make_pose_stamped(nav_placing_pos, nav_placing_ori)
        placing_pose = make_pose_stamped(placing_pos, placing_ori)

        # ------------------------------------------------------------
        # 4. Build plan (UNCHANGED structure)
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # 5. Parameterize + create distribution
        # ------------------------------------------------------------

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

        # ------------------------------------------------------------
        # 6. First execution
        # ------------------------------------------------------------
        print( "first execution (with initial values):" )
        with simulated_robot:
            plan.perform()

        # ------------------------------------------------------------
        # 7. Sample ONCE and convert to dict
        # ------------------------------------------------------------

        def sample_to_dict(prob_circuit, sample_values):
            return dict(zip(
                [v.name for v in prob_circuit.variables],
                sample_values,
            ))

        samples = probabilistic_circuit.sample(10)
        sample_dict = sample_to_dict(probabilistic_circuit, samples[8])
        # ------------------------------------------------------------
        # 8. Update numeric pose lists from sample (NO normalization)
        # ------------------------------------------------------------

        def update_pose_from_sample(sample, base_name, pos, ori):
            pos[0] = float(sample[f"{base_name}.pose.position.x"])
            pos[1] = float(sample[f"{base_name}.pose.position.y"])
            pos[2] = float(sample[f"{base_name}.pose.position.z"])

            ori[0] = float(sample[f"{base_name}.pose.orientation.x"])
            ori[1] = float(sample[f"{base_name}.pose.orientation.y"])
            ori[2] = float(sample[f"{base_name}.pose.orientation.z"])
            ori[3] = float(sample[f"{base_name}.pose.orientation.w"])

        update_pose_from_sample(
            sample_dict,
            "NavigateAction_1.target_location",
            nav_pickup_pos,
            nav_pickup_ori,
        )

        update_pose_from_sample(
            sample_dict,
            "NavigateAction_3.target_location",
            nav_placing_pos,
            nav_placing_ori,
        )

        update_pose_from_sample(
            sample_dict,
            "PlaceAction_4.target_location",
            placing_pos,
            placing_ori,
        )

        # ------------------------------------------------------------
        # 9. Reset milk to original spawn pose
        # ------------------------------------------------------------

        milk_pose = HomogeneousTransformationMatrix.from_xyz_rpy(2.4, 2.5, 1.01, 0, 0, 0)

        with world.modify_world():
            world.add_body(milk_body)
            milk_connection = Connection6DoF.create_with_dofs(parent=world.root, child=milk_body, world=world)
            world.add_connection(milk_connection)
            milk_connection.origin = milk_pose

        # ------------------------------------------------------------
        # 10. Rebuild PoseStamped objects (SECOND RUN)
        # ------------------------------------------------------------

        nav_pose_pickup.pose.position.x = nav_pickup_pos[0]
        nav_pose_pickup.pose.position.y = nav_pickup_pos[1]
        nav_pose_pickup.pose.position.z = nav_pickup_pos[2]
        nav_pose_pickup.pose.orientation.x = nav_pickup_ori[0]
        nav_pose_pickup.pose.orientation.y = nav_pickup_ori[1]
        nav_pose_pickup.pose.orientation.z = nav_pickup_ori[2]
        nav_pose_pickup.pose.orientation.w = nav_pickup_ori[3]

        nav_pose_placing.pose.position.x = nav_placing_pos[0]
        nav_pose_placing.pose.position.y = nav_placing_pos[1]
        nav_pose_placing.pose.position.z = nav_placing_pos[2]
        nav_pose_placing.pose.orientation.x = nav_placing_ori[0]
        nav_pose_placing.pose.orientation.y = nav_placing_ori[1]
        nav_pose_placing.pose.orientation.z = nav_placing_ori[2]
        nav_pose_placing.pose.orientation.w = nav_placing_ori[3]

        placing_pose.pose.position.x = placing_pos[0]
        placing_pose.pose.position.y = placing_pos[1]
        placing_pose.pose.position.z = placing_pos[2]
        placing_pose.pose.orientation.x = placing_ori[0]
        placing_pose.pose.orientation.y = placing_ori[1]
        placing_pose.pose.orientation.z = placing_ori[2]
        placing_pose.pose.orientation.w = placing_ori[3]

        # ------------------------------------------------------------
        # 11. Second execution (with sampled values)
        # Reset iterators in action description nodes before re-executing the plan
        for action_node in plan.actions:
            action_node.action_iter = None


        print( "second execution (with sampled values):" )
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
