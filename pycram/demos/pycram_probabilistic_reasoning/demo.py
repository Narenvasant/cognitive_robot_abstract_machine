import os
import time
import threading
import json
import traceback

import rclpy

from pycram.datastructures.grasp import GraspDescription
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import ParkArmsActionDescription, NavigateActionDescription, PickUpActionDescription, \
    PlaceActionDescription

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


def reset_milk_body(world, milk_body, milk_pose):
    """Reset milk body position by updating its connection origin."""
    with world.modify_world():
        connection = milk_body.parent_connection
        if connection:
            connection.origin = milk_pose


def reset_action_iters(plan):
    """Reset action iterators for all action nodes in the plan."""
    for action_node in plan.actions:
        action_node.action_iter = None


def make_pose_stamped(world, pos, ori):
    """Create a PoseStamped from position and orientation lists."""
    return PoseStamped(
        PyCramPose(
            PyCramVector3(pos[0], pos[1], pos[2]),
            PyCramQuaternion(ori[0], ori[1], ori[2], ori[3]),
        ),
        Header(world.get_body_by_name("map")),
    )


def contains_ellipsis(obj, visited=None, depth=0, max_depth=10):
    """
    Check if an object or its nested attributes contain Ellipsis.

    :param obj: The object to check
    :param visited: Set of object IDs already visited (to prevent circular references)
    :param depth: Current recursion depth
    :param max_depth: Maximum recursion depth to prevent infinite loops
    :return: True if Ellipsis is found, False otherwise
    """
    if depth > max_depth:
        return False

    if visited is None:
        visited = set()

    # Check if this object was already visited
    obj_id = id(obj)
    if obj_id in visited:
        return False

    # Check for Ellipsis
    if obj is ...:
        return True

    # Check collections without recursing into complex objects
    if isinstance(obj, (list, tuple)):
        visited.add(obj_id)
        return any(contains_ellipsis(item, visited, depth + 1, max_depth) for item in obj)

    # Only recurse into simple dataclass-like objects, not complex world entities
    if hasattr(obj, '__dict__') and not isinstance(obj, (Body, type)):
        visited.add(obj_id)
        # Filter out attributes that are likely to cause circular references
        attrs_to_check = {
            k: v for k, v in obj.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }
        return any(
            contains_ellipsis(value, visited, depth + 1, max_depth)
            for value in attrs_to_check.values()
        )

    return False


def simple_plan():
    """
    Create and parameterize a pick-and-place plan, then iterate through samples
    to generate a dataset of successful executions.
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

    # Create milk object
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
    node = rclpy.create_node("pycram_demo")
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    try:
        _tf_publisher = TFPublisher(world=world, node=node)
        _viz_publisher = VizMarkerPublisher(world=world, node=node)

        context = Context(world, robot, None)

        # Define poses with dummy values for first execution
        nav_pose_pickup = PoseStamped(
            PyCramPose(
                PyCramVector3(1.6, 2.5, 0.0),
                PyCramQuaternion(0.0, 0.0, 0.0, 1.0),
            ),
            Header(world.get_body_by_name("map")),
        )

        nav_pose_placing = PoseStamped(
            PyCramPose(
                PyCramVector3(1.6, 1.5, 0.0),
                PyCramQuaternion(0.0, 0.0, 0.0, 1.0),
            ),
            Header(world.get_body_by_name("map")),
        )

        placing_pose = PoseStamped(
            PyCramPose(
                PyCramVector3(2.4, 1.5, 1.0),
                PyCramQuaternion(0.0, 0.0, 0.0, 1.0),
            ),
            Header(world.get_body_by_name("map")),
        )

        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot.right_arm.manipulator,
        )

        # Create initial plan with dummy values
        initial_plan = SequentialPlan(
            context,
            ParkArmsActionDescription(arm=Arms.BOTH),
            NavigateActionDescription(target_location=nav_pose_pickup),
            PickUpActionDescription(
                object_designator=milk_body_1,
                arm=Arms.RIGHT,
                grasp_description=grasp_description,
            ),
            NavigateActionDescription(target_location=nav_pose_placing),
            PlaceActionDescription(
                object_designator=milk_body_1,
                target_location=placing_pose,
                arm=Arms.RIGHT,
            ),
            ParkArmsActionDescription(arm=Arms.BOTH),
        )

        # Execute first iteration with dummy values
        print(f"\n{'='*60}")
        print("Executing initial plan with dummy values...")
        print(f"{'='*60}\n")

        reset_milk_body(world, milk_body_1, milk_pose_1)
        with simulated_robot:
            initial_plan.perform()

        print("✓ Initial execution SUCCESS\n")

        # Now create parameterized plan for iterations
        nav_pose_pickup_param = PoseStamped(
            PyCramPose(
                PyCramVector3(..., ..., 0.0),
                PyCramQuaternion(0, 0, 0, 1),
            ),
            Header(world.get_body_by_name("map")),
        )

        nav_pose_placing_param = PoseStamped(
            PyCramPose(
                PyCramVector3(..., ..., 0.0),
                PyCramQuaternion(0, 0, 0, 1),
            ),
            Header(world.get_body_by_name("map")),
        )

        # Create parameterized plan with Ellipsis for parameterizable fields only
        param_plan = SequentialPlan(
            context,
            ParkArmsActionDescription(arm=Arms.BOTH),
            NavigateActionDescription(target_location=nav_pose_pickup_param),
            PickUpActionDescription(
                object_designator=milk_body_1,
                arm=...,
                grasp_description=grasp_description,
            ),
            NavigateActionDescription(target_location=nav_pose_placing_param),
            PlaceActionDescription(
                object_designator=milk_body_1,
                target_location=placing_pose,
                arm=...,
            ),
            ParkArmsActionDescription(arm=Arms.BOTH),
        )

        # Generate parameterizations using the new API
        print("Generating parameterizations...")

        try:
            # Create action instances from kwargs and parameterize them
            from krrood.probabilistic_knowledge.parameterizer import Parameterizer

            parameterizations = []

            for action_node in param_plan.actions:
                # Check if this action has any Ellipsis in kwargs
                has_ellipsis = any(
                    contains_ellipsis(value)
                    for value in action_node.kwargs.values()
                )

                if has_ellipsis:
                    # Create action instance directly from kwargs (which contain ...)
                    # The action_node.designator_type is the Action class (e.g., NavigateAction, PickUpAction)
                    action_instance = action_node.designator_type(**action_node.kwargs)

                    try:
                        # Parameterize the action instance
                        parameterization = Parameterizer().parameterize(action_instance)
                        if parameterization.variables:
                            parameterizations.append((action_instance, parameterization, action_node))
                    except Exception as e:
                        print(f"Warning: Could not parameterize {action_node.designator_type.__name__}: {type(e).__name__}")
                        traceback.print_exc()
                        continue

            print(f"Found {len(parameterizations)} parameterizable actions")

        except Exception as e:
            print(f"Error generating parameterizations: {type(e).__name__}")
            traceback.print_exc()
            return param_plan

        if not parameterizations:
            print("Error: No parameterizable actions found. Check the plan structure.")
            return param_plan

        print(f"\n{'='*60}")
        print("Starting sample iterations...")
        print(f"{'='*60}\n")

        sample_count = 500
        results = []
        success_samples = []

        for sample_idx in range(sample_count):
            print(f"Testing Sample {sample_idx + 1}/{sample_count}")

            try:
                # Create new actions by sampling parameters
                sampled_params = []
                new_action_instances = []
                parameterized_node_ids = set()

                for action_instance, parameterization, action_node in parameterizations:
                    # Create fully factorized distribution
                    distribution = parameterization.create_fully_factorized_distribution()

                    # Apply conditioning
                    distribution, _ = distribution.conditional(
                        parameterization.assignments_for_conditioning
                    )

                    # Sample from distribution
                    sample = distribution.sample(1)[0]

                    # Create assignment dictionary
                    sample_dict = parameterization.create_assignment_from_variables_and_sample(
                        distribution.variables, sample
                    )

                    # Parameterize the action instance with the sample
                    parameterized_action = parameterization.parameterize_object_with_sample(
                        action_instance, sample_dict
                    )
                    new_action_instances.append(parameterized_action)
                    parameterized_node_ids.add(id(action_node))
                    sampled_params.append((type(action_instance).__name__, sample_dict))

                # Rebuild full action list including non-parameterized actions
                all_action_descriptions = []
                param_idx = 0

                for action_node in param_plan.actions:
                    if id(action_node) in parameterized_node_ids:
                        # Use parameterized action instance
                        all_action_descriptions.append(new_action_instances[param_idx])
                        param_idx += 1
                    else:
                        # Create non-parameterized action from kwargs
                        action = action_node.designator_type(**action_node.kwargs)
                        all_action_descriptions.append(action)

                # Create new plan with all action instances
                new_plan = SequentialPlan(context, *all_action_descriptions)

                # Reset milk body position
                reset_milk_body(world, milk_body_1, milk_pose_1)

                # Execute plan
                with simulated_robot:
                    new_plan.perform()

                print(f"✓ Sample {sample_idx + 1} SUCCESS")
                results.append((sample_idx + 1, "SUCCESS", None))

                # Store successful sample data
                success_sample = {
                    "sample_index": sample_idx + 1,
                    "actions": []
                }

                for action_name, sample_dict in sampled_params:
                    success_sample["actions"].append({
                        "action_type": action_name,
                        "parameters": {str(k.variable.name): v for k, v in sample_dict.items()}
                    })

                success_samples.append(success_sample)

            except Exception as e:
                print(f"✗ Sample {sample_idx + 1} FAILED")
                print(f"Error: {type(e).__name__}: {str(e)}")
                traceback.print_exc()
                results.append((sample_idx + 1, "FAILED", str(e)))

        # Print summary
        success_count = sum(1 for _, status, _ in results if status == "SUCCESS")
        failure_count = len(results) - success_count

        print(f"\n{'='*60}")
        print("EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples tested: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {failure_count}")
        print(f"{'='*60}\n")

        # Save successful samples to file
        dataset_path = os.path.join(os.path.dirname(__file__), "success_samples.jsonl")
        with open(dataset_path, "w", encoding="utf-8") as f:
            for item in success_samples:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(success_samples)} successful samples to: {dataset_path}")

        return param_plan

    finally:
        time.sleep(0.1)
        node.destroy_node()
        rclpy.shutdown()
        thread.join(timeout=2.0)


if __name__ == "__main__":
    simple_plan()
