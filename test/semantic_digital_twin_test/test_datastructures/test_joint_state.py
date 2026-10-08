from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.testing import (
    StateChangeCounter,
    two_arm_robot_world,
    world_setup,
)

# %% applying a joint state to a world


def test_applying_a_joint_state_announces_one_state_change(two_arm_robot_world):
    world = two_arm_robot_world
    right_joint = world.get_connection_by_name("r_joint_1")
    left_joint = world.get_connection_by_name("l_joint_1")
    joint_state = JointState.from_mapping({right_joint: 0.5, left_joint: -0.25})
    counter = StateChangeCounter(_world=world)

    joint_state.apply_to(world)

    assert counter.count == 1
    assert right_joint.position == 0.5
    assert left_joint.position == -0.25


def test_applying_a_joint_state_respects_multiplier_and_offset(world_setup):
    world, l1, l2, _, _, _ = world_setup
    prismatic_connection = world.get_connection(l1, l2)
    prismatic_connection.multiplier = 2.0
    prismatic_connection.offset = 0.5

    JointState.from_mapping({prismatic_connection: 1.5}).apply_to(world)

    assert prismatic_connection.position == 1.5
    assert world.state[prismatic_connection.raw_dof.id].position == 0.5


# %% the degrees of freedom a state has to be driven through


def test_the_targets_are_the_positions_applying_the_state_leaves_on_the_dofs(
    world_setup,
):
    """
    Only a degree of freedom can be driven, so the positions a state names for its
    connections have to be turned into positions for those.

    They are the very ones applying the state leaves behind.
    """
    world, l1, l2, _, _, _ = world_setup
    prismatic_connection = world.get_connection(l1, l2)
    prismatic_connection.multiplier = 2.0
    prismatic_connection.offset = 0.5
    joint_state = JointState.from_mapping({prismatic_connection: 1.5})

    targets = joint_state.degree_of_freedom_targets

    joint_state.apply_to(world)
    assert targets == {
        prismatic_connection.raw_dof.name.name: world.state[
            prismatic_connection.raw_dof.id
        ].position
    }


def test_connections_sharing_one_degree_of_freedom_give_one_target(two_arm_robot_world):
    """
    A mimic linkage's connections all follow one degree of freedom, so a state over
    several of them is driven through that one.
    """
    world = two_arm_robot_world
    follower = world.get_connection_by_name("r_joint_1")
    leader = world.get_connection_by_name("l_joint_1")
    follower.raw_dof = leader.raw_dof
    follower.multiplier = -1.0

    targets = JointState.from_mapping(
        {leader: 0.5, follower: -0.5}
    ).degree_of_freedom_targets

    assert targets == {leader.raw_dof.name.name: 0.5}
