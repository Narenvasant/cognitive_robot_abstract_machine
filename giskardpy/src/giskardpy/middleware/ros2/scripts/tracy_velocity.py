from giskardpy.middleware.ros2.utils import load_xacro

from giskardpy.middleware.ros2 import rospy
from rclpy import Parameter
from rclpy.exceptions import ParameterUninitializedException

from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.middleware.ros2.behavior_tree_config import ClosedLoopBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.configs import (
    TracyVelocityInterface,
    WorldWithTracyConfig,
)


def main():
    rospy.init_node("giskard")
    try:
        rospy.node.declare_parameters(
            namespace="", parameters=[("robot_description", Parameter.Type.STRING)]
        )
        robot_description = rospy.node.get_parameter_or("robot_description").value
    except ParameterUninitializedException as e:
        robot_description = load_xacro(
            "package://iai_tracy_description/urdf/tracy.urdf.xacro"
        )
    giskard = Giskard(
        world_config=WorldWithTracyConfig(urdf=robot_description),
        robot_interface_config=TracyVelocityInterface(),
        behavior_tree_config=ClosedLoopBTConfig(),
        qp_controller_config=QPControllerConfig(
            target_frequency=80, prediction_horizon=30
        ),
    )
    giskard.live()


if __name__ == "__main__":
    main()
