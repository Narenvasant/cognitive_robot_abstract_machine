#!/usr/bin/env python
import rospy

from giskardpy.middleware.ros2.behavior_tree_config import OpenLoopBTConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.configs import (
    WorldWithHSRConfig,
    HSRCollisionAvoidanceConfig,
    HSRJointTrajInterfaceConfig,
)
from giskardpy_ros.ros1.interface import ROS1Wrapper
from middleware import set_middleware

if __name__ == "__main__":
    rospy.init_node("giskard")
    set_middleware(ROS1Wrapper())
    debug_mode = rospy.get_param("~debug_mode", False)
    giskard = Giskard(
        world_config=WorldWithHSRConfig(),
        collision_avoidance_config=HSRCollisionAvoidanceConfig(),
        robot_interface_config=HSRJointTrajInterfaceConfig(),
        behavior_tree_config=OpenLoopBTConfig(debug_mode=debug_mode),
    )
    giskard.live()
