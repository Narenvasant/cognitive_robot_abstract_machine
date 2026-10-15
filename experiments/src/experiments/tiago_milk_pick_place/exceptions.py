"""
What the demo raises when the apartment, the robot or the scene is not what the plan
needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


@dataclass
class UnexpectedTableHeight(DataclassException, ValueError):
    """
    Raised when the table the carton stands on is not at the height the layout assumes.

    Every grasp, lift and release height is measured from that surface.
    """

    table_name: str
    """
    Name of the body that was measured.
    """

    measured_height: float
    """
    The height of the table's collision top, in metres.
    """

    expected_height: float
    """
    The height the layout was written for, in metres.
    """

    tolerance: float
    """
    How far the measurement may be off before this is raised, in metres.
    """

    def error_message(self) -> str:
        return (
            f"{self.table_name} has its top at {self.measured_height:.4f} m, but the "
            f"layout expects {self.expected_height:.4f} m give or take "
            f"{self.tolerance:.4f} m."
        )

    def suggest_correction(self) -> str:
        return (
            "Point the demo at the apartment the layout was written for, or set the "
            "layout's expected_table_top to the measured height."
        )


@dataclass
class NotOverTheTable(DataclassException, ValueError):
    """
    Raised when the carton is to be picked or placed where the table does not reach.
    """

    name: str
    """
    What was to be put there.
    """

    x: float
    """
    Where it would have gone along the world's x-axis, in metres.
    """

    y: float
    """
    Where it would have gone along the world's y-axis, in metres.
    """

    def error_message(self) -> str:
        return f"{self.name} at ({self.x:.3f}, {self.y:.3f}) is not over the table."

    def suggest_correction(self) -> str:
        return "Move it inside the table's collision bounding box in x and y."


@dataclass
class OutOfReach(DataclassException, ValueError):
    """
    Raised when the robot would have to reach further than its arm allows.

    The base does not move in this demo, so everything has to be reachable from where
    the robot stands.
    """

    name: str
    """
    What it was reaching for.
    """

    distance: float
    """
    How far it is from where the robot stands, in metres.
    """

    maximum_reach: float
    """
    How far the arm reaches, in metres.
    """

    def error_message(self) -> str:
        return (
            f"{self.name} is {self.distance:.3f} m from where the robot stands, and the "
            f"arm reaches {self.maximum_reach:.3f} m."
        )

    def suggest_correction(self) -> str:
        return "Stand the robot closer, or move the carton nearer to it."


@dataclass
class ServoOnWrongJoint(DataclassException, ValueError):
    """
    Raised when a position servo drives a different joint in MuJoCo than it was
    declared for.
    """

    servo_name: str
    """
    Name of the servo that was declared.
    """

    declared_joint: str
    """
    Name of the joint it was declared for.
    """

    driven_joint: str
    """
    Name of the joint it actually drives in MuJoCo.
    """

    def error_message(self) -> str:
        return (
            f"{self.servo_name} was declared for {self.declared_joint} but drives "
            f"{self.driven_joint}."
        )

    def suggest_correction(self) -> str:
        return (
            "Report this: the MuJoCo builder has to resolve a servo to the joint it "
            "was declared for."
        )


@dataclass
class CartonDidNotSettle(DataclassException, ValueError):
    """
    Raised when the carton is still moving after the physics was given time to settle.
    """

    drift: float
    """
    How far it moved while settling, in metres.
    """

    tilt_degrees: float
    """
    How far it leans from upright, in degrees.
    """

    def error_message(self) -> str:
        return (
            f"The carton drifted {self.drift:.4f} m and leans {self.tilt_degrees:.1f} "
            "degrees after settling."
        )

    def suggest_correction(self) -> str:
        return "Give the physics longer to settle, or spawn the carton a little higher."


@dataclass
class MissingDrive(DataclassException, ValueError):
    """
    Raised when the robot is not standing on a drive the demo can move it with.
    """

    robot_name: str
    """
    Name of the robot that was looked at.
    """

    def error_message(self) -> str:
        return f"{self.robot_name} has no drive above it."

    def suggest_correction(self) -> str:
        return (
            "Spawn the robot with a RobotSpecification, which puts it on a drive below "
            "odom."
        )
