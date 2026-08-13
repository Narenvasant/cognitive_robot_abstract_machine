"""
Second-generation copy of ``inference.py``, closed-loop stacking with JPT-guided sampling
and causal failure diagnosis against the models ``training/train_jpt2.py`` fits.

Structured exactly like ``inference.py`` -- same world, simulation, segmind setup and
diagnose-and-correct loop -- with two differences:

- The arm parks between the pickup and the place on every attempt, including every
  corrective retry, instead of crossing directly from the pickup pose to the place pose.
  ``inference.py`` itself notes why that park matters (its own comment on
  ``build_stack_plan``: once the stack has some height, the direct crossing can pass
  right through where the already-stacked cubes are) but currently has it disabled; a
  corrective retry repeats that same crossing several times in a row, which is exactly
  where a collision with an already-stacked cube is most likely. ``demo3.py`` collects
  its training data with this same park in place, so the diagnoser here is trained on
  attempts shaped like the ones it corrects.
- Diagnosis is done against ``causal_diagnosis_v2``'s trees, whose causal circuit uses
  ``object_final_z`` -- the height the picked cube actually settled at -- as the effect
  variable instead of ``step_index``. See ``causal_diagnosis_v2.py``'s own docstring for
  why that makes ``object_friction`` a candidate cause on the same footing as every other
  tunable parameter, with nothing spent on a structural role that carries no information
  about success.

Run it with the interpreter whose packages point at this checkout, for example::

    /home/sorin/.virtualenvs/cram2-env/bin/python inference2.py
"""

import dataclasses
import os
import random
import threading
from pathlib import Path
import time
import numpy
import rclpy
from rclpy.executors import MultiThreadedExecutor

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
    ExecutionType,
    MovementType,
)
from coraplex.datastructures.grasp import GraspDescription

from coraplex.execution_environment import ExecutionEnvironment

# Not used directly (this demo never persists anything to a database) -- imported
# purely so ormatic's DAO registry gets populated as a side effect. Without it,
# sample_pickup_instance/sample_place_instance's underlying UnderspecifiedParameters
# machinery fails with NoDAOFoundError the first time it tries to extract features
# from a literal Body/GraspDescription argument, since that internally calls to_dao().
from coraplex.orm.ormatic_interface import Base  # noqa: F401

from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion

from causal_diagnosis_v2 import (
    ActionCausalDiagnoser,
    NoRecommendationAvailable,
    ParameterCorrection,
    PICKUP_CAUSAL_CONFIG,
    PLACE_CAUSAL_CONFIG,
    PICKUP_MODEL_PATH,
    PLACE_MODEL_PATH,
    RootCauseDiagnosis,
)
from parked_arm_detection_gate import (
    ParkedArmDetectionGate,
    RobotArmParkDeviations,
)
from pickup_place_parameterization import (
    ParameterPrior,
    sample_pickup_instance,
    sample_place_instance,
)

from giskardpy.motion_statechart.context import MotionStatechartContext
from physics_simulators.base_simulator import SimulatorCallbackResult
from segmind.detectors.base import SegmindContext
from segmind.detectors.spatial_relation_detector_nodes import SupportDetector
from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.datastructures.definitions import (
    GripperState,
    StaticJointState,
)
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.adapters.multi_sim import MujocoSim, MujocoBody
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.robots.panda import Panda
from semantic_digital_twin.spatial_types.spatial_types import Pose

from cramera.live.bridge import BRIDGE
from cramera.live.runner import start as start_live_viz
from cramera.server import start_in_background as start_viz_frontend


def verify_workspace_matches_demo() -> None:
    """
    Check that the workspace packages were imported from the same checkout this demo
    file lives in.

    Several checkouts of this repository can be installed in different virtualenvs at
    once. Running the demo with the wrong interpreter loads this file from one checkout
    and its imports from another, which surfaces much later as a missing attribute on a
    class that plainly has it.
    """
    # A concrete module rather than the package: the packages here are
    # namespace packages, whose ``__file__`` is None.
    from physics_simulators import mujoco_simulator

    demo_checkout = Path(__file__).resolve().parents[3]
    package_checkout = Path(mujoco_simulator.__file__).resolve().parents[3]
    if demo_checkout == package_checkout:
        return
    raise RuntimeError(
        f"This demo lives in {demo_checkout} but physics_simulators was imported "
        f"from {package_checkout}. Run it with the interpreter whose packages "
        f"point at {demo_checkout}."
    )


verify_workspace_matches_demo()

start_viz_frontend()
start_live_viz()


# %% causal diagnosers, built once from the trained JPTs -- before the ROS/MuJoCo setup
# below, so a missing or untrained model fails immediately rather than after paying for
# that setup.

pickup_diagnoser = ActionCausalDiagnoser(PICKUP_MODEL_PATH, PICKUP_CAUSAL_CONFIG)
"""
Diagnoses failed pickups against the trained :class:`PickUpAction` parameter tree.
"""

place_diagnoser = ActionCausalDiagnoser(PLACE_MODEL_PATH, PLACE_CAUSAL_CONFIG)
"""
Diagnoses failed placements against the trained :class:`PlaceAction` parameter tree.
"""

RANDOM_SEED = int(os.environ.get("INFERENCE2_RANDOM_SEED", "42"))
"""
Seed for the per-attempt parameter sampling, see ``demo2.py``'s own
:data:`RANDOM_SEED` for why this only makes the sampled parameters -- not the run as a
whole -- reproducible.
"""

random.seed(RANDOM_SEED)
numpy.random.seed(RANDOM_SEED)

execition_mode = ExecutionType.SIMULATED

print("Init ROS")
rclpy.init()
node = rclpy.create_node("stretch_demo_node")

executor = MultiThreadedExecutor()
executor.add_node(node)

thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
thread.start()

STACKING_SCENE_PATH: str = os.environ.get(
    "STACKING_SCENE_PATH",
    "/home/nvasant/workspace/ros/src/manipulation_experiments/resources/generated/stacking_scene.xml",
)
"""
Path to the MJCF scene this demo loads. Override via the ``STACKING_SCENE_PATH``
environment variable for checkouts where the scene lives elsewhere.
"""

world = MJCFParser(STACKING_SCENE_PATH).parse()
Panda.from_world(world)
publisher = VizMarkerPublisher(_world=world, node=node).with_tf_publisher()


# It is important to have the ros_node in the context for a real robot
context = Context(
    world=world,
    robot=world.get_semantic_annotations_by_type(Panda)[0],
    ros_node=node,
    evaluate_conditions=False,
)

box = world.get_body_by_name("cube0")
box1 = world.get_body_by_name("cube1")
box2 = world.get_body_by_name("cube2")
box3 = world.get_body_by_name("cube3")
floor = world.get_body_by_name("floor")

print("Perform Plan")

arm = context.robot.get_arms()[0]
gripper = arm.end_effector
physically_simulated_dofs = {c.raw_dof for c in gripper.active_connections} | {
    c.raw_dof for c in arm.active_connections
}

# See demo2.py's own comment on this block: without gravity compensation, the arm
# never registers as converged and Giskard keeps sending corrective commands
# indefinitely, stalling the rest of the plan.
for connection in arm.active_connections:
    connection.child.simulator_additional_properties.append(
        MujocoBody(gravitation_compensation_factor=1.0)
    )

multi_sim = MujocoSim(
    world=world,
    headless=False,
    step_size=0.0001,
    real_time_factor=1,
    physically_simulated_dofs=physically_simulated_dofs,
    sync_rate_hz=100,
)
time_start = time.time()

tool_frame = gripper.tool_frame


def print_positions():
    """
    Prints the tool_frame's and cube's position as seen by the world model (Giskard's
    kinematic belief) side by side with MuJoCo's own live simulated position, so a
    divergence between "where Giskard thinks it is" and "where it actually, physically
    is" is visible directly.
    """
    tool_frame_kinematic = numpy.array(
        world.compute_forward_kinematics(world.root, tool_frame)
        .to_position()
        .evaluate()[:3],
        dtype=float,
    )
    box_kinematic = numpy.array(
        world.compute_forward_kinematics(world.root, box).to_position().evaluate()[:3],
        dtype=float,
    )
    tool_frame_mujoco = numpy.array(
        multi_sim.simulator.get_body_position(tool_frame.name.name).result[:3],
        dtype=float,
    )
    box_mujoco = numpy.array(
        multi_sim.simulator.get_body_position(box.name.name).result[:3], dtype=float
    )
    print(
        f"tool_frame: kinematic={tool_frame_kinematic} mujoco={tool_frame_mujoco} | "
        f"cube: kinematic={box_kinematic} mujoco={box_mujoco}"
    )


NUMBER_OF_ITERATIONS = int(os.environ.get("INFERENCE2_NUMBER_OF_ITERATIONS", "20"))
"""
Number of times the full pickup/stack sequence is repeated.

Much smaller than ``demo3.py``'s: this demo illustrates the diagnose-and-correct loop
rather than collecting a large training dataset, and wide priors make every iteration
slower (a failed cube can trigger several correction attempts before moving on).
"""

MAX_CORRECTION_ATTEMPTS_PER_CUBE = 3
"""
How many times a single cube is retried with a causally corrected sample before its step
is abandoned as a hard failure.
"""

ITERATION_TIME_LIMIT = 120.0
"""
Wall-clock budget (in seconds) for one iteration, checked between cubes.

Twice ``demo3.py``'s: a cube here can go through several correction attempts, each
costing roughly as long as a normal attempt.
"""

CUBE_PICKUP_TIME_LIMIT = 40.0
"""
Wall-clock budget (in seconds) for one cube's *entire* stacking step -- its initial
attempt, every correction retry, and any re-stack of an earlier cube it disturbs along
the way -- checked independently of, and tighter than, :data:`ITERATION_TIME_LIMIT`.

A cube that has rolled out of comfortable reach fails its reach the same way on every
attempt, and no diagnosed correction fixes "the cube is somewhere else"; left bounded
only by the iteration's own budget, one such cube can burn through most of it retrying a
pickup that was never going to succeed, at the expense of the cubes after it.
"""

STACK_HEIGHT_OFFSET = 0.06
"""
Vertical offset (in meters) above a target cube's center at which a placed cube should
end up -- one cube height plus a small clearance margin.
"""

PICKUP_HOVER_HEIGHT = 0.30
"""
World-frame height (in meters above ``world.root``) the tool center point is moved to,
directly above a cube's current x/y, before the pickup's own reach descends onto it.

Comfortably clears the tallest stack this demo can build (4 cubes, each roughly one
:data:`STACK_HEIGHT_OFFSET` tall) plus the gripper's own finger length. Without this, the
reach goes straight from wherever :class:`~coraplex.robot_plans.actions.core.robot_body.ParkArmsAction`
left the arm to the pickup's pre-pose, at whatever height the direct Cartesian path
happens to pass through -- the same class of problem ``build_stack_plan``'s own
carrying-park step already works around for the place side, but which nothing
previously addressed for the pickup side: once a cube ends up resting against a
neighbour (from an earlier overshoot), that direct path clips the neighbour, collision
avoidance fights the goal instead of reaching it, and the arm bends and sweeps through
the cluster instead of picking cleanly. Moving straight above the target first, then
descending, avoids the direct path ever needing to pass through cluttered space.
"""

CARRYING_PARK_JOINT_VELOCITY = 0.1
"""
Joint velocity (in rad/s) for the park move between pickup and place, i.e. the one made
while the gripper is actually holding a cube. See ``demo3.py``'s own identical constant
for why this is slower than :class:`ParkArmsAction`'s own default -- that default is
deliberately fast for a park made with an empty gripper, and the same speed can jerk a
cube, held by nothing but friction, right out of the fingers -- and for why this value is
a further cut past 1.0 and 0.3 rad/s rather than settled on: tune it again if it is still
too fast (or needlessly slow) once tried here too.
"""

CUBE_SPAWN_POSITIONS = {
    "cube0": numpy.array([0.40, 0.10, 0.02487]),
    "cube1": numpy.array([0.40, -0.04, 0.02487]),
    "cube2": numpy.array([0.40, -0.14, 0.02487]),
    "cube3": numpy.array([0.40, -0.24, 0.02487]),
}
"""
Spawn position of every cube.

x/y match the scene's own MJCF definition. z is the height a cube actually settles at
under gravity, not the scene's own drop height (0.06, ~3.5cm higher) -- teleporting
straight to the settled height, confirmed against the real simulator to produce zero
further drift, is what makes a reset land the cube already at rest instead of dropping
and bouncing it onto the table on every single iteration.
"""

CUBE_SPAWN_ORIENTATION = numpy.array([1.0, 0.0, 0.0, 0.0])
"""
Spawn orientation (identity quaternion) of every cube.
"""

WORKSPACE_BOUND = 5.0
"""
Half-extent (in meters) of the region a cube can legitimately be in, in every axis, see
``demo2.py``'s own :data:`WORKSPACE_BOUND`.
"""

# %% wide sampling priors, to make failures common enough to diagnose

WIDE_PICKUP_PARAMETER_PRIORS: dict[str, ParameterPrior] = {
    # Default high=0.24; 0.2 was already noted to spike InfeasibleException failures,
    # so this reaches well past that.
    "pre_approach_linear_velocity": ParameterPrior(
        mean=0.2, std=0.08, low=0.04, high=0.35
    ),
    "final_approach_linear_velocity": ParameterPrior(
        mean=0.16, std=0.09, low=0.01, high=0.35
    ),
    # Default high=0.22; the finger joints' 0.2 m/s physical limit is noted to punt the
    # cube out of the grasp past that point, so this reaches well past it.
    "grasp_closing_velocity": ParameterPrior(mean=0.2, std=0.1, low=0.02, high=0.4),
    # Default high=0.28; 0.18 was already noted to knock the stack down, so this
    # reaches well past that.
    "lift_linear_velocity": ParameterPrior(mean=0.2, std=0.09, low=0.04, high=0.4),
    # Default low=0.15; below the validated 0.3 floor the grasp is called complete
    # before the fingers settle, so this reaches well below both.
    "grasp_stall_minimum_time": ParameterPrior(mean=0.4, std=0.15, low=0.05, high=0.9),
    # Default low=0.15; below roughly 0.3 the cube is noted to slide straight out of
    # the fingers, so this reaches well below that.
    "object_friction": ParameterPrior(mean=1.2, std=0.7, low=0.05, high=2.8),
}
"""
Same fields as :data:`~pickup_place_parameterization.PICKUP_PARAMETER_PRIORS`, widened
to reach past the failure points ``pickup_place_parameterization.py`` notes for each,
deliberately making some fraction of attempts fail so there is something for causal
diagnosis to work with.
"""

WIDE_PLACE_PARAMETER_PRIORS: dict[str, ParameterPrior] = {
    # Default high=0.18; 0.12 was already noted to knock the stack down, so this
    # reaches well past that.
    "transport_linear_velocity": ParameterPrior(
        mean=0.15, std=0.07, low=0.02, high=0.3
    ),
    # Default high=0.14; 0.08 was already noted to scatter the stack, so this
    # reaches well past that.
    "placing_linear_velocity": ParameterPrior(mean=0.12, std=0.06, low=0.01, high=0.25),
    "release_opening_velocity": ParameterPrior(
        mean=0.15, std=0.07, low=0.015, high=0.28
    ),
    # Default high=0.2; 0.14 was already noted to knock the just-placed cube back
    # down, so this reaches well past that.
    "retract_linear_velocity": ParameterPrior(mean=0.17, std=0.07, low=0.02, high=0.3),
}
"""
Same fields as :data:`~pickup_place_parameterization.PLACE_PARAMETER_PRIORS`, widened
the same way as :data:`WIDE_PICKUP_PARAMETER_PRIORS`.
"""


MODEL_FIELD_TO_LIVE_FIELD: dict[str, str] = {
    "grasp_linear_velocity": "final_approach_linear_velocity",
    "grasp_stall_min_time": "grasp_stall_minimum_time",
}
"""
Maps a trained model's variable name (``causal_diagnosis_v2.PICKUP_CAUSAL_CONFIG``/
``PLACE_CAUSAL_CONFIG``, fixed by ``training/models_v2``'s already-fitted trees) to
this checkout's current :class:`PickUpAction`/:class:`PlaceAction` field name, for the
two fields renamed since those trees were fit. Retraining against the current names
directly would let this map be dropped.
"""


def _observed_parameters(action, field_names: tuple[str, ...]) -> dict[str, float]:
    """
    :return: ``action``'s current value for each of ``field_names``, by name.
    """
    return {
        name: getattr(action, MODEL_FIELD_TO_LIVE_FIELD.get(name, name))
        for name in field_names
    }


def diagnose_cube_failure(
    pickup_action: PickUpAction, place_action: PlaceAction
) -> tuple[str, RootCauseDiagnosis] | None:
    """
    Diagnose a failed cube attempt against both of its actions' trees, and report
    whichever is more likely to be the actual root cause.

    A cube ending up on the floor could stem from either action -- a bad grasp or a bad
    placement -- and nothing short of a finer-grained mid-plan check says which. Both
    are diagnosed and the one whose primary cause has the *lower* support probability
    under successful attempts is reported: the more anomalous of the two is the more
    likely explanation.

    :param pickup_action: The failed attempt's pickup, as actually performed.
    :param place_action: The failed attempt's place, as actually performed.
    :return: Which action was implicated and its diagnosis, or ``None`` if neither
        action's tree could recommend a correction.
    """
    candidates: list[tuple[str, RootCauseDiagnosis]] = []
    try:
        candidates.append(
            (
                "pickup",
                pickup_diagnoser.diagnose(
                    _observed_parameters(
                        pickup_action, PICKUP_CAUSAL_CONFIG.cause_names
                    )
                ),
            )
        )
    except NoRecommendationAvailable:
        pass
    try:
        candidates.append(
            (
                "place",
                place_diagnoser.diagnose(
                    _observed_parameters(place_action, PLACE_CAUSAL_CONFIG.cause_names)
                ),
            )
        )
    except NoRecommendationAvailable:
        pass

    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: candidate[1].primary.observed_support_probability,
    )


def apply_correction(
    pickup_action: PickUpAction,
    place_action: PlaceAction,
    action_name: str,
    diagnosis: RootCauseDiagnosis,
    cube_to_stack_on: Body,
) -> tuple[PickUpAction, PlaceAction]:
    """
    Build the corrected retry's actions: every one of ``diagnosis.corrections`` applied
    at once to whichever action it was diagnosed against, plus a freshly re-read place
    target, everything else unchanged from the failed attempt.

    Applying every correction together, not just the primary one, is what actually
    fixes a failure with more than one bad parameter: retrying with only the single
    most-anomalous value corrected leaves the other equally-bad ones untouched, so the
    retry keeps failing (and burns through
    :data:`~inference2.MAX_CORRECTION_ATTEMPTS_PER_CUBE` one parameter at a time instead
    of fixing the actual combination) -- see :meth:`ActionCausalDiagnoser.diagnose`'s own
    docstring for how ``corrections`` is decided.

    The target refresh happens unconditionally, not only when ``action_name`` is
    ``"place"``: whichever action gets corrected, the retry still performs the *same*
    place afterward, and ``place_action.target_location`` was baked in as a fixed pose
    back when this cube's step began (see :func:`_current_place_location`) -- if the
    failed attempt this is correcting nudged ``cube_to_stack_on`` on its way down, the
    uncorrected target would carry that staleness into the retry too.

    The grasp description is refreshed the same way and for the same reason:
    ``pickup_action.grasp_description`` -- its top-down orientation override and its
    :func:`_rotate_gripper_away_from` roll choice alike -- was baked in once, back when
    this cube's step began. The failed attempt this is correcting can have knocked the
    cube (or ``cube_to_stack_on``) into a new position, e.g. now lying flush against a
    neighbour in a direction the original roll choice does not clear; re-deriving it
    fresh from both cubes' current positions is what actually keeps every retry
    reaching for the cube's current top and rolling away from whatever is now nearby,
    not just the first attempt.

    :param pickup_action: The failed attempt's pickup.
    :param place_action: The failed attempt's place.
    :param action_name: Which action ``diagnosis`` was diagnosed against, ``"pickup"``
        or ``"place"``.
    :param diagnosis: The diagnosis to apply.
    :param cube_to_stack_on: The cube this step places onto, read fresh for the
        refreshed target and grasp description.
    """
    corrected_values = {
        MODEL_FIELD_TO_LIVE_FIELD.get(
            correction.variable_name, correction.variable_name
        ): correction.corrected_value
        for correction in diagnosis.corrections
    }
    if action_name == "pickup":
        pickup_action = dataclasses.replace(pickup_action, **corrected_values)
    else:
        place_action = dataclasses.replace(place_action, **corrected_values)
    pickup_action = dataclasses.replace(
        pickup_action,
        grasp_description=pickup_grasp_description(
            pickup_action.object_designator, cube_to_stack_on
        ),
    )
    place_action = dataclasses.replace(
        place_action, target_location=_current_place_location(cube_to_stack_on)
    )
    return pickup_action, place_action


# %% support verification via segmind

INFERENCE_REPORT_PATH = Path(__file__).parent / "inference_report_v2.md"
"""
Markdown file the per-iteration diagnosis-and-correction findings are appended to.

Its own file, separate from ``inference.py``'s ``inference_report.md``: the two runs are
diagnosed against different models and are not comparable iteration-for-iteration.
"""

segmind_context = SegmindContext()
"""
Shared context the support detector accumulates its findings in.
"""

support_detector = SupportDetector()
"""
Detects which bodies currently support which other bodies.
"""

motion_statechart_context = MotionStatechartContext(world=world)
"""
Gives the detector access to the world's bodies and their collision geometry.
"""

detection_gate = ParkedArmDetectionGate(
    arm=RobotArmParkDeviations(world=world, robot=context.robot)
)
"""
Holds every support detection back until the arm has parked out of the way.
"""


def expected_supports() -> list[tuple[Body, Body]]:
    """
    The support relations a fully built stack should have, bottom up.

    The bottom cube is left out: what the scene rests it on is not part of what
    the stacking is judged on, and it is reported among the detected supports
    anyway.
    """
    return [
        (box1, box),
        (box2, box1),
        (box3, box2),
    ]


def detected_supports() -> dict[Body, set[Body]]:
    """
    Every support relation segmind currently sees among the cubes, sampled once
    :class:`ParkedArmDetectionGate` reports the arm parked and the scene settled.

    The detector reports only relations it has not seen before, so its context is
    cleared first to make each iteration's result independent of earlier ones.
    """
    if not detection_gate.wait_for_parked_arm():
        print(
            f"[warning] the arm did not reach its park position within "
            f"{detection_gate.arrival_timeout:.1f}s -- detecting supports anyway"
        )
    segmind_context.latest_support.clear()
    support_detector.update_context_and_events(
        motion_statechart_context, segmind_context, [box, box1, box2, box3]
    )
    return segmind_context.latest_support


def segmind_sees_on_floor(body: Body) -> bool:
    """
    Whether segmind currently sees ``body`` resting directly on the floor, see
    ``demo2.py``'s own :func:`segmind_sees_on_floor` for why this, rather than checking
    support from the cube below, is the between-step check.
    """
    return floor in detected_supports().get(body, set())


STACK_XY_TOLERANCE = 0.02
"""
Greatest horizontal distance (in meters) between a cube's center and the center of the
cube it was stacked onto that still counts as stacked.

The cubes are 50 mm, so a cube more than 20 mm off center overhangs its support by most
of its own width and is on its way down rather than resting.
"""

STACK_Z_TOLERANCE = 0.012
"""
Greatest deviation (in meters) from exactly one cube height above the supporting cube's
center that still counts as stacked.

Stacked and not-stacked are 50 mm apart in height, so this separates them with a wide
margin either side.
"""

ATTEMPT_SETTLE_DURATION = 1.5
"""
Real-time seconds a cube is given to fall and come to rest before an attempt is judged.

Judged immediately, a cube still in the air is neither on the floor nor on the stack, and
a check phrased as "not on the floor" calls that a success.
"""


def cube_edge_length(cube: Body) -> float:
    """
    A cube's own height, read from its collision geometry rather than assumed.

    :param cube: The cube to measure.
    """
    bounding_box = cube.collision.as_bounding_box_collection_in_frame(
        cube
    ).bounding_box()
    return float(bounding_box.max_z - bounding_box.min_z)


def cube_is_stacked_on(cube: Body, supporting_cube: Body) -> bool:
    """
    Whether ``cube`` is resting squarely on top of ``supporting_cube``.

    A positive assertion about where the cube actually is: one cube height above its
    support and horizontally centered on it, both within tolerance. Asking instead
    whether segmind fails to see the cube on the floor answers a different question --
    every pose it cannot classify, including mid-air and wedged against the board,
    passes that test.

    :param cube: The cube that was placed.
    :param supporting_cube: The cube it was meant to be stacked onto.
    """
    placed = numpy.array(cube.global_pose.to_position().evaluate()[:3], dtype=float)
    support = numpy.array(
        supporting_cube.global_pose.to_position().evaluate()[:3], dtype=float
    )
    horizontal_distance = float(numpy.linalg.norm(placed[:2] - support[:2]))
    height_above_support = float(placed[2] - support[2])
    return (
        horizontal_distance <= STACK_XY_TOLERANCE
        and abs(height_above_support - cube_edge_length(cube)) <= STACK_Z_TOLERANCE
    )


def verify_stacked(cube: Body, supporting_cube: Body) -> bool:
    """
    Judge one attempt, and report it whenever segmind judged it differently.

    :param cube: The cube that was placed.
    :param supporting_cube: The cube it was meant to be stacked onto.
    """
    stacked = cube_is_stacked_on(cube, supporting_cube)
    if stacked == (not segmind_sees_on_floor(cube)):
        return stacked

    placed = numpy.array(cube.global_pose.to_position().evaluate()[:3], dtype=float)
    support = numpy.array(
        supporting_cube.global_pose.to_position().evaluate()[:3], dtype=float
    )
    print(
        f"[warning] segmind and geometry disagree about {cube.name.name}: segmind says "
        f"{'on the floor' if segmind_sees_on_floor(cube) else 'off the floor'}, geometry "
        f"says {'stacked' if stacked else 'not stacked'} "
        f"(horizontal {numpy.linalg.norm(placed[:2] - support[:2]):.4f} m, "
        f"height above support {placed[2] - support[2]:+.4f} m, "
        f"expected {cube_edge_length(cube):.4f} m). Going with geometry."
    )
    return stacked


def segmind_approved() -> bool:
    """
    Whether segmind sees the whole stack standing, every expected support at once.

    An independent reading from :func:`full_stack_intact`: it trusts segmind's own
    contact-based detection alone, sampled once at the end of the iteration, rather than
    the per-cube geometric check (:func:`cube_is_stacked_on`) each step's own
    SUCCESS/HARD FAILURE line is decided from. Report both together (see
    :func:`append_inference_report`) rather than only this one -- segmind's detector can
    miss a support that geometry still confirms (the same disagreement
    :func:`verify_stacked` already resolves in geometry's favor per step, just not
    surfaced here before), so this alone flipping to ``False`` does not mean any cube
    actually came down.
    """
    supports = detected_supports()
    return all(
        supporter in supports.get(supported, set())
        for supported, supporter in expected_supports()
    )


def full_stack_intact() -> bool:
    """
    Whether every expected support currently holds, judged the same way each cube's own
    step already was: geometrically, via :func:`cube_is_stacked_on`.

    This is the verdict :func:`append_inference_report` and the main loop's success
    count are based on -- not :func:`segmind_approved` alone -- so the reported "full
    stack" outcome is always consistent with the per-cube SUCCESS/HARD FAILURE lines
    directly above it in the same report, instead of occasionally contradicting them.
    """
    return all(
        cube_is_stacked_on(supported, supporter)
        for supported, supporter in expected_supports()
    )


# %% building and performing one attempt


class FrictionNotApplied(RuntimeError):
    """
    Raised when the sampled object friction could not be pushed onto the simulator, see
    ``demo2.py``'s own :class:`FrictionNotApplied`.
    """

    def __init__(self, body_name: str, reason: str) -> None:
        super().__init__(f"Could not set friction of {body_name}: {reason}")


def _apply_object_friction(pickup_action: PickUpAction) -> None:
    """
    Push ``pickup_action``'s sampled friction onto the simulator's copy of its target
    object, since :class:`PickUpAction` records the value but never applies it itself
    (see ``demo2.py``'s own identical comment on ``_build_stack_plan``).
    """
    friction_result = multi_sim.simulator.set_body_friction(
        pickup_action.object_designator.name.name,
        numpy.array([pickup_action.object_friction, 0.05, 0.0005]),
    )
    if (
        friction_result.type
        is not SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA
    ):
        raise FrictionNotApplied(
            pickup_action.object_designator.name.name, friction_result.info
        )


class UprightGraspDescription(GraspDescription):
    """
    A grasp whose approach is oriented in the world rather than in the object's own
    frame, so it stays a top-down grasp however the object is lying.

    :class:`GraspDescription` builds its pose sequence in the object's frame, so an
    object's own rotation carries into the approach: a cube knocked onto its side turns
    a top-down grasp into a sideways one, which the arm reaches for by bending across the
    row and sweeping its neighbours. A cube presents the same square face whichever way
    up it is, so nothing is lost by ignoring its rotation.

    ..note:: Only sound for objects that can be grasped from any orientation. A shape
        that must be taken by a particular feature -- a handle, a spout -- needs the
        object-relative grasp this replaces.
    """

    def grasp_pose_sequence(self, body: Body) -> list[Pose]:
        """
        Overrides :meth:`GraspDescription.grasp_pose_sequence` to route through this
        class's own :meth:`pose_sequence`, which is what actually discards the body's
        rotation.

        Only matters for callers that reach this method directly (``PickUpAction``'s own
        lift, which uses it to compute the lift-off direction): the reach onto the object
        itself does not call it at all, see :meth:`pose_sequence`.

        :param body: The body being grasped.
        """
        return self.pose_sequence(Pose(reference_frame=body), body)

    def pose_sequence(
        self, target_T_grasp_pose: Pose, body: Body = None, reverse: bool = False
    ) -> list[Pose]:
        """
        Overrides :meth:`GraspDescription.pose_sequence` by resolving
        ``target_T_grasp_pose``'s current position into world coordinates and anchoring
        the result to ``world.root``, discarding whatever rotation the target carries,
        before composing the grasp orientation onto it.

        This is the method that actually matters: ``PickUpAction._grasp_attempt_plan``
        drives ``ReachAction`` with a pose anchored to the object itself, and the base
        implementation multiplies whatever rotation that target carries straight into
        the gripper's target orientation (``target_T_grasp_pose.to_rotation_matrix() @
        ...``). Overriding :meth:`grasp_pose_sequence` alone (as this class first did)
        never touches that path, since ``ReachAction`` calls this method directly rather
        than through it -- the arm kept bending to reach a knocked-over cube's original
        top face exactly as before.

        Anchoring the *result* to ``world.root`` (rather than keeping it anchored to the
        object, as an earlier version of this method did) matters just as much as
        discarding the rotation: the base implementation's own math composes whatever
        reference frame is passed in with *that* frame's current world orientation to
        get the actual gripper target, so anchoring to the object again -- even with an
        identity local rotation -- let its current tilt silently leak back in through
        that composition. Confirmed live: for a cube rolled onto its side (own top-face
        normal measured at roughly ``[-0.90, -0.43, 0]``), the resolved gripper target
        came out at roughly ``[0.90, 0.43, 0]`` -- horizontal, not down -- with an object-
        anchored target, and correctly ``[0, 0, -1]`` once anchored to ``world.root``
        instead.

        Safe for every call this class actually sees: the place target this run builds
        (see ``sample_actions``'s ``place_location``) already carries an identity
        rotation, so there is no legitimate desired orientation being discarded here,
        only the object's own incidental one.

        :param target_T_grasp_pose: The pose of the grasp in the target frame.
        :param body: The body of the grasp.
        :param reverse: If the sequence should be reversed.
        """
        target = target_T_grasp_pose.reference_frame
        world_position = target._world.transform(
            target_T_grasp_pose.to_homogeneous_matrix(), target._world.root
        ).to_position()
        upright_target = Pose(world_position, reference_frame=target._world.root)
        return super().pose_sequence(upright_target, body, reverse)


def _as_upright(grasp_description: GraspDescription) -> UprightGraspDescription:
    """
    The same grasp, described so its approach stays oriented in the world.

    Sampling an action rebuilds its grasp description field by field as a plain
    :class:`GraspDescription` (see ``pickup_place_parameterization``), which drops any
    subclass handed to it, so the sampled action has to be given one back.

    :param grasp_description: The grasp the sampled action came back with.
    """
    return UprightGraspDescription(
        grasp_description.approach_direction,
        grasp_description.vertical_alignment,
        grasp_description.end_effector,
        rotate_gripper=grasp_description.rotate_gripper,
        manipulation_offset=grasp_description.manipulation_offset,
    )


def _rotate_gripper_away_from(object_body: Body, target_body: Body) -> bool:
    """
    Whether the pickup grasp should roll 90 degrees so its fingers' opening axis stays
    perpendicular to the horizontal direction from ``object_body`` to ``target_body``,
    rather than swept toward it.

    ``demo2.py``'s/``demo3.py``'s fixed ``rotate_gripper=True`` keeps the opening axis
    off the spawn row, which runs along y (see their own identical comment) -- correct
    only because every pickup there starts from that row. Here a cube can be retried
    after being knocked out of the row by an earlier failed attempt, landing close to
    the stack or the cube it is meant to stack onto (``target_body``, which for the
    first step is the bottom/support cube); the fixed roll then has even odds of
    sweeping the fingers straight into whichever cube is now nearby instead of away from
    it. Reading both cubes' current x/y position and rolling away from whichever
    direction ``target_body`` actually is keeps the same protection in the ordinary
    spawn-row case (the row runs along y, so this resolves to the same ``True`` demo3.py
    hardcodes) while also covering the knocked-out-of-row case that a fixed choice
    cannot.

    :param object_body: The cube about to be picked, at its current position.
    :param target_body: The cube it is being stacked onto -- the nearest known
        obstacle this pickup should keep the fingers' sweep clear of.
    """
    object_position = numpy.array(
        object_body.global_pose.to_position().evaluate()[:2], dtype=float
    )
    target_position = numpy.array(
        target_body.global_pose.to_position().evaluate()[:2], dtype=float
    )
    toward_target = target_position - object_position
    return bool(abs(toward_target[1]) >= abs(toward_target[0]))


def pickup_grasp_description(
    object_body: Body, target_body: Body
) -> UprightGraspDescription:
    """
    The grasp this pickup is described with, its 90-degree roll chosen from
    ``object_body``'s and ``target_body``'s current positions -- see
    :func:`_rotate_gripper_away_from`.
    """
    return UprightGraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        context.robot.get_arms()[0].end_effector,
        rotate_gripper=_rotate_gripper_away_from(object_body, target_body),
    )


def _current_place_location(target_body: Body) -> Pose:
    """
    :param target_body: The cube being stacked onto.
    :return: A place target directly above ``target_body``'s *current* position, one
        cube height higher.

    Reads ``target_body.global_pose`` fresh on every call rather than once per cube
    step: ``target_body`` can be nudged by a collision during this same step's own
    earlier attempts (or, for the bottom cube, by an entirely different step's attempt
    -- nothing re-verifies it the way :func:`restack_disturbed_cubes` does for cubes in
    ``confirmed_stacks``), and a stale target then aims every later retry at where the
    cube *used to be* rather than where it now is.
    """
    target_pose = target_body.global_pose
    return Pose.from_xyz_rpy(
        x=target_pose.x,
        y=target_pose.y,
        z=target_pose.z + STACK_HEIGHT_OFFSET,
        reference_frame=world.root,
    )


def sample_actions(
    object_body: Body, target_body: Body, picking_arm: Arms
) -> tuple[PickUpAction, PlaceAction]:
    """
    Sample a fresh pickup/place pair from the wide priors, for stacking ``object_body``
    centered above ``target_body``, one cube height higher.
    """
    place_location = _current_place_location(target_body)
    grasp_description = pickup_grasp_description(object_body, target_body)
    pickup_action = sample_pickup_instance(
        object_body, picking_arm, grasp_description, priors=WIDE_PICKUP_PARAMETER_PRIORS
    )
    pickup_action.grasp_description = _as_upright(pickup_action.grasp_description)
    place_action = sample_place_instance(
        object_body, place_location, picking_arm, priors=WIDE_PLACE_PARAMETER_PRIORS
    )
    return pickup_action, place_action


def _pickup_hover_motion(pickup_action: PickUpAction) -> MoveToolCenterPointMotion:
    """
    :return: A position-only move of the tool center point to directly above
        ``pickup_action``'s target cube's *current* x/y, at :data:`PICKUP_HOVER_HEIGHT`.
    """
    cube_pose = pickup_action.object_designator.global_pose
    hover_pose = Pose.from_xyz_rpy(
        x=cube_pose.x, y=cube_pose.y, z=PICKUP_HOVER_HEIGHT, reference_frame=world.root
    )
    return MoveToolCenterPointMotion(
        hover_pose, pickup_action.arm, movement_type=MovementType.TRANSLATION
    )


def build_stack_plan(
    pickup_action: PickUpAction, place_action: PlaceAction
) -> PlanNode:
    """
    Build (without performing) a park/pick/park/place/park plan from already-sampled
    actions, applying the pickup's sampled friction first.
    """
    _apply_object_friction(pickup_action)
    print(
        f"[params] pickup {pickup_action.object_designator.name.name}: "
        f"pre_approach={pickup_action.pre_approach_linear_velocity:.4f}, "
        f"grasp={pickup_action.final_approach_linear_velocity:.4f}, "
        f"closing={pickup_action.grasp_closing_velocity:.4f}, "
        f"lift={pickup_action.lift_linear_velocity:.4f}, "
        f"stall_min_time={pickup_action.grasp_stall_minimum_time:.4f}, "
        f"object_friction={pickup_action.object_friction:.4f}"
    )
    print(
        f"[params] place {pickup_action.object_designator.name.name}: "
        f"transport={place_action.transport_linear_velocity:.4f}, "
        f"placing={place_action.placing_linear_velocity:.4f}, "
        f"release={place_action.release_opening_velocity:.4f}, "
        f"retract={place_action.retract_linear_velocity:.4f}"
    )
    return sequential(
        [
            ParkArmsAction(Arms.BOTH),
            # Moves straight above the target cube's current x/y before the pickup's
            # own reach descends onto it, for the same reason the carrying park below
            # exists on the place side: without this, the reach crosses directly from
            # wherever ParkArmsAction left the arm to the pickup's pre-pose, at
            # whatever height that direct path happens to pass through. Once a cube
            # ends up resting against a neighbour, that direct path clips the
            # neighbour and the arm bends and sweeps through the cluster instead of
            # picking cleanly -- see PICKUP_HOVER_HEIGHT.
            _pickup_hover_motion(pickup_action),
            pickup_action,
            # Parks the held cube clear before transporting it to the place target,
            # rather than crossing directly from the pickup pose to the place pose at
            # whatever height that direct path happens to pass through -- which, once
            # the stack has some height to it, can pass right through where the already
            # stacked cubes are. Kept enabled here (see this module's own docstring),
            # matching how demo3.py collects the training data this run diagnoses
            # against. Slowed to CARRYING_PARK_JOINT_VELOCITY for the same reason
            # demo3.py's does: at full speed this move jerks the cube -- held by
            # nothing but friction -- right out of the fingers.
            ParkArmsAction(Arms.BOTH, max_joint_velocity=CARRYING_PARK_JOINT_VELOCITY),
            place_action,
            ParkArmsAction(Arms.BOTH),
        ],
        context=context,
    )


def perform_attempt(
    pickup_action: PickUpAction,
    place_action: PlaceAction,
    step_name: str,
    supporting_cube: Body,
) -> bool:
    """
    Build and perform one park/pick/place/park attempt, logging and swallowing any
    failure instead of letting it propagate, see ``demo2.py``'s own
    :func:`attempt_stack` for why.

    :param supporting_cube: The cube this attempt stacks onto, which the result is
        judged against.
    :return: Whether :attr:`pickup_action.object_designator` ended up stacked on
        ``supporting_cube``.
    """
    plan = build_stack_plan(pickup_action, place_action)
    try:
        plan.perform()
    except Exception as exc:
        print(f"[warning] {step_name} failed ({type(exc).__name__}: {exc}), moving on")
        try:
            sequential([ParkArmsAction(Arms.BOTH)], context=context).perform()
        except Exception as park_exc:
            print(f"[warning] re-park after {step_name} also failed: {park_exc}")

    time.sleep(ATTEMPT_SETTLE_DURATION)
    return verify_stacked(pickup_action.object_designator, supporting_cube)


def restack_disturbed_cubes(
    confirmed_stacks: list[tuple[Body, Body]],
    picking_arm: Arms,
    disturbed_by: str,
    deadline: float,
) -> bool:
    """
    Re-stack any cube in ``confirmed_stacks`` the gripper has since knocked out of place.

    Called after every attempt made for a *different* cube (the initial one and every
    correction retry alike), since any of those -- not just a failed one -- can nudge a
    cube that was already resting correctly: the approach, the lift, and especially the
    retreat all pass close by the rest of the stack, and a correction retry repeats that
    close pass several times in a row. Checked geometrically (:func:`cube_is_stacked_on`)
    rather than through segmind, for the same reason :func:`verify_stacked` is: a pose
    segmind cannot classify is not evidence the cube is fine.

    Re-stacking one cube can itself disturb whichever cube it rests on, so each is
    re-stacked with only the pairs that precede it in ``confirmed_stacks`` (its own
    support chain) checked afterwards, cascading downward through at most as many levels
    as the stack has cubes.

    :param confirmed_stacks: ``(supported, support)`` pairs already confirmed stacked
        earlier in this iteration, oldest first.
    :param picking_arm: The arm used to re-stack a disturbed cube.
    :param disturbed_by: Name of the attempt that just ran, for the warning if a cube
        turns out to have moved.
    :param deadline: ``time.time()`` value beyond which a re-stack is not attempted --
        whichever of :data:`CUBE_PICKUP_TIME_LIMIT` (for the cube whose attempt this
        disturbance was found during) and :data:`ITERATION_TIME_LIMIT` is tighter.
    :return: Whether every cube in ``confirmed_stacks`` is, still or again, correctly
        stacked.
    """
    all_fine = True
    for index, (supported, support) in enumerate(confirmed_stacks):
        if cube_is_stacked_on(supported, support):
            continue

        if time.time() >= deadline:
            print(
                f"[warning] {supported.name.name} was knocked off {support.name.name} "
                f"during {disturbed_by}, but the time limit is reached -- not "
                "attempting to re-stack it"
            )
            all_fine = False
            continue

        print(
            f"[warning] {supported.name.name} was knocked off {support.name.name} "
            f"during {disturbed_by}; re-stacking it before continuing"
        )
        restack_outcome = attempt_cube_with_correction(
            supported,
            support,
            picking_arm,
            f"re-stack {supported.name.name} onto {support.name.name}",
            confirmed_stacks=confirmed_stacks[:index],
            deadline=deadline,
        )
        if not restack_outcome.final_succeeded:
            print(
                f"[warning] could not re-stack {supported.name.name} onto "
                f"{support.name.name}"
            )
            all_fine = False

    return all_fine


# %% the diagnose-and-correct loop


@dataclasses.dataclass
class CubeAttemptOutcome:
    """
    How one cube's stacking step went, including any diagnosis-and-correction cycle.
    """

    step_label: str
    """
    Which stacking step this outcome is for, e.g. ``"cube1 onto cube0"``.
    """

    initial_succeeded: bool
    """
    Whether the first, uncorrected attempt stayed off the floor.
    """

    correction_attempts: int
    """
    How many corrected retries were performed, 0 if the first attempt already succeeded
    or no diagnosis was available.
    """

    diagnoses: list[tuple[str, RootCauseDiagnosis]]
    """
    Every diagnosis made for this step, one per correction attempt, in order.
    """

    final_succeeded: bool
    """
    Whether the cube ended up off the floor after every attempt made for this step.
    """


@dataclasses.dataclass
class DiagnosisOutcome:
    """
    One causal diagnosis performed during the run, and what happened after applying its
    correction.
    """

    step_label: str
    """
    Which stacking step this diagnosis was for.
    """

    action_name: str
    """
    Which action the diagnosis was made against, ``"pickup"`` or ``"place"``.
    """

    diagnosis: RootCauseDiagnosis
    """
    The diagnosis itself.
    """

    diagnosis_duration_seconds: float
    """
    How long computing this diagnosis took.
    """

    correction_succeeded: bool
    """
    Whether the retry performed with this diagnosis's correction applied stayed off the
    floor.
    """


diagnosis_outcomes: list[DiagnosisOutcome] = []
"""
Every causal diagnosis performed across the whole run, in the order it happened -- see
:func:`diagnosis_summary_lines`.
"""


def attempt_cube_with_correction(
    cube_to_pick: Body,
    cube_to_stack_on: Body,
    picking_arm: Arms,
    step_label: str,
    confirmed_stacks: list[tuple[Body, Body]] | None = None,
    deadline: float = float("inf"),
) -> CubeAttemptOutcome:
    """
    Attempt one cube's stacking step, diagnosing and correcting up to
    :data:`MAX_CORRECTION_ATTEMPTS_PER_CUBE` times if it fails.

    Every attempt made here -- the initial one and every correction retry -- is followed
    by :func:`restack_disturbed_cubes` checking (and, if needed, repairing) every cube in
    ``confirmed_stacks``, since this cube's own approach, grasp or retreat can knock one
    of them aside without this cube's own attempt failing at all. If a disturbed cube
    cannot be repaired, retrying this one further is abandoned too: placing onto (or
    above) a stack that is not actually intact is not worth the remaining attempts.

    Re-stacking a disturbed cube is itself a full pick and place, which can just as
    easily disturb *this* cube if it happens to rest on the one being repaired, so a
    "succeeded" verdict from before the repair is re-checked rather than trusted.

    :param cube_to_pick: The cube this step stacks.
    :param cube_to_stack_on: The cube it is stacked onto.
    :param picking_arm: The arm used for this step and for re-stacking any disturbed
        cube.
    :param step_label: Names this step in log output.
    :param confirmed_stacks: ``(supported, support)`` pairs already confirmed stacked
        earlier in this iteration, oldest first; none by default.
    :param deadline: ``time.time()`` value beyond which neither a correction retry nor a
        re-stack of a disturbed cube is attempted; unbounded by default. The caller
        passes the tighter of :data:`CUBE_PICKUP_TIME_LIMIT` and
        :data:`ITERATION_TIME_LIMIT` for a top-level cube, and this same value straight
        through for any cube re-stacked as a side effect of that one.
    """
    if confirmed_stacks is None:
        confirmed_stacks = []

    def restack_and_reverify(after: str) -> bool:
        """
        Repair any disturbed cube in ``confirmed_stacks``, then re-check whether this
        step's own cube is still standing -- the repair itself may have moved it.

        :return: Whether every cube in ``confirmed_stacks`` and this step's own cube are
            now correctly stacked.
        """
        nonlocal succeeded
        intact = restack_disturbed_cubes(confirmed_stacks, picking_arm, after, deadline)
        if succeeded and intact:
            succeeded = verify_stacked(cube_to_pick, cube_to_stack_on)
            if not succeeded:
                print(
                    f"[warning] {step_label}: {cube_to_pick.name.name} came down while "
                    "a disturbed neighbour was being re-stacked"
                )
        return intact

    pickup_action, place_action = sample_actions(
        cube_to_pick, cube_to_stack_on, picking_arm
    )
    succeeded = perform_attempt(
        pickup_action, place_action, step_label, cube_to_stack_on
    )
    initial_succeeded = succeeded
    print(
        f"[info] {step_label} attempt 1 "
        f"{'is stacked' if succeeded else 'is NOT stacked'}"
    )
    stack_intact = restack_and_reverify(f"{step_label} attempt 1")

    diagnoses: list[tuple[str, RootCauseDiagnosis]] = []
    correction_attempts = 0

    while (
        not succeeded
        and stack_intact
        and correction_attempts < MAX_CORRECTION_ATTEMPTS_PER_CUBE
        and time.time() < deadline
    ):
        diagnosis_start_time = time.time()
        diagnosis_entry = diagnose_cube_failure(pickup_action, place_action)
        diagnosis_duration = time.time() - diagnosis_start_time
        if diagnosis_entry is None:
            print(f"[causal] {step_label}: no correction available, abandoning retries")
            break

        action_name, diagnosis = diagnosis_entry
        diagnoses.append(diagnosis_entry)
        print(
            f"[causal] {step_label} ({action_name}, {diagnosis_duration:.3f}s): "
            f"{diagnosis.explanation()}"
        )

        pickup_action, place_action = apply_correction(
            pickup_action, place_action, action_name, diagnosis, cube_to_stack_on
        )
        correction_attempts += 1
        correction_label = f"{step_label} (correction {correction_attempts})"
        succeeded = perform_attempt(
            pickup_action, place_action, correction_label, cube_to_stack_on
        )
        diagnosis_outcomes.append(
            DiagnosisOutcome(
                step_label=step_label,
                action_name=action_name,
                diagnosis=diagnosis,
                diagnosis_duration_seconds=diagnosis_duration,
                correction_succeeded=succeeded,
            )
        )
        print(
            f"[info] {step_label} correction {correction_attempts} "
            f"{'stayed off the floor' if succeeded else 'ended up on the FLOOR'}"
        )
        stack_intact = restack_and_reverify(correction_label)

    if not stack_intact:
        print(
            f"[warning] {step_label}: abandoning further retries, the stack it builds "
            "on is no longer intact"
        )

    return CubeAttemptOutcome(
        step_label=step_label,
        initial_succeeded=initial_succeeded,
        correction_attempts=correction_attempts,
        diagnoses=diagnoses,
        final_succeeded=succeeded and stack_intact,
    )


def diverged_cubes() -> list[str]:
    """
    The cubes that have left the region the scene could plausibly place them in, see
    ``demo2.py``'s own :func:`diverged_cubes`.
    """
    escaped = []
    for name in CUBE_SPAWN_POSITIONS:
        position = multi_sim.simulator.get_body_position(name).result[:3]
        outside_horizontally = (
            abs(position[0]) > WORKSPACE_BOUND or abs(position[1]) > WORKSPACE_BOUND
        )
        if outside_horizontally or not -WORKSPACE_BOUND < position[2] < WORKSPACE_BOUND:
            escaped.append(name)
    return escaped


class SimulationDidNotRecover(RuntimeError):
    """
    Raised when cubes are still outside the workspace after being reset, see
    ``demo2.py``'s own :class:`SimulationDidNotRecover`.
    """

    def __init__(self, cube_names: list[str], iteration_index: int) -> None:
        super().__init__(
            f"{', '.join(cube_names)} still outside the workspace after the reset at "
            f"the start of iteration {iteration_index}; stopping rather than collecting "
            "attempts against nonsense object poses"
        )


RESET_POSITION_TOLERANCE = 0.01
"""
Greatest distance (in meters) a cube may end up from its own spawn position and still
count as having actually respawned.
"""

MAX_RESET_ATTEMPTS = 3
"""
How many times :func:`reset_cubes` retries teleporting a cube that did not actually end
up at its spawn position, before giving up on it and logging a warning.
"""

RESET_SETTLE_DURATION = 0.5
"""
Real-time seconds a just-teleported cube is given to settle before its position is
checked against its spawn point.
"""


def cubes_not_at_spawn() -> list[str]:
    """
    Names of cubes whose current *horizontal* position is not within
    :data:`RESET_POSITION_TOLERANCE` of their own spawn point.

    Only x/y is checked: even though :data:`CUBE_SPAWN_POSITIONS`' z is now the settled
    resting height rather than a drop height, small numerical settle noise or a cube
    resting fractionally differently than the reference measurement could still make an
    exact z comparison brittle. x/y is what actually says which spawn slot a cube is in,
    and is unaffected by any of that.

    A teleport (``MujocoSimulator.set_body_position``) reports success purely from having
    written the position, not from the cube actually staying there once physics resumes:
    something already occupying that space -- most plausibly the arm itself, left
    wherever the previous iteration's last attempt happened to end, not necessarily clear
    of the spawn row -- can shove a freshly teleported cube straight back out on the very
    next physics step. Checked geometrically after the fact rather than trusted, for the
    same reason :func:`cube_is_stacked_on` is.
    """
    out_of_place = []
    for name, spawn_position in CUBE_SPAWN_POSITIONS.items():
        position = numpy.array(
            multi_sim.simulator.get_body_position(name).result[:3], dtype=float
        )
        horizontal_distance = numpy.linalg.norm(position[:2] - spawn_position[:2])
        if horizontal_distance > RESET_POSITION_TOLERANCE:
            out_of_place.append(name)
    return out_of_place


def reset_robot() -> None:
    """
    Bring the arm and gripper to a standstill and teleport them straight to a
    known-good park/open configuration, independent of whatever state the previous
    iteration left them in.

    :func:`reset_cubes` only ever teleports the cubes. The arm's and gripper's own
    joints are physically simulated too (see ``physically_simulated_dofs`` above), and
    nothing resets *their* velocity, solver warm-start acceleration, or position
    between iterations. A violent collision (a knocked cube, a wedged gripper) leaves
    that residual state behind, and every later iteration then starts from it -- this is
    why, once one collision happens, the robot keeps failing to reach, "dancing", or
    leaving the gripper stuck shut for every iteration after, not just the one the
    collision happened in, even though the cubes themselves come back to their spawn
    position correctly every time.

    A Giskard-planned park (:func:`reset_cubes`'s own best-effort ``ParkArmsAction``) is
    not a substitute: Giskard's own controller can be exactly what a collision left
    wedged, so a controller-driven park can itself fail silently, which is why that call
    is wrapped in a warning rather than trusted. Teleporting the joints directly here
    does not depend on the controller at all, so it recovers even when that park can't.
    """
    for connection in list(arm.active_connections) + list(gripper.active_connections):
        multi_sim.simulator.reset_body_velocity(body_name=connection.child.name.name)

    park_state = arm.get_joint_state_by_type(StaticJointState.PARK)
    open_state = gripper.get_joint_state_by_type(GripperState.OPEN)
    joint_values = {
        connection.name.name: value
        for connection, value in (
            list(zip(park_state.connections, park_state.target_values))
            + list(zip(open_state.connections, open_state.target_values))
        )
    }
    result = multi_sim.simulator.set_joints_values(joint_values)
    if result.type not in (
        SimulatorCallbackResult.ResultType.SUCCESS_WITHOUT_EXECUTION,
        SimulatorCallbackResult.ResultType.SUCCESS_AFTER_EXECUTION_ON_DATA,
    ):
        print(
            "[warning] could not teleport the robot to its park/open configuration: "
            f"{result.info}"
        )


def reset_cubes() -> None:
    """
    Returns every cube to its spawn pose and brings it to a standstill, verifying (and,
    if needed, retrying) that it actually took effect rather than trusting it.

    The arm is parked first, best-effort: a hard iteration abandonment (a diverged
    simulation, a cube that never got stacked, an uncaught exception) can leave it
    anywhere, including hovering right over a cube's spawn point, which would defeat the
    teleport below the moment physics resumes. Call :func:`reset_robot` before this, not
    after: it clears whatever wedged/moving state the arm was actually in, which is what
    lets this best-effort park succeed reliably instead of occasionally failing silently.
    """
    try:
        sequential([ParkArmsAction(Arms.BOTH)], context=context).perform()
    except Exception as park_exc:
        print(f"[warning] could not park the arm before resetting cubes: {park_exc}")

    out_of_place: list[str] = []
    for attempt in range(1, MAX_RESET_ATTEMPTS + 1):
        for name, position in CUBE_SPAWN_POSITIONS.items():
            multi_sim.simulator.reset_body_velocity(body_name=name)
            multi_sim.simulator.set_body_position(body_name=name, position=position)
            multi_sim.simulator.set_body_quaternion(
                body_name=name, quaternion=CUBE_SPAWN_ORIENTATION
            )
        time.sleep(RESET_SETTLE_DURATION)

        out_of_place = cubes_not_at_spawn()
        if not out_of_place:
            return
        print(
            f"[warning] reset attempt {attempt}/{MAX_RESET_ATTEMPTS}: "
            f"{', '.join(out_of_place)} did not settle at spawn -- retrying"
        )

    print(
        f"[warning] {', '.join(out_of_place)} still not at spawn after "
        f"{MAX_RESET_ATTEMPTS} reset attempts; continuing anyway"
    )


def append_inference_report(
    iteration_index: int,
    cube_outcomes: list[CubeAttemptOutcome],
    simulation_diverged: bool,
) -> None:
    """
    Append this iteration's diagnosis-and-correction findings to
    :data:`INFERENCE_REPORT_PATH`.

    Reports both full-stack verdicts, not just segmind's: :func:`full_stack_intact` (the
    same geometric check each cube's own SUCCESS/HARD FAILURE line below is decided
    from) and :func:`segmind_approved` (segmind's independent, contact-based reading).
    They usually agree; printing both, flagged whenever they don't, is what makes the
    disagreement visible in the report itself instead of a console warning easy to miss
    -- see :func:`full_stack_intact`'s own docstring for why the geometric one, not
    segmind's, is what the rest of this run treats as ground truth.
    """
    geometry_approved = full_stack_intact()
    segmind_verdict = segmind_approved()

    lines = [f"\n## Iteration {iteration_index}\n"]
    lines.append(
        f"`full_stack_intact()` (geometry, ground truth): **{geometry_approved}**\n"
    )
    lines.append(
        f"`segmind_approved()` (segmind's own reading): **{segmind_verdict}**\n"
    )
    if geometry_approved != segmind_verdict:
        lines.append(
            "**disagreement**: segmind's contact-based detection missed (or wrongly "
            "saw) a support geometry confirms is otherwise fine -- see "
            "`full_stack_intact()`'s docstring; going with geometry.\n"
        )
    if simulation_diverged:
        lines.append("**simulation diverged -- excluded from results**\n")

    for outcome in cube_outcomes:
        lines.append(f"\n### {outcome.step_label}\n")
        lines.append(
            f"- first attempt: {'success' if outcome.initial_succeeded else 'failure'}"
        )
        for attempt_number, (action_name, diagnosis) in enumerate(
            outcome.diagnoses, start=1
        ):
            lines.append(
                f"- correction {attempt_number} ({action_name}): {diagnosis.explanation()}"
            )
        lines.append(
            f"- final result: {'SUCCESS' if outcome.final_succeeded else 'HARD FAILURE'} "
            f"after {outcome.correction_attempts} correction attempt(s)"
        )

    with INFERENCE_REPORT_PATH.open("a") as report:
        report.write("\n".join(lines) + "\n")
    print(f"[report] iteration {iteration_index} written to {INFERENCE_REPORT_PATH}")


def _correction_payload(correction: ParameterCorrection) -> dict:
    """
    :return: ``correction`` as the JSON-serializable shape the viewer's activity panel
        reads (see ``cramera/web/panels/activity/panel.js``).
    """
    return {
        "variableName": correction.variable_name,
        "observedValue": correction.observed_value,
        "observedSupportProbability": correction.observed_support_probability,
        "correctedValue": correction.corrected_value,
        "correctedSupportProbability": correction.corrected_support_probability,
    }


def _diagnosis_payload(action_name: str, diagnosis: RootCauseDiagnosis) -> dict:
    """
    :param action_name: Which action ``diagnosis`` was diagnosed against, ``"pickup"``
        or ``"place"``.
    :return: ``diagnosis`` as the JSON-serializable shape the viewer's activity panel
        reads.
    """
    return {
        "actionName": action_name,
        "effectVariable": diagnosis.effect_variable_name,
        "primary": _correction_payload(diagnosis.primary),
        "alsoCorrected": [
            _correction_payload(correction) for correction in diagnosis.corrections[1:]
        ],
    }


def _cube_outcome_payload(outcome: CubeAttemptOutcome) -> dict:
    """
    :return: ``outcome`` as the JSON-serializable shape the viewer's activity panel
        reads.
    """
    return {
        "stepLabel": outcome.step_label,
        "finalSucceeded": outcome.final_succeeded,
        "correctionAttempts": outcome.correction_attempts,
        "diagnoses": [
            _diagnosis_payload(action_name, diagnosis)
            for action_name, diagnosis in outcome.diagnoses
        ],
    }


def publish_iteration_activity(
    iteration_index: int,
    cube_outcomes: list[CubeAttemptOutcome],
    simulation_diverged: bool,
    duration_seconds: float,
) -> None:
    """
    Push this iteration's outcome to the live bridge, for the viewer's activity panel.

    Always called, live viewer attached or not: :meth:`Bridge.log_activity` only
    appends to an in-memory list, so this costs nothing the demo itself relies on, and
    a viewer that attaches partway through a run still sees every earlier iteration.

    :param iteration_index: The iteration this outcome is for.
    :param cube_outcomes: Every cube's outcome from this iteration, see
        :func:`append_inference_report`.
    :param simulation_diverged: Whether a cube left the workspace during this
        iteration.
    :param duration_seconds: How long this iteration took, wall-clock.
    """
    BRIDGE.log_activity(
        {
            "iteration": iteration_index,
            "totalIterations": NUMBER_OF_ITERATIONS,
            "durationSeconds": duration_seconds,
            "simulationDiverged": simulation_diverged,
            "fullStackIntact": full_stack_intact(),
            "segmindApproved": segmind_approved(),
            "cubes": [_cube_outcome_payload(outcome) for outcome in cube_outcomes],
        }
    )


def diagnosis_summary_lines(outcomes: list[DiagnosisOutcome]) -> list[str]:
    """
    :param outcomes: Every diagnosis performed during the run, see
        :data:`diagnosis_outcomes`.
    :return: Markdown lines summarizing how often causal diagnosis triggered, how often
        the resulting correction succeeded, and the average time one diagnosis took.
    """
    lines = ["\n## Causal diagnosis summary\n"]
    if not outcomes:
        lines.append("Causal diagnosis was never triggered.\n")
        return lines

    total = len(outcomes)
    succeeded = sum(1 for outcome in outcomes if outcome.correction_succeeded)
    failed = total - succeeded
    average_duration = (
        sum(outcome.diagnosis_duration_seconds for outcome in outcomes) / total
    )

    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| total diagnoses | {total} |")
    lines.append(f"| succeeded after correction | {succeeded} |")
    lines.append(f"| still failed after correction | {failed} |")
    lines.append(f"| average diagnosis time | {average_duration:.3f}s |")
    return lines


def print_and_append_diagnosis_summary(outcomes: list[DiagnosisOutcome]) -> None:
    """
    Print the causal diagnosis summary and append it to :data:`INFERENCE_REPORT_PATH`.

    :param outcomes: Every diagnosis performed during the run, see
        :data:`diagnosis_outcomes`.
    """
    lines = diagnosis_summary_lines(outcomes)
    print("\n".join(lines))
    with INFERENCE_REPORT_PATH.open("a") as report:
        report.write("\n".join(lines) + "\n")


def print_iteration_summary(iteration_index: int) -> None:
    """
    Prints the final z-height of every cube, see ``demo2.py``'s own
    :func:`print_iteration_summary`.
    """
    heights = {
        name: multi_sim.simulator.get_body_position(name).result[2]
        for name in CUBE_SPAWN_POSITIONS
    }
    print(f"--- iteration {iteration_index} final heights: {heights} ---")


multi_sim.start_simulation()

# See demo2.py's own identical comment: without this, the viewer falls back to
# MuJoCo's default camera instead of the scene's intended viewing angle.
viewer = multi_sim.simulator.renderer
if hasattr(viewer, "cam"):
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -20
    viewer.cam.distance = 1.2
    viewer.cam.lookat[:] = [0.3, 0.0, 0.35]

iteration_durations = []
successful_iterations = 0
corrected_success_count = 0
hard_failure_count = 0

with ExecutionEnvironment(
    execution_type=execition_mode,
    collision_avoidance=False,
    real_time_pacing=True,
    # See demo2.py's own identical comment on this budget.
    max_ticks_per_motion_mapping=250,
):
    for iteration in range(1, NUMBER_OF_ITERATIONS + 1):
        iteration_start = time.time()
        print(f"=== starting iteration {iteration}/{NUMBER_OF_ITERATIONS} ===")
        reset_robot()
        reset_cubes()
        time.sleep(1.5)
        still_escaped = diverged_cubes()
        if still_escaped:
            raise SimulationDidNotRecover(still_escaped, iteration)

        cube_outcomes: list[CubeAttemptOutcome] = []
        simulation_diverged = False
        confirmed_stacks: list[tuple[Body, Body]] = []
        iteration_deadline = iteration_start + ITERATION_TIME_LIMIT

        for cube_to_pick, cube_to_stack_on, step_label in [
            (box1, box, "cube1 onto cube0"),
            (box2, box1, "cube2 onto cube1"),
            (box3, box2, "cube3 onto cube2"),
        ]:
            elapsed = time.time() - iteration_start
            if elapsed > ITERATION_TIME_LIMIT:
                print(
                    f"[warning] iteration {iteration} already took {elapsed:.1f}s "
                    f"(limit {ITERATION_TIME_LIMIT:.0f}s), skipping remaining "
                    "attempts and moving to the next iteration"
                )
                break

            # A cube that has rolled out of reach fails the same way on every retry, so
            # its own step gets a tighter budget than the rest of the iteration -- never
            # more than CUBE_PICKUP_TIME_LIMIT, and never more of the iteration's own
            # budget than is actually left.
            cube_deadline = min(
                iteration_deadline, time.time() + CUBE_PICKUP_TIME_LIMIT
            )
            outcome = attempt_cube_with_correction(
                cube_to_pick,
                cube_to_stack_on,
                Arms.LEFT,
                step_label,
                confirmed_stacks=confirmed_stacks,
                deadline=cube_deadline,
            )
            cube_outcomes.append(outcome)
            if outcome.final_succeeded:
                confirmed_stacks.append((cube_to_pick, cube_to_stack_on))
                if outcome.correction_attempts:
                    corrected_success_count += 1
            else:
                hard_failure_count += 1

            # Published as each cube's outcome lands, not just once the whole
            # iteration finishes -- a single cube's pickup+place(+corrections) can
            # itself take a minute or more, so waiting for the iteration to end would
            # leave the activity panel looking stuck on "not live" for that long.
            publish_iteration_activity(
                iteration,
                cube_outcomes,
                simulation_diverged,
                time.time() - iteration_start,
            )

            escaped = diverged_cubes()
            if escaped:
                # See demo2.py's own identical comment: every later step would be
                # planned against a nonsense target pose.
                simulation_diverged = True
                print(
                    f"[warning] simulation DIVERGED: {', '.join(escaped)} left the "
                    f"workspace -- abandoning the rest of iteration {iteration}"
                )
                break
            if not outcome.final_succeeded:
                # Every later step stacks onto this cube, so continuing would only
                # pile onto a cube that is lying on the floor.
                print(
                    f"[warning] {cube_to_pick.name.name} ended up on the floor after "
                    f"correction attempts were exhausted -- abandoning the rest of "
                    f"iteration {iteration}"
                )
                break
            time.sleep(1)

        print_iteration_summary(iteration)
        append_inference_report(iteration, cube_outcomes, simulation_diverged)
        publish_iteration_activity(
            iteration, cube_outcomes, simulation_diverged, time.time() - iteration_start
        )
        if not simulation_diverged and full_stack_intact():
            successful_iterations += 1

        iteration_durations.append(time.time() - iteration_start)
        average_duration = sum(iteration_durations) / len(iteration_durations)
        print(
            f"=== iteration {iteration}/{NUMBER_OF_ITERATIONS} took "
            f"{iteration_durations[-1]:.1f}s (average so far: {average_duration:.1f}s) ==="
        )

print("--- final positions ---")
print_positions()
print(
    f"[summary] {successful_iterations}/{NUMBER_OF_ITERATIONS} iterations fully stacked, "
    f"{corrected_success_count} cube(s) recovered via causal correction, "
    f"{hard_failure_count} cube(s) hard-failed after exhausting corrections"
)
print_and_append_diagnosis_summary(diagnosis_outcomes)

print("Plan finished, keeping the viewer open until it is closed")
while multi_sim.simulator.renderer.is_running():
    time.sleep(0.1)
