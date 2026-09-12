# Tracy experiments

Tracy demos that drive the robot by commanding MuJoCo actuators directly rather than
through Giskard's live closed loop, because Giskard's own QP control loop reads
`world.state` as its belief of the robot's current position, and for a physically
simulated degree of freedom that same state is also written by Giskard's own prior
command -- so Giskard can be satisfied by its own prior write, not by the robot actually
having moved (see `equipment.py`'s own module docstring). Instead, every reach is planned
by Giskard against an isolated scratch copy of the world (`trajectory_planning.py`), and
the resulting trajectory is played back by commanding real MuJoCo actuators
(`real_time_simulation.py`).

## Layout

- `equipment.py` -- parse and mount Tracy (`parse_tracy`, `mount_stationary_robot`,
  `tracy_table_mount_position`), equip it with position servos (`TracyServoTuning`,
  `ServoGains`, `equip_arms_with_servos`, `equip_grippers_with_servos`), gravity
  compensation and self-collision exclusion (`CollisionGroup`), and loose boxes
  (`add_box`, `add_cube`).
- `grasp_contact.py` -- `ContactParameters`: MuJoCo friction and solver settings for a
  grasped object, a resting surface, or a plain cube, applied to bodies.
- `real_time_simulation.py` -- `RealTimeSimulation`, a MuJoCo mirror of a world stepped
  from the calling thread (not MuJoCo's own background thread, whose reads would
  otherwise race a caller's own), paced to the wall clock or, with
  `real_time_factor=None`, as fast as the machine allows.
- `trajectory_planning.py` -- `TrajectoryPlanner`: plan a Cartesian or joint goal
  against a scratch copy of the world and play it back on the real, physically
  simulated one; park arms, open or close a gripper, close it around an object sized to
  the object's width. `RobotiqGripper` names the gripper's own bodies and joints and
  finds the knuckle angle for a given opening.
- `pick_and_place_action.py` -- `PickUpActionMujoco`/`PlaceActionMujoco`, matching the
  real `PickUpAction`/`PlaceAction`'s own field interface but driven by the planner
  above instead of a Giskard motion mapping; `TopDownGraspGeometry` places the tool
  frame so the fingers, not the tool frame, meet the object.

A grasped object is held by real contact friction between the fingers throughout, never
kinematically attached: a poor grasp visibly fails instead of being rescued by a weld.

The ten-milk clutter demo built on this lives in
`experiments/causal_reasoning/tracy_rspn`.
