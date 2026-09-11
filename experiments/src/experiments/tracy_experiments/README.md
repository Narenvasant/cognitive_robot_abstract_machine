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

- `equipment.py` -- parse/mount Tracy, equip it with position-servo actuators, gravity
  compensation, self-collision exclusion; `add_box`/`add_cube`; `table_top_z`.
- `real_time_simulation.py` -- `RealTimeSimulation`, a MuJoCo mirror of a world stepped
  from the calling thread (not MuJoCo's own background thread, whose reads would
  otherwise race a caller's own), paced to the wall clock or, with
  `real_time_factor=None`, as fast as the machine allows.
- `trajectory_planning.py` -- plan-then-execute primitives (`plan_cartesian_trajectory`,
  `plan_joint_trajectory`, `follow_joint_trajectory`, `park_arms`, `set_gripper`,
  `close_gripper_around`): Giskard plans kinematically against a scratch copy of the
  world, then the result is played back on the real, physically simulated one.
- `pick_and_place_action.py` -- `PickUpActionMujoco`/`PlaceActionMujoco`, matching the
  real `PickUpAction`/`PlaceAction`'s own field interface but driven by the primitives
  above instead of a Giskard motion mapping; generic over any body and arm.
- `grasp_contact.py` -- MuJoCo contact-friction tuning (`GRASP_FRICTION`,
  `SURFACE_FRICTION`, solver reference/impedance) for a grasped object and the surface
  it rests on.

A grasped object is held by real contact friction between the fingers throughout, never
kinematically attached: a poor grasp visibly fails instead of being rescued by a weld.

The ten-milk clutter demo built on this lives in
`experiments/causal_reasoning/tracy_rspn`.
