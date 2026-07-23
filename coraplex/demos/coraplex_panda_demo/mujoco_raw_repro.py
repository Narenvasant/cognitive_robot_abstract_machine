"""
Minimal, standalone MuJoCo test -- no ROS, no coraplex, no semantic_digital_twin.
Just: load stacking_scene.xml directly, step physics, watch where cube0 ends up.

This isolates whether the instability is in the scene file itself, or
somewhere in the digital-twin/MujocoBuilder rebuild pipeline that demo.py
actually simulates (MuJoCo runs a rebuilt copy at /tmp/scene.xml, not this
file directly -- so this script is the only way to test the source file
completely on its own).

Run directly: python3 mujoco_raw_repro.py
"""

import time

import mujoco
import mujoco.viewer

SCENE_PATH = (
    "/home/nvasant/workspace/ros/src/manipulation_experiments/resources/generated/"
    "stacking_scene.xml"
)

model = mujoco.MjModel.from_xml_path(SCENE_PATH)
data = mujoco.MjData(model)

cube_names = ["cube0", "cube1", "cube2", "cube3"]
cube_body_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in cube_names
]

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    last_print = 0.0
    while viewer.is_running() and (time.time() - start) < 15:
        mujoco.mj_step(model, data)
        viewer.sync()

        elapsed = time.time() - start
        if elapsed - last_print >= 1.0:
            last_print = elapsed
            print(f"--- t={elapsed:.1f}s ---")
            for name, body_id in zip(cube_names, cube_body_ids):
                pos = data.xpos[body_id]
                print(f"  {name}: pos = [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

print("Done. Final positions:")
for name, body_id in zip(cube_names, cube_body_ids):
    pos = data.xpos[body_id]
    print(f"  {name}: pos = [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")