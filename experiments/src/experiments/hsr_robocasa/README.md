# The HSR in RoboCasa

The Toyota HSR performs RoboCasa's pick-and-place tasks in a RoboCasa kitchen,
simulated in MuJoCo. RoboCasa composes the kitchen and samples the objects, the HSR
replaces RoboCasa's robot, and every object is held by friction alone.

The kitchen is RoboCasa's layout 1, style 1, with everything standing on the floor
lowered from RoboCasa's 0.92 m counters to 0.75 m, where the HSR reaches.

| Task (class in `pick_place.py`) | What the HSR does | Seed | Object |
|---|---|---|---|
| `PickPlaceCounterToStove` | takes the food from the plate and puts it in the pan | 2 | cheese |
| `PickPlaceStoveToCounter` | takes the food from the pan and puts it in the bowl | 12 | lemon |
| `PickPlaceCounterToSink` | takes the object from the counter and drops it in the sink | 0 | can |
| `PickPlaceSinkToCounter` | takes the food out of the sink and puts it on the plate | 11 | yogurt |

## One-time setup

1. **robosuite and RoboCasa**, installed editable into the project's venv without
   their dependencies, so they leave MuJoCo and numpy as the project pins them:

   ```bash
   cd ~/cram2
   git clone https://github.com/ARISE-Initiative/robosuite.git
   git clone https://github.com/robocasa/robocasa.git
   cd ~/cram2/cognitive_robot_abstract_machine
   .venv/bin/python -m pip install --no-deps -e ~/cram2/robosuite -e ~/cram2/robocasa
   .venv/bin/python -m pip install termcolor h5py gymnasium
   ```

   The last line adds the few packages robosuite needs at import.

2. **RoboCasa's assets** (kitchen fixtures and objects, about 6 GB):

   ```bash
   .venv/bin/python -m robocasa.scripts.download_kitchen_assets
   ```

   The loader reads them from `~/robocasa-assets`; if the script put them elsewhere,
   link that folder there.

3. **A one-line fix in RoboCasa.** RoboCasa removes duplicate counter regions with
   `list(set(...))` over XML elements, which hash by memory address, so the same seed
   puts objects in different places in different runs. Keep the order instead:

   ```bash
   sed -i 's/valid_geoms = list(set(valid_geoms))/valid_geoms = list(dict.fromkeys(valid_geoms))/' \
       ~/cram2/robocasa/robocasa/models/fixtures/counter.py
   ```

   Without it, a task can sample a different scene than its seed promises, and its
   full run can fail.

## Every shell

ROS provides `rclpy`, and `~/ros_ws` provides the HSR's description
(`hsr_description`):

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros_ws/install/setup.bash
cd ~/cram2/cognitive_robot_abstract_machine
```

## Running the tests

**Fast tests** (the loader, the lowered layout, the HSR's arm), about 45 s:

```bash
.venv/bin/python -m pytest --orm-build never -q \
    test/semantic_digital_twin_test/test_robocasa \
    test/experiments_test/hsr_robocasa_test
```

The four full runs show as skipped: they only run with `CI=true`.

**Full runs**: each task performed in MuJoCo, about 90 s per task. Cap the memory,
so that a runaway run is killed rather than the machine:

```bash
CI=true systemd-run --user --scope -q -p MemoryMax=14G \
    .venv/bin/python -m pytest --orm-build never -q test/experiments_test/hsr_robocasa_test
```

One task only:

```bash
CI=true .venv/bin/python -m pytest --orm-build never -q \
    test/experiments_test/hsr_robocasa_test -k PickPlaceCounterToSink
```

Run the tests one after the other, never two at the same time: every RoboCasa scene
is written to the same `/tmp/scene.xml`, so parallel runs overwrite each other's
scene.

If a test fails on an ORM import, regenerate the ORM interfaces:

```bash
.venv/bin/python scripts/regenerate_all_orm.py
```

## Watching a task

With MuJoCo's viewer, the stove task:

```bash
.venv/bin/python -m experiments.hsr_robocasa.pick_place
```

Any other task:

```bash
.venv/bin/python -c "
from experiments.hsr_robocasa.pick_place import PickPlaceSinkToCounter
task = PickPlaceSinkToCounter()
task.build_world()
task.run()
"
```
