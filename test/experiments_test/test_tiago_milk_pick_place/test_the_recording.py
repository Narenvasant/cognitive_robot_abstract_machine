"""
Tests for the video the Tiago milk pick-and-place demo can write of a run, and for the
camera it is shot from.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import imageio
import mujoco
import pytest

from experiments.tiago_milk_pick_place.__main__ import (
    DEFAULT_RECORDING_PATH,
    DemoArguments,
)
from experiments.tiago_milk_pick_place.physics import (
    PhysicalMujocoRun,
    SceneRecorder,
    SceneRecording,
    ViewerCamera,
)
from semantic_digital_twin.adapters.mujoco_video_recording import VideoResolution

SCENE_PATH = Path(__file__).parent / "box_on_a_plane.xml"
"""
The scene the recorder is pointed at, which is the smallest one a frame can be drawn
from.
"""


@pytest.fixture
def camera() -> ViewerCamera:
    return ViewerCamera(
        look_at=(1.0, 2.0, 3.0), distance=4.0, azimuth=5.0, elevation=-6.0
    )


def build_run(recording: SceneRecording | None) -> PhysicalMujocoRun:
    """
    A run that has not built any physics, which is all the recording is set up from.

    :param recording: What to record, or nothing.
    :return: The run.
    """
    return PhysicalMujocoRun(
        world=None, robot=None, carton_body_name="milk", recording=recording
    )


# %% the camera the recording is shot from


def test_the_recording_s_camera_orbits_where_the_viewer_looks(camera):
    """
    The recording is shot from where the viewer would have been pointed, so what it
    shows is what someone watching the window would have seen.
    """
    shot_from = camera.free_camera()

    assert tuple(shot_from.lookat) == camera.look_at
    assert shot_from.distance == camera.distance
    assert shot_from.azimuth == camera.azimuth
    assert shot_from.elevation == camera.elevation


def test_the_recording_s_camera_is_not_tied_to_a_body(camera):
    """
    The camera orbits a point of the world rather than riding on the robot.
    """
    assert camera.free_camera().type == mujoco.mjtCamera.mjCAMERA_FREE


# %% how fast the recording plays


def test_a_frame_stands_for_as_much_simulated_time_as_it_is_played_for():
    """
    Frames are taken by the simulation's clock, so the video plays back at the speed the
    robot moves however long the physics took to compute.
    """
    recording = SceneRecording(path=Path("unused.mp4"), frames_per_second=25)

    assert recording.simulated_seconds_per_frame == pytest.approx(1.0 / 25)


# %% setting a run up to record


def test_a_run_asked_to_record_writes_where_it_was_told(tmp_path):
    video = tmp_path / "run.mp4"
    run = build_run(SceneRecording(path=video))

    run.set_up_recording()

    assert run.recorder.recording.path == video


def test_a_run_not_asked_to_record_sets_up_no_recorder():
    run = build_run(None)

    run.set_up_recording()

    assert run.recorder is None


def test_the_recording_is_shot_from_the_run_s_own_camera():
    """
    One camera serves the window and the file, so a run that was pointed somewhere else
    records from there too.
    """
    run = build_run(SceneRecording(path=Path("unused.mp4")))

    run.set_up_recording()

    assert run.recorder.camera is run.camera


# %% what the command line asks for


def test_recording_is_off_unless_it_is_asked_for(monkeypatch):
    monkeypatch.setattr("sys.argv", ["demo"])

    assert DemoArguments.from_command_line().recording is None


def test_a_bare_request_to_record_lands_in_the_downloads_folder(monkeypatch):
    monkeypatch.setattr("sys.argv", ["demo", "--record"])

    assert DemoArguments.from_command_line().recording.path == DEFAULT_RECORDING_PATH


def test_a_request_to_record_somewhere_lands_there(monkeypatch, tmp_path):
    asked_for = tmp_path / "elsewhere.mp4"
    monkeypatch.setattr("sys.argv", ["demo", "--record", str(asked_for)])

    assert DemoArguments.from_command_line().recording.path == asked_for


# %% writing frames to a file


@dataclass
class SteppedScene:
    """
    What the recorder reads a scene from: a simulation that is stepped on another thread
    and guards its own state with a lock.
    """

    model: mujoco.MjModel
    """
    The scene to draw.
    """

    data: mujoco.MjData = field(init=False)
    """
    Its state, which stepping advances.
    """

    lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    """
    What guards that state while a frame is drawn from it.
    """

    def __post_init__(self) -> None:
        self.data = mujoco.MjData(self.model)

    @property
    def simulator(self) -> SteppedScene:
        """
        :return: Itself, standing in for the simulator a simulation hands out.
        """
        return self

    @property
    def _mj_model(self) -> mujoco.MjModel:
        return self.model

    @property
    def _mj_data(self) -> mujoco.MjData:
        return self.data

    @property
    def _model_lock(self) -> threading.Lock:
        return self.lock

    def step(self, steps: int) -> None:
        """
        :param steps: How many steps to advance the scene by.
        """
        for _ in range(steps):
            with self.lock:
                mujoco.mj_step(self.model, self.data)


def can_draw_outside_a_window() -> bool:
    """
    Drawing needs a graphics driver, which the machine running the tests need not have,
    and which answers for itself only by being asked for a context.

    :return: Whether a frame can be drawn at all.
    """
    try:
        mujoco.GLContext(1, 1).free()
    except Exception:
        return False
    return True


@pytest.fixture
def stepped_scene() -> SteppedScene:
    return SteppedScene(model=mujoco.MjModel.from_xml_path(str(SCENE_PATH)))


@pytest.mark.skipif(
    not can_draw_outside_a_window(),
    reason="drawing a frame needs a graphics driver",
)
def test_a_recording_holds_a_frame_for_every_step_of_simulated_time_it_covers(
    stepped_scene, tmp_path
):
    """
    The recorder samples the simulation's own clock, so a scene stepped through a known
    stretch of simulated time comes out as the number of frames that stretch is played
    as.
    """
    video = tmp_path / "scene.mp4"
    recording = SceneRecording(
        path=video,
        frames_per_second=10,
        resolution=VideoResolution(width=128, height=96),
    )
    covered_seconds = 1.0

    with SceneRecorder(
        simulation=stepped_scene, camera=ViewerCamera(), recording=recording
    ) as recorder:
        stepped_scene.step(round(covered_seconds / stepped_scene.model.opt.timestep))

    frames = recording.frames_per_second * covered_seconds
    assert recorder.frames_written == pytest.approx(frames, abs=1)
    assert imageio.v3.improps(video).shape[1:3] == (
        recording.resolution.height,
        recording.resolution.width,
    )
