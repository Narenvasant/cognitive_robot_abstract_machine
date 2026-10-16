"""
The HSR opening and closing a drawer in a RoboCasa kitchen, simulated in MuJoCo with
the drawer moved by nothing but the HSR's hold on its handle.
"""

from enum import Enum

import pytest

pytest.importorskip("robocasa", reason="robocasa is not installed")
pytest.importorskip("robosuite", reason="robosuite is not installed")

from experiments.hsr_robocasa.drawers import CloseDrawer, HSRDrawerTask, OpenDrawer

from .test_pick_place import runs_in_ci

# %% RoboCasa's verdict


class DrawerVerdict(float, Enum):
    """
    How far along its travel RoboCasa's drawer tasks count a drawer as open or as
    closed, in their ``ManipulateDrawer._check_success``.
    """

    OPEN = 0.95
    """
    At least this far open.
    """

    CLOSED = 0.05
    """
    At most this far open.
    """


def how_far_open(experiment: HSRDrawerTask) -> float:
    """
    :return: How far along its travel the drawer of ``experiment`` stands open.
    """
    joint = experiment.drawer.movable_joint
    return joint.position / joint.dof.limits.upper.position


# %% the runs


@runs_in_ci
def test_the_hsr_opens_the_drawer():
    experiment = OpenDrawer(headless=True)
    experiment.build_world()

    experiment.run()

    assert how_far_open(experiment) >= DrawerVerdict.OPEN


@runs_in_ci
def test_the_hsr_closes_the_drawer():
    experiment = CloseDrawer(headless=True)
    experiment.build_world()

    experiment.run()

    assert how_far_open(experiment) <= DrawerVerdict.CLOSED
