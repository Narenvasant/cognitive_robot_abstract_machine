"""
Contact-friction tuning for a MuJoCo-simulated gripper's grasp on a loose object and for
the surface it is released onto, generalized from ``coraplex_panda_demo``'s own
reliably-grasped cube.
"""

from __future__ import annotations

from typing_extensions import Iterable

from experiments.tracy_experiments.equipment import _mujoco_geom_for
from semantic_digital_twin.world_description.world_entity import Body

GRASP_FRICTION = [0.3, 0.05, 0.001]
"""
Contact friction (sliding, torsional, rolling; see
:attr:`~semantic_digital_twin.adapters.multi_sim.MujocoGeom.friction`) given to a loose
object's collision geometry by default.

``0.3`` approximates real sliding friction between painted wood/plastic surfaces
(~0.25-0.4). Contact friction is combined by MuJoCo as the element-wise maximum of the
two participating geoms, so the surface an object rests on needs an equally explicit
:data:`SURFACE_FRICTION` for the object-surface contact to drop below the finger-
dominated grip instead of being pinned at MuJoCo's own ``1.0`` default.
"""

SURFACE_FRICTION = [0.3, 0.005, 0.0001]
"""
Contact friction (sliding, torsional, rolling) given to the surface loose objects rest
on via :func:`apply_contact_friction`.

Matches :data:`GRASP_FRICTION`'s own sliding component so the object-surface contact is
governed by this pair rather than by MuJoCo's own ``1.0`` default. Torsional and rolling
use MuJoCo's own defaults rather than :data:`GRASP_FRICTION`'s grip-stabilizing
multiples of them, since the surface is never pinched between fingers.
"""

GRASP_SOLVER_REFERENCE = [0.008, 1.0]
"""
Contact solver reference (see
:attr:`~semantic_digital_twin.adapters.multi_sim.MujocoGeom.solver_reference`) given to
every loose object, matching ``coraplex_panda_demo``'s cube (``solref="0.008"``).

Stiffer than MuJoCo's own default (``0.02``): a soft contact lets a pinched object sink
into the fingers and then slip back out as the arm lifts, rather than being held solidly
between them.
"""

GRASP_SOLVER_IMPEDANCE = [0.96, 0.99, 0.001, 0.5, 2.0]
"""
Contact solver impedance (see
:attr:`~semantic_digital_twin.adapters.multi_sim.MujocoGeom.solver_impedance`) given to
every loose object, matching ``coraplex_panda_demo``'s cube (``solimp="0.96 0.99"``, the
remaining three values MuJoCo's own defaults).

Harder than MuJoCo's own default (``0.9 0.95``), for the same reason as
:data:`GRASP_SOLVER_REFERENCE`.
"""


def apply_contact_friction(bodies: Iterable[Body], friction: list[float]) -> None:
    """
    Give every collision geometry of every body in ``bodies`` the given contact friction
    (see :attr:`~semantic_digital_twin.adapters.multi_sim.MujocoGeom.friction`), without
    touching solver reference or impedance (see :func:`apply_grasp_contact_parameters`
    for a grasped-object variant that also sets those).

    :param bodies: The bodies to modify in place.
    :param friction: Contact friction to give every body's collision geometry.
    """
    for body in bodies:
        for geometry in body.collision:
            _mujoco_geom_for(geometry).friction = list(friction)


def apply_grasp_contact_parameters(
    bodies: Iterable[Body], friction: list[float]
) -> None:
    """
    Give every body in ``bodies`` the contact parameters that let a gripper pick it up
    and hold it: ``friction`` plus the solver reference and solver impedance of
    ``coraplex_panda_demo``'s own reliably-grasped cube (see :data:`GRASP_FRICTION`,
    :data:`GRASP_SOLVER_REFERENCE`, :data:`GRASP_SOLVER_IMPEDANCE`).

    :param bodies: The bodies to modify in place.
    :param friction: Contact friction to give every body's collision geometry.
    """
    for body in bodies:
        for geometry in body.collision:
            mujoco_geom = _mujoco_geom_for(geometry)
            mujoco_geom.friction = list(friction)
            mujoco_geom.solver_reference = list(GRASP_SOLVER_REFERENCE)
            mujoco_geom.solver_impedance = list(GRASP_SOLVER_IMPEDANCE)
