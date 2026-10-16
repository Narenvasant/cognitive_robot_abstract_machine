"""
Tests for declaring contact parameters on shapes.
"""

from __future__ import annotations

from datetime import timedelta

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.contact import (
    ContactFriction,
    ContactImpedance,
    ContactParameters,
    ContactResistance,
    ContactStiffness,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% list forms


def test_contact_parameters_round_trip_through_their_list_forms():
    friction = ContactFriction(sliding=0.4, torsional=0.05, rolling=0.001)
    stiffness = ContactStiffness(
        time_constant=timedelta(milliseconds=8), damping_ratio=0.9
    )
    impedance = ContactImpedance(
        minimum=0.96, maximum=0.99, width=0.002, midpoint=0.4, power=3.0
    )

    assert ContactFriction(*friction.to_list()) == friction
    assert ContactImpedance(*impedance.to_list()) == impedance
    assert friction.to_list() == [0.4, 0.05, 0.001]
    assert stiffness.to_list() == [0.008, 0.9]
    assert impedance.to_list() == [0.96, 0.99, 0.002, 0.4, 3.0]


# %% which coefficients a contact resolves


def test_a_contact_resolves_sliding_unless_it_is_told_otherwise():
    """
    Resolving sliding alone is what a physics engine does by default, so declaring
    friction changes nothing else about a shape.
    """
    assert ContactParameters(friction=ContactFriction()).resistance is (
        ContactResistance.SLIDING
    )


def test_a_grip_resolves_the_coefficients_it_raises():
    """
    The torsional and rolling friction a grip raises are only felt by a contact that
    resolves them, so the two have to be declared together.
    """
    grip = ContactParameters.create_for_grasped_object()

    assert grip.resistance is ContactResistance.EVERY_DIRECTION
    assert grip.friction.torsional > ContactFriction().torsional
    assert grip.friction.rolling > ContactFriction().rolling


def test_a_surface_is_never_pinched_so_it_only_resolves_sliding():
    assert ContactParameters.create_for_surface().resistance is (
        ContactResistance.SLIDING
    )


# %% applying to bodies


def test_contact_parameters_reach_every_collision_geometry():
    body = Body(name=PrefixedName("box"))
    body.collision = ShapeCollection(
        [
            Box(origin=HomogeneousTransformationMatrix(), scale=Scale(1, 1, 1)),
            Box(origin=HomogeneousTransformationMatrix(), scale=Scale(1, 1, 1)),
        ],
        reference_frame=body,
    )
    contact = ContactParameters.create_for_grasped_object(sliding_friction=0.4)

    contact.apply_to([body])

    for shape in body.collision:
        attached = shape.get_simulator_property_of_type(ContactParameters)
        assert attached == contact
        assert attached is not contact
        assert attached.friction == ContactFriction(
            sliding=0.4, torsional=0.05, rolling=0.001
        )
        assert attached.resistance is ContactResistance.EVERY_DIRECTION


def test_a_grip_applied_over_a_geometrys_own_parameters_carries_its_resistance():
    """
    A shape that already declares contacts keeps its own stiffness, but what the
    contacts resolve comes with the friction they belong to.
    """
    body = Body(name=PrefixedName("box"))
    shape = Box(origin=HomogeneousTransformationMatrix(), scale=Scale(1, 1, 1))
    body.collision = ShapeCollection([shape], reference_frame=body)
    shape.add_simulator_property(ContactParameters(friction=ContactFriction()))

    ContactParameters.create_for_grasped_object().apply_to([body])

    attached = shape.get_simulator_property_of_type(ContactParameters)
    assert attached.resistance is ContactResistance.EVERY_DIRECTION


def test_a_surface_leaves_the_geometrys_own_stiffness_and_impedance():
    body = Body(name=PrefixedName("table"))
    shape = Box(origin=HomogeneousTransformationMatrix(), scale=Scale(1, 1, 1))
    body.collision = ShapeCollection([shape], reference_frame=body)
    own_stiffness = ContactStiffness(time_constant=timedelta(milliseconds=50))
    own = ContactParameters(friction=ContactFriction(), stiffness=own_stiffness)
    shape.add_simulator_property(own)

    ContactParameters.create_for_surface(sliding_friction=0.2).apply_to([body])

    assert shape.get_simulator_property_of_type(ContactParameters) is own
    assert own.friction == ContactFriction(sliding=0.2)
    assert own.stiffness == own_stiffness
    assert own.impedance is None


def test_a_body_without_collision_geometry_is_skipped():
    body = Body(name=PrefixedName("frame"))
    shape = Box(origin=HomogeneousTransformationMatrix(), scale=Scale(1, 1, 1))
    body.visual = ShapeCollection([shape], reference_frame=body)

    ContactParameters.create_for_surface().apply_to([body])

    assert shape.get_simulator_property_of_type(ContactParameters) is None
