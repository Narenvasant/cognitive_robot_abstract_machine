from __future__ import annotations
from random_events.set import Set
from random_events.variable import Continuous, Integer, Symbolic
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.class_diagrams.parameterizer import Parameterizer
from ..dataset.example_classes import (
    Position,
    Orientation,
    Pose,
    Atom,
    Element,
)


def test_parameterize_position():
    """
    Test parameterization of the Position class.
    """
    parameterizer = Parameterizer()
    class_diagram = ClassDiagram([Position])
    wrapped_position = class_diagram.get_wrapped_class(Position)
    variables = parameterizer(wrapped_position)
    expected_variables = [
        Continuous("Position.x"),
        Continuous("Position.y"),
        Continuous("Position.z"),
    ]
    assert variables == expected_variables


def test_parameterize_orientation():
    """
    Test parameterization of the Orientation class.
    """
    parameterizer = Parameterizer()
    class_diagram = ClassDiagram([Orientation])
    wrapped_orientation = class_diagram.get_wrapped_class(Orientation)
    variables = parameterizer(wrapped_orientation)
    expected_variables = [
        Continuous("Orientation.x"),
        Continuous("Orientation.y"),
        Continuous("Orientation.z"),
        Continuous("Orientation.w"),
    ]

    assert variables == expected_variables


def test_parameterize_pose():
    """
    Test parameterization of the Pose class.
    """
    parameterizer = Parameterizer()
    class_diagram = ClassDiagram([Pose, Position, Orientation])
    wrapped_pose = class_diagram.get_wrapped_class(Pose)
    variables = parameterizer(wrapped_pose)
    expected_variables = [
        Continuous("Pose.position.x"),
        Continuous("Pose.position.y"),
        Continuous("Pose.position.z"),
        Continuous("Pose.orientation.x"),
        Continuous("Pose.orientation.y"),
        Continuous("Pose.orientation.z"),
        Continuous("Pose.orientation.w"),
    ]

    assert variables == expected_variables


def test_parameterize_atom():
    """
    Test parameterization of the Atom class.
    """
    parameterizer = Parameterizer()
    class_diagram = ClassDiagram([Atom, Element])
    wrapped_atom = class_diagram.get_wrapped_class(Atom)
    variables = parameterizer(wrapped_atom)
    expected_variables = [
        Symbolic("Atom.element", Set.from_iterable([Element.C, Element.H])),
        Integer("Atom.type"),
        Continuous("Atom.charge"),
    ]

    assert [(type(v), v.name) for v in variables] == [
        (type(v), v.name) for v in expected_variables
    ]



def test_create_fully_factorized_distribution():
    """
    Test for a fully factorized distribution.
    """
    parameterizer = Parameterizer()
    variables = [
        Continuous("Variable.A"),
        Continuous("Variable.B"),
    ]
    probabilistic_circuit = parameterizer.create_fully_factorized_distribution(
        variables
    )
    assert len(probabilistic_circuit.variables) == 2
    assert set(probabilistic_circuit.variables) == set(variables)





