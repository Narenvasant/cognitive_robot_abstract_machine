from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from random_events.product_algebra import SimpleEvent

from krrood.entity_query_language.factories import a, cause, variable
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.learning_method import StratifiedLearning
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    ContinuousInterfaceAttributeError,
    UnknownInterfaceAttributeError,
)
from probabilistic_model.probabilistic_circuit.relational.part_interface import (
    PartInterface,
)
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    RelationalProbabilisticCircuit,
)

from ..dataset.example_classes import (
    DominoBand,
    DominoEnvironment,
    DominoPiece,
    DominoScene,
    SceneRoom,
)
from .test_relational_probabilistic_circuit import scenario  # noqa: F401  # fixture

# %% scenes whose environment places each piece directly


@dataclass
class DirectlyPlacedPieces:
    """
    Draws scenes whose environment decides how likely each piece stands near the
    target, while the number of pieces does not depend on the environment.
    """

    near_rate_in_a_bin: float = 0.8
    """
    How likely a piece of a scene in a bin stands near the target.
    """

    near_rate_on_a_table: float = 0.2
    """
    How likely a piece of a scene on a table stands near the target.
    """

    def near_rate(self, environment: DominoEnvironment) -> float:
        """
        :param environment: Where a scene stands.
        :return: How likely each of its pieces stands near the target.
        """
        if environment is DominoEnvironment.BIN:
            return self.near_rate_in_a_bin
        return self.near_rate_on_a_table

    def scenes(self, scene_count: int, seed: int = 0) -> list[DominoScene]:
        """
        :param scene_count: How many scenes to draw.
        :param seed: Seed of the draw.
        :return: The scenes.
        """
        random_state = np.random.default_rng(seed)
        result = []
        for _ in range(scene_count):
            environment = random_state.choice(list(DominoEnvironment))
            pieces = [
                DominoPiece(
                    band=(
                        DominoBand.NEAR
                        if random_state.random() < self.near_rate(environment)
                        else DominoBand.FAR
                    ),
                    disturbed=bool(random_state.random() < 0.5),
                )
                for _ in range(random_state.integers(2, 5))
            ]
            result.append(DominoScene(environment, pieces))
        return result


@pytest.fixture(scope="module")
def placed_pieces() -> DirectlyPlacedPieces:
    return DirectlyPlacedPieces()


@pytest.fixture(scope="module")
def scenes(placed_pieces) -> list[DominoScene]:
    return placed_pieces.scenes(400)


def interface_with_the_environment() -> PartInterface:
    """
    :return: An interface that shows every piece's template the scene's environment.
    """
    return PartInterface.from_attributes(
        {variable(DominoScene).pieces: [variable(DominoScene).environment]}
    )


@pytest.fixture(scope="module")
def circuit_with_interface(scenes) -> RelationalProbabilisticCircuit:
    return RelationalProbabilisticCircuit(
        learning_method=StratifiedLearning(
            variables=["DominoScene.environment"], method=JointProbabilityTree()
        ),
        part_interface=interface_with_the_environment(),
    ).fit(scenes)


def scene_query(environment) -> object:
    """
    :param environment: What the query gives the environment.
    :return: A resolved query of a scene with three pieces.
    """
    query = a(DominoScene)(
        environment=environment,
        pieces=[a(DominoPiece)(band=..., disturbed=...) for _ in range(3)],
    )
    query.resolve()
    return query


def probability_near(circuit, environment: DominoEnvironment | None) -> float:
    """
    :param circuit: A grounded circuit of :func:`scene_query`.
    :param environment: The environment to condition on, or ``None`` if the query
        gave it.
    :return: The probability that the first piece stands near the target.
    """
    variable_of = {variable.name: variable for variable in circuit.variables}
    near = SimpleEvent.from_data(
        {variable_of["DominoScene.pieces[0].band"]: DominoBand.NEAR}
    ).as_composite_set()
    if environment is None:
        return circuit.probability(near)
    given = SimpleEvent.from_data(
        {variable_of["DominoScene.environment"]: environment}
    ).as_composite_set()
    return circuit.probability(near & given) / circuit.probability(given)


# %% the template sees the environment


def test_the_part_template_is_conditioned_on_the_interface_attribute(
    circuit_with_interface,
):
    template = circuit_with_interface.exchangeable_distribution_templates["pieces"]

    assert [attribute.path for attribute in template.interface_attributes] == [
        "environment"
    ]
    assert template.interface_attributes[0].variable in template.latent_variables


@pytest.mark.parametrize("environment", list(DominoEnvironment))
def test_a_given_environment_shifts_each_piece(
    circuit_with_interface, placed_pieces, environment
):
    circuit = circuit_with_interface.ground(scene_query(environment))

    assert probability_near(circuit, None) == pytest.approx(
        placed_pieces.near_rate(environment), abs=0.08
    )


@pytest.mark.parametrize("environment", list(DominoEnvironment))
def test_an_open_environment_shifts_each_piece_through_its_point_masses(
    circuit_with_interface, placed_pieces, environment
):
    np.random.seed(0)
    circuit = circuit_with_interface.ground(scene_query(...))

    assert probability_near(circuit, environment) == pytest.approx(
        placed_pieces.near_rate(environment), abs=0.08
    )


def test_the_environment_is_a_cause_the_grounded_circuit_is_deterministic_over(
    circuit_with_interface,
):
    query = scene_query(cause)
    query.causes_effect(query.pieces[0].band == DominoBand.NEAR)

    np.random.seed(0)
    result = RelationalCircuitRegistry(
        relational_probabilistic_circuit=circuit_with_interface
    ).get_model(UnderspecifiedParameters(query))

    assert isinstance(result, CausalCircuit)
    assert [variable.name for variable in result.causal_variables] == [
        "DominoScene.environment"
    ]


# %% refused interfaces


def test_a_continuous_interface_attribute_is_refused(scenario):  # noqa: F811
    model = RelationalProbabilisticCircuit(
        part_interface=PartInterface.from_attributes(
            {variable(SceneRoom).objects: [variable(SceneRoom).position.x]}
        )
    )

    with pytest.raises(ContinuousInterfaceAttributeError):
        model.fit(list(scenario))


def test_an_interface_attribute_the_class_lacks_is_refused(scenes):
    model = RelationalProbabilisticCircuit(
        part_interface=PartInterface({"pieces": ["colour"]})
    )

    with pytest.raises(UnknownInterfaceAttributeError):
        model.fit(scenes)
