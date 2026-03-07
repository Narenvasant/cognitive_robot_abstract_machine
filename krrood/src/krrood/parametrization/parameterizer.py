from __future__ import annotations

import typing
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Optional

import numpy as np
from typing_extensions import Any, List

import random_events.variable
from krrood.adapters.json_serializer import leaf_types
from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.factories import and_
from krrood.entity_query_language.query.match import MatchVariable
from krrood.parametrization.random_events_translator import (
    WhereExpressionToRandomEventTranslator,
)
from random_events.product_algebra import Event


@dataclass
class UnderspecifiedParameters:
    """
    A class that extracts all necessary information from an UnderspecifiedVariable and binds it together.
    Instances of this can be used to parameterize objects with underspecified variables using generative models.
    """

    statement: MatchVariable
    """
    The UnderspecifiedVariable to extract information from.
    """

    _random_event_compiler: Optional[WhereExpressionToRandomEventTranslator] = field(
        init=False
    )
    """
    The translator that extracts a random event from the where conditions.
    Only exists if the statement has a where condition.
    """

    truncation_event: Optional[Event] = field(init=False, default=None)
    """
    The where condition as random event.
    Only exists if the statement has a where condition.
    """

    def __post_init__(self):
        self._random_event_compiler = WhereExpressionToRandomEventTranslator(
            and_(*self.statement._where_expressions)
        )
        if self.statement._where_expressions:
            self.truncation_event = self._random_event_compiler.translate()

    @cached_property
    def variables(self) -> Dict[MappedVariable, random_events.variable.Variable]:
        result = self._random_event_compiler.variables
        for literal in self.statement.literals:
            if literal.assigned_variable._type_ not in leaf_types:
                continue
            result[literal.assigned_variable] = (
                random_events.variable.variable_from_name_and_type(
                    literal.assigned_variable._name_, literal.assigned_variable._type_
                )
            )
        return result

    @property
    def assignments_for_conditioning(
        self,
    ) -> Dict[random_events.variable.Variable, Any]:
        """
        :return: A dictionary that contains all facts from the statement and that can be directly used for
        conditioning a probabilistic model.
        """
        return {
            v: v.value
            for v in self._random_event_compiler.variables.keys()
            if v.left._value_ is not None and isinstance(v.value, leaf_types)
        }

    @property
    def random_event_variables(self) -> List[random_events.variable.Variable]:
        """
        :return: A list of all random event variables that are used in the statement.
        """
        return list(self.variables.values())

    def get_variable_by_name(self, name: str) -> random_events.variable.Variable:
        [result] = [v for v in self.random_event_variables if v.name == name]
        return result

    def create_assignment_from_variables_and_sample(
        self,
        variables: typing.Iterable[random_events.variable.Variable],
        sample: np.ndarray,
    ) -> Dict[random_events.variable.Variable, Any]:
        """
        Create an assignment dictionary that can be used to construct a new object from a sample.
        :param variables: The variables from a probabilistic model.
        :param sample: A sample from the same model-
        :return: A dictionary that can be used to construct a new object from a sample.
        """

        result = {}
        for variable_, value in zip(variables, sample):
            parametrization_variable = self.get_variable_by_name(variable_.name)

            if not variable_.is_numeric:
                [value] = [
                    domain_value.element
                    for domain_value in variable_.domain
                    if hash(domain_value) == value
                ]
            else:
                value = value.item()
            result[parametrization_variable] = value

        return result
