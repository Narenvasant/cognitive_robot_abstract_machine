from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing_extensions import List, Optional

from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from random_events.set import Set
from random_events.variable import Continuous, Integer, Symbolic, Variable

from ..class_diagrams.class_diagram import WrappedClass
from ..class_diagrams.wrapped_field import WrappedField


@dataclass
class Parameterizer:
    """
    Parameterizer for creating random event variables from WrappedClass instances.
    """

    def __call__(self, wrapped_class: WrappedClass) -> List[Variable]:
        """
        Create random event variables from a WrappedClass.
        """
        return self._parameterize_wrapped_class(
            wrapped_class, prefix=wrapped_class.clazz.__name__
        )

    def _parameterize_wrapped_class(
        self, wrapped_class: WrappedClass, prefix: str
    ) -> List[Variable]:
        """
        Create variables for all fields of a WrappedClass recursively.
        """
        variables = []
        for wrapped_field in wrapped_class.fields:
            variables.extend(self._parameterize_wrapped_field(wrapped_field, prefix))
        return variables

    def _parameterize_wrapped_field(
        self, wrapped_field: WrappedField, prefix: str
    ) -> List[Variable]:
        """
        Create variables for a single WrappedField.
        """
        field_name = f"{prefix}.{wrapped_field.name}"

        if (
            wrapped_field.type_endpoint is datetime
            or wrapped_field.is_type_type
            or wrapped_field.is_one_to_many_relationship
        ):
            return []

        if wrapped_field.is_one_to_one_relationship and not wrapped_field.is_enum:
            return self._parameterize_relationship(wrapped_field, field_name)

        variable = self._create_variable_from_field(wrapped_field, field_name)
        return [variable] if variable else []

    def _parameterize_relationship(
        self, wrapped_field: WrappedField, field_name: str
    ) -> List[Variable]:
        """
        Create variables for a one-to-one relationship field.
        """
        if not wrapped_field.clazz._class_diagram:
            return []

        target_wrapped_class = wrapped_field.clazz._class_diagram.get_wrapped_class(
            wrapped_field.type_endpoint
        )
        return self._parameterize_wrapped_class(target_wrapped_class, prefix=field_name)

    def _create_variable_from_field(
        self, wrapped_field: WrappedField, field_name: str
    ) -> Optional[Variable]:
        """
        Create a random event variable from a WrappedField based on its type.
        """
        endpoint_type = wrapped_field.type_endpoint

        if wrapped_field.is_enum:
            return Symbolic(field_name, Set.from_iterable(list(endpoint_type)))

        if endpoint_type is int:
            return Integer(field_name)

        if endpoint_type is float:
            return Continuous(field_name)

    def create_fully_factorized_distribution(
        self,
        variables: List[Variable],
    ) -> ProbabilisticCircuit:
        """
        Create a fully factorized probabilistic circuit over the given variables.
        """
        return fully_factorized(
            variables,
            means={v: 0.0 for v in variables if isinstance(v, Continuous)},
            variances={v: 1.0 for v in variables if isinstance(v, Continuous)},
        )
