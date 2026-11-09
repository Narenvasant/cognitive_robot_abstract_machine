"""
Which attributes of an instance the template of each of its exchangeable parts is
conditioned on, beside the aggregation statistics.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

from typing_extensions import Any, Self

from krrood.entity_query_language.core.mapped_variable import Attribute
from probabilistic_model.probabilistic_circuit.relational.part_learning_methods import (
    PartLearningMethods,
)
from random_events.variable import Variable

# %% one attribute a template sees


@dataclass
class InterfaceAttribute:
    """
    An attribute of an instance that the template of one of its parts is conditioned on.
    """

    variable: Variable
    """
    The variable the instance's circuit and the part's template both name the attribute
    by.
    """

    path: str
    """
    The attribute accesses from the instance to the attribute, such as ``"environment"``.
    """

    value_type: type
    """
    The type of the attribute's values.
    """

    def value_of(self, instance: Any) -> Any:
        """
        :param instance: An instance of the class the attribute belongs to.
        :return: What the instance holds in the attribute.
        """
        return functools.reduce(getattr, self.path.split("."), instance)

    def is_determined_by(self, instance: Any) -> bool:
        """
        :param instance: An instance constructed from a query.
        :return: Whether the query gives the attribute a value, rather than leaving it
            open or marking it.
        """
        return isinstance(self.value_of(instance), self.value_type)


# %% the attributes of every part


@dataclass
class PartInterface:
    """
    The instance attributes the template of every exchangeable part, nested ones
    included, is conditioned on beside the aggregation statistics, keyed by the path of
    the part, as :class:`PartLearningMethods` keys its learning methods.
    """

    attribute_paths_by_part: dict[str, list[str]] = field(default_factory=dict)
    """
    Per part path, such as ``"objects"``, the paths of the instance attributes, such as
    ``"environment"``.
    """

    @classmethod
    def from_attributes(cls, attributes: dict[Attribute, list[Attribute]]) -> Self:
        """
        :param attributes: Per attribute access to a part, such as
            ``variable(Room).objects``, the attribute accesses to the instance
            attributes its template sees, such as ``variable(Room).environment``.
        :return: The interface of the parts.
        """
        return cls(
            {
                PartLearningMethods.path_of(part): [
                    PartLearningMethods.path_of(attribute)
                    for attribute in part_attributes
                ]
                for part, part_attributes in attributes.items()
            }
        )

    def attribute_paths_of(self, part: str) -> list[str]:
        """
        :param part: The path of a part.
        :return: The paths of the instance attributes its template sees.
        """
        return self.attribute_paths_by_part.get(part, [])

    def of_part(self, part: str) -> PartInterface:
        """
        :param part: The field name of an exchangeable part.
        :return: The interface of the parts nested in the part.
        """
        prefix = f"{part}."
        return PartInterface(
            {
                path.removeprefix(prefix): attribute_paths
                for path, attribute_paths in self.attribute_paths_by_part.items()
                if path.startswith(prefix)
            }
        )
