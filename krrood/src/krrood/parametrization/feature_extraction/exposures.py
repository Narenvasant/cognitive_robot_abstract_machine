"""
Exposure statistics: what a part of an instance is exposed to through the pairs of the
instance that involve it, such as how many touching pieces a piece has.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

from typing_extensions import Any, Callable, ClassVar, TypeVar

from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import Variable as SymbolicVariable
from krrood.parametrization.feature_extraction.exceptions import (
    InvalidAggregatedFieldError,
)
from krrood.patterns.role import Role
from krrood.utils import get_generic_type_parameters, recursive_subclasses

TExposedPart = TypeVar("TExposedPart", bound=object)
"""
The type of the part an :class:`ExposureStatistic` is a role of.
"""


@dataclass
class _ExposureStatisticDescriptor:
    """
    Registers a method marked with :func:`exposure_statistic` on the class that defines
    it, then replaces itself with the method.
    """

    collection_name: str
    """
    The collection of the instance the part belongs to.
    """

    function: Callable
    """
    The marked method.
    """

    def __set_name__(self, owner: type[ExposureStatistic], name: str) -> None:
        if "exposure_registry" not in owner.__dict__:
            owner.exposure_registry = defaultdict(list)
        owner.exposure_registry[self.collection_name].append(self.function)
        setattr(owner, name, self.function)


def exposure_statistic(
    collection: Attribute,
) -> Callable[[Callable], _ExposureStatisticDescriptor]:
    """
    Marks a method as an exposure statistic of the parts of a collection, such as
    ``@exposure_statistic(variable(Scene).pieces)``.

    :param collection: A symbolic attribute of a variable, naming the collection.
    :raises InvalidAggregatedFieldError: If ``collection`` is not a direct attribute of a
        variable.
    """
    if not (
        isinstance(collection, Attribute)
        and isinstance(collection._child_, SymbolicVariable)
    ):
        raise InvalidAggregatedFieldError(aggregated_field=collection)

    def decorator(function: Callable) -> _ExposureStatisticDescriptor:
        return _ExposureStatisticDescriptor(collection._attribute_name_, function)

    return decorator


@dataclass(eq=False)
class ExposureStatistic(Role[TExposedPart]):
    """
    A role of a part that also holds the instance the part belongs to. Its methods
    marked with :func:`exposure_statistic` summarise the pairs of that instance that
    involve the part. A method returns ``...`` where the instance leaves one of the
    pairs it reads open.
    """

    instance: Any = field(kw_only=True)
    """
    The instance the part belongs to.
    """

    exposure_registry: ClassVar[dict[str, list[Callable]]] = defaultdict(list)
    """
    The methods marked with :func:`exposure_statistic` on this class, by collection.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "exposure_registry" not in cls.__dict__:
            cls.exposure_registry = defaultdict(list)

    @classmethod
    def exposures_of_collection(cls, collection_name: str) -> list[Callable]:
        """
        :param collection_name: A collection of the instance.
        :return: The exposure statistics of its parts, ordered by name.
        """
        return sorted(
            (
                function
                for ancestor in cls.__mro__
                if issubclass(ancestor, ExposureStatistic)
                for function in ancestor.__dict__.get("exposure_registry", {}).get(
                    collection_name, []
                )
            ),
            key=lambda function: function.__name__,
        )

    def values(self, collection_name: str) -> dict[str, Any]:
        """
        :param collection_name: The collection of the instance the part belongs to.
        :return: The value of every exposure statistic of the part, by name.
        """
        return {
            function.__name__: function(self)
            for function in self.exposures_of_collection(collection_name)
        }


@lru_cache(maxsize=None)
def exposure_class_of(part_class: type) -> type[ExposureStatistic] | None:
    """
    :param part_class: The class of a part.
    :return: The most specific :class:`ExposureStatistic` bound to it or to one of its
        superclasses, or ``None``.
    """
    subclasses = list(recursive_subclasses(ExposureStatistic))
    for ancestor in part_class.__mro__:
        for subclass in subclasses:
            bound = get_generic_type_parameters(subclass, ExposureStatistic)
            if bound and bound[0] == ancestor:
                return subclass
    return None
