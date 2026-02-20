from __future__ import annotations

import os.path
from abc import abstractmethod
import logging
from dataclasses import dataclass, fields
from enum import Enum
from functools import cached_property
from itertools import product

from typing_extensions import (
    Any,
    Optional,
    Callable,
    TypeVar,
    Dict,
    Type,
    List,
    Union,
    Iterable,
)

from krrood.entity_query_language.entity import (
    variable,
    evaluate_condition,
    exists,
    set_of,
)
from krrood.entity_query_language.entity_result_processors import an, a
from krrood.entity_query_language.symbolic import Variable, SymbolicExpression
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    KinematicStructureEntity,
)
from ...datastructures.enums import ApproachDirection, VerticalAlignment
from ...datastructures.grasp import GraspDescription
from ...datastructures.pose import PoseStamped
from ...designator import DesignatorDescription
from ...failures import PlanFailure, ConditionNotSatisfied
from ...utils import get_all_values_in_enum

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ActionDescription(DesignatorDescription):
    _pre_perform_callbacks = []
    _post_perform_callbacks = []

    def __post_init__(self):
        pass
        # self._pre_perform_callbacks.append(self._update_robot_params)

    def perform(self) -> Any:
        """
        Full execution: pre-check, plan, post-check
        """
        logger.info(f"Performing action {self.__class__.__name__}")

        for pre_cb in self._pre_perform_callbacks:
            pre_cb(self)

        self.pre_condition()

        result = None
        try:
            result = self.execute()
        except PlanFailure as e:
            raise e
        finally:
            pass
            # for post_cb in self._post_perform_callbacks:
            #     post_cb(self)
            #
            # self.validate_postcondition(result)

        return result

    @abstractmethod
    def execute(self) -> Any:
        """
        Symbolic plan. Should only call motions or sub-actions.
        """
        pass

    @abstractmethod
    def pre_condition(self, bound=True) -> SymbolicExpression:
        pass

    @abstractmethod
    def post_condition(self, bound=True) -> SymbolicExpression:
        pass

    @classmethod
    def pre_perform(cls, func) -> Callable:
        cls._pre_perform_callbacks.append(func)
        return func

    @classmethod
    def post_perform(cls, func) -> Callable:
        cls._post_perform_callbacks.append(func)
        return func

    @cached_property
    def bound_variables(self):
        return self._create_variables(True)

    @cached_property
    def unbound_variables(self):
        return self._create_variables(False)

    @property
    def fields(self):
        self_fields = list(fields(self))
        [self_fields.remove(parent_field) for parent_field in fields(ActionDescription)]
        return self_fields

    def _create_variables(self, bound=True) -> Dict[T, Variable[T] | T]:
        """
        Creates krrood variables for all parameter of this action either bound or unbound.

        :return: A dict with action parameters as keys and variables as values.
        """
        return {
            getattr(self, f.name): variable(
                type(getattr(self, f.name)),
                (
                    [getattr(self, f.name)]
                    if bound
                    else self._find_domain_for_value(getattr(self, f.name), self.world)
                ),
            )
            for f in self.fields
        }

    def evaluate_pre_condition(self) -> bool:
        evaluation = evaluate_condition(self.pre_condition())
        if evaluation:
            return True
        raise ConditionNotSatisfied(True, self.__class__, self.pre_condition())

    def evaluate_post_condition(self) -> bool:
        evaluation = evaluate_condition(self.post_condition())
        if evaluation:
            return True
        raise ConditionNotSatisfied(False, self.__class__, self.post_condition())

    def _find_domain_for_value(self, value: Any, world: World) -> List:
        value_type = type(value)
        if issubclass(value_type, SemanticAnnotation):
            return [
                sa
                for sa in world.semantic_annotations
                if issubclass(type(sa), (value_type, SemanticAnnotation))
            ]
        elif issubclass(value_type, KinematicStructureEntity):
            # return world.kinematic_structure_entities
            return [value]
        elif issubclass(value_type, Enum):
            return get_all_values_in_enum(value_type)
        elif issubclass(value_type, PoseStamped):
            return [value]
        elif issubclass(value_type, GraspDescription):
            return [
                GraspDescription(approach, align, value.manipulator)
                for approach, align in product(
                    get_all_values_in_enum(ApproachDirection),
                    get_all_values_in_enum(VerticalAlignment),
                )
            ]
        logger.warning(f"There is no domain for type {value_type}")
        return []

    def find_possible_parameter(self):
        unbound_condition = self.pre_condition(False)
        query = a(set_of(*self.unbound_variables.values()).where(unbound_condition))
        var_to_field = dict(zip(self.unbound_variables.values(), self.fields))
        for result in query.evaluate():
            bindings = result.data
            yield {var_to_field[k].name: v for k, v in bindings.items()}


ActionType = TypeVar("ActionType", bound=ActionDescription)
type DescriptionType[T] = Union[Iterable[T], T]
