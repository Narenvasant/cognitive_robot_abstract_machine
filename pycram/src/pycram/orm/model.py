from dataclasses import dataclass
from typing import List

import numpy as np
from krrood.ormatic.data_access_objects.alternative_mappings import (
    AlternativeMapping,
    T,
)
from sqlalchemy import TypeDecorator, types
from typing_extensions import Optional

from pycram.datastructures.dataclasses import ExecutionData
from pycram.datastructures.enums import TaskStatus
from pycram.designator import DesignatorDescription
from pycram.failures import PlanFailure
from pycram.plan import (
    ActionDescriptionNode,
    MotionNode,
    PlanNode,
    ActionNode,
    DesignatorNode,)
from pycram.datastructures.dataclasses import Context
from pycram.plans.plan import (
    Plan,
)
from pycram.plans.plan_node import PlanNode
from semantic_digital_twin.world import World

# ----------------------------------------------------------------------------------------------------------------------
#            Map all Designators, that are not self-mapping, here.
#            By default all classes are self-mapping, so you only need to add the ones where not every attribute is
#            supposed to be mapped or where an attribute is from a type, which is not mapped itself.
#            Specify the columns(attributes) that are supposed to be tracked in the database.
#            One attribute equals one column. Please refer to the ORMatic documentation for more information.
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class PlanEdge:
    parent: PlanNode
    child: PlanNode


@dataclass(eq=False)
class PlanMapping(AlternativeMapping[Plan]):
    root: PlanNode
    nodes: List[PlanNode]
    edges: List[PlanEdge]
    context: Context
    initial_world: Optional[World] = None

    @classmethod
    def from_domain_object(cls, obj: Plan):
        return cls(
            root=obj.root,
            nodes=obj.nodes,
            edges=[PlanEdge(edge[0], edge[1]) for edge in obj.edges],
            context=obj.context,
            initial_world=obj.initial_world,
        )

    def to_domain_object(self) -> T:
        result = Plan(context=self.context, initial_world=self.initial_world)
        for node in self.nodes:
            result.add_node(node)

        for edge in self.edges:
            result.add_edge(edge.parent, edge.child)
        return result


class NumpyType(TypeDecorator):
    """
    Type that casts field which are of numpy nd array type
    """

    impl = types.LargeBinary(4 * 1024 * 1024 * 1024 - 1)  # 4 GB max

    def process_bind_param(self, value: np.ndarray, dialect):
        array = value.astype(np.float64)
        return array.tobytes()

    def process_result_value(self, value: impl, dialect) -> Optional[np.ndarray]:
        return np.frombuffer(value, dtype=np.float64)
