import itertools
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing_extensions import (
    Dict,
    Any,
    Type,
    List,
    Optional,
    Union,
    Sequence,
    get_origin,
    get_args,
    get_type_hints,
)

import numpy as np
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from probabilistic_model.probabilistic_model import ProbabilisticModel
from random_events.interval import singleton
from random_events.product_algebra import Event, SimpleEvent
from random_events.set import Set
from random_events.variable import Symbolic, Integer, Variable, Continuous

from pycram.datastructures.dataclasses import Context
from pycram.language import SequentialPlan
from pycram.plan import Plan, DesignatorNode, ResolvedActionNode
from pycram.robot_plans import NavigateAction
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.graph_of_convex_sets import (
    GraphOfConvexSets,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from sortedcontainers import SortedSet


@dataclass
class Parameterizer:
    """
    Robust, dataclass-based Parameterizer for plans.

    Features:
    - Recursively converts dataclass parameters to random_events Variables
    - Handles optional types, sequences, enums, and nested dataclasses
    - Tracks hierarchy of variables per node and per field
    - Supports future structured probabilistic modeling
    """

    plan: Plan
    parameters: Dict[DesignatorNode, Any] = field(default_factory=dict, init=False)
    variables_of_node: Dict[DesignatorNode, List[Variable]] = field(
        default_factory=dict, init=False
    )
    hierarchy_of_node: Dict[DesignatorNode, Dict[str, str]] = field(
        default_factory=dict, init=False
    )
    """
    hierarchy_of_node maps each node to a dictionary:
        variable_name -> dataclass path
        e.g., 'Pose_0.position.x' -> 'Pose.position.x'
    """

    def __post_init__(self):
        """Initialize parameters and variables for all nodes in the plan."""
        self.make_parameters()
        self.make_variables()

    @property
    def variables(self) -> List[Variable]:
        """Return all variables for all plan nodes."""
        return list(itertools.chain.from_iterable(self.variables_of_node.values()))

    def get_variable(self, name: str) -> Variable:
        """Return a variable by its full name."""
        return [v for v in self.variables if v.name == name][0]

    def make_parameters(self):
        """Extract parameters from all DesignatorNodes in the plan."""
        for node in self.plan.nodes:
            if isinstance(node, DesignatorNode):
                self.parameters[node] = node.designator_type._parameters

    def make_variables(self):
        """Convert all parameters into random_events Variables, tracking hierarchy."""
        for index, (node, _) in enumerate(self.parameters.items()):
            variables: List[Variable] = []
            hierarchy: Dict[str, str] = {}

            flattened = node.flattened_parameters()
            for name, param_type in flattened.items():
                full_name = f"{node.designator_type.__name__}_{index}.{name}"
                vars_from_type = self._parameterize_type(
                    param_type, full_name, prefix=name
                )
                variables.extend(vars_from_type)

                # Map variable names to their dataclass paths for hierarchy tracking
                for v in vars_from_type:
                    hierarchy[v.name] = f"{name}"

            self.variables_of_node[node] = variables
            self.hierarchy_of_node[node] = hierarchy

    def _parameterize_type(
        self, typ: Type, prefix: str, prefix_path: str = ""
    ) -> List[Variable]:
        """Recursively convert a type to random_events Variables."""
        variables: List[Variable] = []

        # Handle Optional[T]
        origin = get_origin(typ)
        args = get_args(typ)
        if origin is Union:
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                typ = non_none[0]

        # Handle sequences
        origin = get_origin(typ)
        args = get_args(typ)
        if origin in (list, List, Sequence) and args:
            typ = args[0]

        # Nested dataclass
        if is_dataclass(typ):
            type_hints = get_type_hints(typ)
            for f in fields(typ):
                field_type = type_hints[f.name]
                qualified = f"{prefix}.{f.name}"
                variables.extend(
                    self._parameterize_type(
                        field_type,
                        qualified,
                        prefix_path=(
                            f"{prefix_path}.{f.name}" if prefix_path else f.name
                        ),
                    )
                )

        # Leaf types
        elif issubclass(typ, bool):
            variables.append(Symbolic(prefix, Set.from_iterable([True, False])))
        elif issubclass(typ, Enum):
            variables.append(Symbolic(prefix, Set.from_iterable(list(typ))))
        elif issubclass(typ, int):
            variables.append(Integer(prefix))
        elif issubclass(typ, float):
            variables.append(Continuous(prefix))
        else:
            raise NotImplementedError(
                f"No conversion between {typ} and random_events.Variable"
            )

        return variables

    def plan_from_sample(
        self, model: ProbabilisticModel, sample: np.ndarray, world: World
    ) -> Plan:
        """
        Reconstruct a sequential plan from a sample array of all variables.

        :param model: Probabilistic model used to generate the sample
        :param sample: Sampled values
        :param world: World context to instantiate plan
        :return: SequentialPlan object with resolved actions
        """
        context = Context(
            world, world.get_semantic_annotations_by_type(AbstractRobot)[0], None
        )
        plan = SequentialPlan(context)

        for node in self.variables_of_node:
            values = [
                sample[model.variables.index(var)]
                for var in self.variables_of_node[node]
            ]
            resolved = node.designator_type.reconstruct(values)

            if isinstance(resolved, NavigateAction):
                resolved.target_location.frame_id = world.root

            kwargs = {k: getattr(resolved, k) for k in resolved._parameters}

            plan.add_edge(
                plan.root,
                ResolvedActionNode(
                    designator_ref=resolved,
                    kwargs=kwargs,
                    designator_type=resolved.__class__,
                ),
            )

        return plan

    def create_fully_factorized_distribution(self) -> ProbabilisticCircuit:
        """Return a fully factorized distribution over all plan variables."""
        distribution = fully_factorized(
            self.variables,
            means={v: 0 for v in self.variables},
            variances={v: 1 for v in self.variables},
        )
        return distribution

    def create_restrictions(self) -> SimpleEvent:
        """
        Generate a SimpleEvent representing restrictions from plan nodes.

        :return: SimpleEvent object
        """
        restrictions: Dict[Variable, Any] = {}
        for node, variables in self.variables_of_node.items():
            for variable in variables:
                param_name = variable.name.split(".", 1)[1]
                restriction = node.kwargs.get(param_name, None)
                if restriction is not None:
                    restrictions[variable] = restriction

        event = SimpleEvent(restrictions)
        event.fill_missing_variables(self.variables)
        return event


def collision_free_event(
    world: World, search_space: Optional[BoundingBoxCollection] = None
) -> Event:
    """
    Generate an event describing collision-free regions in the world.
    """
    xy = SpatialVariables.xy

    if search_space is None:
        search_space = BoundingBoxCollection(
            [
                BoundingBox(
                    -np.inf,
                    -np.inf,
                    -np.inf,
                    np.inf,
                    np.inf,
                    np.inf,
                    origin=HomogeneousTransformationMatrix(reference_frame=world.root),
                )
            ]
        )

    search_event = search_space.event
    obstacles = GraphOfConvexSets.obstacles_from_world(world, search_space)

    free_space = search_event - obstacles
    free_space = free_space.marginal(xy)

    z_event = SimpleEvent({SpatialVariables.z.value: singleton(0.0)}).as_composite_set()
    z_event.fill_missing_variables(xy)
    free_space.fill_missing_variables(SortedSet([SpatialVariables.z.value]))
    free_space &= z_event

    return free_space


def update_variables_of_simple_event(
    event: SimpleEvent, new_variables: Dict[Variable, Variable]
) -> SimpleEvent:
    """
    Replace variables in a SimpleEvent according to a mapping.
    """
    return SimpleEvent(
        {new_variables.get(var, var): value for var, value in event.items()}
    )


def update_variables_of_event(
    event: Event, new_variables: Dict[Variable, Variable]
) -> Event:
    """
    Replace variables in all SimpleEvents of an Event according to a mapping.
    """
    return Event(
        [
            update_variables_of_simple_event(se, new_variables)
            for se in event.simple_sets
        ]
    )
