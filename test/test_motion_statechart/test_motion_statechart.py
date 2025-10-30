from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
)
from giskardpy.motion_statechart.motion_statechart_graph import MotionStatechartGraph
import semantic_digital_twin.spatial_types.spatial_types as cas
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName


def test_motion_statechart():
    msg = MotionStatechartGraph()

    node1 = MotionStatechartNode(name=PrefixedName("muh"), motion_statechart=msg)
    node2 = MotionStatechartNode(name=PrefixedName("muh2"), motion_statechart=msg)
    node3 = MotionStatechartNode(name=PrefixedName("muh3"), motion_statechart=msg)

    node1.start_condition = cas.trinary_logic_or(node3, node2)
    assert len(msg.nodes) == 3
    assert len(msg.edges) == 2

    msg.compile()
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.NOT_STARTED
    msg.tick()
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.RUNNING
    msg.tick()
    assert msg.life_cycle_state[node1] == msg.life_cycle_state.NOT_STARTED
    assert msg.life_cycle_state[node2] == msg.life_cycle_state.RUNNING
    assert msg.life_cycle_state[node3] == msg.life_cycle_state.RUNNING


def test_duplicate_name(): ...
