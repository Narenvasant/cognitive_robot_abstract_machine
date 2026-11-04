import os

import pytest

from krrood.entity_query_language.factories import entity, variable, an
from krrood.symbol_graph.symbol_graph import SymbolGraph
from ..dataset.example_classes import KRROODPosition

try:
    import pydot
    import pygraphviz
except ImportError:
    pydot = None
    pygraphviz = None


@pytest.mark.skipif(
    not (pydot and pygraphviz), reason="pydot and graphviz not installed"
)
def test_visualize_symbol_graph():
    SymbolGraph().clear()
    symbol_graph = SymbolGraph()
    symbol_graph.to_dot("symbol_graph.svg", format_="svg", graph_type="type")
    assert len(symbol_graph._class_diagram.wrapped_classes) >= 59
    if os.path.exists("symbol_graph.svg"):
        os.remove("symbol_graph.svg")


def test_memory_leak():
    """
    Test if the SymbolGraph does not artificially keep objects alive that would be
    garbage collected.
    """

    def create_data():
        point = KRROODPosition(1, 2, 3)
        return point

    create_data()

    q = an(entity(variable(KRROODPosition, domain=None)))
    result = list(q.evaluate())

    assert result == []

    assert len(SymbolGraph().wrapped_instances) == 0


def test_dead_instance_is_forgotten_by_the_instance_index():
    """
    Test that removing the dead instances also forgets them in the index from an
    instance's identity to its wrapper, so a new object reusing that identity is not
    taken for the dead one.
    """

    def identity_of_dead_position() -> int:
        position = KRROODPosition(1, 2, 3)
        return id(position)

    identity = identity_of_dead_position()

    SymbolGraph().remove_dead_instances()

    assert identity not in SymbolGraph()._instance_index
