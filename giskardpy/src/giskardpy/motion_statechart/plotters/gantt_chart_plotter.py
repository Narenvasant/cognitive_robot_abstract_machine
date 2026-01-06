from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, TYPE_CHECKING, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from giskardpy.middleware import get_middleware
from giskardpy.motion_statechart.context import ExecutionContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.graph_node import Goal, MotionStatechartNode
from giskardpy.motion_statechart.plotters.styles import (
    LiftCycleStateToColor,
    ObservationStateToColor,
)
from giskardpy.utils.utils import create_path

if TYPE_CHECKING:
    from giskardpy.motion_statechart.motion_statechart import (
        MotionStatechart,
    )


@dataclass
class HistoryGanttChartPlotter:
    """
    Plot a hierarchy-aware Gantt chart of node states.

    Shows parent-child relationships of Goals by ordering rows in
    preorder and by prefixing labels with tree glyphs (├─, └─, │).
    Optional background bands and goal outlines emphasize grouping.
    """

    motion_statechart: MotionStatechart
    context: ExecutionContext | None = None
    second_width_in_cm: float = 2.0

    final_block_buffer: float = 1
    final_block_size: float = 2

    @property
    def x_width_per_control_cycle(self) -> float:
        if self.context is None:
            return 1
        return self.context.dt

    @property
    def total_control_cycles(self) -> int:
        return self.motion_statechart.history.history[-1].control_cycle

    @property
    def x_max(self) -> float:
        return (
            self.total_control_cycles + self.final_block_buffer + self.final_block_size
        )

    @property
    def num_bars(self) -> int:
        return len(self.motion_statechart.history.history[0].life_cycle_state)

    @property
    def use_seconds_for_x_axis(self) -> bool:
        return self.x_width_per_control_cycle != 1.0

    @property
    def figure_height(self) -> float:
        return 0.7 + self.num_bars * 0.25

    @property
    def figure_width(self) -> float:
        if not self.use_seconds_for_x_axis:
            return 0.5 * float((self.total_control_cycles or 0) + 1)
        # 1 inch = 2.54 cm; map seconds to figure width via second_length_in_cm
        inches_per_second = self.second_width_in_cm / 2.54
        return inches_per_second * self.time_span_seconds

    @property
    def time_span_seconds(self) -> float | None:
        return (
            self.total_control_cycles * self.x_width_per_control_cycle
            if self.x_width_per_control_cycle
            else None
        )

    def plot_gantt_chart(self, file_name: str) -> None:
        """
        Render the Gantt chart and save it.

        The chart shows life cycle (top half) and observation state (bottom half)
        per node over time. If a context with dt is provided, the x-axis is in seconds; otherwise, control cycles are used.

        This renders two side-by-side plots:
        - Left: the normal timeline over control cycles or seconds
        - Right: a compact column showing only the final state for each node, with the x label "final"
        Y-axis labels are shown only once on the right plot.
        """

        nodes = self.motion_statechart.nodes
        if len(nodes) == 0:
            get_middleware().logwarn(
                "Gantt chart skipped: no nodes in motion statechart."
            )
            return

        history = self.motion_statechart.history.history
        if len(history) == 0:
            get_middleware().logwarn("Gantt chart skipped: empty StateHistory.")
            return

        ordered_nodes = self._sort_nodes_by_parents()

        # Create two subplots: main timeline and final-state-only column
        fig, (ax_main, ax_final) = plt.subplots(
            1,
            2,
            gridspec_kw={"width_ratios": [10, 1]},
            sharey=True,
            figsize=(self.figure_width, self.figure_height),
        )
        ax_main.grid(True, axis="x", zorder=-1)

        for node_idx, node in enumerate(ordered_nodes):
            self._plot_lifecycle_bar(ax=ax_main, node=node, node_idx=node_idx)
            self._plot_observation_bar(ax=ax_main, node=node, node_idx=node_idx)

            # Draw the final-state-only blocks on the right axis
            self._plot_final_state_column(ax=ax_final, node=node, node_idx=node_idx)

        self._format_axes(
            ax_main=ax_main, ax_final=ax_final, ordered_nodes=ordered_nodes
        )
        self._save_figure(file_name=file_name)

    def _sort_nodes_by_parents(self) -> List[MotionStatechartNode]:

        def return_children_in_order(n: MotionStatechartNode):
            yield n
            if isinstance(n, Goal):
                for c in n.nodes:
                    yield from return_children_in_order(c)

        ordered_: List[MotionStatechartNode] = []
        for root in self.motion_statechart.top_level_nodes:
            ordered_.extend(list(return_children_in_order(root)))
        # reverse list because plt plots bars bottom to top
        return list(reversed(ordered_))

    def _plot_lifecycle_bar(
        self,
        ax: plt.Axes,
        node: MotionStatechartNode,
        node_idx: int,
    ):
        life_cycle_history = (
            self.motion_statechart.history.get_life_cycle_history_of_node(node)
        )
        control_cycle_indices = [
            h.control_cycle for h in self.motion_statechart.history.history
        ]
        self._plot_node_bar(
            ax=ax,
            node_idx=node_idx,
            history=life_cycle_history,
            control_cycle_indices=control_cycle_indices,
            color_map=LiftCycleStateToColor,
            top=True,
        )

    def _plot_observation_bar(
        self,
        ax: plt.Axes,
        node: MotionStatechartNode,
        node_idx: int,
    ):
        obs_history = self.motion_statechart.history.get_observation_history_of_node(
            node
        )
        control_cycle_indices = [
            h.control_cycle for h in self.motion_statechart.history.history
        ]
        self._plot_node_bar(
            ax=ax,
            node_idx=node_idx,
            history=obs_history,
            control_cycle_indices=control_cycle_indices,
            color_map=ObservationStateToColor,
            top=False,
        )

    def _plot_final_state_column(
        self,
        ax: plt.Axes,
        node: MotionStatechartNode,
        node_idx: int,
        column_padding: float = 0.1,
    ) -> None:
        """
        Draw the final state for both lifecycle (top half) and observation (bottom half)
        as a compact column on the right axes.
        """
        # Determine last lifecycle and observation states
        life_cycle_history = (
            self.motion_statechart.history.get_life_cycle_history_of_node(node)
        )
        obs_history = self.motion_statechart.history.get_observation_history_of_node(
            node
        )
        last_lifecycle = life_cycle_history[-1]
        last_observation = obs_history[-1]

        # Column spans from padding to 1 - padding
        width = max(0.0, 1.0 - 2 * column_padding)
        start = column_padding

        # Draw top (lifecycle) and bottom (observation) halves
        self._draw_block(
            ax=ax,
            node_idx=node_idx,
            block_start=start,
            block_width=width,
            color=LiftCycleStateToColor[last_lifecycle],
            top=True,
        )
        self._draw_block(
            ax=ax,
            node_idx=node_idx,
            block_start=start,
            block_width=width,
            color=ObservationStateToColor[last_observation],
            top=False,
        )

    def _plot_node_bar(
        self,
        ax: plt.Axes,
        node_idx: int,
        history: List[LifeCycleValues | ObservationStateValues],
        control_cycle_indices: List[int],
        color_map: Dict[LifeCycleValues | ObservationStateValues, str],
        top: bool,
    ) -> None:
        current_state = history[0]
        start_idx = 0
        for idx, next_state in zip(control_cycle_indices[1:], history[1:]):
            if current_state != next_state:
                life_cycle_width = (idx - start_idx) * self.x_width_per_control_cycle
                self._draw_block(
                    ax=ax,
                    node_idx=node_idx,
                    block_start=start_idx * self.x_width_per_control_cycle,
                    block_width=life_cycle_width,
                    color=color_map[current_state],
                    top=top,
                )
                start_idx = idx
                current_state = next_state
        # plot last stretch until final index
        last_idx = control_cycle_indices[-1]
        life_cycle_width = (last_idx - start_idx) * self.x_width_per_control_cycle
        self._draw_block(
            ax=ax,
            node_idx=node_idx,
            block_start=start_idx * self.x_width_per_control_cycle,
            block_width=life_cycle_width,
            color=color_map[current_state],
            top=top,
        )

    def _draw_block(
        self,
        ax: plt.Axes,
        node_idx,
        block_start,
        block_width,
        color,
        top: bool,
        bar_height: float = 0.8,
    ):
        if top:
            y = node_idx + bar_height / 4
        else:
            y = node_idx - bar_height / 4
        ax.barh(
            y,
            block_width,
            height=bar_height / 2,
            left=block_start,
            color=color,
            zorder=2,
        )

    def _format_axes(
        self,
        ax_main: plt.Axes,
        ax_final: plt.Axes,
        ordered_nodes: List[MotionStatechartNode],
    ) -> None:
        # Configure x-axis for main timeline
        if self.use_seconds_for_x_axis and self.time_span_seconds is not None:
            ax_main.set_xlabel("Time [s]")
            base_ticks = np.arange(0.0, self.time_span_seconds + 1e-9, 0.5).tolist()
            ax_main.set_xlim(0, self.time_span_seconds)
            ax_main.set_xticks(base_ticks)
            ax_main.set_xticklabels([str(t) for t in base_ticks])
        else:
            ax_main.set_xlabel("Control cycle")
            step = max(int(self.x_width_per_control_cycle), 1)
            base_ticks = list(range(0, self.total_control_cycles + 1, step))
            ax_main.set_xlim(0, self.total_control_cycles)
            ax_main.set_xticks(base_ticks)
            ax_main.set_xticklabels([str(t) for t in base_ticks])

        # Configure final-state column x-axis
        ax_final.set_xlim(0.0, 1.0)
        ax_final.set_xticks([0.5])
        ax_final.set_xticklabels(["final"])
        ax_final.grid(False)

        # Y axis labels and limits shown only on the right (final column)
        num_bars = len(self.motion_statechart.history.history[0].life_cycle_state)
        ymin, ymax = -0.8, num_bars - 1 + 0.8
        ax_final.set_ylim(ymin, ymax)

        node_names = []
        for idx, n in enumerate(ordered_nodes):
            prev_depth = 0 if idx == 0 else ordered_nodes[idx - 1].depth
            node_names.append(self._make_label(n, prev_depth))
        node_idx = list(range(len(node_names)))

        ax_main.set_yticks([])
        ax_main.set_ylabel("Nodes")

        ax_final.set_yticks(node_idx, node_names)
        ax_final.set_ylabel("")
        ax_final.yaxis.tick_right()

        plt.tight_layout()

    def _make_label(self, node: MotionStatechartNode, prev_depth: int) -> str:
        depth = node.depth
        if depth == 0:
            return node.unique_name
        diff = depth - prev_depth
        if diff > 0:
            return (
                "│  " * (depth - diff)
                + "└─"  # no space because the formatting is weird otherwise
                * (diff - 1)
                + "└─ "
                + node.unique_name
            )
        else:
            return "│  " * (depth - 1) + "├─ " + node.unique_name

    def _save_figure(self, file_name: str) -> None:
        create_path(file_name)
        plt.savefig(file_name)
        plt.close()
        get_middleware().loginfo(f"Saved gantt chart to {file_name}.")
