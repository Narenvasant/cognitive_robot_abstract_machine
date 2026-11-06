"""
The Tracy clutter picking experiment shown in cramera: every pick runs live in the
viewer against MuJoCo physics, is recorded as a video, a NEEM and a cramera recording,
and every failed pick is explained by the causal answers the recorded attempts give.

What the cramera causal-diagnosis plugin shows is written as JSON documents into one
experiment directory, ``~/.cramera/experiments/tracy_clutter_picking`` by default:

- ``experiment.json``: the recorded attempts and every causal question's answer;
- ``causal_graph.json``: the causal model the questions assume;
- ``probability_tree.json``: a joint probability tree over the recorded attempts;
- ``runs/index.json`` and, per run, ``runs/<run>/answers.json`` (what the causal
  answers say about the run, written when it starts and completed when it ends),
  ``neem.json``, ``diagnosis.json`` and ``video.mp4``.

Run with (the ``iai_tracy_description`` ROS package must be built and sourced, and the
cramera bridge on the path)::

    MUJOCO_GL=egl python -m experiments.causal_reasoning.tracy_clutter_picking.presentation --runs 5
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import imageio
import mujoco
import numpy as np
import pandas as pd
from typing_extensions import Any, Callable, Dict, List, Optional

import experiments.orm.ormatic_interface  # noqa: F401  # maps the domain for fitting
from cramph.executor import ExecutorExtension, TickingExecutorExtension
from cramera_bridge.visualization import LiveVisualization
from cramera_protocol.live_messages import (
    RecordingSaveRequest,
    RecordingState,
    SceneDestination,
)
from experiments.causal_reasoning.do_query import CauseRegion, DoQueryAnswer
from experiments.causal_reasoning.tracy_clutter_picking.backdrop import KitchenBackdrop
from experiments.causal_reasoning.tracy_clutter_picking.dataset import (
    ClutterPickDataset,
    recorded_attempts,
)
from experiments.causal_reasoning.tracy_clutter_picking.do_query import (
    ClosingAxisSideCausesDisturbance,
    ClutterDoQuery,
    ClutterQuestion,
    CrowdingCausesLift,
    FrictionCausesLift,
)
from experiments.causal_reasoning.tracy_clutter_picking.domain import (
    ClosingAxisSide,
    ClutterPickOutcome,
    ClutterPickScene,
    ClutterSceneLayout,
    DistanceBand,
)
from experiments.causal_reasoning.tracy_clutter_picking.episode import PickEpisode
from experiments.causal_reasoning.tracy_clutter_picking.exceptions import (
    EpisodePlanningFailedError,
)
from experiments.causal_reasoning.tracy_clutter_picking.explanations import (
    AnswerRegions,
    DiagnosisStory,
    GeneralReading,
    QuestionKind,
    RunCauses,
    RunReading,
    crowding_of,
    crowding_text,
    friction_text,
    percent,
    points,
)
from experiments.causal_reasoning.tracy_clutter_picking.layout_sampler import (
    ClutterLayoutSampler,
)
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProductUnit,
)
from semantic_digital_twin.adapters.multi_sim import MujocoSim

logger = logging.getLogger(__name__)


# %% where the documents go


class ExperimentFile(StrEnum):
    """
    The files of an experiment directory the plugin reads.
    """

    EXPERIMENT = "experiment.json"
    CAUSAL_GRAPH = "causal_graph.json"
    PROBABILITY_TREE = "probability_tree.json"
    RUNS = "runs"
    RUN_INDEX = "index.json"
    NEEM = "neem.json"
    DIAGNOSIS = "diagnosis.json"
    ANSWERS = "answers.json"
    VIDEO = "video.mp4"


@dataclass
class ExperimentDirectory:
    """
    The directory one experiment's documents are written to.
    """

    path: Path = field(
        default_factory=lambda: Path.home()
        / ".cramera"
        / "experiments"
        / "tracy_clutter_picking"
    )
    """
    The directory.
    """

    def write(self, relative: str | Path, document: Any) -> Path:
        """
        Write a document as JSON, replacing what was there in one step so a reader never
        sees half a file.

        :param relative: Where, below :attr:`path`.
        :param document: What to write.
        :return: The file written.
        """
        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".partial")
        partial.write_text(json.dumps(document, indent=2, default=str))
        partial.replace(target)
        return target

    def run_directory(self, run: str) -> Path:
        """
        :param run: A run's identifier.
        :return: Its directory, created.
        """
        directory = self.path / ExperimentFile.RUNS / run
        directory.mkdir(parents=True, exist_ok=True)
        return directory


# %% the causal analysis of the recorded attempts


@dataclass
class CausalAnalysis:
    """
    The answers the recorded attempts give to the experiment's causal questions, as the
    experiment document the plugin reads; each region of a cause also lists which of
    the cause's possible values it holds, so a run can be looked up in it.
    """

    document: Dict[str, Any]
    """
    The experiment document.
    """

    @classmethod
    def of(cls, dataset: ClutterPickDataset) -> CausalAnalysis:
        """
        :param dataset: The recorded attempts.
        :return: Their answers to the friction, crowding and closing-axis questions.
        """
        count = dataset.recorded_neighbour_count
        frictions = sorted({scene.friction_coefficient for scene in dataset.scenes})
        crowdings = list(range(count + 1))
        sides = [side.value for side in ClosingAxisSide]
        asked = [
            (
                QuestionKind.FRICTION,
                FrictionCausesLift(neighbour_count=count),
                frictions,
            ),
            (
                QuestionKind.CROWDING_ADJUSTED,
                CrowdingCausesLift(neighbour_count=count),
                crowdings,
            ),
            (
                QuestionKind.CROWDING_UNADJUSTED,
                CrowdingCausesLift(neighbour_count=count, adjusted_for=()),
                crowdings,
            ),
            (
                QuestionKind.CLOSING_AXIS_SIDE,
                ClosingAxisSideCausesDisturbance(neighbour_count=count),
                sides,
            ),
        ]
        questions = []
        for kind, question, candidates in asked:
            logger.info("asking: %s", question.asked)
            answer = ClutterDoQuery(question).run(dataset.scenes)
            questions.append(cls.question_document(kind, question, answer, candidates))
        return cls(cls.experiment_document(dataset, questions))

    @staticmethod
    def question_document(
        kind: QuestionKind,
        question: ClutterQuestion,
        answer: DoQueryAnswer,
        candidates: List[Any],
    ) -> Dict[str, Any]:
        """
        :param kind: Which question it is.
        :param question: The question.
        :param answer: Its answer.
        :param candidates: The values the cause can take.
        :return: The question, its answer per region, and the values each region holds.
        """
        return {
            "kind": kind.value,
            "name": question.name,
            "asked": question.asked,
            "adjustedFor": list(question.adjustment_variable_names),
            "trainingExamples": answer.training_example_count,
            "largestShiftFromAdjusting": answer.largest_shift_from_adjusting,
            "regions": [
                {
                    "description": region.description,
                    "share": region.probability,
                    "conditioned": region.conditioned_probability,
                    "adjusted": region.adjusted_probability,
                    "holds": [
                        candidate
                        for candidate in candidates
                        if CausalAnalysis.holds(region, candidate)
                    ],
                }
                for region in answer.regions
            ],
        }

    @staticmethod
    def holds(region: CauseRegion, candidate: Any) -> bool:
        """
        :param region: A region of a cause.
        :param candidate: One value of the cause.
        :return: Whether the region holds it; a symbolic value is read off the region's
            description.
        """
        if isinstance(candidate, str):
            return candidate in region.description
        return candidate in region.cause_values

    @staticmethod
    def experiment_document(
        dataset: ClutterPickDataset, questions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        :param dataset: The recorded attempts.
        :param questions: The questions with their answers.
        :return: The experiment's summary and every answer.
        """
        scenes = dataset.scenes
        per_environment = Counter(scene.environment.value for scene in scenes)
        lifted_per_environment = Counter(
            scene.environment.value for scene in scenes if scene.lifted
        )
        return {
            "title": "Tracy picks a milk carton out of a clutter",
            "summary": (
                "Tracy's left arm picks one milk carton out of a clutter of ten in "
                "MuJoCo, held by the fingertip pads' friction alone. A relational "
                "circuit fitted on the recorded attempts answers causal questions "
                "about why a pick succeeds, adjusting for the environment that "
                "confounds friction and crowding."
            ),
            "dataset": {
                "attempts": len(scenes),
                "successRate": dataset.success_rate,
                "neighbourCount": dataset.recorded_neighbour_count,
                "environments": [
                    {
                        "environment": environment,
                        "attempts": attempts,
                        "successRate": lifted_per_environment[environment] / attempts,
                    }
                    for environment, attempts in sorted(per_environment.items())
                ],
            },
            "questions": questions,
        }

    def question(self, kind: QuestionKind) -> Dict[str, Any]:
        """
        :param kind: Which question.
        :return: Its document.
        """
        [question] = [
            question
            for question in self.document["questions"]
            if question["kind"] == kind.value
        ]
        return question


# %% the causal model the questions assume


class CausalNodeKind(StrEnum):
    """
    What a node of the causal graph is in the experiment's questions.
    """

    CONFOUNDER = "confounder"
    CAUSE = "cause"
    EFFECT = "effect"
    PART = "part"


def causal_graph_document() -> Dict[str, Any]:
    """
    :return: The causal model the experiment's questions assume: the environment decides
        both the grasp friction and how crowded the clutter is, both of which decide
        whether the target comes up, and a neighbour's side of the closing axis decides
        whether the closing fingers disturb it.
    """
    nodes = [
        ("environment", "Environment", CausalNodeKind.CONFOUNDER),
        ("friction", "Grasp friction", CausalNodeKind.CAUSE),
        ("crowding", "Crowding", CausalNodeKind.CAUSE),
        ("grasp_yaw", "Grasp yaw", CausalNodeKind.CAUSE),
        ("lifted", "Target lifted", CausalNodeKind.EFFECT),
        ("side", "Side of closing axis", CausalNodeKind.PART),
        ("disturbed", "Neighbour disturbed", CausalNodeKind.PART),
    ]
    edges = [
        ("environment", "friction"),
        ("environment", "crowding"),
        ("friction", "lifted"),
        ("crowding", "lifted"),
        ("grasp_yaw", "lifted"),
        ("grasp_yaw", "side"),
        ("side", "disturbed"),
    ]
    return {
        "nodes": [
            {"id": identifier, "label": label, "kind": kind.value}
            for identifier, label, kind in nodes
        ],
        "edges": [{"source": source, "target": target} for source, target in edges],
        "adjustments": [
            {
                "cause": "friction",
                "effect": "lifted",
                "adjustedFor": ["environment"],
                "why": "The environment decides both how slippery and how crowded an "
                "attempt is.",
            },
            {
                "cause": "crowding",
                "effect": "lifted",
                "adjustedFor": ["environment"],
                "why": "The environment decides both how crowded and how slippery an "
                "attempt is.",
            },
        ],
    }


# %% a joint probability tree over the recorded attempts


@dataclass
class ProbabilityTreeExport:
    """
    A joint probability tree fitted on the recorded attempts' own columns, with leaves
    large enough to read.
    """

    dataset: ClutterPickDataset
    """
    The recorded attempts.
    """

    minimum_samples_per_leaf: int = 20
    """
    How many attempts a leaf holds at least.
    """

    def table(self) -> pd.DataFrame:
        """
        :return: One row per recorded attempt: its environment, friction, crowding,
            grasp yaw and whether the target came up.
        """
        return pd.DataFrame(
            {
                "environment": [
                    scene.environment.value for scene in self.dataset.scenes
                ],
                "friction": [
                    scene.friction_coefficient for scene in self.dataset.scenes
                ],
                "crowding": [crowding_of(scene) for scene in self.dataset.scenes],
                "grasp_yaw": [scene.grasp_yaw for scene in self.dataset.scenes],
                "lifted": [str(scene.lifted) for scene in self.dataset.scenes],
            }
        )

    def document(self) -> Dict[str, Any]:
        """
        :return: The tree's leaves, each with its share of the attempts, what holds for
            its attempts and how often the target came up in them.
        """
        table = self.table()
        learner = JointProbabilityTree(
            minimum_samples_per_leaf=self.minimum_samples_per_leaf,
            keep_sample_indices=True,
        )
        circuit = learner.fit(table, infer_variables_from_dataframe(table))
        leaves = [
            self.leaf_document(table.iloc[leaf.sample_indices], len(table))
            for leaf in circuit.root.subcircuits
            if isinstance(leaf, ProductUnit)
        ]
        leaves.sort(key=lambda leaf: -leaf["weight"])
        return {
            "attempts": len(table),
            "minimumSamplesPerLeaf": self.minimum_samples_per_leaf,
            "variables": list(table.columns),
            "leaves": leaves,
        }

    @staticmethod
    def leaf_document(rows: pd.DataFrame, attempt_count: int) -> Dict[str, Any]:
        """
        :param rows: The attempts a leaf holds.
        :param attempt_count: How many attempts the tree was fitted on.
        :return: Its share, what each variable is in it, and how often the target came
            up.
        """
        conditions = []
        for column in rows.columns:
            if column == "lifted":
                continue
            values = rows[column]
            if values.dtype == object:
                text = ", ".join(sorted(set(values)))
            elif values.min() == values.max():
                text = f"= {values.min():.4g}"
            else:
                text = f"{values.min():.4g} – {values.max():.4g}"
            conditions.append({"variable": column, "values": text})
        return {
            "weight": len(rows) / attempt_count,
            "samples": len(rows),
            "conditions": conditions,
            "liftedProbability": float((rows["lifted"] == "True").mean()),
        }


# %% why a pick failed


@dataclass
class FactorAssessment:
    """
    How one cause of the lift stood in a run, by the causal answer about it.
    """

    factor: str
    """
    The cause.
    """

    observed: str
    """
    Its value in the run, in words.
    """

    probability_at_observed: Optional[float]
    """
    How likely the target comes up under an intervention setting the cause to that
    value, adjusted for the environment; None where the recorded attempts never had it.
    """

    best_value: str
    """
    The cause's value under which the target comes up most often, in words.
    """

    probability_at_best: float
    """
    How likely the target comes up under that value.
    """

    @property
    def gap(self) -> float:
        """
        :return: How much setting the cause to its best value would raise the chance of
            the lift.
        """
        if self.probability_at_observed is None:
            return 0.0
        return self.probability_at_best - self.probability_at_observed


@dataclass
class FailureDiagnosis:
    """
    Explains a run by the causal answers: which cause made the lift least likely, and
    what would have helped.
    """

    analysis: CausalAnalysis
    """
    The causal answers of the recorded attempts.
    """

    @staticmethod
    def assess(
        factor: str,
        value: Any,
        question: Dict[str, Any],
        text: Callable[[Any], str],
    ) -> FactorAssessment:
        """
        :param factor: The cause's name.
        :param value: Its value in the run.
        :param question: The causal answer about the cause.
        :param text: Puts a value of the cause in words.
        :return: How the cause stood in the run.
        """
        regions = AnswerRegions(question)
        here, best = regions.at(value), regions.best
        return FactorAssessment(
            factor=factor,
            observed=text(value),
            probability_at_observed=None if here is None else here["adjusted"],
            best_value=text(AnswerRegions.value_of(best)),
            probability_at_best=best["adjusted"],
        )

    def diagnose(
        self,
        scene: Optional[ClutterPickScene],
        layout: ClutterSceneLayout,
        failure: Optional[str],
        answers: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        :param scene: The run as an attempt, or None if its motion failed.
        :param layout: The run's layout.
        :param failure: What stopped the run's motion, if anything did.
        :param answers: The run's answers, with its outcome.
        :return: The diagnosis document: whether the pick succeeded, the causes ranked
            by how much their best value would have raised the chance of the lift, the
            primary cause, the recommended correction and the reasoning in steps.
        """
        factors = [
            self.assess(
                "Grasp friction",
                layout.friction_coefficient,
                self.analysis.question(QuestionKind.FRICTION),
                friction_text,
            )
        ]
        if scene is not None:
            factors.append(
                self.assess(
                    "Crowding",
                    crowding_of(scene),
                    self.analysis.question(QuestionKind.CROWDING_ADJUSTED),
                    crowding_text,
                )
            )
        factors.sort(key=lambda factor: -factor.gap)
        lifted = scene is not None and scene.lifted
        document: Dict[str, Any] = {
            "lifted": lifted,
            "motionFailure": failure,
            "adjustedFor": ["environment"],
            "factors": [
                dict(dataclasses.asdict(factor), gap=factor.gap) for factor in factors
            ],
            "disturbedNeighbours": self.disturbed_neighbours(scene),
        }
        if lifted:
            document["verdict"] = "The pick worked: the carton came up."
        elif failure is not None:
            document["verdict"] = "The motion stopped before the grasp was tried."
        else:
            primary = factors[0]
            document["primaryCause"] = primary.factor
            document["verdict"] = (
                f"The pick failed, mainly because of the {primary.factor.lower()}."
            )
            document["explanation"] = (
                f"With the {primary.factor.lower()} at {primary.observed}, a pick lifts "
                f"the carton {percent(primary.probability_at_observed)} of the time; at "
                f"{primary.best_value} it would lift it "
                f"{percent(primary.probability_at_best)} of the time."
            )
            change = {
                "Grasp friction": f"Grip with a friction of {primary.best_value}",
                "Crowding": "Clear the cartons right next to the target before the "
                "pick",
            }[primary.factor]
            document["correction"] = (
                f"{change}: the causal answers say that alone raises the chance of the "
                f"lift by {points(primary.gap)}."
            )
        document["steps"] = DiagnosisStory().steps(document, answers)
        return document

    def disturbed_neighbours(
        self, scene: Optional[ClutterPickScene]
    ) -> List[Dict[str, Any]]:
        """
        :param scene: The run as an attempt.
        :return: Each neighbour the pick shoved aside, with how likely a neighbour on
            its side of the closing axis is disturbed.
        """
        if scene is None:
            return []
        regions = AnswerRegions(self.analysis.question(QuestionKind.CLOSING_AXIS_SIDE))
        result = []
        for index, neighbour in enumerate(scene.neighbours):
            if not neighbour.disturbed:
                continue
            side = neighbour.closing_axis_side.value
            region = regions.at(side)
            result.append(
                {
                    "neighbour": index,
                    "side": side,
                    "displacement": neighbour.displacement,
                    "probabilityDisturbedOnThisSide": (
                        None if region is None else region["adjusted"]
                    ),
                }
            )
        return result


# %% recording a run


@dataclass
class NeemLog(TickingExecutorExtension):
    """
    Logs every plan node starting and ending in a run, stamped with the simulation's
    clock: the run's narrative for its NEEM.
    """

    simulation: MujocoSim = field(kw_only=True)
    """
    The simulation whose clock stamps the events.
    """

    events: List[Dict[str, Any]] = field(default_factory=list)
    """
    The logged events, in order.
    """

    reported: Any = field(default=None, init=False)
    """
    The newest state of the statechart whose changes were logged.
    """

    def after_tick(self, executor: Any) -> None:
        history = executor.statechart.state_history
        moment = history.latest_item
        if moment is None or moment is self.reported:
            return
        self.reported = moment
        now = self.simulation.simulator.current_simulation_time
        for node in history.nodes_started_in_latest_item():
            self.events.append(self.event(node, "started", now))
        for node in history.nodes_ended_in_latest_item():
            self.events.append(self.event(node, "ended", now))

    @staticmethod
    def event(node: Any, what: str, now: float) -> Dict[str, Any]:
        """
        :param node: A plan node.
        :param what: Whether it started or ended.
        :param now: The simulation time.
        :return: The event.
        """
        return {
            "time": round(float(now), 3),
            "event": what,
            "node": type(node).__name__,
            "state": (
                str(node.life_cycle_state.name)
                if hasattr(node, "life_cycle_state")
                else ""
            ),
        }


@dataclass
class EpisodeVideo:
    """
    Renders the scene camera into a video while a run's physics advances, one frame per
    thirtieth of a second of simulated time.
    """

    path: Path
    """
    The video file.
    """

    camera_name: str
    """
    The camera rendered.
    """

    frames_per_second: int = 30
    """
    Frames per second of simulated time.
    """

    width: int = 640
    """
    Width of a frame, in pixels.
    """

    height: int = 480
    """
    Height of a frame, in pixels.
    """

    stopped: threading.Event = field(default_factory=threading.Event, init=False)
    """
    Set when the recording is to stop.
    """

    thread: Optional[threading.Thread] = field(default=None, init=False)
    """
    The thread rendering the frames.
    """

    frame_count: int = field(default=0, init=False)
    """
    How many frames were written.
    """

    def start(self, simulation: MujocoSim) -> None:
        """
        :param simulation: The running simulation to render.
        """
        self.thread = threading.Thread(
            target=self.record, args=(simulation,), daemon=True, name="episode-video"
        )
        self.thread.start()

    def stop(self) -> None:
        """
        Stop recording and finish the file.
        """
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()

    def record(self, simulation: MujocoSim) -> None:
        """
        Render one frame per :attr:`frames_per_second` of simulated time with a single
        renderer kept for the whole run.

        :param simulation: The running simulation to render.
        """
        simulator = simulation.simulator
        writer = imageio.get_writer(
            self.path, fps=self.frames_per_second, codec="libx264", quality=8
        )
        renderer = mujoco.Renderer(simulator._mj_model, self.height, self.width)
        next_frame_time = 0.0
        try:
            while not self.stopped.is_set():
                now = simulator.current_simulation_time
                if now < next_frame_time:
                    time.sleep(0.002)
                    continue
                next_frame_time = now + 1.0 / self.frames_per_second
                with simulator._model_lock:
                    renderer.update_scene(simulator._mj_data, self.camera_name)
                writer.append_data(renderer.render())
                self.frame_count += 1
        finally:
            renderer.close()
            writer.close()


@dataclass
class PresentedEpisode(PickEpisode):
    """
    A pick whose plan is published to cramera live, logged for its NEEM and recorded as
    a video.
    """

    visualization: Optional[LiveVisualization] = field(default=None, init=False)
    """
    The live visualization of the run.
    """

    neem: Optional[NeemLog] = field(default=None, init=False)
    """
    The run's event log.
    """

    video: Optional[EpisodeVideo] = field(default=None, init=False)
    """
    The run's video.
    """

    def executor_extensions(self) -> List[ExecutorExtension]:
        self.neem = NeemLog(simulation=self.simulation)
        extensions: List[ExecutorExtension] = [self.neem]
        if self.visualization is not None:
            extensions.append(self.visualization.progress())
        return extensions

    def _pick(self, simulation: MujocoSim) -> None:
        if self.video is not None:
            self.video.start(simulation)
        super()._pick(simulation)

    def perform(self) -> ClutterPickOutcome:
        """
        Does what :meth:`PickEpisode.perform` does, and finishes the video once the
        outcome is measured.
        """
        try:
            return super().perform()
        finally:
            if self.video is not None:
                self.video.stop()


# %% the presentation


@dataclass
class Presentation:
    """
    Writes the experiment's analysis, then runs picks one after the other, each live in
    cramera, recorded and, where it fails, diagnosed.
    """

    directory: ExperimentDirectory = field(default_factory=ExperimentDirectory)
    """
    Where the documents go.
    """

    seed: int = 0
    """
    Seed of the runs' layouts.
    """

    live_port: int = 8765
    """
    The port the live visualization is served on.
    """

    pause_between_runs: float = 3.0
    """
    How long, in seconds, the viewer shows a finished run before the next one starts.
    """

    backdrop: Optional[KitchenBackdrop] = None
    """
    The kitchen stood around Tracy in every run, if any.
    """

    runs: List[Dict[str, Any]] = field(default_factory=list, init=False)
    """
    The index of the runs done so far.
    """

    def prepare_analysis(self, dataset: ClutterPickDataset) -> CausalAnalysis:
        """
        Fit the causal answers and the probability tree on the recorded attempts and
        write them, each answer with its takeaway in words.

        :param dataset: The recorded attempts.
        :return: The analysis.
        """
        written = self.directory.path / ExperimentFile.EXPERIMENT
        if written.is_file():
            analysis = CausalAnalysis(json.loads(written.read_text()))
        else:
            analysis = CausalAnalysis.of(dataset)
        questions = analysis.document["questions"]
        for question in questions:
            question["takeaway"] = GeneralReading().takeaway(question, questions)
        self.directory.write(ExperimentFile.EXPERIMENT, analysis.document)
        self.directory.write(ExperimentFile.CAUSAL_GRAPH, causal_graph_document())
        self.directory.write(
            ExperimentFile.PROBABILITY_TREE, ProbabilityTreeExport(dataset).document()
        )
        return analysis

    def run(self, run_count: int) -> None:
        """
        :param run_count: How many picks to run.
        """
        dataset = recorded_attempts().load()
        analysis = self.prepare_analysis(dataset)
        diagnosis = FailureDiagnosis(analysis)
        reading = RunReading(
            {kind: analysis.question(kind) for kind in QuestionKind}, dataset.scenes
        )
        sampler = ClutterLayoutSampler(np.random.default_rng(self.seed))
        for number in range(1, run_count + 1):
            self.run_once(f"run_{number:03d}", sampler.sample(), diagnosis, reading)
            time.sleep(self.pause_between_runs)

    def run_once(
        self,
        run: str,
        layout: ClutterSceneLayout,
        diagnosis: FailureDiagnosis,
        reading: RunReading,
    ) -> None:
        """
        :param run: The run's identifier.
        :param layout: The layout picked in.
        :param diagnosis: What explains a failed pick.
        :param reading: What the causal answers say about a run.
        """
        directory = self.directory.run_directory(run)
        started = datetime.now().isoformat(timespec="seconds")
        episode = PresentedEpisode(headless=True)
        scene = episode.prepare(layout)
        if self.backdrop is not None:
            self.backdrop.add_to(scene.world)
        episode.visualization = LiveVisualization(
            scene.world, port=self.live_port
        ).start()
        episode.video = EpisodeVideo(
            directory / ExperimentFile.VIDEO, scene.camera_name
        )
        causes = RunCauses.of(layout)
        answers = reading.document(run, causes)
        answers_path = Path(ExperimentFile.RUNS) / run / ExperimentFile.ANSWERS
        self.directory.write(answers_path, answers)
        self.runs.append(
            {
                "run": run,
                "started": started,
                "state": "running",
                "environment": layout.environment.value,
                "friction": layout.friction_coefficient,
                "crowding": causes.adjacent_neighbours,
            }
        )
        self.write_index()
        failure, outcome = None, None
        try:
            outcome = episode.perform()
        except EpisodePlanningFailedError as error:
            failure = str(error)
            logger.warning("%s: the motion failed: %s", run, failure)
        recorded_scene = None if outcome is None else outcome.to_scene(layout)
        answers = reading.with_outcome(answers, recorded_scene, failure)
        self.directory.write(answers_path, answers)
        diagnosis_document = diagnosis.diagnose(
            recorded_scene, layout, failure, answers
        )
        self.directory.write(
            Path(ExperimentFile.RUNS) / run / ExperimentFile.DIAGNOSIS,
            diagnosis_document,
        )
        neem_path = Path(ExperimentFile.RUNS) / run / ExperimentFile.NEEM
        self.directory.write(
            neem_path,
            self.neem_document(
                run, started, layout, outcome, recorded_scene, episode, None, failure
            ),
        )
        self.runs[-1] = {
            "run": run,
            "started": started,
            "state": "finished",
            "lifted": bool(outcome is not None and outcome.lifted),
            "liftHeight": None if outcome is None else outcome.lift_height,
            "environment": layout.environment.value,
            "friction": layout.friction_coefficient,
            "crowding": None if recorded_scene is None else crowding_of(recorded_scene),
            "verdict": diagnosis_document["verdict"],
            "primaryCause": diagnosis_document.get("primaryCause"),
            "recording": None,
            "video": f"{ExperimentFile.RUNS}/{run}/{ExperimentFile.VIDEO}",
        }
        self.write_index()
        bundle = self.save_recording(episode.visualization, run, started)
        episode.visualization.stop()
        self.directory.write(
            neem_path,
            self.neem_document(
                run, started, layout, outcome, recorded_scene, episode, bundle, failure
            ),
        )
        self.runs[-1]["recording"] = bundle
        self.write_index()
        logger.info("%s: %s", run, diagnosis_document["verdict"])

    def write_index(self) -> None:
        """
        Write the runs' index.
        """
        self.directory.write(
            Path(ExperimentFile.RUNS) / ExperimentFile.RUN_INDEX, {"runs": self.runs}
        )

    @staticmethod
    def save_recording(
        visualization: LiveVisualization, run: str, started: str
    ) -> Optional[str]:
        """
        :param visualization: The run's live visualization.
        :param run: The run's identifier.
        :param started: When the run started, in ISO format.
        :return: The name of the cramera recording the run was saved as, if any.
        """
        name = f"tracy_clutter_{run}_{started.replace('-', '').replace(':', '')}"
        publisher = visualization.publisher
        status = publisher.recording_status()
        if status.state is not RecordingState.RECORDING or status.frame_count == 0:
            logger.warning("%s: cramera recorded no frames to save", run)
            return None
        publisher.finish_recording()
        publisher.save_recording(
            RecordingSaveRequest(name=name, destination=SceneDestination.LOCAL)
        )
        return name

    @staticmethod
    def neem_document(
        run: str,
        started: str,
        layout: ClutterSceneLayout,
        outcome: Optional[ClutterPickOutcome],
        scene: Optional[ClutterPickScene],
        episode: PresentedEpisode,
        bundle: Optional[str],
        failure: Optional[str],
    ) -> Dict[str, Any]:
        """
        :return: The run's NEEM: what was tried, what happened step by step, and what
            came of it.
        """
        return {
            "run": run,
            "started": started,
            "task": "Pick the marked milk carton out of the clutter with the left arm",
            "robot": "Tracy",
            "simulation": "MuJoCo, contact friction only, no attachment",
            "parameters": {
                "environment": layout.environment.value,
                "frictionCoefficient": layout.friction_coefficient,
                "graspYaw": layout.grasp_yaw,
                "targetIndex": layout.target_index,
                "cartons": len(layout.objects),
            },
            "events": [] if episode.neem is None else episode.neem.events,
            "outcome": (
                None
                if outcome is None
                else {
                    "lifted": outcome.lifted,
                    "liftHeight": outcome.lift_height,
                    "neighbourDisplacements": outcome.neighbour_displacements,
                }
            ),
            "neighbours": (
                []
                if scene is None
                else [
                    {
                        "category": neighbour.category.value,
                        "distanceBand": neighbour.distance_band.value,
                        "closingAxisSide": neighbour.closing_axis_side.value,
                        "displacement": neighbour.displacement,
                        "disturbed": neighbour.disturbed,
                    }
                    for neighbour in scene.neighbours
                ]
            ),
            "motionFailure": failure,
            "videoFrames": 0 if episode.video is None else episode.video.frame_count,
            "recording": bundle,
        }


def main() -> None:
    """
    Run the presentation from the command line.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--pause",
        type=float,
        default=3.0,
        help="seconds the viewer shows a finished run before the next one starts",
    )
    parser.add_argument("--directory", type=Path, default=None)
    parser.add_argument(
        "--kitchen",
        type=Path,
        default=None,
        help="the object library's objects/articulated folder, to stand a kitchen "
        "around Tracy",
    )
    arguments = parser.parse_args()
    presentation = Presentation(
        seed=arguments.seed,
        live_port=arguments.port,
        pause_between_runs=arguments.pause,
    )
    if arguments.kitchen is not None:
        presentation.backdrop = KitchenBackdrop(arguments.kitchen)
    if arguments.directory is not None:
        presentation.directory = ExperimentDirectory(arguments.directory)
    presentation.run(arguments.runs)


if __name__ == "__main__":
    main()
