# Tracy picks one milk out of ten: relational causal circuits against a flat tree

## What this experiment is for

Relational sum-product networks (RSPNs) are less expressive than most machine-learning
models. The argument this experiment makes is that they are expressive *enough*: a
relational circuit fitted on a robot's own recorded attempts can answer causal questions
about a cluttered pick -- questions a flat model cannot even pose -- and the answers
hold up on a task a robot actually executes.

Tracy's left arm picks one milk carton out of a ten-carton clutter in MuJoCo. The carton
is held by contact friction between the fingertip pads alone; there is no kinematic
attachment, so a poor grasp visibly fails. Every attempt is recorded as a relational
scene and fed to two pipelines that are asked the very same `cause`/`causes_effect`
EQL queries:

- **relational circuit** -- an RSPN fitted on the scenes' relational structure,
  grounded per query into a circuit over exactly the queried objects, registered as a
  `CausalCircuit`;
- **flat-table tree** -- a joint probability tree (JPT) fitted on the same attempts
  flattened into one fixed-width table, registered as a `CausalCircuit` the same way.

`results.md` is the comparison: which questions each pipeline answers and what, how
many models each fitted, how long, how big, how well each explains held-out attempts,
how fast each answers, and how often the pick came up in the first place.

The scene classes follow the shape of the
[GraspClutter6D](https://sites.google.com/view/graspclutter6d) dataset (an environment
kind, many objects with known poses, a grasp scored by the friction it relies on), so
its scenes replace the ten-milk mock once they are annotated in the semantic digital
twin; `graspclutter6d.py` already reads its BOP-format scene files into a layout.

## The domain

`domain.py` holds the classes every other module works on.

| class | what it is |
|---|---|
| `ClutterPickScene` | one recorded attempt: the environment it stood in, where the target stood, the grasp's friction coefficient and yaw, whether the target came up and how far it rose, and its `neighbours` |
| `ClutteredObject` | one neighbour as an exchangeable part of the scene: category, position relative to the target, yaw, distance to the target and its `DistanceBand`, which `ClosingAxisSide` of the fingers it stands on, how far the pick shoved it and whether that counts as `disturbed` |
| `ClutterPickSceneAggregations` | the aggregation statistics over the neighbours the relational model derives: `crowding_count`, how many neighbours stand adjacent |
| `ClutterSceneLayout` / `PlacedObject` | the input side of an attempt: every carton's absolute pose, which one is the target, the grasp's friction and yaw |
| `ClutterPickOutcome` | what an attempt did: lift height, lifted or not, every neighbour's displacement; `to_scene` turns a layout and its outcome into a `ClutterPickScene` |
| `NeighbourThresholds` | the distances and angles that turn measured geometry into bands and sides |
| `FrictionLadder` | the friction levels an attempt can be given |

**Why friction.** GraspNet-1Billion and GraspClutter6D score a grasp by the smallest
friction coefficient it still closes under, so friction is the natural causal knob of a
grasp dataset. An attempt's friction coefficient is set on the cartons' geoms *and* on
the picking gripper's fingertip pads: MuJoCo gives a contact the larger of its two
geoms' friction, so a slippery carton only slips if the pads closing on it are no
grippier. The ladder sits around the coefficient below which a carton slips out of the
pads, and every level is exactly representable in single precision, because a circuit's
support is read back in single precision and a level that rounds there would no longer
match the point its own leaves sit on.

**Why the environment is a confounder.** A clutter stands on a *table* or in a *bin*
(`layout_sampler.py`). A bin packs the cartons more tightly *and* holds only the
slippery ones, so in the recorded attempts friction and crowding are correlated without
either causing the other; a question that marks the environment as a `confounder` has
it summed out by backdoor adjustment.

## The demo and the data

| file | what it holds |
|---|---|
| `scene.py` | `MilkClutterWorld`: Tracy, its table, the cartons, a fixed camera and a light, built from a layout and equipped with position servos |
| `episode.py` | `PickEpisode`: park, pick the target with `PickUpActionMujoco`, measure how far it rose and how far each neighbour moved |
| `demo.py` | watch one attempt in the viewer, or render before/after screenshots headless |
| `layout_sampler.py` | `ClutterLayoutSampler`: a jittered grid of ten cartons in a table or bin environment, a random target, a friction level and a grasp yaw |
| `collect_data.py` | record many attempts headless into `recorded/milk_clutter_attempts.json`, rewritten after every attempt |
| `synthetic.py` | a closed-form stand-in for the attempt with the same causal structure, so the pipelines are tested without a simulator |
| `dataset.py` | the attempts on disk, their train/test split, and success rates grouped by any key |
| `graspclutter6d.py` | reading a BOP-format GraspClutter6D scene (`scene_gt.json`, `scene_camera.json`) into a layout |

The MuJoCo stack the demo drives (parsing and mounting Tracy, servos, self-collision
exclusion, the real-time simulation, trajectory planning against a scratch copy of the
world, the pick and place actions, contact tuning) lives in
`experiments/tracy_experiments`.

## The pipelines

| file | what it holds |
|---|---|
| `flat_table.py` | `SceneSchema`, how EQL names every attribute, and `FlatTable`, the attempts flattened into one row each with one block of columns per neighbour index |
| `pipelines.py` | `CausalQueryPipeline` and its two implementations, `RelationalPipeline` and `FlatTablePipeline` |
| `queries.py` | the question catalogue, each question one EQL query that reads the same for either pipeline |
| `evaluation.py` | asking every question to every pipeline and recording what came of it |
| `report.py` | rendering the comparison as Markdown |
| `run_pipeline.py` | the whole comparison end to end |

**One model per cause.** Backdoor adjustment needs the circuit to be
support-deterministic over the cause: no sum unit may mix branches that overlap on it.
A fit guarantees that by stratifying its training rows on the cause's exact value, and
stratifying on two causes at once cannot serve both (two partitions sharing a value of
one of them overlap on it). Each pipeline therefore keeps one plain model for
everything that is not a causal query -- scoring held-out attempts -- and fits one
further model per cause variable it is asked about, the first time it is asked. A cause
on a neighbour attribute stratifies the neighbour template (relational) or the one
column of that neighbour index (flat).

**The questions.** Three kinds of cause, each asked about a clutter of the recorded
size and again about a clutter of another size:

1. *friction → lifted*, adjusting for the environment: which grasp friction makes the
   target come up;
2. *crowding count → lifted*, adjusting for the environment: how many adjacent
   neighbours the target can have and still come up -- the cause is an aggregation over
   the exchangeable parts;
3. *closing-axis side → disturbed*, for one neighbour: whether standing where the
   fingers close causes that neighbour to be shoved aside -- cause and effect both live
   on one part.

The relational circuit grounds itself for whatever objects a query names, so it answers
about 4 or 12 neighbours from attempts recorded with 9; the flat table has columns for
9 neighbours and nothing else, so it refuses those.

## Reading `results.md`

- **How often the pick came up** -- the recorded attempts before any model, grouped by
  environment, friction level and crowding: the picking efficiency in clutter.
- **Which questions each pipeline can answer** -- one row per question; an answer is
  put into words (the most effective setting of the cause and how likely the effect
  then is, against the least effective), a refusal says why.
- **Fit and likelihood** -- models fitted, training seconds, circuit size, held-out
  coverage (a tree's leaves span only the ranges they saw) and mean log-likelihood on
  the covered attempts and on the attempts both pipelines cover.
- **Seconds per question** -- the first ask (including the cause model's fit) and the
  same question asked again with every model fitted.
- **What the results show** -- the findings read off the numbers above.
- One section per question with the full interventional table: for every region of the
  cause, its population share, the naive conditional probability of the effect, and
  the backdoor-adjusted interventional probability.

## Running it

The `iai_tracy_description` ROS package must be built and sourced for anything that
builds the MuJoCo scene; headless runs need MuJoCo's EGL backend (`export
MUJOCO_GL=egl`).

```bash
# watch one attempt in the viewer (add --headless --screenshots DIR for images only)
python -m experiments.causal_reasoning.tracy_rspn.demo --seed 3

# record attempts (about 15 s each headless; the file is rewritten after every attempt)
python -m experiments.causal_reasoning.tracy_rspn.collect_data \
    experiments/src/experiments/causal_reasoning/tracy_rspn/recorded/milk_clutter_attempts.json \
    --attempts 300

# fit, score and question both pipelines; needs the experiments ORM interface
python scripts/regenerate_all_orm.py
python -m experiments.causal_reasoning.tracy_rspn.run_pipeline
```

The tests under `test/causal_reasoning_test/test_tracy_rspn` run the pipelines on the
synthetic attempts, so they need no simulator; the ones that build the MuJoCo scene are
skipped where Tracy's description is not installed.
