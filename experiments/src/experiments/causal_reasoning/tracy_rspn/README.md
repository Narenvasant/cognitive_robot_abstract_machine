# Tracy picks one milk out of ten: relational causal circuits against a flat tree

Tracy's left arm picks one milk carton out of a ten-carton clutter in MuJoCo, held by
contact friction alone (no kinematic attachment). Every attempt is recorded as a
relational scene -- the attempt's own attributes plus one exchangeable part per
neighbouring carton -- shaped after the [GraspClutter6D](https://sites.google.com/view/graspclutter6d)
dataset, so its scenes replace the mock once they are annotated.

The recorded attempts feed two causal-query pipelines that are asked the very same
`cause`/`causes_effect` EQL queries:

- **relational circuit**: a relational probabilistic circuit (RSPN) fitted on the
  scenes' relational structure, grounded per query, and registered as a
  `CausalCircuit`;
- **flat-table tree**: a joint probability tree fitted on the same attempts flattened
  into one fixed-width table, registered the same way.

`results.md` reports, per question, whether each pipeline answered and what, and, per
pipeline, how many models it fitted, how long, how big, how well it explains held-out
attempts, and how fast it answered.

## Layout

| file | what it holds |
|---|---|
| `domain.py` | the scene, object, layout and outcome classes, the aggregation statistics, and the friction ladder |
| `layout_sampler.py` | random ten-carton layouts in a *table* or a *bin* environment (the confounder) |
| `scene.py` | the MuJoCo world: Tracy, its table, the cartons, a camera and a light |
| `episode.py` | one attempt: park, pick with `PickUpActionMujoco`, measure lift and neighbour displacement |
| `demo.py` | watch one attempt (`python -m experiments.causal_reasoning.tracy_rspn.demo`) |
| `collect_data.py` | record many attempts headless into `recorded/milk_clutter_attempts.json` |
| `synthetic.py` | a closed-form stand-in for the attempt, for tests without a simulator |
| `dataset.py` | the attempts on disk, and their train/test split |
| `flat_table.py` | flattening attempts into the fixed-width table, named like EQL names the attributes |
| `pipelines.py` | the two pipelines behind one interface, one model per cause each |
| `queries.py` | the question catalogue |
| `evaluation.py`, `report.py` | asking every question, and writing the comparison as Markdown |
| `run_pipeline.py` | the whole comparison, end to end |
| `graspclutter6d.py` | reading a BOP-format GraspClutter6D scene into a layout |

## Running it

The `iai_tracy_description` ROS package must be built and sourced for anything that
builds the MuJoCo scene.

```bash
# watch one attempt in the viewer (add --headless --screenshots DIR for images only)
python -m experiments.causal_reasoning.tracy_rspn.demo --seed 3

# record attempts (rewrites the file after every attempt)
python -m experiments.causal_reasoning.tracy_rspn.collect_data \
    experiments/src/experiments/causal_reasoning/tracy_rspn/recorded/milk_clutter_attempts.json \
    --attempts 300

# fit, score and question both pipelines; needs the experiments ORM interface
python scripts/regenerate_all_orm.py
python -m experiments.causal_reasoning.tracy_rspn.run_pipeline
```

Headless runs need MuJoCo's EGL backend: `export MUJOCO_GL=egl`.

## How friction enters

An attempt's friction coefficient is set on the cartons' geoms *and* on the picking
gripper's fingertip pads: MuJoCo gives a contact the larger of its two geoms' friction,
so a slippery carton only slips if the pads closing on it are no grippier. The ladder
of levels sits around the coefficient below which a carton slips out of the pads, and
every level is exactly representable in single precision, because a circuit's support
is read back in single precision.
