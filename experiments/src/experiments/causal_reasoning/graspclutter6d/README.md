# GraspClutter6D: relational circuit against flat-table trees

## What this experiment is for

A robot that has to clear a cluttered bin wants to know what it can change. Not "are
scenes with many small objects easier to clear" — that is a correlation, and it is
confounded by everything that makes a scene small-and-easy in the first place — but "if I
were to make this scene one with fewer large objects in it, would every object still have
a grasp". That is an interventional question, and answering it needs a model you can
intervene on.

This experiment asks that kind of question of the
[GraspClutter6D dataset](https://sites.google.com/view/graspclutter6d) and compares four
ways of building a model to answer it. The dataset is unusually well suited to it: a
thousand real cluttered bin, shelf and table scenes, each photographed from thirteen
poses by four cameras, each object annotated with its ground-truth 6D pose and the share
of it every camera actually sees, and each object model annotated with analytic antipodal
grasps that were then checked for collision against every scene the object stands in. So
"can this object still be grasped where it lies" is not a proxy the experiment invents;
it is ground truth the dataset computed, and it depends on the clutter around the object
rather than on the object alone.

The comparison is between one relational model and three flat ones, and the interesting
part is not which of them is most accurate. On the columns they share, several of them
are the same tree and give the same answer to the third decimal. The interesting part is
which questions each of them can be asked at all, and what its answers about individual
objects are worth once you notice that the order the objects were written down in was
arbitrary.

## The data

The dataset ships as a 203 GB archive of images plus a handful of small annotation files,
and this experiment reads only the small ones.

Per scene, three files in the
[BOP format](https://github.com/thodan/bop_toolkit):

- `scene_camera.json` — per frame, the camera's intrinsics and its pose in the scene's
  own world frame.
- `scene_gt.json` — per frame, the pose of every object instance relative to that camera,
  in the order the annotators labelled them.
- `scene_gt_info.json` — per frame, how much of every object instance the camera actually
  sees (`visib_fract`), which is the dataset's own measure of occlusion.

Per object model, `models_info.json` gives the mesh's diameter, and `grasp_label/`
gives every antipodal grasp candidate with the friction coefficient it needs. Per scene,
`collision_label/` says which of those candidates collide with the rest of that
particular scene. An object counts as graspable in a scene when at least one of its
candidates is antipodal at the friction coefficient the dataset's own toolkit defaults to
(0.4) and does not collide.

954 of the thousand scenes are listed under one of the dataset's two object catalogues
and are the ones used here. They hold 2 to 20 object instances each, drawn from 200
object models, and 52 annotated frames each.

The three per-scene annotation files are read either from an extracted copy of the
dataset or, without downloading anything, from a dataset server over http — the server's
address and the dataset's location on it come from
`SEMANTIC_DIGITAL_TWIN_DATASET_SERVER` and `SEMANTIC_DIGITAL_TWIN_DATASET_ROOT`. Asking
the server for one of a scene's annotation files brings the other two with it and leaves
the scene's images where they are, so reading all 954 scenes costs well under a gigabyte.
The grasp and collision labels have to be on the machine, and reading them is slow enough
(about 20 minutes for the whole dataset) that the counts read off them are kept in an
index beside the dataset and computed once.

## The domain

`domain.py` holds the classes every other module works on. A scene is the relational
example: its own attributes, plus two lists of exchangeable parts.

| class | what it is |
|---|---|
| `GraspClutterScene` | one scene: which object catalogue it draws from, how far its clutter is spread (`extent`) and stacked (`height_span`), whether every object in it keeps a grasp, and its `objects` and `viewpoints` |
| `GraspClutterObject` | one object instance as an exchangeable part: its size, diameter, visibility, occlusion level and whether it is still graspable |
| `GraspClutterViewpoint` | one annotated camera frame as an exchangeable part: which camera took it, how far it stood, and whether it saw the scene clearly |
| `GraspClutterSceneAggregations` | the counts the relational model derives over the parts: small and occluded objects over the objects, near and clear viewpoints over the viewpoints |

Two things about this shape are worth saying out loud, because both are choices.

**Every continuous measurement has a discrete counterpart.** An object carries both a
`diameter` in meters and a `size` of small, medium or large; a viewpoint carries both a
`distance` and a `proximity` of near or far. This is not redundancy for its own sake: an
aggregation statistic counts the parts matching a condition, and a condition on a part
that a query leaves open can only compare for equality, so a count of "small objects"
needs a field that is literally equal to "small". The boundaries between the levels are
not chosen. They are read off the dataset's own distribution, so that each level holds an
equal share of the values — object sizes are the catalogue's own terciles, occlusion is
the terciles of the visibility the cameras record, and a viewpoint's proximity and clarity
are the medians of the frames themselves.

**Where the scene's geometry is measured.** The dataset calibrates each camera's world
frame separately, so the 52 frames of a scene do not agree on where the origin sits —
they disagree by up to about 17 cm. They do agree, to within about 2 cm, on where the
objects lie relative to each other, which is all `extent` and `height_span` are about, so
both are measured per frame and averaged over the frames rather than read off whichever
frame happens to be first.

## Why three flat tables

A scene has between 2 and 20 object instances and no canonical order over them. The
annotation file lists them in the order they were labelled, and nothing ties a position
in that list to an identity. A flat learner, which needs a fixed set of columns, therefore
has three choices, and each is a pipeline here:

- keep only the scene's own scalars, and lose every question about the parts;
- add the aggregation counts, which is exactly the table the relational circuit's
  class-level circuit is fitted on, so on those columns the two are the same tree;
- unroll the parts by position and pad, which makes a column mean whatever part a scene
  happens to list there.

Every query lists one object and one viewpoint with all their attributes open. For the
relational circuit that is what makes grounding retain the scene's counts as variables (a
query with an empty object list is a scene with no objects, whose counts are zero). A flat
table ignores a part a query merely lists, and refuses a query that constrains a column it
does not have: sets one of its attributes, or marks it as cause, confounder or effect.

## The pipelines

| file | what it holds |
|---|---|
| `annotations.py` | reading a scene's three BOP files, from disk or from the dataset server, and the geometry derived from them |
| `grasp_labels.py` | counting the grasps a scene leaves each of its objects, and the index that keeps those counts |
| `dataset.py` | building the scenes, the levels their measurements are read as, and a synthetic generator of the same shape for tests |
| `flat_table.py` | `SceneSchema`, how EQL names every attribute; `FlatTable`, the scenes as one row each in one of the three `TableLayout`s; `SceneView`, how much of a scene a likelihood is taken over |
| `pipelines.py` | `CausalQueryPipeline` and its two implementations, `RelationalPipeline` and `FlatTablePipeline`, the latter once per layout |
| `queries.py` | the question catalogue, each question one EQL query that reads the same for every pipeline |
| `evaluation.py` | asking every question to every pipeline and recording what came of it (`evaluate`), then the three studies: `permutation_study` reorders every scene's objects and viewpoints and asks the object questions again, `split_study` repeats the comparison over several random splits, `learning_curve` fits on growing shares of the scenes |
| `report.py` | rendering the comparison and the studies as Markdown |
| `run_pipeline.py` | the whole comparison end to end |

**One model per cause.** Backdoor adjustment needs the circuit to be
support-deterministic over the cause: no sum unit may mix branches that overlap on it. A
fit guarantees that by stratifying its training rows on the cause's exact value.
Stratifying on two variables at once cannot serve both, so each pipeline keeps one plain
model for everything that is not a causal query — scoring held-out scenes, for instance —
and fits one further model per cause variable the first time it is asked about that
cause. A cause on an object attribute stratifies the object template in the relational
pipeline, or that position's column in the unrolled tree; the other tables have no column
for it.

The fewest rows a leaf may hold is given as a share of the rows the model is fitted on
rather than as a count, so the same setting holds for a class circuit over a few hundred
scenes and for an object template over their tens of thousands of parts.

**The questions.**

1. *Small objects, occluded objects and clear viewpoints cause a scene to leave every
   object graspable*, each adjusting for `extent`, how far the clutter is spread out — a
   scene spread thin over a table is both easier to reach into and differently composed
   than one heaped in a bin. The cause is a count over exchangeable parts.
2. *The object catalogue causes a scene to leave every object graspable*, once adjusting
   for `extent`, which every pipeline has, and once for the small-object count, which the
   scalars-only tree does not. The cause is an attribute of the scene itself.
3. *The object catalogue causes one object to be heavily occluded*, and *the
   occluded-object count causes one object to lose every grasp*. The cause is
   scene-level, the effect one object's own attribute.
4. *An object's size causes it to lose every grasp.* Cause and effect both live on one
   object.

**The studies.** A single split cannot tell the pipelines apart, so three things are
measured around it. Reordering every scene's objects and viewpoints at random and asking
the object questions again shows what an answer about "object 0" is worth: nothing about a
relational circuit can depend on the order, while an unrolled table's column holds a
different object of every scene afterwards. Repeating the comparison over several random
splits gives the spread of every number. Fitting on a growing share of the scenes shows
how much data each pipeline needs to explain a whole scene, objects and viewpoints
included, which only the relational circuit and the unrolled tree can score at all.

## Running it

The scenes' annotations come from an extracted copy of the dataset or from the dataset
server:

```bash
export SEMANTIC_DIGITAL_TWIN_DATASET_SERVER="http://<host>:<port>/datasets"
export SEMANTIC_DIGITAL_TWIN_DATASET_ROOT="/raid/users/tom_sch/datasets"
```

The grasp and collision labels have to be on the machine. They are two of the dataset's
own archives, about 5 GB to download and rather more to extract:

```bash
python - <<'EOF'
import huggingface_hub, pathlib, py7zr
directory = pathlib.Path.home() / "graspclutter6d-dataset"
for name in ("split_info.7z", "models_eval.7z", "collision_label.7z", "grasp_label.7z"):
    archive = huggingface_hub.hf_hub_download(
        "GraspClutter6D/GraspClutter6D", name, repo_type="dataset",
        local_dir=str(directory),
    )
    with py7zr.SevenZipFile(archive, mode="r") as opened:
        opened.extractall(path=str(directory))
EOF
```

Then:

```bash
# fit, score and question every pipeline, reorder the parts three times, repeat over
# five splits and measure the learning curve
python scripts/regenerate_all_orm.py
python -m experiments.causal_reasoning.graspclutter6d.run_pipeline

# a quicker look: a fifth of the scenes, one ordering, two splits
python -m experiments.causal_reasoning.graspclutter6d.run_pipeline \
    --scenes 200 --orderings 1 --splits 2
```

The first run reads the grasp labels for every scene, which takes about twenty minutes;
every run after that reads the index it leaves behind.

The tests under `test/experiments_test/causal_reasoning_test/test_graspclutter6d` run the
pipelines and the studies on synthetic scenes of the same shape, and read one real scene's
annotations from a trimmed copy checked in beside them, so they need neither the dataset
nor network access.
