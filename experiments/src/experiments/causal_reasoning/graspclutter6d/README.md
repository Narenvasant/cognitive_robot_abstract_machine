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

The comparison is between two models that treat objects as exchangeable and three flat
ones, and the interesting part is not which of them is most accurate. On the columns
they share, several of them are the same tree and give the same answer to the third
decimal. The interesting part is which questions each of them can be asked at all, what
its answers about individual objects are worth once you notice that the order the
objects were written down in was arbitrary, and — on a synthetic model of the same
domain whose interventional probabilities are known by construction — how far those
answers are from the truth.

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
and are the ones used here. They hold 5 to 20 object instances each, drawn from 200
object models, and 52 annotated frames each.

The three per-scene annotation files are read either from an extracted copy of the
dataset or, without downloading anything, from a dataset server over http — the server's
address and the dataset's location on it come from
`SEMANTIC_DIGITAL_TWIN_DATASET_SERVER` and `SEMANTIC_DIGITAL_TWIN_DATASET_ROOT`. Asking
the server for one of a scene's annotation files brings the other two with it and leaves
the scene's images where they are, so reading all 954 scenes costs well under a gigabyte.
The grasp and collision labels have to be on the machine, and reading them is slow enough
(about 20 minutes for the whole dataset) that the counts read off them are kept in an
index beside the dataset and computed once. The built scenes themselves are kept in a
database through the package's ORM (`scene_store.py`), one row per scene, object and
viewpoint, so that every run after the first reads 954 scenes back in three seconds
instead of touching the server or the labels at all; `GRASPCLUTTER6D_DATABASE_URI` names
the database, and a file beside the dataset is used when it does not.

## The domain

`domain.py` holds the classes every other module works on. A scene is the relational
example: its own attributes, plus two lists of exchangeable parts.

| class | what it is |
|---|---|
| `GraspClutterScene` | one scene: which object catalogue it draws from, how far its clutter is spread (`extent`) and stacked (`height_span`), whether every object in it keeps a grasp, and its `objects` and `viewpoints` |
| `GraspClutterObject` | one object instance as an exchangeable part: its size, diameter, visibility, occlusion level and whether it is still graspable |
| `GraspClutterViewpoint` | one annotated camera frame as an exchangeable part: which camera took it, how far it stood, and whether it saw the scene clearly |
| `GraspClutterSceneAggregations` | the counts the relational model derives over the parts: all, small and occluded objects over the objects, near and clear viewpoints over the viewpoints |

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

## Why three flat tables, and a fourth model in between

A scene has between 5 and 20 object instances and no canonical order over them. The
annotation file lists them in the order they were labelled, and nothing ties a position
in that list to an identity. A flat learner, which needs a fixed set of columns, therefore
has three choices, and each is a pipeline here:

- keep only the scene's own scalars, and lose every question about the parts;
- add the aggregation counts, which is exactly the table the relational circuit's
  class-level circuit is fitted on, so on those columns the two are the same tree;
- unroll the parts by position and pad, which makes a column mean whatever part a scene
  happens to list there.

The viewpoints are different. The recording rig numbers a scene's 52 frames in a fixed
sequence, four cameras per pose, so frame *i* is the same physical camera in every scene
and a column that addresses a viewpoint by position addresses a real thing. Whether a
part is exchangeable is a fact about the data, one relation at a time, and the fourth
model declares it that way: the **hybrid circuit** treats the objects as an exchangeable
part, with the relational circuit's template and counts, and the viewpoints as
positional columns of its scene-level tree. It answers a question about an object with
the relational circuit and every other question with the tree.

Every query lists one object and one viewpoint with all their attributes open. For the
relational circuit that is what makes grounding retain the scene's counts as variables (a
query with an empty object list is a scene with no objects, whose counts are zero). A flat
table ignores a part a query merely lists, and refuses a query that constrains a column it
does not have: sets one of its attributes, or marks it as cause, confounder or effect.

A sixth estimator is not a circuit at all: **regression adjustment**, the textbook
backdoor estimator, a logistic regression of the effect on the cause and the confounders
over the propositional table, averaged at each value of the cause over the confounders'
distribution in the data. It can be asked exactly the questions the propositional tree
can, and it is there so that the comparison is not only between configurations of one
learner.

## The pipelines

| file | what it holds |
|---|---|
| `annotations.py` | reading a scene's three BOP files, from disk or from the dataset server, and the geometry derived from them |
| `grasp_labels.py` | counting the grasps a scene leaves each of its objects, and the index that keeps those counts |
| `dataset.py` | building the scenes, the levels their measurements are read as, and a synthetic generator of the same shape for tests |
| `scene_store.py` | keeping the built scenes in a database and reading them back |
| `synthetic_scm.py` | a structural causal model over the same domain, whose interventional probabilities are computed by forcing a cause in the mechanism |
| `baselines.py` | regression adjustment on the propositional table |
| `flat_table.py` | `SceneSchema`, how EQL names every attribute; `FlatTable`, the scenes as one row each in one of the three `TableLayout`s; `SceneView`, how much of a scene a likelihood is taken over |
| `pipelines.py` | `CausalQueryPipeline` and its three implementations: `RelationalPipeline`, `HybridPipeline`, and `FlatTablePipeline` once per layout |
| `queries.py` | the question catalogue, each question one EQL query that reads the same for every pipeline |
| `evaluation.py` | asking every question to every pipeline and recording what came of it (`evaluate`), then the studies: `permutation_study` reorders every scene's parts and asks the object questions again, `split_study` repeats the comparison over several random splits, `learning_curve` fits on growing shares of the scenes, `ground_truth_study` scores every pipeline against the synthetic model's truth, `monte_carlo_study` follows the relational circuit's answers as grounding draws more samples, `scaling_study` measures cost against the number of objects |
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

**The relational circuit's fit does not depend on the listing order, exactly.** The tree
learner breaks ties by row order, and every scene's parts are pooled into the template's
rows, so a naive refit on reordered scenes could differ in the last digit. The relational
pipeline therefore sorts every scene's parts canonically before fitting and before
scoring; a test refits it on three reorderings of the same scenes and asserts that the
refusals, the regions and the probabilities are the same to floating-point precision.

**The questions.** Fourteen, in four kinds.

1. *Small objects, occluded objects and clear viewpoints cause a scene to leave every
   object graspable* — each asked three times: adjusting for `extent`, how far the
   clutter is spread out; adjusting for the number of objects, which drives both how
   many of them are small and how many chances there are for one to lose every grasp,
   a textbook backdoor; and adjusting for both. The cause is a count over exchangeable
   parts.
2. *The object catalogue causes a scene to leave every object graspable*, once adjusting
   for `extent`, which every pipeline has, and once for the small-object count, which the
   scalars-only tree does not. The cause is an attribute of the scene itself.
3. *The object catalogue causes one object to be heavily occluded*, and *the
   occluded-object count causes one object to lose every grasp*. The cause is
   scene-level, the effect one object's own attribute.
4. *An object's size causes it to lose every grasp.* Cause and effect both live on one
   object.

**What an answer is.** Backdoor adjustment gives the effect's probability under every
region of the cause the model distinguishes, and a count has up to twenty of them, the
outermost held by one or two scenes. Every region is therefore reported with *n*, the
number of training scenes it holds, and a Wilson interval over that n; a region below the
support threshold (ten scenes) is marked † and takes no part in any summary. The
headline statistics are not the argmax over regions, which moves with the split, but a
*trend* — Spearman's rank correlation between the cause's value and the adjusted
probability over the supported regions — and a *contrast*, the adjusted probability at
the highest supported region minus at the lowest, with Newcombe's interval.

**The studies.** A single split cannot tell the pipelines apart, so six things are
measured around it.

- *Reordering.* Every scene's objects and viewpoints are put in a random order, twenty
  times over, the pipelines that model the parts are refitted each time and asked the
  object questions again, and the parts in the dataset's own order is the baseline every
  reordering is measured against. Reported as a distribution: the spread of the adjusted
  probability per region, the share of reorderings in which the most effective region
  moved, and the share in which the trend changed sign.
- *Splits.* Optionally (`--splits N`), the comparison repeated over random splits for
  the spread of every trend and contrast; off by default, since it multiplies the run
  time and adds error bars rather than findings.
- *Learning curve.* Fitting on a growing share of the scenes shows how much data each
  pipeline needs to explain a whole scene.
- *Ground truth.* Scenes are sampled from a structural causal model over the same
  domain (`synthetic_scm.py`). Its one confounder is the number of objects, which drives
  both the share of small objects and each object's chance of keeping a grasp; the
  strength of that confounding and the typical number of objects are swept; and the
  scenes list their small objects first, so a positional column is systematically
  misleading. The true interventional probability of any question is the effect's rate
  over 200,000 scenes sampled with the cause forced in the mechanism — the `do`
  operator by construction, needing no inference. Every pipeline's answer is scored
  against it, in the model's own order and under reorderings.
- *Grounding samples.* The relational circuit is asked two questions with grounding
  drawing from 50 to 32,000 samples for the counts the query leaves open, to find where
  the answers settle: inference on the grounded circuit is exact, marginalising the open
  counts is a consistent estimate, and the paper should say which is which.
- *Scaling.* Fit time, query time and circuit size against the number of objects per
  scene, on the synthetic model, for the pipelines that model the parts.

## What the results show

Numbers from `results.md`: one 763/191 split with seed 0 for the questions and timings,
five splits for the spreads, three random orderings of the parts. Every number below is
in there, in the table it came from.

**Which questions each pipeline can ask.** The relational circuit and the unrolled tree
answer all eight questions; the propositional tree answers the five whose cause and effect
are scene attributes or counts and refuses the three about an individual object, because
it has no column for one; the scalars-only tree answers one question in eight, the
catalogue question adjusted for extent, because that is the only one that mentions
nothing it lacks. This is the same pattern on every one of the five splits. The relational
circuit refused the object-size question on one of the five splits as not
support-deterministic: the grounded circuit mixes one copy of the object template per
sampled count, and on that split the copies overlapped on `size`, so backdoor adjustment
had no disjoint regions to intervene on and the circuit said so rather than answer.

**Where every pipeline has the columns, every pipeline answers alike.** On the five
scene-level questions the relational circuit and the propositional tree agree to the
third decimal, as they must: the relational circuit's class circuit *is* the
propositional tree, fitted on the same eight columns. The unrolled tree agrees with them
on the most effective setting on four of the five and is within 0.01 on the fifth. What
the shared answers say about GraspClutter6D: the dataset's own catalogue of 200 objects
leaves every object graspable more often than the YCB-Video objects or a mix of both
(0.41 against 0.36 and 0.33, adjusting for how far the clutter is spread), and adjusting
for the small-object count instead moves that by a hundredth. The count questions have
mild trends and noisy extremes. Going from no small objects to eleven or twelve takes the
adjusted probability from 0.45 to 0.20–0.24, and the occluded-object count runs the other
way, from 0.16–0.19 at four to six occluded objects to 0.52–0.67 at sixteen to eighteen;
but the settings the report names as *most effective* — 15 small objects at 0.50, 19
occluded objects at 1.00 — are strata of four and of one scene, and the five splits
disagree on them (small-object count: 1, 3 or 15). Read the trend, not the extreme. The
naive and adjusted columns agree almost everywhere: within a stratum of the cause, the
clutter's extent carried no further information about graspability.

**The answers about an object depend on which object "object 0" is, unless the model
treats objects as exchangeable.** Asked how many occluded objects cause one object to
lose every grasp, the relational circuit gives a monotone answer about an exchangeable
object: 0.06 with no occluded objects rising smoothly to 0.16 with twenty. The unrolled
tree, asked the same question about the object listed first, answers 0.33 at zero
occluded objects falling to 0.00 at seventeen — the opposite direction, and about a
different thing. Reordering the parts three times moves the unrolled tree's adjusted
probabilities by up to 0.33 and flips its most effective setting on two of the three
object questions; the relational circuit's answers do not move at all, on any question,
under any ordering, because nothing about an exchangeable part can depend on where it was
listed. Over the five splits the relational circuit finds the same most effective setting
of the occluded-count question every time (20), and the unrolled tree alternates between
0 and 1.

**Whole-scene likelihood, and what the dataset's order is worth.** In the order the
dataset lists the parts, the unrolled tree explains the held-out scenes it covers far
better than the relational circuit: a mean whole-scene log-likelihood of +31 against −14
over the scenes both cover. Under any random reordering that number falls to −88, and
its coverage from 67% to 41–46%, while the relational circuit's stays at −16 and 84%. The
reason is that a scene's frames are numbered by the recording rig, four cameras per pose
in a fixed sequence, so which camera took frame *i* is the same in every scene and the
viewpoint columns address a real thing: in the dataset's order that column is
deterministic, worth log 4 for each of the 52 frames on its own, and the camera's
distance to the objects at position *i* is close to the same across scenes as well. The
relational circuit treats a viewpoint as exchangeable and pays for the camera every time.
This is the one place the flat table's position-as-identity assumption is right, and it
is right about the rig, not about the scene; the object columns, which carry no such
identity, are the ones whose causal answers flip. A flat learner has no way to tell the
two kinds of column apart, and a relational model has no way to use the first kind.

**Cost.** The relational circuit's plain fit takes 146 seconds against 35 for the
propositional tree and 486 for the unrolled tree, whose 363,000 nodes are forty times
the relational circuit's 36,000. Once fitted, the trees answer a question in 0.2 to
15 seconds (the clear-viewpoint question, over a count that runs from 0 to 52, takes the
propositional tree 116 seconds) and the relational circuit in 75 to 217, the difference
being Monte-Carlo grounding over the counts the query leaves open.

What the comparison does not show: that the relational circuit gives better causal
answers than a flat tree on questions both can pose from the same counts. On those
columns it is that tree. What it shows is which questions a flat learner can pose at all,
that its answers about parts are answers about a listing order, and that the one thing
the listing order does encode here — the camera rig — is exactly the thing an
exchangeable model cannot see.

## Running it

The scenes' annotations come from an extracted copy of the dataset or from the dataset
server, and the built scenes are kept in a database:

```bash
export SEMANTIC_DIGITAL_TWIN_DATASET_SERVER="http://<host>:<port>/datasets"
export SEMANTIC_DIGITAL_TWIN_DATASET_ROOT="/raid/users/tom_sch/datasets"
# optional; a SQLite file beside the dataset is used otherwise
export GRASPCLUTTER6D_DATABASE_URI="postgresql+psycopg://<user>:<password>@localhost:5432/graspclutter6d"
```

The grasp and collision labels have to be on the machine for the first build. They are
two of the dataset's own archives, about 5 GB to download and rather more to extract:

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
# fit, score and question every pipeline, score against the synthetic model's truth,
# reorder the parts twenty times, follow the grounding samples, measure the learning
# curve and the scaling; the report is rewritten after every study
python scripts/regenerate_all_orm.py
python -m experiments.causal_reasoning.graspclutter6d.run_pipeline

# a quicker look: a fifth of the scenes, three orderings
python -m experiments.causal_reasoning.graspclutter6d.run_pipeline \
    --scenes 200 --orderings 3

# with error bars from five random splits, at several times the run time
python -m experiments.causal_reasoning.graspclutter6d.run_pipeline --splits 5

# after a change to the domain classes, build the scenes afresh
python -m experiments.causal_reasoning.graspclutter6d.run_pipeline --rebuild
```

The first build reads the grasp labels for every scene, which takes about twenty minutes
and writes the scenes to the database; every run after that reads them back in seconds.

The tests under `test/experiments_test/causal_reasoning_test/test_graspclutter6d` run the
pipelines and every study on synthetic scenes, and read one real scene's annotations from
a trimmed copy checked in beside them, so they need neither the dataset nor network
access.
