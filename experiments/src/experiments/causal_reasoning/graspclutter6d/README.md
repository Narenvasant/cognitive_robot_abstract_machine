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

The pipelines, the studies and the report are not this experiment's own: they live in
the shared `experiments.causal_reasoning.comparison` package and work on any relational
example a `RelationalDomain` describes, so that every dataset compared this way is
compared the same way. This package supplies the domain, the data, the questions and the
words.

| file | what it holds |
|---|---|
| `annotations.py` | reading a scene's three BOP files, from disk or from the dataset server, and the geometry derived from them |
| `grasp_labels.py` | counting the grasps a scene leaves each of its objects, and the index that keeps those counts |
| `dataset.py` | building the scenes, the levels their measurements are read as, a synthetic generator of the same shape for tests, and the graspability summaries the report opens with |
| `scene_store.py` | keeping the built scenes in a database and reading them back |
| `domain.py` | the scene, its parts and their aggregation counts, and `scene_domain()`, the scene as the shared comparison sees it |
| `queries.py` | the question catalogue, each question one EQL query that reads the same for every pipeline |
| `synthetic_scm.py` | a structural causal model over the same domain, whose interventional probabilities are computed by forcing a cause in the mechanism, and `SceneTruth`, the model as the known truth the shared ground-truth study scores against |
| `run_pipeline.py` | the whole comparison end to end: the `Experiment` handed to the shared runner, and the report's prose |

And in the shared package:

| file | what it holds |
|---|---|
| `domain.py` | `RelationalDomain`, what an example and its exchangeable parts are; `ExampleView`, how much of an example a likelihood is taken over |
| `flat_table.py` | `Schema`, how EQL names every attribute; `FlatTable`, the examples as one row each in one of the three `TableLayout`s |
| `pipelines.py` | `CausalQueryPipeline` and its three implementations: `RelationalPipeline`, `HybridPipeline`, and `FlatTablePipeline` once per layout |
| `baselines.py` | regression adjustment on the propositional table |
| `queries.py` | what every question is made of: `CausalQueryCase`, `Confounder`, and the open-part queries |
| `dataset.py` | `ExampleDataset`: splitting, reordering the parts, and the effect's rate |
| `evaluation.py` | asking every question to every pipeline and recording what came of it (`evaluate`), then the studies: `permutation_study` reorders every example's parts and asks the part questions again, `split_study` repeats the comparison over several random splits, `learning_curve` fits on growing shares of the examples, `ground_truth_study` scores every pipeline against a `KnownTruth`, `monte_carlo_study` follows the relational circuit's answers as grounding draws more samples, `scaling_study` measures cost against the number of parts |
| `report.py` | rendering the comparison and the studies as Markdown, around the `ReportText` an experiment writes |
| `run.py` | `Experiment`, `RunSettings` and `run`, the studies in order with the report written after each |

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

Numbers from `results.md`: one 763/191 split with seed 0 for the questions, the
timings and the likelihoods; twenty random orderings of the parts; five settings of the
synthetic model with 400 scenes each; a support threshold of ten training scenes per
cause region. Every number below is in there, in the table it came from.

**Which questions each pipeline can ask.** The relational circuit, the hybrid circuit
and the unrolled tree answer all fourteen questions; the propositional tree and the
regression baseline answer the eleven whose cause and effect are scene attributes or
counts and refuse the three about an individual object, because they have no column
for one; the scalars-only tree answers one question in fourteen, the catalogue question
adjusted for extent, the only one that mentions nothing it lacks.

**Where every pipeline has the columns, every pipeline answers alike.** On the eleven
scene-level questions the relational circuit and the propositional tree agree to the
third decimal, as they must: the relational circuit's class circuit *is* the
propositional tree, fitted on the same columns. The hybrid circuit and the unrolled tree
give the same trend and, to within a few hundredths, the same contrast on every one of
them; where they name another most effective region, it is a tie between two regions
at 0.5 or an argmax over sparse regions, and the contrast says so with its interval.
What the shared answers say about GraspClutter6D: the dataset's own catalogue of 200
objects leaves every object graspable more often than a mix of both catalogues (0.41
against 0.33 adjusting for extent, 0.39 against 0.27 adjusting for the small-object
count; contrast 0.07 [-0.00, 0.15] and 0.13 [0.05, 0.20]); the small-object count has
no effect once the number of objects is adjusted for (contrast 0.07 [-0.18, 0.31] from
none to fourteen, trend -0.4); more occluded objects go with *more* scenes that leave
every object graspable (0.34 [-0.00, 0.57] from two to sixteen, trend 0.7); and more
clear viewpoints go with fewer (-0.28 [-0.39, -0.15] from none to fifty-two, trend
-0.7). The last two are the confounding the dataset is known for: the bin scenes are
photographed from fewer clear poses and stacked deeper, and the number of objects
drives both the counts and the graspability.

**Adjusting for the number of objects is what changes the answers.** With extent alone
adjusted for, the adjusted probabilities equal the naive ones to the third decimal in
almost every region: the spread of the clutter carries no information about
graspability beyond the count itself. Adjusting for the number of objects moves single
regions by up to 0.15 (twelve small objects: 0.20 naive, 0.11 adjusted; sixteen occluded
objects: 0.52 naive, 0.38 adjusted) and the small-object contrast from -0.09 to +0.07.
The regression baseline agrees with the circuits on the direction of every trend but
not on its size: it puts the occluded-object contrast at 0.61 [0.26, 0.78] adjusting for
the number of objects where the circuits put it at 0.20 to 0.42, which is what a
logistic model does to a relation that is not monotone.

**The answers about an object are the answers the synthetic model checks.** Read off
the relational circuit, the catalogue makes an exchangeable object heavily occluded
with probability 0.36 against 0.32; the occluded-object count leaves an object's own
chance of losing every grasp between 0.08 and 0.13 (trend 0.3, contrast 0.02 [-0.26,
0.23]); and a small object loses every grasp a shade more often than a large one (0.12
against 0.10, contrast 0.02 [0.00, 0.03] over the 11,000 training objects). The
unrolled tree answers the same three questions about "object 0" and gets the sign of
two of them the other way (the ycb-video catalogue, a large object) with contrasts of
0.05 and 0.08 whose intervals exclude the relational circuit's. Against the synthetic
model, whose interventional probabilities are known, the relational circuit's error on
these three questions is 0.033, 0.009 and 0.007 and the unrolled tree's 0.133, 0.054
and 0.044; over every question and setting the relational circuit's support-weighted
error is 0.026 and its rank correlation with the truth 0.58, the propositional tree's
0.058 and 0.40, the unrolled tree's 0.064 and 0.26. Where the pipelines share columns
their error is the same (0.125 on the small-object question, the count's extremes being
rare within every stratum of the confounder); where they do not, the circuit is three
to five times closer to the truth, and its error grows only from 0.05 to 0.08 as the
scenes grow from five to twenty objects.

**Reordering the parts moves the unrolled tree's answers and nothing else.** Over
twenty random orderings the unrolled tree's most effective region moves in 88% of them
and its adjusted probability on the occluded-count question ranges by 0.50; the
relational and hybrid circuits' answers about objects do not move by 1e-9, because an
exchangeable object has no position. The dataset's own order is not arbitrary for the
viewpoints, whose frames the recording rig numbers, and the whole-scene likelihood shows
it: the hybrid circuit, which holds the viewpoints by position and the objects as
exchangeable, scores held-out scenes at 83.3 nats against the unrolled tree's 26.9 and
the relational circuit's -17.2, and reordering the parts costs it 124 nats and the
unrolled tree 120, while the relational circuit's likelihood stays where it was. A
position means something for a viewpoint and nothing for an object, and the three
models sit exactly where that puts them.

**Data and grounding samples.** The relational circuit's templates pool every object
and every frame of every training scene, so at a fifth of the data it already covers
36% of the held-out scenes, against the unrolled tree's 13%, and 84% against 64% at four
fifths. Grounding needs samples in proportion to how many distinct values the open
counts take: the object-level question is settled from fifty samples, the small-object
count question adjusted for the number of objects only from 8,000, which is why the
relational circuit answers the count questions in minutes.

**Cost.** The relational circuit is 37,013 nodes and 190 seconds of fitting, the hybrid
circuit 8,668 and 44, the propositional tree 10,429 and 43, the unrolled tree 364,486
and 523. Once fitted, a scene-level question costs every circuit the same backdoor
adjustment over the cause's regions times the confounders' (a two-confounder question
takes 2 to 30 minutes on every one of them), and an object question costs the
relational and hybrid circuits two to four minutes of grounding against the unrolled
tree's seconds. On synthetic scenes of growing size the relational circuit's templates
grow from 2,330 to 10,016 nodes between five and fifty objects and its fit from 16 to
44 seconds, the unrolled tree from 7,422 to 116,486 nodes and 6 to 276 seconds, one
block of columns per position.

What the comparison does not show: that the relational circuit gives better causal
answers on scene-level questions than a flat tree given the same counts. It cannot,
because on those columns it is that tree. What it shows is which questions a flat
learner can pose at all, that its answers about parts are answers about the listing
order, and that the relational circuit is the one model here whose answers about an
object are both order-free and, where the truth is known, close to it.

`comparison.md` puts these numbers beside the other two datasets the same pipeline was
run on.

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
