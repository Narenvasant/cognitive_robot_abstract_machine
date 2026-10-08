# Tracy clutter picking

## What this experiment is for

Relational sum-product networks are less expressive than most machine-learning models.
The argument this experiment makes is that they are expressive *enough*: a relational
circuit fitted on a robot's own recorded attempts can answer causal questions about a
cluttered pick, including questions whose cause or effect is one object of the clutter
rather than the attempt as a whole.

Tracy's left arm picks one milk carton out of a clutter of ten in MuJoCo. The carton is
held by contact friction between the fingertip pads alone, with no kinematic attachment,
so a poor grasp visibly fails. Every attempt was recorded as a relational scene, and the
recorded attempts are what this experiment asks its questions of.

There is one model here, the relational circuit. Nothing is compared against anything
else.

## The domain

One attempt is a `ClutterPickScene`: the environment it stood in, where the target
stood, the friction of the grasp, how the gripper was turned, whether the target came up
and how far it rose. Every other object of the clutter is a `ClutteredObject` held in
the scene's `neighbours` field, an exchangeable part: what kind of object it is, where it
stands relative to the target, which band that distance falls in, which side of the
fingers' closing axis it stands on, how far it moved, and whether the attempt shoved it
aside.

`ClutterPickSceneAggregations.crowding_count` counts the neighbours standing adjacent to
the target. It is a statistic over the parts rather than a column of the attempt, which
is what lets a question be asked about the crowding of a clutter whose size was never
fixed in advance.

## The questions

Each question in `do_query.py` marks one cause and one effect in the query itself. The
circuit grounds against a clutter of the size the question asks about, and the effect is
read off every region of the cause twice: once by conditioning alone, once with backdoor
adjustment. The gap between those two is the point.

- **Friction causes the lift**, adjusting for the environment. The environment decides
  both how slippery and how crowded an attempt is, which is what makes it a confounder
  rather than a nuisance.
- **Crowding causes the lift**, adjusting for the environment, and again adjusting for
  nothing, so the difference adjusting makes is visible. The cause is a count over the
  parts.
- **A neighbour's side of the closing axis causes that neighbour to be disturbed.**
  Cause and effect both live on one part, so this is a question about the clutter's
  structure rather than about the attempt as a whole.

## Grouping the fit

A cause has to come out support-deterministic or the registration rejects the model: two
branches of the circuit must never claim the same value of the cause. The fit is
therefore grouped by the cause, and `CauseStratification` records where the cause sits so
that the right circuit is grouped. An attribute of the attempt or a statistic over its
parts groups the class circuit; one neighbour's own attribute groups the template holding
the neighbours instead, since the class circuit has no column for it.

## Where the attempts come from

`dataset.recorded_attempts()` names the attempts recorded in MuJoCo, hosted in their own
repository and fetched on first use. `ClutterPickDataset` saves and loads them.

`synthetic.py` is a closed-form stand-in with the same shape and the same causal
structure: friction lets the fingers hold the target, every adjacent neighbour takes a
share of that hold away, and the environment drives both. It needs no simulator, so the
tests fit on it and run anywhere.

## What is not here

The code that drives Tracy in MuJoCo to record a fresh set of attempts is not in this
package. It was written against a generation of `coraplex` whose grasp vocabulary
(`GraspDescription`, `ApproachDirection`, `VerticalAlignment`, `ViewManager`) current
`main` no longer has, and reconciling it with the grasp API that replaced it is a
separate piece of work. The recorded attempts it produced are hosted, so the questions
above can be asked without it.
