---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(chapter:causal-structure)=
# Causal Structure

{ref}`chapter:causal-reasoning` answers an interventional query on a circuit once the
query says which variables to adjust for. This chapter covers the step before it:
learning the causal structure from the training data, and letting that structure say
which variables to adjust for. It is implemented in `probabilistic_model.causal_structure`.

## Causal circuits and the causal structure

The two packages answer different questions and work together.

|  | `probabilistic_circuit.causal` | `causal_structure` |
|---|---|---|
| answers | *how strong* is the effect of a cause, given the variables to adjust for | *which* variables to adjust for, and whether the effect can be identified at all |
| central class | `CausalCircuit`, a fitted circuit with its cause and effect variables | `CausalGraph` and `AdjustmentSelector`, learned from the circuit's training rows |
| input | a fitted circuit, a cause, an effect, an adjustment set | the training rows, and background knowledge such as temporal order |
| output | the interventional circuit $P(Y \mid do(X))$ by backdoor adjustment | a valid adjustment set, a few candidate sets, or a refusal |
| checks | support determinism: is the circuit able to answer the query exactly | the adjustment criterion: does adjusting for the set answer the causal question |
| what it trusts | the adjustment set it is handed | the independences in the data and the background knowledge |

A `CausalCircuit` computes the backdoor formula exactly for whatever adjustment set it
is given, and has no way to tell whether that set is the right one: a missing confounder
or an adjusted-for mediator gives a wrong number with no warning. The causal structure
supplies or checks that set. Estimation stays in the circuit; nothing in
`causal_structure` changes how an answer is computed.

## The pieces

- `graph.CausalGraph` holds a graph whose edges may be directed or undirected. An
  undirected edge is one the data cannot orient, so the graph stands for every
  directed acyclic graph that orients those edges without adding a collider:
  `member_graphs()` lists them.
- `identification.AdjustmentSelector` checks the adjustment criterion (Shpitser,
  VanderWeele and Robins, 2010; Perković et al., 2015) in every member graph. Its
  `identify` returns one of:
  - `IdentifiedEffect`: the best set that is valid in every member graph, ranked by
    how many of its combinations with the cause the training rows support only thinly,
    then by size;
  - `PartiallyIdentifiedEffect`: one candidate set per parent set of the cause, when
    the member graphs disagree on the cause's parents (Maathuis, Kalisch and Bühlmann,
    2009);
  - `UnidentifiedEffect`: no set of the eligible variables works.

  `check_declared` raises `InvalidAdjustmentSet` for a declared set that fails in some
  member graph.
- `learning.PCStructureLearner` runs the PC algorithm through causal-learn on the rows,
  discretized into a few levels per column, and `learning.RelationalStructureLearner`
  does the same on a relational circuit's class table and part tables. A part attribute
  becomes one template node for every part of its collection.
- `background_knowledge.BackgroundKnowledge` carries what the data cannot settle:
  - tiers, where a variable is never caused by one in a later tier;
  - required and forbidden edges;
  - deterministic copies, which the tests skip and the graph restores as children of
    their source.

## Using it from a query

`krrood`'s `ProbabilisticBackend` takes a `causal_structure_registry`. With one, a query
that marks `cause` and `causes_effect` but no `confounder` has its adjustment set chosen
from the learned graph, a marked `confounder` set is checked, and an effect the graph
does not identify raises `EffectNotIdentified`.
`ProbabilisticBackend.interventional_range` answers a partially identified effect with
the lowest and highest probability across the candidate sets. A query with
`causes_effect` and no `cause` lets `rank_causes` rank every variable that may be an
ancestor of the effect. See the EQL causality guide for examples.

## What the data has to supply

Learning the structure from observational data has limits:

- **Direction.** Without background knowledge only the edges into colliders are
  oriented. Temporal order, such as "the environment is chosen before the friction",
  orients most of the rest.
- **Deterministic copies.** A variable computed from another breaks the independence
  tests. Testing two variables given the copy finds them independent and deletes a real
  edge, so copies have to be declared.
- **Sample size.** A discrete independence test over many cells has little power, so
  small samples miss real edges and the chosen set can be invalid.
- **Hidden confounders.** PC assumes every common cause is recorded.
- **The relational assumption.** A relational circuit lets an instance's attributes
  affect its parts only through the aggregation statistics, and the relational learner
  follows it.
- **Interventions on counts.** Setting an aggregation statistic is ambiguous once it
  has parents among the part attributes, since many configurations of the parts give
  the same count (Spirtes and Scheines, 2004).

## Evaluation

```{note}
Filled in from the structure study once it completes.
```
