# Tracy clutter picking: relational circuit against flat-table tree

Tracy's left arm picks one milk carton out of a ten-carton clutter in MuJoCo, holding it by contact friction alone. Every attempt is recorded as a relational scene: the attempt's own attributes (environment, grasp friction, grasp yaw, whether the target came up) and one exchangeable part per neighbouring carton (its position relative to the target, its distance band, which side of the fingers' closing axis it stands on, and how far the pick shoved it).

Two pipelines were fitted on the same recorded attempts and asked the same `cause`/`causes_effect` EQL queries:

- **relational circuit**: a relational probabilistic circuit fitted on the scenes' relational structure -- one circuit over the attempt's own attributes and its aggregation statistics, one template over a neighbour's attributes -- grounded per query into a circuit over exactly the queried objects and registered as a causal circuit;
- **flat-table tree**: a joint probability tree fitted on the same attempts flattened into one fixed-width table, one block of columns per neighbour index, registered as a causal circuit the same way.

Both answer a query by backdoor adjustment: the model is stratified so it is support-deterministic over the cause, the effect's probability is read off every region of the cause, and any variable the query marks as a confounder is summed out of that reading.

## Setup

- recorded attempts: 300 (240 to fit on, 60 held out)
- neighbours per attempt: 9
- attempts whose target was lifted: 65.3%
- fewest training rows per leaf: 15 in a cause-specific model, 50 in the plain model that scores held-out attempts

## How often the pick came up

The recorded attempts themselves, before any model: the share of attempts whose target was still held at the end, grouped by the environment the clutter stood in, by the grasp's friction coefficient, and by how many neighbours stood adjacent to the target (closer than the fingers' sweep). This is the picking efficiency in clutter the models are asked to explain.

| environment | attempts | lifted |
|---|---|---|
| bin | 158 | 45.6% |
| table | 142 | 87.3% |

| friction coefficient | attempts | lifted |
|---|---|---|
| 0.125 | 76 | 0.0% |
| 0.1875 | 78 | 87.2% |
| 0.25 | 79 | 77.2% |
| 0.375 | 24 | 100.0% |
| 0.5 | 17 | 100.0% |
| 0.75 | 26 | 100.0% |

| adjacent neighbours | attempts | lifted |
|---|---|---|
| 0 | 130 | 86.9% |
| 1 | 28 | 85.7% |
| 2 | 73 | 45.2% |
| 3 | 48 | 41.7% |
| 4 | 16 | 25.0% |
| 5 | 5 | 40.0% |


## Which questions each pipeline can answer

One row per question, one column per pipeline. An answered cell says, in words, which setting of the cause makes the effect most likely after adjustment and how likely, against the least favourable setting; a refused cell says why the pipeline could not answer at all.

| question | relational circuit | flat-table tree |
|---|---|---|
| In a clutter of 9 neighbours, which grasp friction coefficient causes the target to be lifted, adjusting for the environment? | answered: with a grasp friction coefficient of 0.375, the target is lifted with probability 1.00, the highest of any setting; with a grasp friction coefficient of 0.125 it is only 0.00. | answered: with a grasp friction coefficient of 0.375, the target is lifted with probability 1.00, the highest of any setting; with a grasp friction coefficient of 0.125 it is only 0.00. |
| In a clutter of 9 neighbours, how many of them standing adjacent to the target causes it to be lifted, adjusting for the environment? | answered: with 0 adjacent neighbours, the target is lifted with probability 0.88, the highest of any setting; with 4 adjacent neighbours it is only 0.23. | answered: with 0 adjacent neighbours, the target is lifted with probability 0.88, the highest of any setting; with 4 adjacent neighbours it is only 0.23. |
| In a clutter of 9 neighbours, does neighbour 0 standing along the fingers' closing axis cause it to be disturbed by the pick? | answered: with neighbour 0 standing along the closing axis, neighbour 0 is disturbed with probability 0.17, the highest of any setting; with neighbour 0 standing across the closing axis it is only 0.01. | answered: with neighbour 0 standing along the closing axis, neighbour 0 is disturbed with probability 0.22, the highest of any setting; with neighbour 0 standing across the closing axis it is only 0.03. |
| In a clutter of 4 neighbours, which grasp friction coefficient causes the target to be lifted, adjusting for the environment? | answered: with a grasp friction coefficient of 0.375, the target is lifted with probability 1.00, the highest of any setting; with a grasp friction coefficient of 0.125 it is only 0.00. | answered: with a grasp friction coefficient of 0.375, the target is lifted with probability 1.00, the highest of any setting; with a grasp friction coefficient of 0.125 it is only 0.00. |
| In a clutter of 12 neighbours, how many of them standing adjacent to the target causes it to be lifted, adjusting for the environment? | answered: with 0 adjacent neighbours, the target is lifted with probability 0.88, the highest of any setting; with 4 adjacent neighbours it is only 0.23. | refused: the fitted table has no column for the queried variables. |
| In a clutter of 12 neighbours, does neighbour 11 standing along the fingers' closing axis cause it to be disturbed by the pick? | answered: with neighbour 11 standing along the closing axis, neighbour 11 is disturbed with probability 0.17, the highest of any setting; with neighbour 11 standing across the closing axis it is only 0.01. | refused: the fitted table has no column for the queried variables. |

The three questions about the recorded clutter size can be put to either pipeline; the three about other clutter sizes have no columns in the flat table, so only a model that grounds itself for the queried objects can answer them.

## Fit and likelihood

What each pipeline cost and how well it explains attempts it never saw. *Models fitted* counts the plain model plus one support-deterministic model per distinct cause the questions asked about; *training seconds* and the *nodes*/*edges* of every fitted circuit are summed over them. *Held-out coverage* is the share of held-out attempts that lie inside the plain model's support at all -- a tree's leaves span only the value ranges they were fitted on, so an attempt with any attribute outside every leaf's range has zero likelihood. The *mean log-likelihood* is over the covered attempts only, on an attempt's observed attributes (its own scalars and every neighbour's); the last column restricts it to the attempts both pipelines cover, so the two numbers are over the same rows.

| pipeline | models fitted | training seconds | nodes | edges | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by both) |
|---|---|---|---|---|---|---|---|
| relational circuit | 5 | 8.01 | 9942 | 9932 | 30.0% | 117.90 | 130.07 |
| flat-table tree | 4 | 3.48 | 6838 | 6834 | 56.7% | 81.03 | 79.96 |

## Seconds per question

Wall-clock time from asking to the answer or the refusal. The *first ask* of a cause includes fitting that cause's own support-deterministic model; *asked again* repeats the question with every model fitted, so only grounding (for the relational circuit), verification and backdoor adjustment remain. A refusal is fast when it is a schema check; a relational answer about a larger clutter grounds a larger circuit and takes longer.

| question | relational circuit, first ask | relational circuit, asked again | flat-table tree, first ask | flat-table tree, asked again |
|---|---|---|---|---|
| friction_causes_lift_9_neighbours | 5.37 | 4.82 | 1.59 | 0.16 |
| crowding_causes_lift_9_neighbours | 6.18 | 5.43 | 1.22 | 0.19 |
| closing_axis_side_causes_disturbance_of_neighbour_0_of_9 | 7.92 | 6.90 | 0.70 | 0.10 |
| friction_causes_lift_4_neighbours | 2.57 | 2.60 | 0.15 | 1.11 |
| crowding_causes_lift_12_neighbours | 8.21 | 8.11 | 0.04 | 0.05 |
| closing_axis_side_causes_disturbance_of_neighbour_11_of_12 | 11.71 | 8.82 | 0.26 | 0.26 |

## What the results show

- The relational circuit answered 6 of 6 questions.
- The flat-table tree answered 4 of 6 questions, refusing `crowding_causes_lift_12_neighbours` because the fitted table has no column for the queried variables; `closing_axis_side_causes_disturbance_of_neighbour_11_of_12` because the fitted table has no column for the queried variables.
- On `friction_causes_lift_9_neighbours`, both pipelines find a grasp friction coefficient of 0.375 the most effective setting (adjusted probabilities 1.00, 1.00).
- On `crowding_causes_lift_9_neighbours`, both pipelines find 0 adjacent neighbours the most effective setting (adjusted probabilities 0.88, 0.88).
- On `closing_axis_side_causes_disturbance_of_neighbour_0_of_9`, both pipelines find neighbour 0 standing along the closing axis the most effective setting (adjusted probabilities 0.17, 0.22).
- On `friction_causes_lift_4_neighbours`, both pipelines find a grasp friction coefficient of 0.375 the most effective setting (adjusted probabilities 1.00, 1.00).
- The flat-table tree covers the most held-out attempts (56.7%); on the attempts both cover, the relational circuit assigns the higher mean log-likelihood (130.07).
- The relational circuit takes 6.11 seconds per answered question on average once its models are fitted.
- The flat-table tree takes 0.39 seconds per answered question on average once its models are fitted.

## In a clutter of 9 neighbours, which grasp friction coefficient causes the target to be lifted, adjusting for the environment?

One row per region of the cause the model distinguishes. *P(region)* is how much of the recorded population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0.125 | 0.242 | 0.000 | 0.000 |
| 0.1875 | 0.271 | 0.862 | 0.885 |
| 0.25 | 0.275 | 0.758 | 0.790 |
| 0.375 | 0.075 | 1.000 | 1.000 |
| 0.5 | 0.062 | 1.000 | 1.000 |
| 0.75 | 0.075 | 1.000 | 1.000 |

EQL's own `cause` search settles on 0.1875: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### flat-table tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0.125 | 0.242 | 0.000 | 0.000 |
| 0.1875 | 0.271 | 0.862 | 0.862 |
| 0.25 | 0.275 | 0.758 | 0.757 |
| 0.375 | 0.075 | 1.000 | 1.000 |
| 0.5 | 0.062 | 1.000 | 1.000 |
| 0.75 | 0.075 | 1.000 | 1.000 |

EQL's own `cause` search settles on 0.1875: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.86).


## In a clutter of 9 neighbours, how many of them standing adjacent to the target causes it to be lifted, adjusting for the environment?

One row per region of the cause the model distinguishes. *P(region)* is how much of the recorded population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.421 | 0.881 | 0.881 |
| 1 | 0.100 | 0.875 | 0.875 |
| 2 | 0.233 | 0.446 | 0.446 |
| 3 | 0.171 | 0.415 | 0.415 |
| 4 | 0.054 | 0.231 | 0.231 |
| 5 | 0.021 | 0.400 | 0.400 |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### flat-table tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.421 | 0.881 | 0.881 |
| 1 | 0.100 | 0.875 | 0.875 |
| 2 | 0.233 | 0.446 | 0.446 |
| 3 | 0.171 | 0.415 | 0.415 |
| 4 | 0.054 | 0.231 | 0.231 |
| 5 | 0.021 | 0.400 | 0.400 |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).


## In a clutter of 9 neighbours, does neighbour 0 standing along the fingers' closing axis cause it to be disturbed by the pick?

One row per region of the cause the model distinguishes. *P(region)* is how much of the recorded population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| across | 0.496 | 0.012 | 0.012 |
| along | 0.504 | 0.170 | 0.170 |

EQL's own `cause` search settles on along: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.17).

### flat-table tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| across | 0.471 | 0.027 | 0.027 |
| along | 0.529 | 0.220 | 0.220 |

EQL's own `cause` search settles on along: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.22).


## In a clutter of 4 neighbours, which grasp friction coefficient causes the target to be lifted, adjusting for the environment?

One row per region of the cause the model distinguishes. *P(region)* is how much of the recorded population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0.125 | 0.242 | 0.000 | 0.000 |
| 0.1875 | 0.271 | 0.862 | 0.885 |
| 0.25 | 0.275 | 0.758 | 0.790 |
| 0.375 | 0.075 | 1.000 | 1.000 |
| 0.5 | 0.062 | 1.000 | 1.000 |
| 0.75 | 0.075 | 1.000 | 1.000 |

EQL's own `cause` search settles on 0.1875: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### flat-table tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0.125 | 0.242 | 0.000 | 0.000 |
| 0.1875 | 0.271 | 0.862 | 0.862 |
| 0.25 | 0.275 | 0.758 | 0.757 |
| 0.375 | 0.075 | 1.000 | 1.000 |
| 0.5 | 0.062 | 1.000 | 1.000 |
| 0.75 | 0.075 | 1.000 | 1.000 |

EQL's own `cause` search settles on 0.1875: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.86).


## In a clutter of 12 neighbours, how many of them standing adjacent to the target causes it to be lifted, adjusting for the environment?

One row per region of the cause the model distinguishes. *P(region)* is how much of the recorded population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.421 | 0.881 | 0.881 |
| 1 | 0.100 | 0.875 | 0.875 |
| 2 | 0.233 | 0.446 | 0.446 |
| 3 | 0.171 | 0.415 | 0.415 |
| 4 | 0.054 | 0.231 | 0.231 |
| 5 | 0.021 | 0.400 | 0.400 |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.88).

### flat-table tree

Refused: the fitted table has no column for the queried variables.


## In a clutter of 12 neighbours, does neighbour 11 standing along the fingers' closing axis cause it to be disturbed by the pick?

One row per region of the cause the model distinguishes. *P(region)* is how much of the recorded population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| across | 0.496 | 0.012 | 0.012 |
| along | 0.504 | 0.170 | 0.170 |

EQL's own `cause` search settles on along: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.17).

### flat-table tree

Refused: the fitted table has no column for the queried variables.

