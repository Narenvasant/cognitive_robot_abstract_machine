# Tracy clutter picking: relational circuit against flat-table tree

Both pipelines were fitted on the same recorded attempts and asked the same `cause`/`causes_effect` EQL queries.

- recorded attempts: 300 (240 to fit on, 60 held out)
- neighbours per attempt: 9
- attempts whose target was lifted: 65.3%

## How often the pick came up

| environment | attempts | lifted % |
|---|---|---|
| bin | 158 | 45.6 |
| table | 142 | 87.3 |

| friction coefficient | attempts | lifted % |
|---|---|---|
| 0.125 | 76 | 0.0 |
| 0.1875 | 78 | 87.2 |
| 0.25 | 79 | 77.2 |
| 0.375 | 24 | 100.0 |
| 0.5 | 17 | 100.0 |
| 0.75 | 26 | 100.0 |

| adjacent neighbours | attempts | lifted % |
|---|---|---|
| 0 | 130 | 86.9 |
| 1 | 28 | 85.7 |
| 2 | 73 | 45.2 |
| 3 | 48 | 41.7 |
| 4 | 16 | 25.0 |
| 5 | 5 | 40.0 |


## Which questions each pipeline can answer

| question | relational circuit | flat-table tree |
|---|---|---|
| In a clutter of 9 neighbours, which grasp friction coefficient causes the target to be lifted, adjusting for the environment? | answered: do(cause = 0.375) gives the effect with probability 1.000; EQL's search settles on 0.1875 (0.885) | answered: do(cause = 0.375) gives the effect with probability 1.000; EQL's search settles on 0.1875 (0.862) |
| In a clutter of 9 neighbours, how many of them standing adjacent to the target causes it to be lifted, adjusting for the environment? | answered: do(cause = 0) gives the effect with probability 0.881; EQL's search settles on 0 (0.881) | answered: do(cause = 0) gives the effect with probability 0.881; EQL's search settles on 0 (0.881) |
| In a clutter of 9 neighbours, does neighbour 0 standing along the fingers' closing axis cause it to be disturbed by the pick? | answered: do(cause = along) gives the effect with probability 0.134; EQL's search settles on along (0.134) | answered: do(cause = along) gives the effect with probability 0.220; EQL's search settles on along (0.220) |
| In a clutter of 4 neighbours, which grasp friction coefficient causes the target to be lifted, adjusting for the environment? | answered: do(cause = 0.375) gives the effect with probability 1.000; EQL's search settles on 0.1875 (0.885) | answered: do(cause = 0.375) gives the effect with probability 1.000; EQL's search settles on 0.1875 (0.862) |
| In a clutter of 12 neighbours, how many of them standing adjacent to the target causes it to be lifted, adjusting for the environment? | answered: do(cause = 0) gives the effect with probability 0.881; EQL's search settles on 0 (0.881) | refused: the fitted table has no column for the queried variables |
| In a clutter of 12 neighbours, does neighbour 11 standing along the fingers' closing axis cause it to be disturbed by the pick? | answered: do(cause = along) gives the effect with probability 0.134; EQL's search settles on along (0.134) | refused: the fitted table has no column for the queried variables |

An answer names the cause region whose intervention makes the effect most likely after adjustment, and the region EQL's own `cause` search settles on: the region most probable once the effect is required to hold, from which the query's samples are drawn.

## Fit and likelihood

| pipeline | models fitted | training seconds | nodes | edges | held-out coverage % | mean log-likelihood (covered) | mean log-likelihood (covered by both) |
|---|---|---|---|---|---|---|---|
| relational circuit | 5 | 7.98 | 7737 | 7727 | 30.0 | 118.34 | 130.01 |
| flat-table tree | 4 | 3.01 | 5922 | 5918 | 56.7 | 81.03 | 79.96 |

Coverage is the share of held-out attempts inside the model's support; a tree's leaves span only the ranges they saw. Log-likelihoods are over an attempt's observed attributes, its own and its neighbours'.

## Seconds per question

| question | relational circuit, first ask | relational circuit, asked again | flat-table tree, first ask | flat-table tree, asked again |
|---|---|---|---|---|
| friction_causes_lift_9_neighbours | 5.91 | 3.37 | 1.14 | 0.75 |
| crowding_causes_lift_9_neighbours | 4.91 | 4.99 | 1.15 | 0.15 |
| closing_axis_side_causes_disturbance_of_neighbour_0_of_9 | 5.50 | 4.81 | 0.69 | 0.09 |
| friction_causes_lift_4_neighbours | 2.05 | 2.01 | 0.12 | 0.12 |
| crowding_causes_lift_12_neighbours | 6.42 | 6.50 | 0.04 | 0.04 |
| closing_axis_side_causes_disturbance_of_neighbour_11_of_12 | 7.36 | 5.70 | 0.28 | 0.28 |

The first ask includes fitting the cause-specific model the first time that cause is asked about; asked again, only grounding, verification and adjustment remain.

## In a clutter of 9 neighbours, which grasp friction coefficient causes the target to be lifted, adjusting for the environment?

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0.125 | 0.242 | 0.000 | 0.000 |
| 0.1875 | 0.271 | 0.862 | 0.885 |
| 0.25 | 0.275 | 0.758 | 0.790 |
| 0.375 | 0.075 | 1.000 | 1.000 |
| 0.5 | 0.062 | 1.000 | 1.000 |
| 0.75 | 0.075 | 1.000 | 1.000 |

### flat-table tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0.125 | 0.242 | 0.000 | 0.000 |
| 0.1875 | 0.271 | 0.862 | 0.862 |
| 0.25 | 0.275 | 0.758 | 0.757 |
| 0.375 | 0.075 | 1.000 | 1.000 |
| 0.5 | 0.062 | 1.000 | 1.000 |
| 0.75 | 0.075 | 1.000 | 1.000 |


## In a clutter of 9 neighbours, how many of them standing adjacent to the target causes it to be lifted, adjusting for the environment?

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.421 | 0.881 | 0.881 |
| 1 | 0.100 | 0.875 | 0.875 |
| 2 | 0.233 | 0.446 | 0.446 |
| 3 | 0.171 | 0.415 | 0.415 |
| 4 | 0.054 | 0.231 | 0.231 |
| 5 | 0.021 | 0.400 | 0.400 |

### flat-table tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.421 | 0.881 | 0.881 |
| 1 | 0.100 | 0.875 | 0.875 |
| 2 | 0.233 | 0.446 | 0.446 |
| 3 | 0.171 | 0.415 | 0.415 |
| 4 | 0.054 | 0.231 | 0.231 |
| 5 | 0.021 | 0.400 | 0.400 |


## In a clutter of 9 neighbours, does neighbour 0 standing along the fingers' closing axis cause it to be disturbed by the pick?

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| across | 0.497 | 0.010 | 0.010 |
| along | 0.503 | 0.134 | 0.134 |

### flat-table tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| across | 0.471 | 0.027 | 0.027 |
| along | 0.529 | 0.220 | 0.220 |


## In a clutter of 4 neighbours, which grasp friction coefficient causes the target to be lifted, adjusting for the environment?

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0.125 | 0.242 | 0.000 | 0.000 |
| 0.1875 | 0.271 | 0.862 | 0.885 |
| 0.25 | 0.275 | 0.758 | 0.790 |
| 0.375 | 0.075 | 1.000 | 1.000 |
| 0.5 | 0.062 | 1.000 | 1.000 |
| 0.75 | 0.075 | 1.000 | 1.000 |

### flat-table tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0.125 | 0.242 | 0.000 | 0.000 |
| 0.1875 | 0.271 | 0.862 | 0.862 |
| 0.25 | 0.275 | 0.758 | 0.757 |
| 0.375 | 0.075 | 1.000 | 1.000 |
| 0.5 | 0.062 | 1.000 | 1.000 |
| 0.75 | 0.075 | 1.000 | 1.000 |


## In a clutter of 12 neighbours, how many of them standing adjacent to the target causes it to be lifted, adjusting for the environment?

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.421 | 0.881 | 0.881 |
| 1 | 0.100 | 0.875 | 0.875 |
| 2 | 0.233 | 0.446 | 0.446 |
| 3 | 0.171 | 0.415 | 0.415 |
| 4 | 0.054 | 0.231 | 0.231 |
| 5 | 0.021 | 0.400 | 0.400 |

### flat-table tree

Refused: the fitted table has no column for the queried variables.


## In a clutter of 12 neighbours, does neighbour 11 standing along the fingers' closing axis cause it to be disturbed by the pick?

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| across | 0.497 | 0.010 | 0.010 |
| along | 0.503 | 0.134 | 0.134 |

### flat-table tree

Refused: the fitted table has no column for the queried variables.

