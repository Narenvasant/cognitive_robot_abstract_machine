# Mutagenesis: relational circuit against flat-table tree

The CTU Mutagenesis dataset records 188 nitroaromatic molecules, each as its own attributes (the `ind1` structural indicator, `logp`, `lumo`, and whether it tested mutagenic) with one exchangeable part per atom (element, atom-type code, partial charge, number of bonds) and one per bond (its type). A molecule has between 14 and 40 atoms, and its atoms have no canonical order.

Two pipelines were fitted on the same molecules and asked the same `cause`/`causes_effect` EQL queries:

- **relational circuit**: a relational probabilistic circuit fitted on the molecules' relational structure, one circuit over the molecule's own attributes and its aggregation counts (chlorine atoms, branching atoms, double bonds, aromatic bonds), one template over an atom's attributes and one over a bond's, grounded per query into a circuit over exactly the queried molecule, atoms and bonds and registered as a causal circuit;
- **flat-table tree**: a joint probability tree fitted on the same molecules flattened into one table of the molecule's own attributes and the same four aggregation counts, registered as a causal circuit the same way. There is no fixed-width unrolling of the atoms in which one column would mean the same thing in two molecules, so the table is the classic propositional summary of a relational example: scalars plus counts, and no atoms or bonds.

Both answer a query by backdoor adjustment: the model is stratified so it is support-deterministic over the cause, the effect's probability is read off every region of the cause, and any variable the query marks as a confounder is summed out of that reading. Every query lists one atom and one bond with all their attributes open, which is what makes grounding retain the molecule's counts as variables; the flat table ignores parts a query says nothing about and refuses a query that constrains one.

## Setup

- molecules: 188 (150 to fit on, 38 held out)
- molecules that tested mutagenic: 66.5%
- fewest training rows per leaf: 15 in a cause-specific model, 50 in the plain model that scores held-out molecules

## How often a molecule is mutagenic

The molecules themselves, before any model: the share that tested mutagenic, grouped by the `ind1` indicator, by how many branching atoms (atoms with three or four bonds, the ring-fusion and branch points of the molecular graph) the molecule has, and by how many of its bonds are aromatic. This is the signal the models are asked to explain.

| ind1 | molecules | mutagenic |
|---|---|---|
| False | 85 | 30.6% |
| True | 103 | 96.1% |

| branching atoms | molecules | mutagenic |
|---|---|---|
| 7 | 7 | 0.0% |
| 8 | 12 | 33.3% |
| 9 | 12 | 16.7% |
| 10 | 13 | 15.4% |
| 11 | 7 | 14.3% |
| 12 | 5 | 40.0% |
| 13 | 16 | 68.8% |
| 14 | 21 | 61.9% |
| 15 | 20 | 85.0% |
| 16 | 12 | 83.3% |
| 17 | 19 | 100.0% |
| 18 | 14 | 100.0% |
| 19 | 7 | 100.0% |
| 20 | 1 | 100.0% |
| 21 | 15 | 100.0% |
| 22 | 4 | 100.0% |
| 24 | 2 | 100.0% |
| 25 | 1 | 100.0% |

| aromatic bonds | molecules | mutagenic |
|---|---|---|
| 5 | 1 | 0.0% |
| 6 | 32 | 21.9% |
| 10 | 11 | 27.3% |
| 11 | 17 | 35.3% |
| 12 | 60 | 70.0% |
| 14 | 3 | 100.0% |
| 15 | 4 | 100.0% |
| 16 | 7 | 100.0% |
| 17 | 16 | 100.0% |
| 18 | 1 | 100.0% |
| 19 | 21 | 100.0% |
| 21 | 1 | 100.0% |
| 22 | 1 | 100.0% |
| 24 | 10 | 100.0% |
| 26 | 2 | 100.0% |
| 30 | 1 | 100.0% |


## Which questions each pipeline can answer

One row per question, one column per pipeline. An answered cell says, in words, which setting of the cause makes the effect most likely after adjustment and how likely, against the least favourable setting; a refused cell says why the pipeline could not answer at all.

| question | relational circuit | flat-table tree |
|---|---|---|
| How many branching atoms cause a molecule to be mutagenic, adjusting for the ind1 indicator? | answered: with 17 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 7 branching atoms it is only 0.00. | answered: with 17 branching atoms, the molecule is mutagenic with probability 1.00, the highest of any setting; with 7 branching atoms it is only 0.00. |
| How many aromatic bonds cause a molecule to be mutagenic, adjusting for the ind1 indicator? | answered: with 14 aromatic bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 5 aromatic bonds it is only 0.00. | answered: with 14 aromatic bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 5 aromatic bonds it is only 0.00. |
| How many double bonds cause a molecule to be mutagenic, adjusting for the ind1 indicator? | answered: with 9 double bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 5 double bonds it is only 0.50. | answered: with 9 double bonds, the molecule is mutagenic with probability 1.00, the highest of any setting; with 5 double bonds it is only 0.50. |
| Does the ind1 indicator cause a molecule to be mutagenic, adjusting for its branching-atom count? | answered: with ind1 = True, the molecule is mutagenic with probability 0.94, the highest of any setting; with ind1 = False it is only 0.37. | answered: with ind1 = True, the molecule is mutagenic with probability 0.94, the highest of any setting; with ind1 = False it is only 0.37. |
| Does the ind1 indicator cause atom 0 of a molecule to be carbon? | answered: with ind1 = True, atom 0 is carbon with probability 0.52, the highest of any setting; with ind1 = False it is only 0.44. | refused: the fitted table has no column for the queried variables. |
| How many branching atoms cause atom 0 of a molecule to be terminal, with a single bond? | answered: with 8 branching atoms, atom 0 is terminal with probability 0.51, the highest of any setting; with 25 branching atoms it is only 0.34. | refused: the fitted table has no column for the queried variables. |
| Does the element of atom 0 of a molecule cause it to be terminal, with a single bond? | refused: the cause regions read off the model overlap. | refused: the fitted table has no column for the queried variables. |

The questions whose cause and effect are both molecule-level attributes or counts can be put to either pipeline. A question whose effect is one atom's own attribute has no column in the flat table, so only a model that grounds itself for the queried atoms can answer it. A question whose cause is one atom's own attribute is refused by both. The flat table has no column for it. The relational circuit, grounding with the molecule's counts left open, mixes one copy of the atom template per sampled count, and those copies overlap on the atom's element without being identical (a copy for a molecule with no chlorine has no chlorine atom, a copy for one with some has). Support-determinism verification only inspects a sum whose children are disjoint somewhere, so it lets that mixture through, and the regions then read off it carry more probability together than one; the experiment refuses an answer whose regions do not partition the cause rather than report it.

## Fit and likelihood

What each pipeline cost and how well it explains molecules it never saw. *Models fitted* counts the plain model plus one support-deterministic model per distinct cause the questions asked about; *training seconds* and the *nodes*/*edges* of every fitted circuit are summed over them, which for the relational circuit includes the atom and bond templates. *Held-out coverage* is the share of held-out molecules that lie inside the plain model's support at all, since a tree's leaves span only the value ranges they were fitted on. The *mean log-likelihood* is over the covered molecules only, on the molecule's own attributes and its four counts, the variables both pipelines model; the last column restricts it to the molecules both pipelines cover, so the two numbers are over the same rows.

| pipeline | models fitted | training seconds | nodes | edges | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by both) |
|---|---|---|---|---|---|---|---|
| relational circuit | 6 | 28.20 | 7507 | 7489 | 89.5% | -8.24 | -8.24 |
| flat-table tree | 5 | 6.90 | 678 | 673 | 89.5% | -8.24 | -8.24 |

A held-out molecule is more than its scalars and counts: it is also every one of its atoms and bonds. Only a pipeline that models the parts can score those, as the class circuit over the scalars and counts times each part template over one atom or bond given the counts.

| pipeline | held-out coverage (whole molecule) | mean log-likelihood (whole molecule, covered) |
|---|---|---|
| relational circuit | 60.5% | 5.56 |

## Seconds per question

Wall-clock time from asking to the answer or the refusal. The *first ask* of a cause includes fitting that cause's own support-deterministic model; *asked again* repeats the question with every model fitted, so only grounding (for the relational circuit), verification and backdoor adjustment remain. A refusal is fast when it is a schema check; a relational answer draws Monte-Carlo samples for every count the query leaves open and grounds one atom template per sampled value, which is where its time goes.

| question | relational circuit, first ask | relational circuit, asked again | flat-table tree, first ask | flat-table tree, asked again |
|---|---|---|---|---|
| branching_atom_count_causes_mutagenicity | 7.35 | 2.62 | 2.48 | 0.30 |
| aromatic_bond_count_causes_mutagenicity | 6.32 | 2.63 | 1.26 | 0.23 |
| double_bond_count_causes_mutagenicity | 6.09 | 3.00 | 1.35 | 0.13 |
| indicator_causes_mutagenicity | 5.07 | 1.71 | 2.08 | 0.15 |
| indicator_causes_carbon_atom_0 | 3.22 | 3.93 | 0.00 | 0.00 |
| branching_atom_count_causes_terminal_atom_0 | 3.06 | 3.92 | 0.00 | 0.00 |
| element_causes_terminal_atom_0 | 5.03 | 1.05 | 0.00 | 0.00 |

## What the results show

- The relational circuit answered 6 of 7 questions, refusing `element_causes_terminal_atom_0` because the cause regions read off the model overlap.
- The flat-table tree answered 4 of 7 questions, refusing `indicator_causes_carbon_atom_0` because the fitted table has no column for the queried variables; `branching_atom_count_causes_terminal_atom_0` because the fitted table has no column for the queried variables; `element_causes_terminal_atom_0` because the fitted table has no column for the queried variables.
- On `branching_atom_count_causes_mutagenicity`, both pipelines find 17 branching atoms the most effective setting (adjusted probabilities 1.00, 1.00).
- On `aromatic_bond_count_causes_mutagenicity`, both pipelines find 14 aromatic bonds the most effective setting (adjusted probabilities 1.00, 1.00).
- On `double_bond_count_causes_mutagenicity`, both pipelines find 9 double bonds the most effective setting (adjusted probabilities 1.00, 1.00).
- On `indicator_causes_mutagenicity`, both pipelines find ind1 = True the most effective setting (adjusted probabilities 0.94, 0.94).
- On a molecule's own attributes and counts, the pipelines assign the same mean log-likelihood (-8.24) to the held-out molecules and cover the same share of them (89.5%): the relational circuit's class-level circuit and the flat-table tree are fitted on the same rows with the same settings, so they are the same tree. The relational circuit differs in what it models besides: the atoms and bonds.
- The relational circuit takes 2.97 seconds per answered question on average once its models are fitted.
- The flat-table tree takes 0.20 seconds per answered question on average once its models are fitted.

## How many branching atoms cause a molecule to be mutagenic, adjusting for the ind1 indicator?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 7 | 0.047 | 0.000 | 0.000 |
| 8 | 0.053 | 0.375 | 0.375 |
| 9 | 0.060 | 0.111 | 0.111 |
| 10 | 0.073 | 0.182 | 0.182 |
| 11 | 0.033 | 0.200 | 0.200 |
| 12 | 0.027 | 0.250 | 0.250 |
| 13 | 0.073 | 0.727 | 0.727 |
| 14 | 0.113 | 0.588 | 0.588 |
| 15 | 0.113 | 0.824 | 0.824 |
| 16 | 0.080 | 0.833 | 0.833 |
| 17 | 0.093 | 1.000 | 1.000 |
| 18 | 0.080 | 1.000 | 1.000 |
| 19 | 0.027 | 1.000 | 1.000 |
| 20 | 0.007 | 1.000 | 1.000 |
| 21 | 0.080 | 1.000 | 1.000 |
| 22 | 0.020 | 1.000 | 1.000 |
| 24 | 0.013 | 1.000 | 1.000 |
| 25 | 0.007 | 1.000 | 1.000 |

EQL's own `cause` search settles on 17: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).

### flat-table tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 7 | 0.047 | 0.000 | 0.000 |
| 8 | 0.053 | 0.375 | 0.375 |
| 9 | 0.060 | 0.111 | 0.111 |
| 10 | 0.073 | 0.182 | 0.182 |
| 11 | 0.033 | 0.200 | 0.200 |
| 12 | 0.027 | 0.250 | 0.250 |
| 13 | 0.073 | 0.727 | 0.727 |
| 14 | 0.113 | 0.588 | 0.588 |
| 15 | 0.113 | 0.824 | 0.824 |
| 16 | 0.080 | 0.833 | 0.833 |
| 17 | 0.093 | 1.000 | 1.000 |
| 18 | 0.080 | 1.000 | 1.000 |
| 19 | 0.027 | 1.000 | 1.000 |
| 20 | 0.007 | 1.000 | 1.000 |
| 21 | 0.080 | 1.000 | 1.000 |
| 22 | 0.020 | 1.000 | 1.000 |
| 24 | 0.013 | 1.000 | 1.000 |
| 25 | 0.007 | 1.000 | 1.000 |

EQL's own `cause` search settles on 17: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 1.00).


## How many aromatic bonds cause a molecule to be mutagenic, adjusting for the ind1 indicator?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 5 | 0.007 | 0.000 | 0.000 |
| 6 | 0.167 | 0.240 | 0.240 |
| 10 | 0.053 | 0.250 | 0.250 |
| 11 | 0.100 | 0.333 | 0.333 |
| 12 | 0.347 | 0.712 | 0.713 |
| 14 | 0.007 | 1.000 | 1.000 |
| 15 | 0.020 | 1.000 | 1.000 |
| 16 | 0.027 | 1.000 | 1.000 |
| 17 | 0.080 | 1.000 | 1.000 |
| 18 | 0.007 | 1.000 | 1.000 |
| 19 | 0.120 | 1.000 | 1.000 |
| 22 | 0.007 | 1.000 | 1.000 |
| 24 | 0.040 | 1.000 | 1.000 |
| 26 | 0.013 | 1.000 | 1.000 |
| 30 | 0.007 | 1.000 | 1.000 |

EQL's own `cause` search settles on 12: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.71).

### flat-table tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 5 | 0.007 | 0.000 | 0.000 |
| 6 | 0.167 | 0.240 | 0.240 |
| 10 | 0.053 | 0.250 | 0.250 |
| 11 | 0.100 | 0.333 | 0.333 |
| 12 | 0.347 | 0.712 | 0.713 |
| 14 | 0.007 | 1.000 | 1.000 |
| 15 | 0.020 | 1.000 | 1.000 |
| 16 | 0.027 | 1.000 | 1.000 |
| 17 | 0.080 | 1.000 | 1.000 |
| 18 | 0.007 | 1.000 | 1.000 |
| 19 | 0.120 | 1.000 | 1.000 |
| 22 | 0.007 | 1.000 | 1.000 |
| 24 | 0.040 | 1.000 | 1.000 |
| 26 | 0.013 | 1.000 | 1.000 |
| 30 | 0.007 | 1.000 | 1.000 |

EQL's own `cause` search settles on 12: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.71).


## How many double bonds cause a molecule to be mutagenic, adjusting for the ind1 indicator?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 2 | 0.573 | 0.547 | 0.537 |
| 3 | 0.087 | 0.692 | 0.692 |
| 4 | 0.220 | 0.848 | 0.823 |
| 5 | 0.013 | 0.500 | 0.500 |
| 6 | 0.060 | 0.889 | 0.889 |
| 7 | 0.007 | 1.000 | 1.000 |
| 8 | 0.033 | 0.800 | 0.800 |
| 9 | 0.007 | 1.000 | 1.000 |

EQL's own `cause` search settles on 2: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.54).

### flat-table tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 2 | 0.573 | 0.547 | 0.537 |
| 3 | 0.087 | 0.692 | 0.692 |
| 4 | 0.220 | 0.848 | 0.823 |
| 5 | 0.013 | 0.500 | 0.500 |
| 6 | 0.060 | 0.889 | 0.889 |
| 7 | 0.007 | 1.000 | 1.000 |
| 8 | 0.033 | 0.800 | 0.800 |
| 9 | 0.007 | 1.000 | 1.000 |

EQL's own `cause` search settles on 2: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.54).


## Does the ind1 indicator cause a molecule to be mutagenic, adjusting for its branching-atom count?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| False | 0.453 | 0.309 | 0.371 |
| True | 0.547 | 0.951 | 0.938 |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.94).

### flat-table tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| False | 0.453 | 0.309 | 0.371 |
| True | 0.547 | 0.951 | 0.938 |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.94).


## Does the ind1 indicator cause atom 0 of a molecule to be carbon?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| False | 0.453 | 0.441 | 0.441 |
| True | 0.547 | 0.522 | 0.522 |

EQL's own `cause` search settles on True: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.52).

### flat-table tree

Refused: the fitted table has no column for the queried variables.


## How many branching atoms cause atom 0 of a molecule to be terminal, with a single bond?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 7 | 0.047 | 0.450 | 0.450 |
| 8 | 0.053 | 0.514 | 0.514 |
| 9 | 0.060 | 0.467 | 0.467 |
| 10 | 0.073 | 0.450 | 0.450 |
| 11 | 0.033 | 0.463 | 0.463 |
| 12 | 0.027 | 0.467 | 0.467 |
| 13 | 0.073 | 0.392 | 0.392 |
| 14 | 0.113 | 0.453 | 0.453 |
| 15 | 0.113 | 0.455 | 0.455 |
| 16 | 0.080 | 0.445 | 0.445 |
| 17 | 0.093 | 0.410 | 0.410 |
| 18 | 0.080 | 0.410 | 0.410 |
| 19 | 0.027 | 0.405 | 0.405 |
| 20 | 0.007 | 0.411 | 0.411 |
| 21 | 0.080 | 0.405 | 0.405 |
| 22 | 0.020 | 0.450 | 0.450 |
| 24 | 0.013 | 0.368 | 0.368 |
| 25 | 0.007 | 0.342 | 0.342 |

EQL's own `cause` search settles on 15: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.46).

### flat-table tree

Refused: the fitted table has no column for the queried variables.


## Does the element of atom 0 of a molecule cause it to be terminal, with a single bond?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

Refused: the cause regions read off the model overlap.

### flat-table tree

Refused: the fitted table has no column for the queried variables.

