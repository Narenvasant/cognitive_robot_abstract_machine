# GraspClutter6D: relational circuit against flat-table trees

The GraspClutter6D dataset records a thousand real, densely cluttered bin, shelf and table scenes, each photographed from thirteen poses by four cameras, with the ground-truth pose and the visible share of every object instance in every frame, and with analytic antipodal grasps annotated on every object model and checked for collision against every scene it stands in. A scene here is its own attributes (which object catalogue it is built from, how far its clutter is spread, how far it is stacked, and whether every object in it keeps a grasp) with one exchangeable part per object instance (size, diameter, visibility, occlusion, graspability) and one per camera frame (camera, distance, proximity, clarity). A scene holds between two and twenty object instances, and they have no canonical order; the annotation file lists them in the order they were labelled, and nothing ties a position to an identity.

Four pipelines were fitted on the same scenes and asked the same `cause`/`causes_effect` EQL queries:

- **relational circuit**: a relational probabilistic circuit fitted on the scenes' relational structure, one circuit over the scene's own attributes and its aggregation counts (small objects, occluded objects, near viewpoints, clear viewpoints), one template over an object's attributes and one over a viewpoint's, grounded per query into a circuit over exactly the queried scene, objects and viewpoints and registered as a causal circuit;
- **propositional tree**: a joint probability tree fitted on the scenes flattened into one table of the scene's own attributes and the same four counts, the classic propositional summary of a relational example, registered as a causal circuit the same way;
- **unrolled tree**: the same tree on a table that also carries every object's and viewpoint's attributes under the part's position, padded with an absent marker past a scene's last part, so that a column means whatever part a scene happens to list at that position;
- **scalars-only tree**: the same tree on the scene's own attributes alone, what a flat learner sees without the relational feature extraction.

Every flat tree answers a query by backdoor adjustment on a table column; the relational circuit does the same on the variable of a grounded circuit. In both, the model is stratified so it is support-deterministic over the cause, the effect's probability is read off every region of the cause, and any variable the query marks as a confounder is summed out of that reading. Every query lists one object and one viewpoint with all their attributes open, which is what makes grounding retain the scene's counts as variables; a flat table ignores parts a query says nothing about and refuses a query that constrains a column it does not have.

## Setup

- scenes: 954 (763 to fit on, 191 held out)
- scenes leaving every object graspable: 37.6%
- fewest training rows per leaf, as a share of the rows fitted on: 0.05 in a cause-specific model, 0.15 in the plain model that scores held-out scenes
- split seed: 0

## How often a scene leaves every object graspable

The scenes themselves, before any model: the share that leave every object instance with at least one antipodal grasp the rest of the scene does not block, grouped by the object catalogue the scene is built from, by how many of its objects are small, and by how many of them the cameras do not see whole. This is the signal the models are asked to explain.

| object catalogue | scenes | every object graspable |
|---|---|---|
| grasp | 487 | 43.1% |
| mixed | 161 | 28.0% |
| ycb-video | 306 | 34.0% |

| small objects | scenes | every object graspable |
|---|---|---|
| 0 | 198 | 47.5% |
| 1 | 45 | 40.0% |
| 2 | 65 | 35.4% |
| 3 | 67 | 43.3% |
| 4 | 84 | 33.3% |
| 5 | 81 | 38.3% |
| 6 | 78 | 30.8% |
| 7 | 91 | 33.0% |
| 8 | 60 | 31.7% |
| 9 | 53 | 39.6% |
| 10 | 40 | 42.5% |
| 11 | 28 | 28.6% |
| 12 | 21 | 23.8% |
| 13 | 14 | 28.6% |
| 14 | 16 | 31.2% |
| 15 | 4 | 50.0% |
| 16 | 3 | 33.3% |
| 17 | 2 | 0.0% |
| 18 | 1 | 0.0% |
| 20 | 3 | 0.0% |

| occluded objects | scenes | every object graspable |
|---|---|---|
| 0 | 3 | 33.3% |
| 1 | 6 | 33.3% |
| 2 | 18 | 22.2% |
| 3 | 19 | 36.8% |
| 4 | 25 | 16.0% |
| 5 | 30 | 36.7% |
| 6 | 55 | 20.0% |
| 7 | 86 | 32.6% |
| 8 | 92 | 41.3% |
| 9 | 109 | 35.8% |
| 10 | 116 | 42.2% |
| 11 | 121 | 42.1% |
| 12 | 73 | 43.8% |
| 13 | 67 | 34.3% |
| 14 | 54 | 37.0% |
| 15 | 33 | 48.5% |
| 16 | 26 | 42.3% |
| 17 | 10 | 60.0% |
| 18 | 8 | 50.0% |
| 19 | 1 | 100.0% |
| 20 | 2 | 50.0% |


## Which questions each pipeline can answer

One row per question, one column per pipeline. An answered cell says, in words, which setting of the cause makes the effect most likely after adjustment and how likely, against the least favourable setting; a refused cell says why the pipeline could not answer at all.

| question | relational circuit | propositional tree | unrolled tree | scalars-only tree |
|---|---|---|---|---|
| How many small objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out? | answered: with 15 small objects, every object of the scene stays graspable with probability 0.50, the highest of any setting; with 20 small objects it is only 0.00. | answered: with 15 small objects, every object of the scene stays graspable with probability 0.50, the highest of any setting; with 17 small objects it is only 0.00. | answered: with 15 small objects, every object of the scene stays graspable with probability 0.50, the highest of any setting; with 17 small objects it is only 0.00. | refused: the fitted table has no column for the queried variables. |
| How many occluded objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out? | answered: with 19 occluded objects, every object of the scene stays graspable with probability 1.00, the highest of any setting; with 4 occluded objects it is only 0.16. | answered: with 19 occluded objects, every object of the scene stays graspable with probability 1.00, the highest of any setting; with 4 occluded objects it is only 0.16. | answered: with 20 occluded objects, every object of the scene stays graspable with probability 1.00, the highest of any setting; with 4 occluded objects it is only 0.16. | refused: the fitted table has no column for the queried variables. |
| How many clear viewpoints cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out? | answered: with 12 clear viewpoints, every object of the scene stays graspable with probability 1.00, the highest of any setting; with 16 clear viewpoints it is only 0.00. | answered: with 14 clear viewpoints, every object of the scene stays graspable with probability 1.00, the highest of any setting; with 16 clear viewpoints it is only 0.00. | answered: with 14 clear viewpoints, every object of the scene stays graspable with probability 1.00, the highest of any setting; with 16 clear viewpoints it is only 0.00. | refused: the fitted table has no column for the queried variables. |
| Does the object catalogue a scene is built from cause every object of it to stay graspable, adjusting for its spread (extent)? | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.41, the highest of any setting; with the mixed catalogue it is only 0.33. | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.41, the highest of any setting; with the mixed catalogue it is only 0.33. | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.41, the highest of any setting; with the mixed catalogue it is only 0.33. | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.41, the highest of any setting; with the mixed catalogue it is only 0.33. |
| Does the object catalogue a scene is built from cause every object of it to stay graspable, adjusting for its small-object count? | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.39, the highest of any setting; with the ycb-video catalogue it is only 0.32. | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.39, the highest of any setting; with the ycb-video catalogue it is only 0.32. | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.40, the highest of any setting; with the mixed catalogue it is only 0.30. | refused: the fitted table has no column for the queried variables. |
| Does the object catalogue a scene is built from cause object 0 of it to be heavily occluded? | answered: with the grasp catalogue, object 0 is heavily occluded with probability 0.31, the highest of any setting; with the ycb-video catalogue it is only 0.28. | refused: the fitted table has no column for the queried variables. | answered: with the ycb-video catalogue, object 0 is heavily occluded with probability 0.29, the highest of any setting; with the mixed catalogue it is only 0.23. | refused: the fitted table has no column for the queried variables. |
| How many occluded objects cause object 0 of a scene to lose every grasp? | answered: with 20 occluded objects, object 0 loses every grasp with probability 0.16, the highest of any setting; with 0 occluded objects it is only 0.06. | refused: the fitted table has no column for the queried variables. | answered: with 0 occluded objects, object 0 loses every grasp with probability 0.33, the highest of any setting; with 17 occluded objects it is only 0.00. | refused: the fitted table has no column for the queried variables. |
| Does the size of object 0 of a scene cause it to lose every grasp? | answered: with object 0 being medium, object 0 loses every grasp with probability 0.11, the highest of any setting; with object 0 being small it is only 0.09. | refused: the fitted table has no column for the queried variables. | answered: with object 0 being large, object 0 loses every grasp with probability 0.15, the highest of any setting; with object 0 being small it is only 0.08. | refused: the fitted table has no column for the queried variables. |

A question about counts needs the counts: the scalars-only tree refuses it. A question whose effect is one object's own attribute needs the objects: the propositional tree refuses it, the unrolled tree answers it about whatever object the scenes list at that position, and the relational circuit answers it about an exchangeable object. What an answer about "object 0" is worth is what the reordering below measures.

## Fit and likelihood

What each pipeline cost. *Models fitted* counts the plain model plus one support-deterministic model per distinct cause the questions asked about and the pipeline could fit; *training seconds* and the *nodes*/*edges* of every fitted circuit are summed over them, which for the relational circuit includes the object and viewpoint templates.

| pipeline | models fitted | training seconds | nodes | edges |
|---|---|---|---|---|
| relational circuit | 6 | 145.58 | 35832 | 35814 |
| propositional tree | 5 | 34.99 | 9585 | 9580 |
| unrolled tree | 6 | 485.96 | 363289 | 363283 |
| scalars-only tree | 2 | 0.04 | 526 | 524 |

How well each explains scenes it never saw, on three views of a scene: its own scalars, which every pipeline models; its scalars and counts; and the whole scene, objects and viewpoints included, which only the pipelines that model the parts can score. The relational circuit scores a whole scene as its class circuit over the scalars and counts times each part template over one object or viewpoint given the counts; the unrolled tree scores it as one row. *Held-out coverage* is the share of held-out scenes that lie inside the plain model's support at all, since a tree's leaves span only the value ranges they were fitted on, and a whole scene is covered only if every one of its objects and viewpoints is. The *mean log-likelihood* is over the covered scenes only; the last column restricts it to the scenes every pipeline in the table covers, so the numbers are over the same rows.

### scalars

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|
| relational circuit | 100.0% | 0.13 | 0.21 |
| propositional tree | 100.0% | 0.13 | 0.21 |
| unrolled tree | 100.0% | -0.05 | -0.02 |
| scalars-only tree | 95.8% | 0.20 | 0.20 |

### scalars and counts

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|
| relational circuit | 88.0% | -10.50 | -10.44 |
| propositional tree | 88.0% | -10.50 | -10.44 |
| unrolled tree | 96.3% | -10.93 | -10.71 |

### whole scene

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|
| relational circuit | 84.3% | -15.99 | -13.73 |
| unrolled tree | 67.0% | 27.15 | 30.83 |

## Seconds per question

Wall-clock time from asking to the answer or the refusal. The *first ask* of a cause includes fitting that cause's own support-deterministic model; *asked again* repeats the question with every model fitted, so only grounding (for the relational circuit), verification and backdoor adjustment remain. A refusal is fast when it is a schema check; a relational answer draws Monte-Carlo samples for every count the query leaves open and grounds one part template per sampled value, which is where its time goes.

| question | relational circuit, first ask | relational circuit, asked again | propositional tree, first ask | propositional tree, asked again | unrolled tree, first ask | unrolled tree, asked again | scalars-only tree, first ask | scalars-only tree, asked again |
|---|---|---|---|---|---|---|---|---|
| small_object_count_causes_graspability | 113.88 | 90.84 | 8.54 | 2.69 | 99.58 | 7.64 | 0.00 | 0.00 |
| occluded_object_count_causes_graspability | 118.66 | 97.37 | 11.34 | 4.58 | 112.12 | 12.57 | 0.00 | 0.00 |
| clear_viewpoint_count_causes_graspability | 220.85 | 217.01 | 126.84 | 116.46 | 368.81 | 136.62 | 0.00 | 0.00 |
| catalogue_causes_graspability_adjusting_extent | 149.77 | 129.81 | 7.03 | 0.21 | 24.20 | 1.07 | 0.25 | 0.22 |
| catalogue_causes_graspability_adjusting_small_object_count | 132.80 | 134.93 | 1.34 | 1.17 | 2.20 | 2.02 | 0.00 | 0.00 |
| catalogue_causes_occluded_object_0 | 143.97 | 143.17 | 0.00 | 0.00 | 1.16 | 2.93 | 0.00 | 0.00 |
| occluded_object_count_causes_blocked_object_0 | 109.91 | 110.76 | 0.00 | 0.00 | 7.66 | 7.42 | 0.00 | 0.00 |
| size_causes_blocked_object_0 | 100.93 | 74.46 | 0.00 | 0.00 | 24.17 | 1.03 | 0.00 | 0.00 |

## Does the order of the objects matter?

Every scene's objects and viewpoints were put in a random order, 3 times over, and each time the pipelines that model the parts were refitted on the same split and asked the questions about objects again. A relational circuit treats the objects as exchangeable, so nothing about it can depend on the order; an unrolled table's column `objects[0]` holds a different object of every scene after each reordering. *Best regions* lists every most effective cause region found over the orderings; *largest difference* is, over the cause regions every answered ordering distinguishes, the widest gap between orderings in the effect's adjusted probability.

| question | pipeline | orderings answered | best regions | largest difference in adjusted P(effect) |
|---|---|---|---|---|
| catalogue_causes_occluded_object_0 | relational circuit | 3 of 3 | grasp | 0.00 |
| occluded_object_count_causes_blocked_object_0 | relational circuit | 3 of 3 | 20 | 0.00 |
| size_causes_blocked_object_0 | relational circuit | 3 of 3 | medium | 0.00 |
| catalogue_causes_occluded_object_0 | unrolled tree | 3 of 3 | grasp, mixed | 0.08 |
| occluded_object_count_causes_blocked_object_0 | unrolled tree | 3 of 3 | 0, 1 | 0.33 |
| size_causes_blocked_object_0 | unrolled tree | 3 of 3 | small | 0.03 |

The whole-scene likelihood of the same held-out scenes under each ordering:

| pipeline | ordering 0, coverage / mean log-likelihood | ordering 1, coverage / mean log-likelihood | ordering 2, coverage / mean log-likelihood | largest difference |
|---|---|---|---|---|
| relational circuit | 83.8% / -16.57 | 84.3% / -15.99 | 84.3% / -15.99 | 0.58 |
| unrolled tree | 44.0% / -88.16 | 40.8% / -88.07 | 45.5% / -89.76 | 1.68 |

## Over several splits

The comparison repeated over 5 random splits (seeds 0, 1, 2, 3, 4), mean ± standard deviation. The likelihoods are over the scenes every pipeline modelling the view covers.

### scalars

| pipeline | held-out coverage | mean log-likelihood (covered by all) |
|---|---|---|
| relational circuit | 0.995 ± 0.003 | 0.22 ± 0.05 |
| propositional tree | 0.995 ± 0.003 | 0.22 ± 0.05 |
| unrolled tree | 0.995 ± 0.003 | -0.04 ± 0.04 |
| scalars-only tree | 0.976 ± 0.010 | 0.19 ± 0.03 |

### scalars and counts

| pipeline | held-out coverage | mean log-likelihood (covered by all) |
|---|---|---|
| relational circuit | 0.841 ± 0.037 | -10.26 ± 0.25 |
| propositional tree | 0.841 ± 0.037 | -10.26 ± 0.25 |
| unrolled tree | 0.933 ± 0.023 | -10.56 ± 0.22 |

### whole scene

| pipeline | held-out coverage | mean log-likelihood (covered by all) |
|---|---|---|
| relational circuit | 0.818 ± 0.032 | -12.98 ± 2.34 |
| unrolled tree | 0.618 ± 0.048 | 36.61 ± 7.20 |

Per question, how many splits each pipeline answered and which most effective cause regions it found across them:

| question | relational circuit | propositional tree | unrolled tree | scalars-only tree |
|---|---|---|---|---|
| small_object_count_causes_graspability | 5 of 5: 1, 15, 3 | 5 of 5: 1, 15, 3 | 5 of 5: 15, 3 | 0 of 5 |
| occluded_object_count_causes_graspability | 5 of 5: 17, 19 | 5 of 5: 17, 19, 20 | 5 of 5: 17, 19, 20 | 0 of 5 |
| clear_viewpoint_count_causes_graspability | 5 of 5: 12, 14, 29, 30, 35 | 5 of 5: 14, 30 | 5 of 5: 14, 17, 35 | 0 of 5 |
| catalogue_causes_graspability_adjusting_extent | 5 of 5: grasp | 5 of 5: grasp | 5 of 5: grasp | 5 of 5: grasp |
| catalogue_causes_graspability_adjusting_small_object_count | 5 of 5: grasp | 5 of 5: grasp | 5 of 5: grasp | 0 of 5 |
| catalogue_causes_occluded_object_0 | 5 of 5: grasp, mixed | 0 of 5 | 5 of 5: grasp, ycb-video | 0 of 5 |
| occluded_object_count_causes_blocked_object_0 | 5 of 5: 20 | 0 of 5 | 5 of 5: 0, 1 | 0 of 5 |
| size_causes_blocked_object_0 | 4 of 5: medium, small | 0 of 5 | 5 of 5: large | 0 of 5 |

## How much training data it takes

Every pipeline's plain model fitted on a growing share of the scenes and scored on the same held-out fifth, over 3 splits, mean ± standard deviation of the held-out coverage and of the mean log-likelihood over the covered scenes. The relational circuit's templates pool every object of every training scene, and every frame of it, where the unrolled tree sees one row per scene.

### scalars and counts

| training share | relational circuit, coverage | relational circuit, mean log-likelihood | propositional tree, coverage | propositional tree, mean log-likelihood | unrolled tree, coverage | unrolled tree, mean log-likelihood |
|---|---|---|---|---|---|---|
| 20.0% | 0.494 ± 0.013 | -9.35 ± 0.28 | 0.494 ± 0.013 | -9.35 ± 0.28 | 0.597 ± 0.030 | -9.84 ± 0.23 |
| 40.0% | 0.705 ± 0.033 | -10.03 ± 0.07 | 0.705 ± 0.033 | -10.03 ± 0.07 | 0.813 ± 0.005 | -10.66 ± 0.11 |
| 60.0% | 0.784 ± 0.040 | -10.19 ± 0.28 | 0.784 ± 0.040 | -10.19 ± 0.28 | 0.913 ± 0.024 | -10.92 ± 0.13 |
| 80.0% | 0.857 ± 0.040 | -10.41 ± 0.27 | 0.857 ± 0.040 | -10.41 ± 0.27 | 0.946 ± 0.021 | -11.02 ± 0.08 |

### whole scene

| training share | relational circuit, coverage | relational circuit, mean log-likelihood | unrolled tree, coverage | unrolled tree, mean log-likelihood |
|---|---|---|---|---|
| 20.0% | 0.401 ± 0.039 | -9.30 ± 0.45 | 0.136 ± 0.064 | 34.89 ± 9.33 |
| 40.0% | 0.644 ± 0.026 | -12.30 ± 1.90 | 0.339 ± 0.017 | 36.94 ± 5.35 |
| 60.0% | 0.745 ± 0.040 | -13.93 ± 2.25 | 0.513 ± 0.048 | 35.88 ± 6.06 |
| 80.0% | 0.827 ± 0.038 | -16.39 ± 1.86 | 0.644 ± 0.045 | 28.96 ± 4.74 |

## What the results show

- The relational circuit answered 8 of 8 questions.
- The propositional tree answered 5 of 8 questions, refusing `catalogue_causes_occluded_object_0` because the fitted table has no column for the queried variables; `occluded_object_count_causes_blocked_object_0` because the fitted table has no column for the queried variables; `size_causes_blocked_object_0` because the fitted table has no column for the queried variables.
- The unrolled tree answered 8 of 8 questions.
- The scalars-only tree answered 1 of 8 questions, refusing `small_object_count_causes_graspability` because the fitted table has no column for the queried variables; `occluded_object_count_causes_graspability` because the fitted table has no column for the queried variables; `clear_viewpoint_count_causes_graspability` because the fitted table has no column for the queried variables; `catalogue_causes_graspability_adjusting_small_object_count` because the fitted table has no column for the queried variables; `catalogue_causes_occluded_object_0` because the fitted table has no column for the queried variables; `occluded_object_count_causes_blocked_object_0` because the fitted table has no column for the queried variables; `size_causes_blocked_object_0` because the fitted table has no column for the queried variables.
- On `small_object_count_causes_graspability`, every pipeline that answered finds 15 small objects the most effective setting (adjusted probabilities: relational circuit 0.50, propositional tree 0.50, unrolled tree 0.50).
- On `occluded_object_count_causes_graspability`, the pipelines disagree on the most effective setting: the relational circuit says 19 occluded objects (1.00); the propositional tree says 19 occluded objects (1.00); the unrolled tree says 20 occluded objects (1.00).
- On `clear_viewpoint_count_causes_graspability`, the pipelines disagree on the most effective setting: the relational circuit says 12 clear viewpoints (1.00); the propositional tree says 14 clear viewpoints (1.00); the unrolled tree says 14 clear viewpoints (1.00).
- On `catalogue_causes_graspability_adjusting_extent`, every pipeline that answered finds the grasp catalogue the most effective setting (adjusted probabilities: relational circuit 0.41, propositional tree 0.41, unrolled tree 0.41, scalars-only tree 0.41).
- On `catalogue_causes_graspability_adjusting_small_object_count`, every pipeline that answered finds the grasp catalogue the most effective setting (adjusted probabilities: relational circuit 0.39, propositional tree 0.39, unrolled tree 0.40).
- On `catalogue_causes_occluded_object_0`, the pipelines disagree on the most effective setting: the relational circuit says the grasp catalogue (0.31); the unrolled tree says the ycb-video catalogue (0.29).
- On `occluded_object_count_causes_blocked_object_0`, the pipelines disagree on the most effective setting: the relational circuit says 20 occluded objects (0.16); the unrolled tree says 0 occluded objects (0.33).
- On `size_causes_blocked_object_0`, the pipelines disagree on the most effective setting: the relational circuit says object 0 being medium (0.11); the unrolled tree says object 0 being large (0.15).
- On the scalars, the relational circuit assigns the highest mean log-likelihood (0.21, against propositional tree 0.21, unrolled tree -0.02, scalars-only tree 0.20) to the held-out scenes every pipeline covers; coverage: relational circuit 100.0%, propositional tree 100.0%, unrolled tree 100.0%, scalars-only tree 95.8%.
- On the scalars and counts, the relational circuit assigns the highest mean log-likelihood (-10.44, against propositional tree -10.44, unrolled tree -10.71) to the held-out scenes every pipeline covers; coverage: relational circuit 88.0%, propositional tree 88.0%, unrolled tree 96.3%.
- On the whole scene, the unrolled tree assigns the highest mean log-likelihood (30.83, against relational circuit -13.73) to the held-out scenes every pipeline covers; coverage: relational circuit 84.3%, unrolled tree 67.0%.
- Reordering the objects moved the relational circuit's adjusted effect probabilities by up to 0.00 and its whole-scene mean log-likelihood by 0.58.
- Reordering the objects moved the unrolled tree's adjusted effect probabilities by up to 0.33 and its whole-scene mean log-likelihood by 1.68; it changed the most effective setting of `catalogue_causes_occluded_object_0`, `occluded_object_count_causes_blocked_object_0`.
- The relational circuit takes 124.79 seconds per answered question on average once its models are fitted.
- The propositional tree takes 25.02 seconds per answered question on average once its models are fitted.
- The unrolled tree takes 21.41 seconds per answered question on average once its models are fitted.
- The scalars-only tree takes 0.22 seconds per answered question on average once its models are fitted.

## How many small objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.194 | 0.446 | 0.446 |
| 1 | 0.046 | 0.429 | 0.429 |
| 2 | 0.063 | 0.333 | 0.333 |
| 3 | 0.072 | 0.491 | 0.491 |
| 4 | 0.087 | 0.333 | 0.333 |
| 5 | 0.087 | 0.409 | 0.409 |
| 6 | 0.088 | 0.313 | 0.313 |
| 7 | 0.102 | 0.359 | 0.359 |
| 8 | 0.068 | 0.327 | 0.327 |
| 9 | 0.055 | 0.381 | 0.381 |
| 10 | 0.038 | 0.414 | 0.414 |
| 11 | 0.033 | 0.240 | 0.240 |
| 12 | 0.020 | 0.200 | 0.200 |
| 13 | 0.016 | 0.333 | 0.333 |
| 14 | 0.018 | 0.357 | 0.357 |
| 15 | 0.005 | 0.500 | 0.500 |
| 16 | 0.004 | 0.333 | 0.333 |
| 17 | 0.003 | 0.000 | 0.000 |
| 20 | 0.003 | 0.000 | 0.000 |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### propositional tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.194 | 0.446 | 0.446 |
| 1 | 0.046 | 0.429 | 0.429 |
| 2 | 0.063 | 0.333 | 0.333 |
| 3 | 0.072 | 0.491 | 0.491 |
| 4 | 0.087 | 0.333 | 0.333 |
| 5 | 0.087 | 0.409 | 0.409 |
| 6 | 0.088 | 0.313 | 0.313 |
| 7 | 0.102 | 0.359 | 0.359 |
| 8 | 0.068 | 0.327 | 0.327 |
| 9 | 0.055 | 0.381 | 0.381 |
| 10 | 0.038 | 0.414 | 0.414 |
| 11 | 0.033 | 0.240 | 0.240 |
| 12 | 0.020 | 0.200 | 0.200 |
| 13 | 0.016 | 0.333 | 0.333 |
| 14 | 0.018 | 0.357 | 0.357 |
| 15 | 0.005 | 0.500 | 0.500 |
| 16 | 0.004 | 0.333 | 0.333 |
| 17 | 0.003 | 0.000 | 0.000 |
| 20 | 0.003 | 0.000 | 0.000 |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### unrolled tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.194 | 0.446 | 0.446 |
| 1 | 0.046 | 0.429 | 0.429 |
| 2 | 0.063 | 0.333 | 0.333 |
| 3 | 0.072 | 0.491 | 0.491 |
| 4 | 0.087 | 0.333 | 0.333 |
| 5 | 0.087 | 0.409 | 0.409 |
| 6 | 0.088 | 0.313 | 0.313 |
| 7 | 0.102 | 0.359 | 0.359 |
| 8 | 0.068 | 0.327 | 0.327 |
| 9 | 0.055 | 0.381 | 0.381 |
| 10 | 0.038 | 0.414 | 0.414 |
| 11 | 0.033 | 0.240 | 0.240 |
| 12 | 0.020 | 0.200 | 0.200 |
| 13 | 0.016 | 0.333 | 0.333 |
| 14 | 0.018 | 0.357 | 0.357 |
| 15 | 0.005 | 0.500 | 0.500 |
| 16 | 0.004 | 0.333 | 0.333 |
| 17 | 0.003 | 0.000 | 0.000 |
| 20 | 0.003 | 0.000 | 0.000 |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.


## How many occluded objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.004 | 0.333 | 0.333 |
| 1 | 0.005 | 0.250 | 0.250 |
| 2 | 0.016 | 0.167 | 0.182 |
| 3 | 0.022 | 0.353 | 0.312 |
| 4 | 0.033 | 0.160 | 0.160 |
| 5 | 0.033 | 0.400 | 0.400 |
| 6 | 0.055 | 0.190 | 0.190 |
| 7 | 0.093 | 0.324 | 0.324 |
| 8 | 0.094 | 0.431 | 0.431 |
| 9 | 0.114 | 0.345 | 0.345 |
| 10 | 0.125 | 0.453 | 0.453 |
| 11 | 0.122 | 0.409 | 0.409 |
| 12 | 0.080 | 0.426 | 0.426 |
| 13 | 0.067 | 0.333 | 0.333 |
| 14 | 0.056 | 0.372 | 0.372 |
| 15 | 0.033 | 0.400 | 0.400 |
| 16 | 0.028 | 0.524 | 0.524 |
| 17 | 0.010 | 0.625 | 0.625 |
| 18 | 0.008 | 0.667 | 0.667 |
| 19 | 0.001 | 1.000 | 1.000 |
| 20 | 0.001 | 1.000 | 1.000 |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### propositional tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.004 | 0.333 | 0.333 |
| 1 | 0.005 | 0.250 | 0.250 |
| 2 | 0.016 | 0.167 | 0.182 |
| 3 | 0.022 | 0.353 | 0.312 |
| 4 | 0.033 | 0.160 | 0.160 |
| 5 | 0.033 | 0.400 | 0.400 |
| 6 | 0.055 | 0.190 | 0.190 |
| 7 | 0.093 | 0.324 | 0.324 |
| 8 | 0.094 | 0.431 | 0.431 |
| 9 | 0.114 | 0.345 | 0.345 |
| 10 | 0.125 | 0.453 | 0.453 |
| 11 | 0.122 | 0.409 | 0.409 |
| 12 | 0.080 | 0.426 | 0.426 |
| 13 | 0.067 | 0.333 | 0.333 |
| 14 | 0.056 | 0.372 | 0.372 |
| 15 | 0.033 | 0.400 | 0.400 |
| 16 | 0.028 | 0.524 | 0.524 |
| 17 | 0.010 | 0.625 | 0.625 |
| 18 | 0.008 | 0.667 | 0.667 |
| 19 | 0.001 | 1.000 | 1.000 |
| 20 | 0.001 | 1.000 | 1.000 |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### unrolled tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.004 | 0.333 | 0.333 |
| 1 | 0.005 | 0.250 | 0.250 |
| 2 | 0.016 | 0.167 | 0.182 |
| 3 | 0.022 | 0.353 | 0.312 |
| 4 | 0.033 | 0.160 | 0.160 |
| 5 | 0.033 | 0.400 | 0.400 |
| 6 | 0.055 | 0.190 | 0.190 |
| 7 | 0.093 | 0.324 | 0.324 |
| 8 | 0.094 | 0.431 | 0.431 |
| 9 | 0.114 | 0.345 | 0.345 |
| 10 | 0.125 | 0.453 | 0.453 |
| 11 | 0.122 | 0.409 | 0.409 |
| 12 | 0.080 | 0.426 | 0.426 |
| 13 | 0.067 | 0.333 | 0.333 |
| 14 | 0.056 | 0.372 | 0.372 |
| 15 | 0.033 | 0.400 | 0.400 |
| 16 | 0.028 | 0.524 | 0.524 |
| 17 | 0.010 | 0.625 | 0.625 |
| 18 | 0.008 | 0.667 | 0.667 |
| 19 | 0.001 | 1.000 | 1.000 |
| 20 | 0.001 | 1.000 | 1.000 |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.


## How many clear viewpoints cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.237 | 0.481 | 0.481 |
| 1 | 0.014 | 0.545 | 0.545 |
| 2 | 0.024 | 0.556 | 0.588 |
| 3 | 0.010 | 0.625 | 0.500 |
| 4 | 0.004 | 0.667 | 0.667 |
| 5 | 0.009 | 0.429 | 0.429 |
| 6 | 0.007 | 0.600 | 0.600 |
| 7 | 0.013 | 0.500 | 0.500 |
| 8 | 0.007 | 0.400 | 0.400 |
| 9 | 0.008 | 0.500 | 0.500 |
| 10 | 0.013 | 0.400 | 0.400 |
| 11 | 0.013 | 0.100 | 0.100 |
| 12 | 0.007 | 0.800 | 1.000 |
| 13 | 0.012 | 0.222 | 0.222 |
| 14 | 0.004 | 1.000 | 1.000 |
| 15 | 0.013 | 0.500 | 0.444 |
| 16 | 0.003 | 0.000 | 0.000 |
| 17 | 0.005 | 0.500 | 0.667 |
| 18 | 0.014 | 0.273 | 0.200 |
| 19 | 0.007 | 0.800 | 0.667 |
| 20 | 0.004 | 0.333 | 0.333 |
| 21 | 0.009 | 0.714 | 0.714 |
| 22 | 0.007 | 0.600 | 0.600 |
| 23 | 0.012 | 0.778 | 0.750 |
| 24 | 0.004 | 0.667 | 0.667 |
| 25 | 0.009 | 0.571 | 0.571 |
| 26 | 0.005 | 0.500 | 0.500 |
| 27 | 0.010 | 0.750 | 0.714 |
| 28 | 0.005 | 0.500 | 0.500 |
| 29 | 0.004 | 0.667 | 0.667 |
| 30 | 0.008 | 0.667 | 0.600 |
| 31 | 0.004 | 0.333 | 0.333 |
| 32 | 0.007 | 0.400 | 0.400 |
| 33 | 0.007 | 0.400 | 0.250 |
| 34 | 0.012 | 0.111 | 0.111 |
| 35 | 0.007 | 1.000 | 1.000 |
| 36 | 0.008 | 0.500 | 0.400 |
| 37 | 0.013 | 0.400 | 0.333 |
| 38 | 0.008 | 0.500 | 0.500 |
| 39 | 0.009 | 0.286 | 0.286 |
| 40 | 0.014 | 0.273 | 0.250 |
| 41 | 0.009 | 0.286 | 0.286 |
| 42 | 0.013 | 0.500 | 0.500 |
| 43 | 0.018 | 0.357 | 0.357 |
| 44 | 0.017 | 0.308 | 0.308 |
| 45 | 0.020 | 0.400 | 0.385 |
| 46 | 0.029 | 0.136 | 0.136 |
| 47 | 0.026 | 0.300 | 0.300 |
| 48 | 0.038 | 0.172 | 0.172 |
| 49 | 0.042 | 0.313 | 0.313 |
| 50 | 0.059 | 0.111 | 0.111 |
| 51 | 0.047 | 0.139 | 0.139 |
| 52 | 0.093 | 0.197 | 0.197 |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.48).

### propositional tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.237 | 0.481 | 0.481 |
| 1 | 0.014 | 0.545 | 0.545 |
| 2 | 0.024 | 0.556 | 0.588 |
| 3 | 0.010 | 0.625 | 0.500 |
| 4 | 0.004 | 0.667 | 0.667 |
| 5 | 0.009 | 0.429 | 0.429 |
| 6 | 0.007 | 0.600 | 0.600 |
| 7 | 0.013 | 0.500 | 0.500 |
| 8 | 0.007 | 0.400 | 0.400 |
| 9 | 0.008 | 0.500 | 0.500 |
| 10 | 0.013 | 0.400 | 0.400 |
| 11 | 0.013 | 0.100 | 0.100 |
| 12 | 0.007 | 0.800 | 1.000 |
| 13 | 0.012 | 0.222 | 0.222 |
| 14 | 0.004 | 1.000 | 1.000 |
| 15 | 0.013 | 0.500 | 0.444 |
| 16 | 0.003 | 0.000 | 0.000 |
| 17 | 0.005 | 0.500 | 0.667 |
| 18 | 0.014 | 0.273 | 0.200 |
| 19 | 0.007 | 0.800 | 0.667 |
| 20 | 0.004 | 0.333 | 0.333 |
| 21 | 0.009 | 0.714 | 0.714 |
| 22 | 0.007 | 0.600 | 0.600 |
| 23 | 0.012 | 0.778 | 0.750 |
| 24 | 0.004 | 0.667 | 0.667 |
| 25 | 0.009 | 0.571 | 0.571 |
| 26 | 0.005 | 0.500 | 0.500 |
| 27 | 0.010 | 0.750 | 0.714 |
| 28 | 0.005 | 0.500 | 0.500 |
| 29 | 0.004 | 0.667 | 0.667 |
| 30 | 0.008 | 0.667 | 0.600 |
| 31 | 0.004 | 0.333 | 0.333 |
| 32 | 0.007 | 0.400 | 0.400 |
| 33 | 0.007 | 0.400 | 0.250 |
| 34 | 0.012 | 0.111 | 0.111 |
| 35 | 0.007 | 1.000 | 1.000 |
| 36 | 0.008 | 0.500 | 0.400 |
| 37 | 0.013 | 0.400 | 0.333 |
| 38 | 0.008 | 0.500 | 0.500 |
| 39 | 0.009 | 0.286 | 0.286 |
| 40 | 0.014 | 0.273 | 0.250 |
| 41 | 0.009 | 0.286 | 0.286 |
| 42 | 0.013 | 0.500 | 0.500 |
| 43 | 0.018 | 0.357 | 0.357 |
| 44 | 0.017 | 0.308 | 0.308 |
| 45 | 0.020 | 0.400 | 0.385 |
| 46 | 0.029 | 0.136 | 0.136 |
| 47 | 0.026 | 0.300 | 0.300 |
| 48 | 0.038 | 0.172 | 0.172 |
| 49 | 0.042 | 0.312 | 0.312 |
| 50 | 0.059 | 0.111 | 0.111 |
| 51 | 0.047 | 0.139 | 0.139 |
| 52 | 0.093 | 0.197 | 0.197 |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.48).

### unrolled tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.237 | 0.481 | 0.481 |
| 1 | 0.014 | 0.545 | 0.545 |
| 2 | 0.024 | 0.556 | 0.588 |
| 3 | 0.010 | 0.625 | 0.500 |
| 4 | 0.004 | 0.667 | 0.667 |
| 5 | 0.009 | 0.429 | 0.429 |
| 6 | 0.007 | 0.600 | 0.600 |
| 7 | 0.013 | 0.500 | 0.500 |
| 8 | 0.007 | 0.400 | 0.400 |
| 9 | 0.008 | 0.500 | 0.500 |
| 10 | 0.013 | 0.400 | 0.400 |
| 11 | 0.013 | 0.100 | 0.100 |
| 12 | 0.007 | 0.800 | 1.000 |
| 13 | 0.012 | 0.222 | 0.222 |
| 14 | 0.004 | 1.000 | 1.000 |
| 15 | 0.013 | 0.500 | 0.444 |
| 16 | 0.003 | 0.000 | 0.000 |
| 17 | 0.005 | 0.500 | 0.667 |
| 18 | 0.014 | 0.273 | 0.200 |
| 19 | 0.007 | 0.800 | 0.667 |
| 20 | 0.004 | 0.333 | 0.333 |
| 21 | 0.009 | 0.714 | 0.714 |
| 22 | 0.007 | 0.600 | 0.600 |
| 23 | 0.012 | 0.778 | 0.750 |
| 24 | 0.004 | 0.667 | 0.667 |
| 25 | 0.009 | 0.571 | 0.571 |
| 26 | 0.005 | 0.500 | 0.500 |
| 27 | 0.010 | 0.750 | 0.714 |
| 28 | 0.005 | 0.500 | 0.500 |
| 29 | 0.004 | 0.667 | 0.667 |
| 30 | 0.008 | 0.667 | 0.600 |
| 31 | 0.004 | 0.333 | 0.333 |
| 32 | 0.007 | 0.400 | 0.400 |
| 33 | 0.007 | 0.400 | 0.250 |
| 34 | 0.012 | 0.111 | 0.111 |
| 35 | 0.007 | 1.000 | 1.000 |
| 36 | 0.008 | 0.500 | 0.400 |
| 37 | 0.013 | 0.400 | 0.333 |
| 38 | 0.008 | 0.500 | 0.500 |
| 39 | 0.009 | 0.286 | 0.286 |
| 40 | 0.014 | 0.273 | 0.250 |
| 41 | 0.009 | 0.286 | 0.286 |
| 42 | 0.013 | 0.500 | 0.500 |
| 43 | 0.018 | 0.357 | 0.357 |
| 44 | 0.017 | 0.308 | 0.308 |
| 45 | 0.020 | 0.400 | 0.385 |
| 46 | 0.029 | 0.136 | 0.136 |
| 47 | 0.026 | 0.300 | 0.300 |
| 48 | 0.038 | 0.172 | 0.172 |
| 49 | 0.042 | 0.312 | 0.312 |
| 50 | 0.059 | 0.111 | 0.111 |
| 51 | 0.047 | 0.139 | 0.139 |
| 52 | 0.093 | 0.197 | 0.197 |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.48).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.


## Does the object catalogue a scene is built from cause every object of it to stay graspable, adjusting for its spread (extent)?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| grasp | 0.506 | 0.407 | 0.407 |
| mixed | 0.169 | 0.333 | 0.333 |
| ycb-video | 0.325 | 0.355 | 0.355 |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.41).

### propositional tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| grasp | 0.506 | 0.407 | 0.407 |
| mixed | 0.169 | 0.333 | 0.333 |
| ycb-video | 0.325 | 0.355 | 0.355 |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.41).

### unrolled tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| grasp | 0.506 | 0.407 | 0.407 |
| mixed | 0.169 | 0.333 | 0.333 |
| ycb-video | 0.325 | 0.355 | 0.355 |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.41).

### scalars-only tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| grasp | 0.506 | 0.407 | 0.407 |
| mixed | 0.169 | 0.333 | 0.333 |
| ycb-video | 0.325 | 0.355 | 0.355 |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.41).


## Does the object catalogue a scene is built from cause every object of it to stay graspable, adjusting for its small-object count?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| grasp | 0.506 | 0.407 | 0.391 |
| mixed | 0.169 | 0.333 | 0.332 |
| ycb-video | 0.325 | 0.355 | 0.324 |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.39).

### propositional tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| grasp | 0.506 | 0.407 | 0.392 |
| mixed | 0.169 | 0.333 | 0.332 |
| ycb-video | 0.325 | 0.355 | 0.324 |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.39).

### unrolled tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| grasp | 0.506 | 0.407 | 0.398 |
| mixed | 0.169 | 0.333 | 0.304 |
| ycb-video | 0.325 | 0.355 | 0.340 |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.40).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.


## Does the object catalogue a scene is built from cause object 0 of it to be heavily occluded?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| grasp | 0.506 | 0.307 | 0.307 |
| mixed | 0.169 | 0.288 | 0.288 |
| ycb-video | 0.325 | 0.277 | 0.277 |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.31).

### propositional tree

Refused: the fitted table has no column for the queried variables.

### unrolled tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| grasp | 0.506 | 0.277 | 0.277 |
| mixed | 0.169 | 0.233 | 0.233 |
| ycb-video | 0.325 | 0.286 | 0.286 |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.28).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.


## How many occluded objects cause object 0 of a scene to lose every grasp?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.004 | 0.063 | 0.063 |
| 1 | 0.005 | 0.064 | 0.064 |
| 2 | 0.016 | 0.069 | 0.069 |
| 3 | 0.022 | 0.081 | 0.081 |
| 4 | 0.033 | 0.088 | 0.088 |
| 5 | 0.033 | 0.091 | 0.091 |
| 6 | 0.055 | 0.097 | 0.097 |
| 7 | 0.093 | 0.104 | 0.104 |
| 8 | 0.094 | 0.109 | 0.109 |
| 9 | 0.114 | 0.116 | 0.116 |
| 10 | 0.125 | 0.119 | 0.119 |
| 11 | 0.122 | 0.123 | 0.123 |
| 12 | 0.080 | 0.119 | 0.119 |
| 13 | 0.067 | 0.117 | 0.117 |
| 14 | 0.056 | 0.117 | 0.117 |
| 15 | 0.033 | 0.122 | 0.122 |
| 16 | 0.028 | 0.118 | 0.118 |
| 17 | 0.010 | 0.113 | 0.113 |
| 18 | 0.008 | 0.114 | 0.114 |
| 19 | 0.001 | 0.130 | 0.130 |
| 20 | 0.001 | 0.161 | 0.161 |

EQL's own `cause` search settles on 11: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.12).

### propositional tree

Refused: the fitted table has no column for the queried variables.

### unrolled tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| 0 | 0.004 | 0.333 | 0.333 |
| 1 | 0.005 | 0.250 | 0.250 |
| 2 | 0.016 | 0.083 | 0.083 |
| 3 | 0.022 | 0.176 | 0.176 |
| 4 | 0.033 | 0.160 | 0.160 |
| 5 | 0.033 | 0.080 | 0.080 |
| 6 | 0.055 | 0.119 | 0.119 |
| 7 | 0.093 | 0.085 | 0.085 |
| 8 | 0.094 | 0.125 | 0.125 |
| 9 | 0.114 | 0.092 | 0.092 |
| 10 | 0.125 | 0.126 | 0.126 |
| 11 | 0.122 | 0.097 | 0.097 |
| 12 | 0.080 | 0.098 | 0.098 |
| 13 | 0.067 | 0.098 | 0.098 |
| 14 | 0.056 | 0.070 | 0.070 |
| 15 | 0.033 | 0.040 | 0.040 |
| 16 | 0.028 | 0.048 | 0.048 |
| 17 | 0.010 | 0.000 | 0.000 |
| 18 | 0.008 | 0.000 | 0.000 |
| 19 | 0.001 | 0.000 | 0.000 |
| 20 | 0.001 | 0.000 | 0.000 |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.13).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.


## Does the size of object 0 of a scene cause it to lose every grasp?

One row per region of the cause the model distinguishes. *P(region)* is how much of the training population that region holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for. Where the two columns agree, the confounder carried no extra information within that region.

### relational circuit

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| large | 0.513 | 0.108 | 0.108 |
| medium | 0.416 | 0.109 | 0.109 |
| small | 0.071 | 0.094 | 0.094 |

EQL's own `cause` search settles on large: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.11).

### propositional tree

Refused: the fitted table has no column for the queried variables.

### unrolled tree

| cause region | P(region) | naive P(effect) | adjusted P(effect | do(cause)) |
|---|---|---|---|
| large | 0.233 | 0.152 | 0.152 |
| medium | 0.402 | 0.094 | 0.094 |
| small | 0.364 | 0.076 | 0.076 |

EQL's own `cause` search settles on medium: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.09).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

