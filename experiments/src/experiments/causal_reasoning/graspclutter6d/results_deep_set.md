# GraspClutter6D: relational circuit against flat-table trees

The GraspClutter6D dataset records a thousand real, densely cluttered bin, shelf and table scenes, each photographed from thirteen poses by four cameras, with the ground-truth pose and the visible share of every object instance in every frame, and with analytic antipodal grasps annotated on every object model and checked for collision against every scene it stands in. A scene here is its own attributes (which object catalogue it is built from, how far its clutter is spread, how far it is stacked, and whether every object in it keeps a grasp) with one exchangeable part per object instance (size, diameter, visibility, occlusion, graspability) and one per camera frame (camera, distance, proximity, clarity). A scene holds between five and twenty object instances, and they have no canonical order; the annotation file lists them in the order they were labelled, and nothing ties a position to an identity.

Five pipelines were fitted on the same scenes and asked the same `cause`/`causes_effect` EQL queries:

- **relational circuit**: a relational probabilistic circuit fitted on the scenes' relational structure, one circuit over the scene's own attributes and its aggregation counts (small objects, occluded objects, near viewpoints, clear viewpoints), one template over an object's attributes and one over a viewpoint's, grounded per query into a circuit over exactly the queried scene, objects and viewpoints and registered as a causal circuit;
- **hybrid circuit**: the same, except that the viewpoints, whose position in a scene the recording rig fixes, are held by position rather than pooled into a template;
- **propositional tree**: a joint probability tree fitted on the scenes flattened into one table of the scene's own attributes and the same four counts, the classic propositional summary of a relational example, registered as a causal circuit the same way;
- **unrolled tree**: the same tree on a table that also carries every object's and viewpoint's attributes under the part's position, padded with an absent marker past a scene's last part, so that a column means whatever part a scene happens to list at that position;
- **scalars-only tree**: the same tree on the scene's own attributes alone, what a flat learner sees without the relational feature extraction.

Every flat tree answers a query by backdoor adjustment on a table column; the relational circuit does the same on the variable of a grounded circuit. In both, the model is stratified so it is support-deterministic over the cause, the effect's probability is read off every region of the cause, and any variable the query marks as a confounder is summed out of that reading. Every query lists one object and one viewpoint with all their attributes open, which is what makes grounding retain the scene's counts as variables; a flat table ignores parts a query says nothing about and refuses a query that constrains a column it does not have.

## Setup

- scenes: 954 (763 to fit on, 191 held out)
- scenes where every object stays graspable: 37.6%
- fewest training rows per leaf, as a share of the rows fitted on: 0.0 in a cause-specific model, 0.0 in the plain model that scores held-out scenes
- split seed: 0
- fewest training scenes a cause region may hold for its effect to be read as an answer: 10; a region below that is marked † in the tables and takes no part in any summary

## How often every object stays graspable

The scenes themselves, before any model: the share where every object stays graspable, grouped by the object catalogue the scene is built from, by how many of its objects are small, by how many of them the cameras do not see whole, and by how many objects it holds at all. This is the signal the models are asked to explain.

| object catalogue | scenes | effect |
|---|---|---|
| grasp | 487 | 43.1% |
| mixed | 161 | 28.0% |
| ycb-video | 306 | 34.0% |

| small objects | scenes | effect |
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

| occluded objects | scenes | effect |
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

| objects | scenes | effect |
|---|---|---|
| 5 | 1 | 100.0% |
| 7 | 1 | 0.0% |
| 8 | 1 | 100.0% |
| 9 | 2 | 0.0% |
| 10 | 66 | 42.4% |
| 11 | 69 | 55.1% |
| 12 | 81 | 46.9% |
| 13 | 123 | 41.5% |
| 14 | 135 | 40.0% |
| 15 | 121 | 32.2% |
| 16 | 114 | 33.3% |
| 17 | 91 | 33.0% |
| 18 | 62 | 30.6% |
| 19 | 46 | 26.1% |
| 20 | 41 | 24.4% |


## Which questions each pipeline can answer

One row per question, one column per pipeline. An answered cell says, in words, which setting of the cause makes the effect most likely after adjustment and how likely, against the least favourable setting, over the regions that hold enough training scenes to be read; a refused cell says why the pipeline could not answer at all.

| question | neural adjustment |
|---|---|
| How many small objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out? | answered: with 13 small objects, every object of the scene stays graspable with probability 0.39, the highest of any setting; with 7 small objects it is only 0.34. |
| How many small objects cause every object of a scene to stay graspable, adjusting for the number of objects? | answered: with 11 small objects, every object of the scene stays graspable with probability 0.36, the highest of any setting; with 14 small objects it is only 0.33. |
| How many small objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out and the number of objects? | answered: with 14 small objects, every object of the scene stays graspable with probability 0.42, the highest of any setting; with 2 small objects it is only 0.31. |
| How many occluded objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out? | answered: with 2 occluded objects, every object of the scene stays graspable with probability 0.47, the highest of any setting; with 14 occluded objects it is only 0.27. |
| How many occluded objects cause every object of a scene to stay graspable, adjusting for the number of objects? | answered: with 16 occluded objects, every object of the scene stays graspable with probability 0.68, the highest of any setting; with 2 occluded objects it is only 0.19. |
| How many occluded objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out and the number of objects? | answered: with 16 occluded objects, every object of the scene stays graspable with probability 0.68, the highest of any setting; with 2 occluded objects it is only 0.17. |
| How many clear viewpoints cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out? | answered: with 18 clear viewpoints, every object of the scene stays graspable with probability 0.49, the highest of any setting; with 52 clear viewpoints it is only 0.16. |
| How many clear viewpoints cause every object of a scene to stay graspable, adjusting for the number of objects? | answered: with 18 clear viewpoints, every object of the scene stays graspable with probability 0.54, the highest of any setting; with 52 clear viewpoints it is only 0.17. |
| How many clear viewpoints cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out and the number of objects? | answered: with 18 clear viewpoints, every object of the scene stays graspable with probability 0.51, the highest of any setting; with 52 clear viewpoints it is only 0.15. |
| Does the object catalogue a scene is built from cause every object of it to stay graspable, adjusting for its spread (extent)? | answered: with the ycb-video catalogue, every object of the scene stays graspable with probability 0.39, the highest of any setting; with the mixed catalogue it is only 0.34. |
| Does the object catalogue a scene is built from cause every object of it to stay graspable, adjusting for its small-object count? | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.39, the highest of any setting; with the mixed catalogue it is only 0.33. |
| Does the object catalogue a scene is built from cause object 0 of it to be heavily occluded? | answered: with the mixed catalogue, object 0 is heavily occluded with probability 0.32, the highest of any setting; with the grasp catalogue it is only 0.31. |
| How many occluded objects cause object 0 of a scene to lose every grasp? | answered: with 0 occluded objects, object 0 loses every grasp with probability 0.25, the highest of any setting; with 20 occluded objects it is only 0.03. |
| Does the size of object 0 of a scene cause it to lose every grasp? | answered: with object 0 being small, object 0 loses every grasp with probability 0.15, the highest of any setting; with object 0 being large it is only 0.11. |

A question about counts needs the counts: the scalars-only tree refuses it. A question whose effect is one object's own attribute needs the objects: the propositional tree refuses it, the unrolled tree answers it about whatever object the scenes list at that position, and the relational circuit answers it about an exchangeable object. What an answer about "object 0" is worth is what the reordering below measures.

## Trend and contrast

The most effective setting is an argmax over up to twenty sparse regions and moves with the split. Two summaries that do not: *trend* is Spearman's rank correlation between the cause's value and the adjusted probability over the supported regions, for a numeric cause; *contrast* is the adjusted probability at the highest supported region minus at the lowest (for a symbolic cause, at the most effective minus at the least), with Newcombe's interval from the Wilson intervals of the two regions' support.

| question | neural adjustment, trend | neural adjustment, contrast |
|---|---|---|
| small_object_count_causes_graspability_adjusting_extent | 0.25 | 0.01 [-0.21, 0.27] (0 → 14) |
| small_object_count_causes_graspability_adjusting_object_count | 0.20 | -0.02 [-0.22, 0.25] (0 → 14) |
| small_object_count_causes_graspability_adjusting_extent_and_object_count | 0.90 | 0.09 [-0.14, 0.35] (0 → 14) |
| occluded_object_count_causes_graspability_adjusting_extent | -0.95 | -0.14 [-0.44, 0.18] (2 → 16) |
| occluded_object_count_causes_graspability_adjusting_object_count | 1.00 | 0.49 [0.14, 0.70] (2 → 16) |
| occluded_object_count_causes_graspability_adjusting_extent_and_object_count | 1.00 | 0.51 [0.16, 0.71] (2 → 16) |
| clear_viewpoint_count_causes_graspability_adjusting_extent | -0.92 | -0.20 [-0.29, -0.07] (0 → 52) |
| clear_viewpoint_count_causes_graspability_adjusting_object_count | -0.89 | -0.18 [-0.28, -0.06] (0 → 52) |
| clear_viewpoint_count_causes_graspability_adjusting_extent_and_object_count | -0.92 | -0.27 [-0.37, -0.15] (0 → 52) |
| catalogue_causes_graspability_adjusting_extent | - | 0.06 [-0.05, 0.16] (mixed → ycb-video) |
| catalogue_causes_graspability_adjusting_small_object_count | - | 0.06 [-0.03, 0.15] (mixed → grasp) |
| catalogue_causes_occluded_object_0 | - | 0.01 [-0.02, 0.03] (grasp → mixed) |
| occluded_object_count_causes_blocked_object_0 | -1.00 | -0.22 [-0.39, -0.01] (0 → 20) |
| size_causes_blocked_object_0 | - | 0.03 [0.02, 0.05] (large → small) |

## What adjusting for changes

The same count question under each set of confounders it was asked with, read off the neural adjustment. *n* is how many training scenes hold that value of the cause; † marks a region below the support threshold.

### small_object_count

| cause region | n | naive | adjusted for how far the clutter is spread out | adjusted for the number of objects | adjusted for how far the clutter is spread out and the number of objects |
|---|---|---|---|---|---|
| 0 | 148 | 0.446 | 0.379 | 0.351 | 0.327 |
| 1 | 35 | 0.429 | 0.370 | 0.351 | 0.316 |
| 2 | 48 | 0.333 | 0.368 | 0.348 | 0.307 |
| 3 | 55 | 0.491 | 0.369 | 0.344 | 0.310 |
| 4 | 66 | 0.333 | 0.364 | 0.338 | 0.314 |
| 5 | 66 | 0.409 | 0.353 | 0.334 | 0.320 |
| 6 | 67 | 0.313 | 0.342 | 0.336 | 0.326 |
| 7 | 78 | 0.359 | 0.338 | 0.344 | 0.333 |
| 8 | 52 | 0.327 | 0.339 | 0.354 | 0.341 |
| 9 | 42 | 0.381 | 0.346 | 0.361 | 0.351 |
| 10 | 29 | 0.414 | 0.357 | 0.364 | 0.362 |
| 11 | 25 | 0.240 | 0.371 | 0.364 | 0.375 |
| 12 | 15 | 0.200 | 0.385 | 0.359 | 0.389 |
| 13 | 12 | 0.333 | 0.394 | 0.348 | 0.403 |
| 14 | 14 | 0.357 | 0.389 | 0.333 | 0.416 |
| 15 † | 4 | 0.500 | 0.353 | 0.315 | 0.429 |
| 16 † | 3 | 0.333 | 0.273 | 0.297 | 0.439 |
| 17 † | 2 | 0.000 | 0.164 | 0.279 | 0.444 |
| 20 † | 2 | 0.000 | 0.010 | 0.246 | 0.419 |

### occluded_object_count

| cause region | n | naive | adjusted for how far the clutter is spread out | adjusted for the number of objects | adjusted for how far the clutter is spread out and the number of objects |
|---|---|---|---|---|---|
| 0 † | 3 | 0.333 | 0.500 | 0.193 | 0.178 |
| 1 † | 4 | 0.250 | 0.479 | 0.188 | 0.174 |
| 2 | 12 | 0.167 | 0.473 | 0.189 | 0.171 |
| 3 | 17 | 0.353 | 0.472 | 0.196 | 0.171 |
| 4 | 25 | 0.160 | 0.470 | 0.207 | 0.180 |
| 5 | 25 | 0.400 | 0.466 | 0.224 | 0.201 |
| 6 | 42 | 0.190 | 0.459 | 0.250 | 0.229 |
| 7 | 71 | 0.324 | 0.448 | 0.280 | 0.262 |
| 8 | 72 | 0.431 | 0.429 | 0.310 | 0.297 |
| 9 | 87 | 0.345 | 0.403 | 0.341 | 0.333 |
| 10 | 95 | 0.453 | 0.372 | 0.371 | 0.369 |
| 11 | 93 | 0.409 | 0.340 | 0.400 | 0.407 |
| 12 | 61 | 0.426 | 0.309 | 0.430 | 0.447 |
| 13 | 51 | 0.333 | 0.284 | 0.464 | 0.493 |
| 14 | 43 | 0.372 | 0.275 | 0.513 | 0.552 |
| 15 | 25 | 0.400 | 0.289 | 0.589 | 0.617 |
| 16 | 21 | 0.524 | 0.333 | 0.679 | 0.680 |
| 17 † | 8 | 0.625 | 0.396 | 0.761 | 0.738 |
| 18 † | 6 | 0.667 | 0.464 | 0.829 | 0.790 |
| 19 † | 1 | 1.000 | 0.531 | 0.881 | 0.834 |
| 20 † | 1 | 1.000 | 0.597 | 0.920 | 0.870 |

### clear_viewpoint_count

| cause region | n | naive | adjusted for how far the clutter is spread out | adjusted for the number of objects | adjusted for how far the clutter is spread out and the number of objects |
|---|---|---|---|---|---|
| 0 | 181 | 0.481 | 0.355 | 0.352 | 0.423 |
| 1 | 11 | 0.545 | 0.411 | 0.421 | 0.459 |
| 2 | 18 | 0.556 | 0.450 | 0.476 | 0.489 |
| 3 † | 8 | 0.625 | 0.437 | 0.466 | 0.492 |
| 4 † | 3 | 0.667 | 0.406 | 0.441 | 0.469 |
| 5 † | 7 | 0.429 | 0.369 | 0.414 | 0.430 |
| 6 † | 5 | 0.600 | 0.331 | 0.390 | 0.385 |
| 7 | 10 | 0.500 | 0.306 | 0.373 | 0.349 |
| 8 † | 5 | 0.400 | 0.293 | 0.363 | 0.332 |
| 9 † | 6 | 0.500 | 0.293 | 0.363 | 0.332 |
| 10 | 10 | 0.400 | 0.304 | 0.371 | 0.343 |
| 11 | 10 | 0.100 | 0.324 | 0.386 | 0.362 |
| 12 † | 5 | 0.800 | 0.351 | 0.406 | 0.385 |
| 13 † | 9 | 0.222 | 0.380 | 0.428 | 0.410 |
| 14 † | 3 | 1.000 | 0.408 | 0.451 | 0.436 |
| 15 | 10 | 0.500 | 0.434 | 0.475 | 0.460 |
| 16 † | 2 | 0.000 | 0.455 | 0.497 | 0.481 |
| 17 † | 4 | 0.500 | 0.472 | 0.519 | 0.498 |
| 18 | 11 | 0.273 | 0.486 | 0.538 | 0.514 |
| 19 † | 5 | 0.800 | 0.497 | 0.554 | 0.527 |
| 20 † | 3 | 0.333 | 0.507 | 0.568 | 0.538 |
| 21 † | 7 | 0.714 | 0.514 | 0.579 | 0.548 |
| 22 † | 5 | 0.600 | 0.519 | 0.588 | 0.554 |
| 23 † | 9 | 0.778 | 0.521 | 0.592 | 0.558 |
| 24 † | 3 | 0.667 | 0.519 | 0.593 | 0.557 |
| 25 † | 7 | 0.571 | 0.514 | 0.588 | 0.553 |
| 26 † | 4 | 0.500 | 0.505 | 0.578 | 0.545 |
| 27 † | 8 | 0.750 | 0.493 | 0.565 | 0.533 |
| 28 † | 4 | 0.500 | 0.479 | 0.547 | 0.518 |
| 29 † | 3 | 0.667 | 0.463 | 0.528 | 0.502 |
| 30 † | 6 | 0.667 | 0.446 | 0.508 | 0.483 |
| 31 † | 3 | 0.333 | 0.429 | 0.487 | 0.464 |
| 32 † | 5 | 0.400 | 0.412 | 0.466 | 0.444 |
| 33 † | 5 | 0.400 | 0.395 | 0.446 | 0.425 |
| 34 † | 9 | 0.111 | 0.378 | 0.428 | 0.406 |
| 35 † | 5 | 1.000 | 0.362 | 0.410 | 0.388 |
| 36 † | 6 | 0.500 | 0.347 | 0.393 | 0.371 |
| 37 | 10 | 0.400 | 0.332 | 0.377 | 0.354 |
| 38 † | 6 | 0.500 | 0.317 | 0.361 | 0.337 |
| 39 † | 7 | 0.286 | 0.303 | 0.345 | 0.321 |
| 40 | 11 | 0.273 | 0.290 | 0.330 | 0.305 |
| 41 † | 7 | 0.286 | 0.277 | 0.315 | 0.290 |
| 42 | 10 | 0.500 | 0.264 | 0.300 | 0.274 |
| 43 | 14 | 0.357 | 0.252 | 0.285 | 0.259 |
| 44 | 13 | 0.308 | 0.240 | 0.271 | 0.245 |
| 45 | 15 | 0.400 | 0.228 | 0.257 | 0.231 |
| 46 | 22 | 0.136 | 0.217 | 0.243 | 0.217 |
| 47 | 20 | 0.300 | 0.206 | 0.230 | 0.204 |
| 48 | 29 | 0.172 | 0.196 | 0.217 | 0.192 |
| 49 | 32 | 0.312 | 0.186 | 0.204 | 0.180 |
| 50 | 45 | 0.111 | 0.177 | 0.192 | 0.169 |
| 51 | 36 | 0.139 | 0.167 | 0.180 | 0.159 |
| 52 | 71 | 0.197 | 0.159 | 0.169 | 0.150 |


## Fit and likelihood

What each pipeline cost. *Models fitted* counts the plain model plus one support-deterministic model per distinct cause the questions asked about and the pipeline could fit; *training seconds* and the *nodes*/*edges* of every fitted circuit are summed over them, which for the relational circuit includes the part templates.

| pipeline | models fitted | training seconds | nodes | edges |
|---|---|---|---|---|
| neural adjustment | 15 | 8.89 | 0 | 0 |

How well each explains scenes it never saw, on three views of one scene: its own scalars, which every pipeline models; its scalars and counts; and the whole scene, parts included, which only the pipelines that model the parts can score. The relational circuit scores a whole scene as its class circuit over the scalars and counts times each part template over one part given the counts; the unrolled tree scores it as one row. *Held-out coverage* is the share of held-out scenes that lie inside the plain model's support at all, since a tree's leaves span only the value ranges they were fitted on, and a whole scene is covered only if every one of its parts is. The *mean log-likelihood* is over the covered scenes only; the last column restricts it to the scenes every pipeline in the table covers, so the numbers are over the same rows.

### scalars

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|

### scalars and counts

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|

### whole scene

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|

## Seconds per question

Wall-clock time from asking to the answer or the refusal. The *first ask* of a cause includes fitting that cause's own support-deterministic model; *asked again* repeats the question with every model fitted, so only grounding (for the relational circuit), verification and backdoor adjustment remain. A refusal is fast when it is a schema check; a relational answer draws Monte-Carlo samples for every count the query leaves open and grounds one part template per sampled value, which is where its time goes.

| question | neural adjustment, first ask | neural adjustment, asked again |
|---|---|---|
| small_object_count_causes_graspability_adjusting_extent | 0.24 | - |
| small_object_count_causes_graspability_adjusting_object_count | 0.26 | - |
| small_object_count_causes_graspability_adjusting_extent_and_object_count | 0.19 | - |
| occluded_object_count_causes_graspability_adjusting_extent | 0.30 | - |
| occluded_object_count_causes_graspability_adjusting_object_count | 0.24 | - |
| occluded_object_count_causes_graspability_adjusting_extent_and_object_count | 0.16 | - |
| clear_viewpoint_count_causes_graspability_adjusting_extent | 0.46 | - |
| clear_viewpoint_count_causes_graspability_adjusting_object_count | 0.43 | - |
| clear_viewpoint_count_causes_graspability_adjusting_extent_and_object_count | 0.36 | - |
| catalogue_causes_graspability_adjusting_extent | 0.23 | - |
| catalogue_causes_graspability_adjusting_small_object_count | 0.22 | - |
| catalogue_causes_occluded_object_0 | 1.06 | - |
| occluded_object_count_causes_blocked_object_0 | 1.61 | - |
| size_causes_blocked_object_0 | 1.02 | - |

## Does the order of the parts matter?

Every scene's parts were put in a random order, 20 times over, and each time the pipelines that model the parts were refitted on the same split and asked the questions about parts again; the parts in the order the dataset lists them is the baseline every reordering is measured against. A relational circuit treats the objects as exchangeable, so nothing about it can depend on the order; an unrolled table's column `objects[0]` holds a different object of every scene after each reordering. Per question and pipeline: how many reorderings were answered; over the cause regions every answered ordering distinguishes, the mean standard deviation and the widest range of the adjusted probability; the share of reorderings whose most effective region is not the dataset-order one; and the share whose trend changed sign.

| question | pipeline | reorderings answered | mean sd of adjusted P(effect) | widest range | argmax moved | trend sign flipped |
|---|---|---|---|---|---|---|

The whole-scene likelihood of the same held-out scenes with the parts in the order the dataset lists them, and over the reorderings. The dataset's order is not arbitrary throughout: a scene's frames are numbered by the recording rig, four cameras per pose in a fixed sequence, so which camera took frame *i* is the same in every scene, and a column that addresses a viewpoint by position addresses a real thing. Its objects carry no such order. *Largest drop* is how far below the dataset-order likelihood the worst reordering took each pipeline.

| pipeline | dataset order, coverage / mean log-likelihood | reorderings, coverage / mean log-likelihood (mean ± sd) | largest drop |
|---|---|---|---|

## Over several splits

The comparison repeated over 3 random splits (seeds 0, 1, 2), mean ± standard deviation. The likelihoods are over the scenes every pipeline modelling the view covers.

### scalars

| pipeline | held-out coverage | mean log-likelihood (covered by all) |
|---|---|---|

### scalars and counts

| pipeline | held-out coverage | mean log-likelihood (covered by all) |
|---|---|---|

### whole scene

| pipeline | held-out coverage | mean log-likelihood (covered by all) |
|---|---|---|

Per question, how many splits each pipeline answered, and the mean ± standard deviation over the splits of its trend and of its contrast:

| question | neural adjustment, answered | neural adjustment, trend | neural adjustment, contrast |
|---|---|---|---|
| small_object_count_causes_graspability_adjusting_extent | 3 of 3 | 0.03 ± 0.22 | -0.03 ± 0.04 |
| small_object_count_causes_graspability_adjusting_object_count | 3 of 3 | 0.29 ± 0.24 | -0.03 ± 0.02 |
| small_object_count_causes_graspability_adjusting_extent_and_object_count | 3 of 3 | 0.76 ± 0.19 | 0.07 ± 0.03 |
| occluded_object_count_causes_graspability_adjusting_extent | 3 of 3 | -0.95 ± 0.00 | -0.18 ± 0.03 |
| occluded_object_count_causes_graspability_adjusting_object_count | 3 of 3 | 1.00 ± 0.00 | 0.41 ± 0.06 |
| occluded_object_count_causes_graspability_adjusting_extent_and_object_count | 3 of 3 | 1.00 ± 0.00 | 0.39 ± 0.09 |
| clear_viewpoint_count_causes_graspability_adjusting_extent | 3 of 3 | -0.93 ± 0.02 | -0.20 ± 0.02 |
| clear_viewpoint_count_causes_graspability_adjusting_object_count | 3 of 3 | -0.93 ± 0.03 | -0.19 ± 0.05 |
| clear_viewpoint_count_causes_graspability_adjusting_extent_and_object_count | 3 of 3 | -0.96 ± 0.03 | -0.29 ± 0.07 |
| catalogue_causes_graspability_adjusting_extent | 3 of 3 | - | 0.12 ± 0.05 |
| catalogue_causes_graspability_adjusting_small_object_count | 3 of 3 | - | 0.15 ± 0.06 |
| catalogue_causes_occluded_object_0 | 3 of 3 | - | 0.01 ± 0.00 |
| occluded_object_count_causes_blocked_object_0 | 3 of 3 | -0.97 ± 0.03 | -0.17 ± 0.05 |
| size_causes_blocked_object_0 | 3 of 3 | - | 0.02 ± 0.01 |

## How much training data it takes

Every pipeline's plain model fitted on a growing share of the scenes and scored on the same held-out fifth, over 0 splits, mean ± standard deviation of the held-out coverage and of the mean log-likelihood over the covered scenes. The relational circuit's templates pool every part of every training scene, where the unrolled tree sees one row per scene. Every object of every training scene, and every frame of it, goes into the templates.

### scalars and counts

| training share |
|---|

### whole scene

| training share |
|---|

## Error against known truth

Scenes sampled from a structural causal model over the same domain, whose interventional probabilities are known by construction. The cause is forced to a value in the mechanism and the effect's rate read off 200,000 forced scenes. The number of objects is the model's one confounder, driving both the causes and the effect, and the scenes list their small objects first, so a column that addresses an object by position is systematically misleading. A count's extreme values are rare in the data and rarer still within every stratum of the confounder, so no estimator recovers their interventional probability well; the questions whose cause or effect lives on one object are where the pipelines differ. Every pipeline was fitted on 400 scenes per setting and asked the questions; *mean* and *max absolute error* are over every supported cause region of every answered question, the *support-weighted* error weighs each region by the training rows it holds, *worst ordering* is the mean absolute error under the reordering of the parts the pipeline did worst on, and *rank correlation* is Spearman's between the answered and the true probabilities over a question's regions.

| pipeline | questions answered | mean abs. error | support-weighted abs. error | max abs. error | mean abs. error, worst ordering | rank correlation with truth |
|---|---|---|---|---|---|---|
| neural adjustment | 100.0% | 0.069 | 0.033 | 0.382 | 0.069 | 0.37 |

Mean absolute error per setting of the model:

| objects per scene | confounding strength | neural adjustment |
|---|---|---|
| 5 | 0.6 | 0.061 |
| 10 | 0.0 | 0.048 |
| 10 | 0.3 | 0.068 |
| 10 | 0.6 | 0.073 |
| 20 | 0.6 | 0.085 |

Mean absolute error per question, over every setting:

| question | neural adjustment |
|---|---|
| small_object_count_causes_graspability_adjusting_object_count | 0.123 |
| occluded_object_count_causes_graspability_adjusting_object_count | 0.054 |
| clear_viewpoint_count_causes_graspability_adjusting_object_count | 0.070 |
| catalogue_causes_graspability_adjusting_object_count | 0.038 |
| catalogue_causes_occluded_object_0 | 0.046 |
| occluded_object_count_causes_blocked_object_0 | 0.027 |
| size_causes_blocked_object_0 | 0.026 |

## Cost against the number of objects

The pipelines that model the parts, fitted on synthetic scenes of growing size (400 scenes each) and asked one question about a part. The relational circuit's part templates pool every part of every scene into one circuit, so their size follows the number of distinct part attributes, not the number of parts; the unrolled table carries one block of columns per position, so its tree grows with the widest scene. *First ask* includes fitting the cause-specific model, *asked again* is grounding and adjustment alone.

| parts per scene |
|---|

## What the results show

- The neural adjustment answered 14 of 14 questions.

## How many small objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.379 | [0.30, 0.46] |
| 1 | 35 | 0.046 | 0.429 | 0.370 | [0.23, 0.54] |
| 2 | 48 | 0.063 | 0.333 | 0.368 | [0.25, 0.51] |
| 3 | 55 | 0.072 | 0.491 | 0.369 | [0.25, 0.50] |
| 4 | 66 | 0.087 | 0.333 | 0.364 | [0.26, 0.48] |
| 5 | 66 | 0.087 | 0.409 | 0.353 | [0.25, 0.47] |
| 6 | 67 | 0.088 | 0.313 | 0.342 | [0.24, 0.46] |
| 7 | 78 | 0.102 | 0.359 | 0.338 | [0.24, 0.45] |
| 8 | 52 | 0.068 | 0.327 | 0.339 | [0.23, 0.47] |
| 9 | 42 | 0.055 | 0.381 | 0.346 | [0.22, 0.50] |
| 10 | 29 | 0.038 | 0.414 | 0.357 | [0.21, 0.54] |
| 11 | 25 | 0.033 | 0.240 | 0.371 | [0.21, 0.57] |
| 12 | 15 | 0.020 | 0.200 | 0.385 | [0.19, 0.63] |
| 13 | 12 | 0.016 | 0.333 | 0.394 | [0.18, 0.66] |
| 14 | 14 | 0.018 | 0.357 | 0.389 | [0.19, 0.64] |
| 15 † | 4 | 0.005 | 0.500 | 0.353 | [0.08, 0.77] |
| 16 † | 3 | 0.004 | 0.333 | 0.273 | [0.04, 0.76] |
| 17 † | 2 | 0.003 | 0.000 | 0.164 | [0.01, 0.76] |
| 20 † | 2 | 0.003 | 0.000 | 0.010 | [0.00, 0.66] |

EQL's own `cause` search settles on 13: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.39).


## How many small objects cause every object of a scene to stay graspable, adjusting for the number of objects?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.351 | [0.28, 0.43] |
| 1 | 35 | 0.046 | 0.429 | 0.351 | [0.21, 0.52] |
| 2 | 48 | 0.063 | 0.333 | 0.348 | [0.23, 0.49] |
| 3 | 55 | 0.072 | 0.491 | 0.344 | [0.23, 0.48] |
| 4 | 66 | 0.087 | 0.333 | 0.338 | [0.24, 0.46] |
| 5 | 66 | 0.087 | 0.409 | 0.334 | [0.23, 0.45] |
| 6 | 67 | 0.088 | 0.313 | 0.336 | [0.23, 0.46] |
| 7 | 78 | 0.102 | 0.359 | 0.344 | [0.25, 0.45] |
| 8 | 52 | 0.068 | 0.327 | 0.354 | [0.24, 0.49] |
| 9 | 42 | 0.055 | 0.381 | 0.361 | [0.23, 0.51] |
| 10 | 29 | 0.038 | 0.414 | 0.364 | [0.21, 0.54] |
| 11 | 25 | 0.033 | 0.240 | 0.364 | [0.21, 0.56] |
| 12 | 15 | 0.020 | 0.200 | 0.359 | [0.17, 0.61] |
| 13 | 12 | 0.016 | 0.333 | 0.348 | [0.15, 0.62] |
| 14 | 14 | 0.018 | 0.357 | 0.333 | [0.15, 0.59] |
| 15 † | 4 | 0.005 | 0.500 | 0.315 | [0.07, 0.74] |
| 16 † | 3 | 0.004 | 0.333 | 0.297 | [0.05, 0.77] |
| 17 † | 2 | 0.003 | 0.000 | 0.279 | [0.03, 0.82] |
| 20 † | 2 | 0.003 | 0.000 | 0.246 | [0.03, 0.80] |

EQL's own `cause` search settles on 11: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.36).


## How many small objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out and the number of objects?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.327 | [0.26, 0.41] |
| 1 | 35 | 0.046 | 0.429 | 0.316 | [0.19, 0.48] |
| 2 | 48 | 0.063 | 0.333 | 0.307 | [0.20, 0.45] |
| 3 | 55 | 0.072 | 0.491 | 0.310 | [0.20, 0.44] |
| 4 | 66 | 0.087 | 0.333 | 0.314 | [0.21, 0.43] |
| 5 | 66 | 0.087 | 0.409 | 0.320 | [0.22, 0.44] |
| 6 | 67 | 0.088 | 0.313 | 0.326 | [0.23, 0.44] |
| 7 | 78 | 0.102 | 0.359 | 0.333 | [0.24, 0.44] |
| 8 | 52 | 0.068 | 0.327 | 0.341 | [0.23, 0.48] |
| 9 | 42 | 0.055 | 0.381 | 0.351 | [0.22, 0.50] |
| 10 | 29 | 0.038 | 0.414 | 0.362 | [0.21, 0.54] |
| 11 | 25 | 0.033 | 0.240 | 0.375 | [0.21, 0.57] |
| 12 | 15 | 0.020 | 0.200 | 0.389 | [0.19, 0.63] |
| 13 | 12 | 0.016 | 0.333 | 0.403 | [0.18, 0.67] |
| 14 | 14 | 0.018 | 0.357 | 0.416 | [0.20, 0.66] |
| 15 † | 4 | 0.005 | 0.500 | 0.429 | [0.12, 0.81] |
| 16 † | 3 | 0.004 | 0.333 | 0.439 | [0.10, 0.85] |
| 17 † | 2 | 0.003 | 0.000 | 0.444 | [0.08, 0.89] |
| 20 † | 2 | 0.003 | 0.000 | 0.419 | [0.07, 0.88] |

EQL's own `cause` search settles on 14: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.42).


## How many occluded objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.500 | [0.13, 0.87] |
| 1 † | 4 | 0.005 | 0.250 | 0.479 | [0.14, 0.84] |
| 2 | 12 | 0.016 | 0.167 | 0.473 | [0.23, 0.73] |
| 3 | 17 | 0.022 | 0.353 | 0.472 | [0.26, 0.69] |
| 4 | 25 | 0.033 | 0.160 | 0.470 | [0.29, 0.66] |
| 5 | 25 | 0.033 | 0.400 | 0.466 | [0.29, 0.65] |
| 6 | 42 | 0.055 | 0.190 | 0.459 | [0.32, 0.61] |
| 7 | 71 | 0.093 | 0.324 | 0.448 | [0.34, 0.56] |
| 8 | 72 | 0.094 | 0.431 | 0.429 | [0.32, 0.54] |
| 9 | 87 | 0.114 | 0.345 | 0.403 | [0.31, 0.51] |
| 10 | 95 | 0.125 | 0.453 | 0.372 | [0.28, 0.47] |
| 11 | 93 | 0.122 | 0.409 | 0.340 | [0.25, 0.44] |
| 12 | 61 | 0.080 | 0.426 | 0.309 | [0.21, 0.43] |
| 13 | 51 | 0.067 | 0.333 | 0.284 | [0.18, 0.42] |
| 14 | 43 | 0.056 | 0.372 | 0.275 | [0.16, 0.42] |
| 15 | 25 | 0.033 | 0.400 | 0.289 | [0.15, 0.49] |
| 16 | 21 | 0.028 | 0.524 | 0.333 | [0.17, 0.55] |
| 17 † | 8 | 0.010 | 0.625 | 0.396 | [0.15, 0.71] |
| 18 † | 6 | 0.008 | 0.667 | 0.464 | [0.17, 0.79] |
| 19 † | 1 | 0.001 | 1.000 | 0.531 | [0.06, 0.95] |
| 20 † | 1 | 0.001 | 1.000 | 0.597 | [0.08, 0.96] |

EQL's own `cause` search settles on 2: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.47).


## How many occluded objects cause every object of a scene to stay graspable, adjusting for the number of objects?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.193 | [0.02, 0.71] |
| 1 † | 4 | 0.005 | 0.250 | 0.188 | [0.03, 0.65] |
| 2 | 12 | 0.016 | 0.167 | 0.189 | [0.06, 0.47] |
| 3 | 17 | 0.022 | 0.353 | 0.196 | [0.07, 0.43] |
| 4 | 25 | 0.033 | 0.160 | 0.207 | [0.09, 0.40] |
| 5 | 25 | 0.033 | 0.400 | 0.224 | [0.10, 0.42] |
| 6 | 42 | 0.055 | 0.190 | 0.250 | [0.14, 0.40] |
| 7 | 71 | 0.093 | 0.324 | 0.280 | [0.19, 0.39] |
| 8 | 72 | 0.094 | 0.431 | 0.310 | [0.22, 0.42] |
| 9 | 87 | 0.114 | 0.345 | 0.341 | [0.25, 0.45] |
| 10 | 95 | 0.125 | 0.453 | 0.371 | [0.28, 0.47] |
| 11 | 93 | 0.122 | 0.409 | 0.400 | [0.31, 0.50] |
| 12 | 61 | 0.080 | 0.426 | 0.430 | [0.31, 0.55] |
| 13 | 51 | 0.067 | 0.333 | 0.464 | [0.33, 0.60] |
| 14 | 43 | 0.056 | 0.372 | 0.513 | [0.37, 0.65] |
| 15 | 25 | 0.033 | 0.400 | 0.589 | [0.40, 0.76] |
| 16 | 21 | 0.028 | 0.524 | 0.679 | [0.47, 0.84] |
| 17 † | 8 | 0.010 | 0.625 | 0.761 | [0.42, 0.93] |
| 18 † | 6 | 0.008 | 0.667 | 0.829 | [0.43, 0.97] |
| 19 † | 1 | 0.001 | 1.000 | 0.881 | [0.16, 1.00] |
| 20 † | 1 | 0.001 | 1.000 | 0.920 | [0.17, 1.00] |

EQL's own `cause` search settles on 16: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.68).


## How many occluded objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out and the number of objects?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.178 | [0.02, 0.70] |
| 1 † | 4 | 0.005 | 0.250 | 0.174 | [0.02, 0.64] |
| 2 | 12 | 0.016 | 0.167 | 0.171 | [0.05, 0.45] |
| 3 | 17 | 0.022 | 0.353 | 0.171 | [0.06, 0.40] |
| 4 | 25 | 0.033 | 0.160 | 0.180 | [0.08, 0.37] |
| 5 | 25 | 0.033 | 0.400 | 0.201 | [0.09, 0.39] |
| 6 | 42 | 0.055 | 0.190 | 0.229 | [0.13, 0.38] |
| 7 | 71 | 0.093 | 0.324 | 0.262 | [0.17, 0.37] |
| 8 | 72 | 0.094 | 0.431 | 0.297 | [0.20, 0.41] |
| 9 | 87 | 0.114 | 0.345 | 0.333 | [0.24, 0.44] |
| 10 | 95 | 0.125 | 0.453 | 0.369 | [0.28, 0.47] |
| 11 | 93 | 0.122 | 0.409 | 0.407 | [0.31, 0.51] |
| 12 | 61 | 0.080 | 0.426 | 0.447 | [0.33, 0.57] |
| 13 | 51 | 0.067 | 0.333 | 0.493 | [0.36, 0.63] |
| 14 | 43 | 0.056 | 0.372 | 0.552 | [0.40, 0.69] |
| 15 | 25 | 0.033 | 0.400 | 0.617 | [0.42, 0.78] |
| 16 | 21 | 0.028 | 0.524 | 0.680 | [0.47, 0.84] |
| 17 † | 8 | 0.010 | 0.625 | 0.738 | [0.40, 0.92] |
| 18 † | 6 | 0.008 | 0.667 | 0.790 | [0.40, 0.96] |
| 19 † | 1 | 0.001 | 1.000 | 0.834 | [0.14, 0.99] |
| 20 † | 1 | 0.001 | 1.000 | 0.870 | [0.16, 1.00] |

EQL's own `cause` search settles on 16: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.68).


## How many clear viewpoints cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.355 | [0.29, 0.43] |
| 1 | 11 | 0.014 | 0.545 | 0.411 | [0.18, 0.69] |
| 2 | 18 | 0.024 | 0.556 | 0.450 | [0.25, 0.67] |
| 3 † | 8 | 0.010 | 0.625 | 0.437 | [0.17, 0.74] |
| 4 † | 3 | 0.004 | 0.667 | 0.406 | [0.09, 0.83] |
| 5 † | 7 | 0.009 | 0.429 | 0.369 | [0.12, 0.71] |
| 6 † | 5 | 0.007 | 0.600 | 0.331 | [0.09, 0.72] |
| 7 | 10 | 0.013 | 0.500 | 0.306 | [0.11, 0.61] |
| 8 † | 5 | 0.007 | 0.400 | 0.293 | [0.07, 0.70] |
| 9 † | 6 | 0.008 | 0.500 | 0.293 | [0.08, 0.67] |
| 10 | 10 | 0.013 | 0.400 | 0.304 | [0.11, 0.61] |
| 11 | 10 | 0.013 | 0.100 | 0.324 | [0.12, 0.62] |
| 12 † | 5 | 0.007 | 0.800 | 0.351 | [0.09, 0.74] |
| 13 † | 9 | 0.012 | 0.222 | 0.380 | [0.15, 0.68] |
| 14 † | 3 | 0.004 | 1.000 | 0.408 | [0.09, 0.83] |
| 15 | 10 | 0.013 | 0.500 | 0.434 | [0.19, 0.71] |
| 16 † | 2 | 0.003 | 0.000 | 0.455 | [0.08, 0.89] |
| 17 † | 4 | 0.005 | 0.500 | 0.472 | [0.14, 0.84] |
| 18 | 11 | 0.014 | 0.273 | 0.486 | [0.24, 0.74] |
| 19 † | 5 | 0.007 | 0.800 | 0.497 | [0.17, 0.83] |
| 20 † | 3 | 0.004 | 0.333 | 0.507 | [0.13, 0.88] |
| 21 † | 7 | 0.009 | 0.714 | 0.514 | [0.21, 0.81] |
| 22 † | 5 | 0.007 | 0.600 | 0.519 | [0.18, 0.84] |
| 23 † | 9 | 0.012 | 0.778 | 0.521 | [0.24, 0.79] |
| 24 † | 3 | 0.004 | 0.667 | 0.519 | [0.13, 0.88] |
| 25 † | 7 | 0.009 | 0.571 | 0.514 | [0.21, 0.81] |
| 26 † | 4 | 0.005 | 0.500 | 0.505 | [0.15, 0.85] |
| 27 † | 8 | 0.010 | 0.750 | 0.493 | [0.21, 0.78] |
| 28 † | 4 | 0.005 | 0.500 | 0.479 | [0.14, 0.84] |
| 29 † | 3 | 0.004 | 0.667 | 0.463 | [0.11, 0.86] |
| 30 † | 6 | 0.008 | 0.667 | 0.446 | [0.16, 0.78] |
| 31 † | 3 | 0.004 | 0.333 | 0.429 | [0.10, 0.84] |
| 32 † | 5 | 0.007 | 0.400 | 0.412 | [0.12, 0.78] |
| 33 † | 5 | 0.007 | 0.400 | 0.395 | [0.12, 0.77] |
| 34 † | 9 | 0.012 | 0.111 | 0.378 | [0.15, 0.68] |
| 35 † | 5 | 0.007 | 1.000 | 0.362 | [0.10, 0.74] |
| 36 † | 6 | 0.008 | 0.500 | 0.347 | [0.10, 0.71] |
| 37 | 10 | 0.013 | 0.400 | 0.332 | [0.13, 0.63] |
| 38 † | 6 | 0.008 | 0.500 | 0.317 | [0.09, 0.69] |
| 39 † | 7 | 0.009 | 0.286 | 0.303 | [0.09, 0.66] |
| 40 | 11 | 0.014 | 0.273 | 0.290 | [0.11, 0.58] |
| 41 † | 7 | 0.009 | 0.286 | 0.277 | [0.08, 0.63] |
| 42 | 10 | 0.013 | 0.500 | 0.264 | [0.09, 0.57] |
| 43 | 14 | 0.018 | 0.357 | 0.252 | [0.10, 0.51] |
| 44 | 13 | 0.017 | 0.308 | 0.240 | [0.09, 0.51] |
| 45 | 15 | 0.020 | 0.400 | 0.228 | [0.09, 0.48] |
| 46 | 22 | 0.029 | 0.136 | 0.217 | [0.09, 0.42] |
| 47 | 20 | 0.026 | 0.300 | 0.206 | [0.08, 0.42] |
| 48 | 29 | 0.038 | 0.172 | 0.196 | [0.09, 0.37] |
| 49 | 32 | 0.042 | 0.312 | 0.186 | [0.09, 0.35] |
| 50 | 45 | 0.059 | 0.111 | 0.177 | [0.09, 0.31] |
| 51 | 36 | 0.047 | 0.139 | 0.167 | [0.08, 0.32] |
| 52 | 71 | 0.093 | 0.197 | 0.159 | [0.09, 0.26] |

EQL's own `cause` search settles on 18: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.49).


## How many clear viewpoints cause every object of a scene to stay graspable, adjusting for the number of objects?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.352 | [0.29, 0.42] |
| 1 | 11 | 0.014 | 0.545 | 0.421 | [0.19, 0.69] |
| 2 | 18 | 0.024 | 0.556 | 0.476 | [0.27, 0.69] |
| 3 † | 8 | 0.010 | 0.625 | 0.466 | [0.19, 0.76] |
| 4 † | 3 | 0.004 | 0.667 | 0.441 | [0.10, 0.85] |
| 5 † | 7 | 0.009 | 0.429 | 0.414 | [0.15, 0.74] |
| 6 † | 5 | 0.007 | 0.600 | 0.390 | [0.11, 0.76] |
| 7 | 10 | 0.013 | 0.500 | 0.373 | [0.15, 0.67] |
| 8 † | 5 | 0.007 | 0.400 | 0.363 | [0.10, 0.75] |
| 9 † | 6 | 0.008 | 0.500 | 0.363 | [0.11, 0.72] |
| 10 | 10 | 0.013 | 0.400 | 0.371 | [0.15, 0.66] |
| 11 | 10 | 0.013 | 0.100 | 0.386 | [0.16, 0.68] |
| 12 † | 5 | 0.007 | 0.800 | 0.406 | [0.12, 0.77] |
| 13 † | 9 | 0.012 | 0.222 | 0.428 | [0.18, 0.72] |
| 14 † | 3 | 0.004 | 1.000 | 0.451 | [0.10, 0.85] |
| 15 | 10 | 0.013 | 0.500 | 0.475 | [0.22, 0.74] |
| 16 † | 2 | 0.003 | 0.000 | 0.497 | [0.09, 0.90] |
| 17 † | 4 | 0.005 | 0.500 | 0.519 | [0.16, 0.86] |
| 18 | 11 | 0.014 | 0.273 | 0.538 | [0.27, 0.78] |
| 19 † | 5 | 0.007 | 0.800 | 0.554 | [0.20, 0.86] |
| 20 † | 3 | 0.004 | 0.333 | 0.568 | [0.16, 0.90] |
| 21 † | 7 | 0.009 | 0.714 | 0.579 | [0.26, 0.85] |
| 22 † | 5 | 0.007 | 0.600 | 0.588 | [0.22, 0.88] |
| 23 † | 9 | 0.012 | 0.778 | 0.592 | [0.29, 0.83] |
| 24 † | 3 | 0.004 | 0.667 | 0.593 | [0.17, 0.91] |
| 25 † | 7 | 0.009 | 0.571 | 0.588 | [0.26, 0.85] |
| 26 † | 4 | 0.005 | 0.500 | 0.578 | [0.19, 0.89] |
| 27 † | 8 | 0.010 | 0.750 | 0.565 | [0.26, 0.83] |
| 28 † | 4 | 0.005 | 0.500 | 0.547 | [0.17, 0.87] |
| 29 † | 3 | 0.004 | 0.667 | 0.528 | [0.14, 0.89] |
| 30 † | 6 | 0.008 | 0.667 | 0.508 | [0.19, 0.82] |
| 31 † | 3 | 0.004 | 0.333 | 0.487 | [0.12, 0.87] |
| 32 † | 5 | 0.007 | 0.400 | 0.466 | [0.15, 0.81] |
| 33 † | 5 | 0.007 | 0.400 | 0.446 | [0.14, 0.80] |
| 34 † | 9 | 0.012 | 0.111 | 0.428 | [0.18, 0.72] |
| 35 † | 5 | 0.007 | 1.000 | 0.410 | [0.12, 0.78] |
| 36 † | 6 | 0.008 | 0.500 | 0.393 | [0.13, 0.74] |
| 37 | 10 | 0.013 | 0.400 | 0.377 | [0.15, 0.67] |
| 38 † | 6 | 0.008 | 0.500 | 0.361 | [0.11, 0.72] |
| 39 † | 7 | 0.009 | 0.286 | 0.345 | [0.11, 0.69] |
| 40 | 11 | 0.014 | 0.273 | 0.330 | [0.13, 0.62] |
| 41 † | 7 | 0.009 | 0.286 | 0.315 | [0.10, 0.66] |
| 42 | 10 | 0.013 | 0.500 | 0.300 | [0.11, 0.60] |
| 43 | 14 | 0.018 | 0.357 | 0.285 | [0.12, 0.55] |
| 44 | 13 | 0.017 | 0.308 | 0.271 | [0.10, 0.54] |
| 45 | 15 | 0.020 | 0.400 | 0.257 | [0.10, 0.51] |
| 46 | 22 | 0.029 | 0.136 | 0.243 | [0.11, 0.45] |
| 47 | 20 | 0.026 | 0.300 | 0.230 | [0.10, 0.45] |
| 48 | 29 | 0.038 | 0.172 | 0.217 | [0.11, 0.39] |
| 49 | 32 | 0.042 | 0.312 | 0.204 | [0.10, 0.37] |
| 50 | 45 | 0.059 | 0.111 | 0.192 | [0.10, 0.33] |
| 51 | 36 | 0.047 | 0.139 | 0.180 | [0.09, 0.33] |
| 52 | 71 | 0.093 | 0.197 | 0.169 | [0.10, 0.27] |

EQL's own `cause` search settles on 18: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.54).


## How many clear viewpoints cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out and the number of objects?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.423 | [0.35, 0.50] |
| 1 | 11 | 0.014 | 0.545 | 0.459 | [0.22, 0.72] |
| 2 | 18 | 0.024 | 0.556 | 0.489 | [0.28, 0.70] |
| 3 † | 8 | 0.010 | 0.625 | 0.492 | [0.21, 0.78] |
| 4 † | 3 | 0.004 | 0.667 | 0.469 | [0.11, 0.86] |
| 5 † | 7 | 0.009 | 0.429 | 0.430 | [0.16, 0.75] |
| 6 † | 5 | 0.007 | 0.600 | 0.385 | [0.11, 0.76] |
| 7 | 10 | 0.013 | 0.500 | 0.349 | [0.14, 0.65] |
| 8 † | 5 | 0.007 | 0.400 | 0.332 | [0.09, 0.72] |
| 9 † | 6 | 0.008 | 0.500 | 0.332 | [0.10, 0.70] |
| 10 | 10 | 0.013 | 0.400 | 0.343 | [0.13, 0.64] |
| 11 | 10 | 0.013 | 0.100 | 0.362 | [0.14, 0.66] |
| 12 † | 5 | 0.007 | 0.800 | 0.385 | [0.11, 0.76] |
| 13 † | 9 | 0.012 | 0.222 | 0.410 | [0.17, 0.71] |
| 14 † | 3 | 0.004 | 1.000 | 0.436 | [0.10, 0.85] |
| 15 | 10 | 0.013 | 0.500 | 0.460 | [0.21, 0.73] |
| 16 † | 2 | 0.003 | 0.000 | 0.481 | [0.09, 0.90] |
| 17 † | 4 | 0.005 | 0.500 | 0.498 | [0.15, 0.85] |
| 18 | 11 | 0.014 | 0.273 | 0.514 | [0.26, 0.76] |
| 19 † | 5 | 0.007 | 0.800 | 0.527 | [0.19, 0.84] |
| 20 † | 3 | 0.004 | 0.333 | 0.538 | [0.14, 0.89] |
| 21 † | 7 | 0.009 | 0.714 | 0.548 | [0.23, 0.83] |
| 22 † | 5 | 0.007 | 0.600 | 0.554 | [0.20, 0.86] |
| 23 † | 9 | 0.012 | 0.778 | 0.558 | [0.27, 0.81] |
| 24 † | 3 | 0.004 | 0.667 | 0.557 | [0.15, 0.90] |
| 25 † | 7 | 0.009 | 0.571 | 0.553 | [0.24, 0.83] |
| 26 † | 4 | 0.005 | 0.500 | 0.545 | [0.17, 0.87] |
| 27 † | 8 | 0.010 | 0.750 | 0.533 | [0.24, 0.81] |
| 28 † | 4 | 0.005 | 0.500 | 0.518 | [0.16, 0.86] |
| 29 † | 3 | 0.004 | 0.667 | 0.502 | [0.13, 0.88] |
| 30 † | 6 | 0.008 | 0.667 | 0.483 | [0.18, 0.80] |
| 31 † | 3 | 0.004 | 0.333 | 0.464 | [0.11, 0.86] |
| 32 † | 5 | 0.007 | 0.400 | 0.444 | [0.14, 0.80] |
| 33 † | 5 | 0.007 | 0.400 | 0.425 | [0.13, 0.79] |
| 34 † | 9 | 0.012 | 0.111 | 0.406 | [0.16, 0.70] |
| 35 † | 5 | 0.007 | 1.000 | 0.388 | [0.11, 0.76] |
| 36 † | 6 | 0.008 | 0.500 | 0.371 | [0.12, 0.73] |
| 37 | 10 | 0.013 | 0.400 | 0.354 | [0.14, 0.65] |
| 38 † | 6 | 0.008 | 0.500 | 0.337 | [0.10, 0.70] |
| 39 † | 7 | 0.009 | 0.286 | 0.321 | [0.10, 0.67] |
| 40 | 11 | 0.014 | 0.273 | 0.305 | [0.12, 0.60] |
| 41 † | 7 | 0.009 | 0.286 | 0.290 | [0.08, 0.64] |
| 42 | 10 | 0.013 | 0.500 | 0.274 | [0.09, 0.58] |
| 43 | 14 | 0.018 | 0.357 | 0.259 | [0.10, 0.52] |
| 44 | 13 | 0.017 | 0.308 | 0.245 | [0.09, 0.52] |
| 45 | 15 | 0.020 | 0.400 | 0.231 | [0.09, 0.48] |
| 46 | 22 | 0.029 | 0.136 | 0.217 | [0.09, 0.42] |
| 47 | 20 | 0.026 | 0.300 | 0.204 | [0.08, 0.42] |
| 48 | 29 | 0.038 | 0.172 | 0.192 | [0.09, 0.37] |
| 49 | 32 | 0.042 | 0.312 | 0.180 | [0.08, 0.34] |
| 50 | 45 | 0.059 | 0.111 | 0.169 | [0.09, 0.30] |
| 51 | 36 | 0.047 | 0.139 | 0.159 | [0.07, 0.31] |
| 52 | 71 | 0.093 | 0.197 | 0.150 | [0.09, 0.25] |

EQL's own `cause` search settles on 18: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.51).


## Does the object catalogue a scene is built from cause every object of it to stay graspable, adjusting for its spread (extent)?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.407 | 0.391 | [0.34, 0.44] |
| mixed | 129 | 0.169 | 0.333 | 0.335 | [0.26, 0.42] |
| ycb-video | 248 | 0.325 | 0.355 | 0.394 | [0.33, 0.46] |

EQL's own `cause` search settles on ycb-video: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.39).


## Does the object catalogue a scene is built from cause every object of it to stay graspable, adjusting for its small-object count?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.407 | 0.393 | [0.35, 0.44] |
| mixed | 129 | 0.169 | 0.333 | 0.330 | [0.25, 0.41] |
| ycb-video | 248 | 0.325 | 0.355 | 0.386 | [0.33, 0.45] |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.39).


## Does the object catalogue a scene is built from cause object 0 of it to be heavily occluded?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 5536 | 0.499 | 0.337 | 0.311 | [0.30, 0.32] |
| mixed | 1905 | 0.172 | 0.349 | 0.318 | [0.30, 0.34] |
| ycb-video | 3650 | 0.329 | 0.301 | 0.312 | [0.30, 0.33] |

EQL's own `cause` search settles on mixed: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.32).


## How many occluded objects cause object 0 of a scene to lose every grasp?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 33 | 0.003 | 0.091 | 0.247 | [0.13, 0.42] |
| 1 | 47 | 0.004 | 0.213 | 0.233 | [0.13, 0.37] |
| 2 | 156 | 0.014 | 0.128 | 0.215 | [0.16, 0.29] |
| 3 | 197 | 0.018 | 0.208 | 0.194 | [0.15, 0.26] |
| 4 | 313 | 0.028 | 0.163 | 0.175 | [0.14, 0.22] |
| 5 | 295 | 0.027 | 0.112 | 0.156 | [0.12, 0.20] |
| 6 | 543 | 0.049 | 0.157 | 0.138 | [0.11, 0.17] |
| 7 | 919 | 0.083 | 0.141 | 0.126 | [0.11, 0.15] |
| 8 | 936 | 0.084 | 0.112 | 0.120 | [0.10, 0.14] |
| 9 | 1205 | 0.109 | 0.118 | 0.118 | [0.10, 0.14] |
| 10 | 1378 | 0.124 | 0.108 | 0.118 | [0.10, 0.14] |
| 11 | 1402 | 0.126 | 0.115 | 0.118 | [0.10, 0.14] |
| 12 | 945 | 0.085 | 0.098 | 0.117 | [0.10, 0.14] |
| 13 | 839 | 0.076 | 0.088 | 0.114 | [0.09, 0.14] |
| 14 | 744 | 0.067 | 0.101 | 0.105 | [0.09, 0.13] |
| 15 | 447 | 0.040 | 0.083 | 0.091 | [0.07, 0.12] |
| 16 | 383 | 0.035 | 0.044 | 0.074 | [0.05, 0.10] |
| 17 | 152 | 0.014 | 0.020 | 0.057 | [0.03, 0.11] |
| 18 | 117 | 0.011 | 0.034 | 0.043 | [0.02, 0.10] |
| 19 | 20 | 0.002 | 0.000 | 0.033 | [0.00, 0.21] |
| 20 | 20 | 0.002 | 0.000 | 0.025 | [0.00, 0.20] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.25).


## Does the size of object 0 of a scene cause it to lose every grasp?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| large | 3602 | 0.325 | 0.103 | 0.115 | [0.10, 0.13] |
| medium | 3578 | 0.323 | 0.108 | 0.125 | [0.11, 0.14] |
| small | 3911 | 0.353 | 0.122 | 0.150 | [0.14, 0.16] |

EQL's own `cause` search settles on small: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.15).

