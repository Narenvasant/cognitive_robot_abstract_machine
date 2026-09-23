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
- fewest training rows per leaf, as a share of the rows fitted on: 0.05 in a cause-specific model, 0.15 in the plain model that scores held-out scenes
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

| question | relational circuit | hybrid circuit | propositional tree | unrolled tree | scalars-only tree | regression adjustment | neural adjustment |
|---|---|---|---|---|---|---|---|
| How many small objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out? | answered: with 3 small objects, every object of the scene stays graspable with probability 0.49, the highest of any setting; with 12 small objects it is only 0.20. | answered: with 3 small objects, every object of the scene stays graspable with probability 0.49, the highest of any setting; with 12 small objects it is only 0.20. | answered: with 3 small objects, every object of the scene stays graspable with probability 0.49, the highest of any setting; with 12 small objects it is only 0.20. | answered: with 3 small objects, every object of the scene stays graspable with probability 0.49, the highest of any setting; with 12 small objects it is only 0.20. | refused: the fitted table has no column for the queried variables. | answered: with 0 small objects, every object of the scene stays graspable with probability 0.42, the highest of any setting; with 14 small objects it is only 0.31. | answered: with 13 small objects, every object of the scene stays graspable with probability 0.39, the highest of any setting; with 7 small objects it is only 0.34. |
| How many small objects cause every object of a scene to stay graspable, adjusting for the number of objects? | answered: with 14 small objects, every object of the scene stays graspable with probability 0.51, the highest of any setting; with 12 small objects it is only 0.11. | answered: with 14 small objects, every object of the scene stays graspable with probability 0.51, the highest of any setting; with 12 small objects it is only 0.11. | answered: with 14 small objects, every object of the scene stays graspable with probability 0.51, the highest of any setting; with 12 small objects it is only 0.11. | answered: with 14 small objects, every object of the scene stays graspable with probability 0.51, the highest of any setting; with 12 small objects it is only 0.11. | refused: the fitted table has no column for the queried variables. | answered: with 0 small objects, every object of the scene stays graspable with probability 0.42, the highest of any setting; with 14 small objects it is only 0.31. | answered: with 11 small objects, every object of the scene stays graspable with probability 0.36, the highest of any setting; with 14 small objects it is only 0.33. |
| How many small objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out and the number of objects? | answered: with 14 small objects, every object of the scene stays graspable with probability 0.51, the highest of any setting; with 12 small objects it is only 0.11. | answered: with 14 small objects, every object of the scene stays graspable with probability 0.51, the highest of any setting; with 12 small objects it is only 0.11. | answered: with 14 small objects, every object of the scene stays graspable with probability 0.51, the highest of any setting; with 12 small objects it is only 0.11. | answered: with 14 small objects, every object of the scene stays graspable with probability 0.51, the highest of any setting; with 12 small objects it is only 0.11. | refused: the fitted table has no column for the queried variables. | answered: with 0 small objects, every object of the scene stays graspable with probability 0.38, the highest of any setting; with 14 small objects it is only 0.38. | answered: with 14 small objects, every object of the scene stays graspable with probability 0.42, the highest of any setting; with 2 small objects it is only 0.31. |
| How many occluded objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out? | answered: with 16 occluded objects, every object of the scene stays graspable with probability 0.52, the highest of any setting; with 4 occluded objects it is only 0.16. | answered: with 16 occluded objects, every object of the scene stays graspable with probability 0.52, the highest of any setting; with 4 occluded objects it is only 0.16. | answered: with 16 occluded objects, every object of the scene stays graspable with probability 0.52, the highest of any setting; with 4 occluded objects it is only 0.16. | answered: with 16 occluded objects, every object of the scene stays graspable with probability 0.52, the highest of any setting; with 4 occluded objects it is only 0.16. | refused: the fitted table has no column for the queried variables. | answered: with 16 occluded objects, every object of the scene stays graspable with probability 0.47, the highest of any setting; with 2 occluded objects it is only 0.27. | answered: with 2 occluded objects, every object of the scene stays graspable with probability 0.47, the highest of any setting; with 14 occluded objects it is only 0.27. |
| How many occluded objects cause every object of a scene to stay graspable, adjusting for the number of objects? | answered: with 12 occluded objects, every object of the scene stays graspable with probability 0.47, the highest of any setting; with 4 occluded objects it is only 0.07. | answered: with 16 occluded objects, every object of the scene stays graspable with probability 0.61, the highest of any setting; with 4 occluded objects it is only 0.10. | answered: with 12 occluded objects, every object of the scene stays graspable with probability 0.47, the highest of any setting; with 4 occluded objects it is only 0.07. | answered: with 16 occluded objects, every object of the scene stays graspable with probability 0.55, the highest of any setting; with 4 occluded objects it is only 0.09. | refused: the fitted table has no column for the queried variables. | answered: with 16 occluded objects, every object of the scene stays graspable with probability 0.71, the highest of any setting; with 2 occluded objects it is only 0.10. | answered: with 16 occluded objects, every object of the scene stays graspable with probability 0.68, the highest of any setting; with 2 occluded objects it is only 0.19. |
| How many occluded objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out and the number of objects? | answered: with 12 occluded objects, every object of the scene stays graspable with probability 0.45, the highest of any setting; with 4 occluded objects it is only 0.09. | answered: with 16 occluded objects, every object of the scene stays graspable with probability 0.61, the highest of any setting; with 4 occluded objects it is only 0.10. | answered: with 12 occluded objects, every object of the scene stays graspable with probability 0.46, the highest of any setting; with 4 occluded objects it is only 0.08. | answered: with 16 occluded objects, every object of the scene stays graspable with probability 0.55, the highest of any setting; with 4 occluded objects it is only 0.09. | refused: the fitted table has no column for the queried variables. | answered: with 16 occluded objects, every object of the scene stays graspable with probability 0.70, the highest of any setting; with 2 occluded objects it is only 0.10. | answered: with 16 occluded objects, every object of the scene stays graspable with probability 0.68, the highest of any setting; with 2 occluded objects it is only 0.17. |
| How many clear viewpoints cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out? | answered: with 2 clear viewpoints, every object of the scene stays graspable with probability 0.59, the highest of any setting; with 11 clear viewpoints it is only 0.10. | answered: with 2 clear viewpoints, every object of the scene stays graspable with probability 0.59, the highest of any setting; with 11 clear viewpoints it is only 0.10. | answered: with 2 clear viewpoints, every object of the scene stays graspable with probability 0.59, the highest of any setting; with 11 clear viewpoints it is only 0.10. | answered: with 2 clear viewpoints, every object of the scene stays graspable with probability 0.59, the highest of any setting; with 11 clear viewpoints it is only 0.10. | refused: the fitted table has no column for the queried variables. | answered: with 0 clear viewpoints, every object of the scene stays graspable with probability 0.53, the highest of any setting; with 52 clear viewpoints it is only 0.24. | answered: with 18 clear viewpoints, every object of the scene stays graspable with probability 0.38, the highest of any setting; with 7 clear viewpoints it is only 0.38. |
| How many clear viewpoints cause every object of a scene to stay graspable, adjusting for the number of objects? | answered: with 15 clear viewpoints, every object of the scene stays graspable with probability 0.60, the highest of any setting; with 11 clear viewpoints it is only 0.08. | answered: with 15 clear viewpoints, every object of the scene stays graspable with probability 0.60, the highest of any setting; with 11 clear viewpoints it is only 0.08. | answered: with 15 clear viewpoints, every object of the scene stays graspable with probability 0.60, the highest of any setting; with 11 clear viewpoints it is only 0.08. | answered: with 15 clear viewpoints, every object of the scene stays graspable with probability 0.60, the highest of any setting; with 11 clear viewpoints it is only 0.08. | refused: the fitted table has no column for the queried variables. | answered: with 0 clear viewpoints, every object of the scene stays graspable with probability 0.55, the highest of any setting; with 52 clear viewpoints it is only 0.23. | answered: with 37 clear viewpoints, every object of the scene stays graspable with probability 0.38, the highest of any setting; with 7 clear viewpoints it is only 0.38. |
| How many clear viewpoints cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out and the number of objects? | answered: with 15 clear viewpoints, every object of the scene stays graspable with probability 0.52, the highest of any setting; with 50 clear viewpoints it is only 0.11. | answered: with 1 clear viewpoints, every object of the scene stays graspable with probability 0.60, the highest of any setting; with 11 clear viewpoints it is only 0.09. | answered: with 1 clear viewpoints, every object of the scene stays graspable with probability 0.53, the highest of any setting; with 50 clear viewpoints it is only 0.12. | answered: with 1 clear viewpoints, every object of the scene stays graspable with probability 0.63, the highest of any setting; with 11 clear viewpoints it is only 0.09. | refused: the fitted table has no column for the queried variables. | answered: with 0 clear viewpoints, every object of the scene stays graspable with probability 0.55, the highest of any setting; with 52 clear viewpoints it is only 0.22. | answered: with 18 clear viewpoints, every object of the scene stays graspable with probability 0.38, the highest of any setting; with 52 clear viewpoints it is only 0.38. |
| Does the object catalogue a scene is built from cause every object of it to stay graspable, adjusting for its spread (extent)? | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.41, the highest of any setting; with the mixed catalogue it is only 0.33. | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.41, the highest of any setting; with the mixed catalogue it is only 0.33. | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.41, the highest of any setting; with the mixed catalogue it is only 0.33. | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.41, the highest of any setting; with the mixed catalogue it is only 0.33. | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.41, the highest of any setting; with the mixed catalogue it is only 0.33. | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.41, the highest of any setting; with the mixed catalogue it is only 0.34. | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.38, the highest of any setting; with the mixed catalogue it is only 0.38. |
| Does the object catalogue a scene is built from cause every object of it to stay graspable, adjusting for its small-object count? | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.39, the highest of any setting; with the mixed catalogue it is only 0.27. | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.40, the highest of any setting; with the mixed catalogue it is only 0.30. | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.40, the highest of any setting; with the mixed catalogue it is only 0.28. | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.40, the highest of any setting; with the mixed catalogue it is only 0.30. | refused: the fitted table has no column for the queried variables. | answered: with the grasp catalogue, every object of the scene stays graspable with probability 0.40, the highest of any setting; with the mixed catalogue it is only 0.34. | answered: with the ycb-video catalogue, every object of the scene stays graspable with probability 0.38, the highest of any setting; with the mixed catalogue it is only 0.38. |
| Does the object catalogue a scene is built from cause object 0 of it to be heavily occluded? | answered: with the grasp catalogue, object 0 is heavily occluded with probability 0.36, the highest of any setting; with the ycb-video catalogue it is only 0.32. | answered: with the grasp catalogue, object 0 is heavily occluded with probability 0.36, the highest of any setting; with the ycb-video catalogue it is only 0.32. | refused: the fitted table has no column for the queried variables. | answered: with the ycb-video catalogue, object 0 is heavily occluded with probability 0.29, the highest of any setting; with the mixed catalogue it is only 0.23. | refused: the fitted table has no column for the queried variables. | refused: the fitted table has no column for the queried variables. | answered: with the mixed catalogue, object 0 is heavily occluded with probability 0.35, the highest of any setting; with the ycb-video catalogue it is only 0.34. |
| How many occluded objects cause object 0 of a scene to lose every grasp? | answered: with 10 occluded objects, object 0 loses every grasp with probability 0.13, the highest of any setting; with 2 occluded objects it is only 0.08. | answered: with 10 occluded objects, object 0 loses every grasp with probability 0.13, the highest of any setting; with 2 occluded objects it is only 0.08. | refused: the fitted table has no column for the queried variables. | answered: with 3 occluded objects, object 0 loses every grasp with probability 0.18, the highest of any setting; with 15 occluded objects it is only 0.04. | refused: the fitted table has no column for the queried variables. | refused: the fitted table has no column for the queried variables. | answered: with 0 occluded objects, object 0 loses every grasp with probability 0.25, the highest of any setting; with 20 occluded objects it is only 0.03. |
| Does the size of object 0 of a scene cause it to lose every grasp? | answered: with object 0 being small, object 0 loses every grasp with probability 0.12, the highest of any setting; with object 0 being large it is only 0.10. | answered: with object 0 being small, object 0 loses every grasp with probability 0.12, the highest of any setting; with object 0 being large it is only 0.10. | refused: the fitted table has no column for the queried variables. | answered: with object 0 being large, object 0 loses every grasp with probability 0.15, the highest of any setting; with object 0 being small it is only 0.08. | refused: the fitted table has no column for the queried variables. | refused: the fitted table has no column for the queried variables. | answered: with object 0 being small, object 0 loses every grasp with probability 0.17, the highest of any setting; with object 0 being large it is only 0.16. |

A question about counts needs the counts: the scalars-only tree refuses it. A question whose effect is one object's own attribute needs the objects: the propositional tree refuses it, the unrolled tree answers it about whatever object the scenes list at that position, and the relational circuit answers it about an exchangeable object. What an answer about "object 0" is worth is what the reordering below measures.

## Trend and contrast

The most effective setting is an argmax over up to twenty sparse regions and moves with the split. Two summaries that do not: *trend* is Spearman's rank correlation between the cause's value and the adjusted probability over the supported regions, for a numeric cause; *contrast* is the adjusted probability at the highest supported region minus at the lowest (for a symbolic cause, at the most effective minus at the least), with Newcombe's interval from the Wilson intervals of the two regions' support.

| question | relational circuit, trend | relational circuit, contrast | hybrid circuit, trend | hybrid circuit, contrast | propositional tree, trend | propositional tree, contrast | unrolled tree, trend | unrolled tree, contrast | scalars-only tree, trend | scalars-only tree, contrast | regression adjustment, trend | regression adjustment, contrast | neural adjustment, trend | neural adjustment, contrast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small_object_count_causes_graspability_adjusting_extent | -0.52 | -0.09 [-0.30, 0.18] (0 → 14) | -0.53 | -0.09 [-0.30, 0.18] (0 → 14) | -0.53 | -0.09 [-0.30, 0.18] (0 → 14) | -0.52 | -0.09 [-0.30, 0.18] (0 → 14) | - | - | -1.00 | -0.11 [-0.31, 0.16] (0 → 14) | 0.25 | 0.01 [-0.21, 0.27] (0 → 14) |
| small_object_count_causes_graspability_adjusting_object_count | -0.42 | 0.07 [-0.18, 0.31] (0 → 14) | -0.37 | 0.06 [-0.19, 0.30] (0 → 14) | -0.36 | 0.06 [-0.18, 0.31] (0 → 14) | -0.39 | 0.06 [-0.19, 0.30] (0 → 14) | - | - | -1.00 | -0.11 [-0.30, 0.16] (0 → 14) | 0.20 | -0.02 [-0.22, 0.25] (0 → 14) |
| small_object_count_causes_graspability_adjusting_extent_and_object_count | -0.42 | 0.08 [-0.17, 0.32] (0 → 14) | -0.37 | 0.06 [-0.19, 0.30] (0 → 14) | -0.36 | 0.08 [-0.17, 0.32] (0 → 14) | -0.39 | 0.06 [-0.18, 0.31] (0 → 14) | - | - | -1.00 | -0.00 [-0.22, 0.26] (0 → 14) | 0.90 | 0.09 [-0.14, 0.35] (0 → 14) |
| occluded_object_count_causes_graspability_adjusting_extent | 0.68 | 0.34 [-0.00, 0.57] (2 → 16) | 0.68 | 0.34 [-0.00, 0.57] (2 → 16) | 0.64 | 0.34 [-0.00, 0.57] (2 → 16) | 0.64 | 0.34 [-0.00, 0.57] (2 → 16) | - | - | 1.00 | 0.20 [-0.14, 0.47] (2 → 16) | -0.95 | -0.14 [-0.44, 0.18] (2 → 16) |
| occluded_object_count_causes_graspability_adjusting_object_count | 0.76 | 0.20 [-0.13, 0.44] (2 → 16) | 0.76 | 0.42 [0.07, 0.64] (2 → 16) | 0.78 | 0.21 [-0.13, 0.45] (2 → 16) | 0.81 | 0.36 [0.02, 0.59] (2 → 16) | - | - | 1.00 | 0.61 [0.26, 0.78] (2 → 16) | 1.00 | 0.49 [0.14, 0.70] (2 → 16) |
| occluded_object_count_causes_graspability_adjusting_extent_and_object_count | 0.78 | 0.17 [-0.16, 0.42] (2 → 16) | 0.74 | 0.42 [0.07, 0.64] (2 → 16) | 0.76 | 0.17 [-0.16, 0.43] (2 → 16) | 0.82 | 0.36 [0.02, 0.59] (2 → 16) | - | - | 1.00 | 0.60 [0.26, 0.77] (2 → 16) | 1.00 | 0.51 [0.16, 0.71] (2 → 16) |
| clear_viewpoint_count_causes_graspability_adjusting_extent | -0.66 | -0.28 [-0.39, -0.15] (0 → 52) | -0.66 | -0.28 [-0.39, -0.15] (0 → 52) | -0.66 | -0.28 [-0.39, -0.15] (0 → 52) | -0.66 | -0.28 [-0.39, -0.15] (0 → 52) | - | - | -1.00 | -0.29 [-0.40, -0.16] (0 → 52) | -0.28 | -0.00 [-0.13, 0.13] (0 → 52) |
| clear_viewpoint_count_causes_graspability_adjusting_object_count | -0.71 | -0.32 [-0.42, -0.19] (0 → 52) | -0.70 | -0.30 [-0.40, -0.17] (0 → 52) | -0.71 | -0.32 [-0.42, -0.19] (0 → 52) | -0.70 | -0.31 [-0.41, -0.19] (0 → 52) | - | - | -1.00 | -0.33 [-0.43, -0.19] (0 → 52) | -0.08 | -0.00 [-0.13, 0.13] (0 → 52) |
| clear_viewpoint_count_causes_graspability_adjusting_extent_and_object_count | -0.58 | -0.29 [-0.40, -0.16] (0 → 52) | -0.66 | -0.29 [-0.39, -0.16] (0 → 52) | -0.59 | -0.29 [-0.40, -0.16] (0 → 52) | -0.65 | -0.31 [-0.41, -0.18] (0 → 52) | - | - | -1.00 | -0.33 [-0.44, -0.20] (0 → 52) | -0.92 | -0.00 [-0.13, 0.13] (0 → 52) |
| catalogue_causes_graspability_adjusting_extent | - | 0.07 [-0.00, 0.15] (mixed → grasp) | - | 0.07 [-0.02, 0.16] (mixed → grasp) | - | 0.07 [-0.02, 0.16] (mixed → grasp) | - | 0.07 [-0.02, 0.16] (mixed → grasp) | - | 0.07 [-0.02, 0.16] (mixed → grasp) | - | 0.07 [-0.03, 0.16] (mixed → grasp) | - | 0.00 [-0.10, 0.09] (mixed → grasp) |
| catalogue_causes_graspability_adjusting_small_object_count | - | 0.13 [0.05, 0.20] (mixed → grasp) | - | 0.09 [-0.00, 0.18] (mixed → grasp) | - | 0.12 [0.02, 0.20] (mixed → grasp) | - | 0.09 [-0.00, 0.18] (mixed → grasp) | - | - | - | 0.06 [-0.04, 0.15] (mixed → grasp) | - | 0.00 [-0.10, 0.10] (mixed → ycb-video) |
| catalogue_causes_occluded_object_0 | - | 0.04 [-0.05, 0.13] (ycb-video → grasp) | - | 0.04 [-0.05, 0.13] (ycb-video → grasp) | - | - | - | 0.05 [-0.04, 0.14] (mixed → ycb-video) | - | - | - | - | - | 0.00 [-0.02, 0.03] (ycb-video → mixed) |
| occluded_object_count_causes_blocked_object_0 | 0.31 | 0.02 [-0.26, 0.23] (2 → 16) | 0.31 | 0.02 [-0.26, 0.23] (2 → 16) | - | - | -0.49 | -0.04 [-0.31, 0.16] (2 → 16) | - | - | - | - | -1.00 | -0.22 [-0.39, -0.01] (0 → 20) |
| size_causes_blocked_object_0 | - | 0.02 [0.00, 0.03] (large → small) | - | 0.02 [0.00, 0.03] (large → small) | - | - | - | 0.08 [0.02, 0.14] (small → large) | - | - | - | - | - | 0.02 [-0.00, 0.03] (large → small) |

## What adjusting for changes

The same count question under each set of confounders it was asked with, read off the relational circuit. *n* is how many training scenes hold that value of the cause; † marks a region below the support threshold.

### small_object_count

| cause region | n | naive | adjusted for how far the clutter is spread out | adjusted for the number of objects | adjusted for how far the clutter is spread out and the number of objects |
|---|---|---|---|---|---|
| 0 | 148 | 0.446 | 0.446 | 0.441 | 0.433 |
| 1 | 35 | 0.429 | 0.429 | 0.432 | 0.432 |
| 2 | 48 | 0.333 | 0.333 | 0.389 | 0.389 |
| 3 | 55 | 0.491 | 0.491 | 0.452 | 0.441 |
| 4 | 66 | 0.333 | 0.333 | 0.336 | 0.337 |
| 5 | 66 | 0.409 | 0.409 | 0.403 | 0.404 |
| 6 | 67 | 0.313 | 0.313 | 0.330 | 0.325 |
| 7 | 78 | 0.359 | 0.359 | 0.354 | 0.357 |
| 8 | 52 | 0.327 | 0.327 | 0.313 | 0.302 |
| 9 | 42 | 0.381 | 0.381 | 0.381 | 0.372 |
| 10 | 29 | 0.414 | 0.414 | 0.417 | 0.416 |
| 11 | 25 | 0.240 | 0.240 | 0.266 | 0.268 |
| 12 | 15 | 0.200 | 0.200 | 0.108 | 0.109 |
| 13 | 12 | 0.333 | 0.333 | 0.291 | 0.293 |
| 14 | 14 | 0.357 | 0.357 | 0.512 | 0.514 |
| 15 † | 4 | 0.500 | 0.500 | 0.498 | 0.496 |
| 16 † | 3 | 0.333 | 0.333 | 0.572 | 0.015 |
| 17 † | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 20 † | 2 | 0.000 | 0.000 | 0.000 | 0.000 |

### occluded_object_count

| cause region | n | naive | adjusted for how far the clutter is spread out | adjusted for the number of objects | adjusted for how far the clutter is spread out and the number of objects |
|---|---|---|---|---|---|
| 0 † | 3 | 0.333 | 0.333 | 0.342 | 0.268 |
| 1 † | 4 | 0.250 | 0.250 | 0.143 | 0.124 |
| 2 | 12 | 0.167 | 0.182 | 0.180 | 0.213 |
| 3 | 17 | 0.353 | 0.312 | 0.241 | 0.210 |
| 4 | 25 | 0.160 | 0.160 | 0.070 | 0.095 |
| 5 | 25 | 0.400 | 0.400 | 0.328 | 0.315 |
| 6 | 42 | 0.190 | 0.190 | 0.197 | 0.197 |
| 7 | 71 | 0.324 | 0.324 | 0.285 | 0.282 |
| 8 | 72 | 0.431 | 0.431 | 0.390 | 0.372 |
| 9 | 87 | 0.345 | 0.345 | 0.297 | 0.285 |
| 10 | 95 | 0.453 | 0.453 | 0.435 | 0.436 |
| 11 | 93 | 0.409 | 0.409 | 0.404 | 0.404 |
| 12 | 61 | 0.426 | 0.426 | 0.468 | 0.453 |
| 13 | 51 | 0.333 | 0.333 | 0.369 | 0.373 |
| 14 | 43 | 0.372 | 0.372 | 0.408 | 0.394 |
| 15 | 25 | 0.400 | 0.400 | 0.414 | 0.414 |
| 16 | 21 | 0.524 | 0.524 | 0.379 | 0.380 |
| 17 † | 8 | 0.625 | 0.625 | 0.764 | 0.761 |
| 18 † | 6 | 0.667 | 0.667 | 0.503 | 0.504 |
| 19 † | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| 20 † | 1 | 1.000 | 1.000 | 1.000 | 1.000 |

### clear_viewpoint_count

| cause region | n | naive | adjusted for how far the clutter is spread out | adjusted for the number of objects | adjusted for how far the clutter is spread out and the number of objects |
|---|---|---|---|---|---|
| 0 | 181 | 0.481 | 0.481 | 0.492 | 0.487 |
| 1 | 11 | 0.545 | 0.545 | 0.526 | 0.521 |
| 2 | 18 | 0.556 | 0.588 | 0.501 | 0.494 |
| 3 † | 8 | 0.625 | 0.500 | 0.744 | 0.593 |
| 4 † | 3 | 0.667 | 0.667 | 0.756 | 0.764 |
| 5 † | 7 | 0.429 | 0.429 | 0.181 | 0.131 |
| 6 † | 5 | 0.600 | 0.600 | 0.532 | 0.672 |
| 7 | 10 | 0.500 | 0.500 | 0.433 | 0.370 |
| 8 † | 5 | 0.400 | 0.400 | 0.441 | 0.317 |
| 9 † | 6 | 0.500 | 0.500 | 0.583 | 0.612 |
| 10 | 10 | 0.400 | 0.400 | 0.383 | 0.298 |
| 11 | 10 | 0.100 | 0.100 | 0.077 | 0.138 |
| 12 † | 5 | 0.800 | 1.000 | 0.842 | 1.000 |
| 13 † | 9 | 0.222 | 0.222 | 0.263 | 0.191 |
| 14 † | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| 15 | 10 | 0.500 | 0.444 | 0.603 | 0.525 |
| 16 † | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 17 † | 4 | 0.500 | 0.667 | 0.510 | 0.606 |
| 18 | 11 | 0.273 | 0.200 | 0.316 | 0.186 |
| 19 † | 5 | 0.800 | 0.667 | 0.815 | 0.259 |
| 20 † | 3 | 0.333 | 0.333 | 0.327 | 0.323 |
| 21 † | 7 | 0.714 | 0.714 | 0.709 | 0.544 |
| 22 † | 5 | 0.600 | 0.600 | 0.681 | 0.787 |
| 23 † | 9 | 0.778 | 0.750 | 0.790 | 0.816 |
| 24 † | 3 | 0.667 | 0.667 | 0.670 | 0.662 |
| 25 † | 7 | 0.571 | 0.571 | 0.383 | 0.372 |
| 26 † | 4 | 0.500 | 0.500 | 0.458 | 0.559 |
| 27 † | 8 | 0.750 | 0.714 | 0.791 | 0.739 |
| 28 † | 4 | 0.500 | 0.500 | 0.500 | 0.567 |
| 29 † | 3 | 0.667 | 0.667 | 0.827 | 0.748 |
| 30 † | 6 | 0.667 | 0.600 | 0.625 | 0.603 |
| 31 † | 3 | 0.333 | 0.333 | 0.324 | 0.415 |
| 32 † | 5 | 0.400 | 0.400 | 0.258 | 0.275 |
| 33 † | 5 | 0.400 | 0.250 | 0.615 | 0.574 |
| 34 † | 9 | 0.111 | 0.111 | 0.149 | 0.164 |
| 35 † | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| 36 † | 6 | 0.500 | 0.400 | 0.490 | 0.214 |
| 37 | 10 | 0.400 | 0.333 | 0.417 | 0.378 |
| 38 † | 6 | 0.500 | 0.500 | 0.635 | 0.743 |
| 39 † | 7 | 0.286 | 0.286 | 0.306 | 0.310 |
| 40 | 11 | 0.273 | 0.250 | 0.334 | 0.290 |
| 41 † | 7 | 0.286 | 0.286 | 0.147 | 0.229 |
| 42 | 10 | 0.500 | 0.500 | 0.401 | 0.186 |
| 43 | 14 | 0.357 | 0.357 | 0.427 | 0.380 |
| 44 | 13 | 0.308 | 0.308 | 0.261 | 0.319 |
| 45 | 15 | 0.400 | 0.385 | 0.366 | 0.313 |
| 46 | 22 | 0.136 | 0.136 | 0.130 | 0.152 |
| 47 | 20 | 0.300 | 0.300 | 0.335 | 0.414 |
| 48 | 29 | 0.172 | 0.172 | 0.175 | 0.203 |
| 49 | 32 | 0.312 | 0.312 | 0.272 | 0.273 |
| 50 | 45 | 0.111 | 0.111 | 0.103 | 0.114 |
| 51 | 36 | 0.139 | 0.139 | 0.128 | 0.128 |
| 52 | 71 | 0.197 | 0.197 | 0.171 | 0.195 |


## Fit and likelihood

What each pipeline cost. *Models fitted* counts the plain model plus one support-deterministic model per distinct cause the questions asked about and the pipeline could fit; *training seconds* and the *nodes*/*edges* of every fitted circuit are summed over them, which for the relational circuit includes the part templates.

| pipeline | models fitted | training seconds | nodes | edges |
|---|---|---|---|---|
| relational circuit | 6 | 187.16 | 37013 | 36995 |
| hybrid circuit | 1 | 41.19 | 8668 | 8664 |
| propositional tree | 5 | 40.75 | 10429 | 10424 |
| unrolled tree | 6 | 495.87 | 364486 | 364480 |
| scalars-only tree | 2 | 0.04 | 526 | 524 |
| regression adjustment | 12 | 7.85 | 0 | 0 |
| neural adjustment | 15 | 8.44 | 0 | 0 |

How well each explains scenes it never saw, on three views of one scene: its own scalars, which every pipeline models; its scalars and counts; and the whole scene, parts included, which only the pipelines that model the parts can score. The relational circuit scores a whole scene as its class circuit over the scalars and counts times each part template over one part given the counts; the unrolled tree scores it as one row. *Held-out coverage* is the share of held-out scenes that lie inside the plain model's support at all, since a tree's leaves span only the value ranges they were fitted on, and a whole scene is covered only if every one of its parts is. The *mean log-likelihood* is over the covered scenes only; the last column restricts it to the scenes every pipeline in the table covers, so the numbers are over the same rows.

### scalars

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|
| relational circuit | 100.0% | 0.06 | 0.10 |
| hybrid circuit | 100.0% | -0.09 | -0.06 |
| propositional tree | 100.0% | 0.06 | 0.10 |
| unrolled tree | 100.0% | -0.05 | -0.02 |
| scalars-only tree | 95.8% | 0.20 | 0.20 |

### scalars and counts

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|
| relational circuit | 91.1% | -12.79 | -12.76 |
| hybrid circuit | 94.2% | -13.34 | -13.26 |
| propositional tree | 91.1% | -12.79 | -12.76 |
| unrolled tree | 95.8% | -13.26 | -13.09 |

### whole scene

| pipeline | held-out coverage | mean log-likelihood (covered) | mean log-likelihood (covered by all) |
|---|---|---|---|
| relational circuit | 87.4% | -19.18 | -17.22 |
| hybrid circuit | 84.3% | 78.74 | 83.33 |
| unrolled tree | 67.0% | 24.85 | 26.91 |

## Seconds per question

Wall-clock time from asking to the answer or the refusal. The *first ask* of a cause includes fitting that cause's own support-deterministic model; *asked again* repeats the question with every model fitted, so only grounding (for the relational circuit), verification and backdoor adjustment remain. A refusal is fast when it is a schema check; a relational answer draws Monte-Carlo samples for every count the query leaves open and grounds one part template per sampled value, which is where its time goes.

| question | relational circuit, first ask | relational circuit, asked again | hybrid circuit, first ask | hybrid circuit, asked again | propositional tree, first ask | propositional tree, asked again | unrolled tree, first ask | unrolled tree, asked again | scalars-only tree, first ask | scalars-only tree, asked again | regression adjustment, first ask | regression adjustment, asked again | neural adjustment, first ask | neural adjustment, asked again |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small_object_count_causes_graspability_adjusting_extent | 162.10 | 121.23 | 43.22 | 7.67 | 10.47 | 2.19 | 102.86 | 5.93 | 0.00 | 0.00 | 0.03 | - | 0.23 | - |
| small_object_count_causes_graspability_adjusting_object_count | 167.47 | 160.43 | 20.77 | 22.25 | 18.30 | 18.27 | 22.75 | 25.09 | 0.01 | 0.00 | 0.03 | - | 0.27 | - |
| small_object_count_causes_graspability_adjusting_extent_and_object_count | 324.67 | 305.65 | 100.84 | 102.42 | 100.28 | 100.83 | 130.60 | 132.94 | 0.00 | 0.00 | 0.04 | - | 0.20 | - |
| occluded_object_count_causes_graspability_adjusting_extent | 167.77 | 129.99 | 49.40 | 8.32 | 14.31 | 4.92 | 112.81 | 11.60 | 0.00 | 0.00 | 0.03 | - | 0.31 | - |
| occluded_object_count_causes_graspability_adjusting_object_count | 183.23 | 170.57 | 23.91 | 23.48 | 20.70 | 20.39 | 26.28 | 27.53 | 0.00 | 0.01 | 0.03 | - | 0.26 | - |
| occluded_object_count_causes_graspability_adjusting_extent_and_object_count | 448.56 | 440.44 | 97.94 | 100.54 | 151.48 | 150.49 | 97.81 | 101.35 | 0.00 | 0.00 | 0.04 | - | 0.17 | - |
| clear_viewpoint_count_causes_graspability_adjusting_extent | 286.74 | 238.43 | 210.99 | 124.85 | 126.43 | 119.96 | 368.07 | 133.67 | 0.00 | 0.00 | 0.05 | - | 0.24 | - |
| clear_viewpoint_count_causes_graspability_adjusting_object_count | 253.46 | 242.33 | 101.42 | 101.46 | 96.18 | 95.06 | 110.05 | 109.50 | 0.01 | 0.00 | 0.06 | - | 0.27 | - |
| clear_viewpoint_count_causes_graspability_adjusting_extent_and_object_count | 1751.47 | 1727.74 | 1675.29 | 1676.96 | 1236.53 | 1209.40 | 1113.49 | 1122.68 | 0.00 | 0.00 | 0.06 | - | 0.24 | - |
| catalogue_causes_graspability_adjusting_extent | 204.66 | 175.48 | 15.83 | 0.77 | 8.51 | 0.23 | 27.61 | 1.15 | 0.26 | 0.23 | 0.03 | - | 0.35 | - |
| catalogue_causes_graspability_adjusting_small_object_count | 188.89 | 187.34 | 1.76 | 1.76 | 1.24 | 1.20 | 2.21 | 5.14 | 0.00 | 0.00 | 0.03 | - | 0.35 | - |
| catalogue_causes_occluded_object_0 | 215.83 | 215.76 | 249.60 | 220.05 | 0.00 | 0.00 | 1.08 | 1.13 | 0.00 | 0.00 | 0.01 | - | 2.09 | - |
| occluded_object_count_causes_blocked_object_0 | 163.61 | 164.03 | 197.78 | 170.33 | 0.00 | 0.00 | 5.70 | 5.81 | 0.01 | 0.00 | 0.01 | - | 1.76 | - |
| size_causes_blocked_object_0 | 114.26 | 82.78 | 116.23 | 89.79 | 0.01 | 0.00 | 27.73 | 4.33 | 0.00 | 0.00 | 0.01 | - | 1.79 | - |

## Does the order of the parts matter?

Every scene's parts were put in a random order, 20 times over, and each time the pipelines that model the parts were refitted on the same split and asked the questions about parts again; the parts in the order the dataset lists them is the baseline every reordering is measured against. A relational circuit treats the objects as exchangeable, so nothing about it can depend on the order; an unrolled table's column `objects[0]` holds a different object of every scene after each reordering. Per question and pipeline: how many reorderings were answered; over the cause regions every answered ordering distinguishes, the mean standard deviation and the widest range of the adjusted probability; the share of reorderings whose most effective region is not the dataset-order one; and the share whose trend changed sign.

| question | pipeline | reorderings answered | mean sd of adjusted P(effect) | widest range | argmax moved | trend sign flipped |
|---|---|---|---|---|---|---|
| catalogue_causes_occluded_object_0 | relational circuit | 20 of 20 | 0.000 | 0.00 | 0.0% | - |
| occluded_object_count_causes_blocked_object_0 | relational circuit | 20 of 20 | 0.000 | 0.00 | 0.0% | 0.0% |
| size_causes_blocked_object_0 | relational circuit | 20 of 20 | 0.000 | 0.00 | 0.0% | - |
| catalogue_causes_occluded_object_0 | hybrid circuit | 20 of 20 | 0.000 | 0.00 | 0.0% | - |
| occluded_object_count_causes_blocked_object_0 | hybrid circuit | 20 of 20 | 0.000 | 0.00 | 0.0% | 0.0% |
| size_causes_blocked_object_0 | hybrid circuit | 20 of 20 | 0.000 | 0.00 | 0.0% | - |
| catalogue_causes_occluded_object_0 | unrolled tree | 20 of 20 | 0.028 | 0.15 | 95.0% | - |
| occluded_object_count_causes_blocked_object_0 | unrolled tree | 20 of 20 | 0.049 | 0.50 | 75.0% | 0.0% |
| size_causes_blocked_object_0 | unrolled tree | 20 of 20 | 0.019 | 0.09 | 95.0% | - |

The whole-scene likelihood of the same held-out scenes with the parts in the order the dataset lists them, and over the reorderings. The dataset's order is not arbitrary throughout: a scene's frames are numbered by the recording rig, four cameras per pose in a fixed sequence, so which camera took frame *i* is the same in every scene, and a column that addresses a viewpoint by position addresses a real thing. Its objects carry no such order. *Largest drop* is how far below the dataset-order likelihood the worst reordering took each pipeline.

| pipeline | dataset order, coverage / mean log-likelihood | reorderings, coverage / mean log-likelihood (mean ± sd) | largest drop |
|---|---|---|---|
| relational circuit | 87.4% / -19.18 | 0.874 ± 0.000 / -19.18 ± 0.00 | 0.00 |
| hybrid circuit | 84.3% / 78.74 | 0.589 ± 0.021 / -42.60 ± 1.36 | 123.61 |
| unrolled tree | 67.0% / 24.85 | 0.452 ± 0.023 / -91.71 ± 1.78 | 120.39 |

## How much training data it takes

Every pipeline's plain model fitted on a growing share of the scenes and scored on the same held-out fifth, over 3 splits, mean ± standard deviation of the held-out coverage and of the mean log-likelihood over the covered scenes. The relational circuit's templates pool every part of every training scene, where the unrolled tree sees one row per scene. Every object of every training scene, and every frame of it, goes into the templates.

### scalars and counts

| training share | relational circuit, coverage | relational circuit, mean log-likelihood | hybrid circuit, coverage | hybrid circuit, mean log-likelihood | propositional tree, coverage | propositional tree, mean log-likelihood | unrolled tree, coverage | unrolled tree, mean log-likelihood |
|---|---|---|---|---|---|---|---|---|
| 20.0% | 0.445 ± 0.035 | -11.55 ± 0.30 | 0.572 ± 0.043 | -12.24 ± 0.21 | 0.445 ± 0.035 | -11.55 ± 0.30 | 0.565 ± 0.047 | -12.25 ± 0.22 |
| 40.0% | 0.674 ± 0.014 | -12.16 ± 0.10 | 0.806 ± 0.004 | -13.10 ± 0.12 | 0.674 ± 0.014 | -12.16 ± 0.10 | 0.805 ± 0.007 | -13.09 ± 0.12 |
| 60.0% | 0.812 ± 0.015 | -12.64 ± 0.15 | 0.888 ± 0.015 | -13.37 ± 0.13 | 0.812 ± 0.015 | -12.64 ± 0.15 | 0.904 ± 0.015 | -13.31 ± 0.12 |
| 80.0% | 0.873 ± 0.028 | -12.79 ± 0.07 | 0.932 ± 0.009 | -13.50 ± 0.17 | 0.873 ± 0.028 | -12.79 ± 0.07 | 0.941 ± 0.018 | -13.39 ± 0.10 |

### whole scene

| training share | relational circuit, coverage | relational circuit, mean log-likelihood | hybrid circuit, coverage | hybrid circuit, mean log-likelihood | unrolled tree, coverage | unrolled tree, mean log-likelihood |
|---|---|---|---|---|---|---|
| 20.0% | 0.363 ± 0.030 | -11.92 ± 1.34 | 0.419 ± 0.056 | 82.11 ± 9.65 | 0.134 ± 0.065 | 32.78 ± 9.29 |
| 40.0% | 0.597 ± 0.023 | -15.45 ± 0.75 | 0.620 ± 0.036 | 86.44 ± 4.98 | 0.339 ± 0.017 | 34.50 ± 5.43 |
| 60.0% | 0.764 ± 0.031 | -17.53 ± 1.77 | 0.736 ± 0.016 | 84.53 ± 2.37 | 0.513 ± 0.048 | 33.52 ± 6.08 |
| 80.0% | 0.838 ± 0.028 | -19.49 ± 0.28 | 0.803 ± 0.034 | 82.45 ± 4.87 | 0.644 ± 0.045 | 26.61 ± 4.72 |

## Error against known truth

Scenes sampled from a structural causal model over the same domain, whose interventional probabilities are known by construction. The cause is forced to a value in the mechanism and the effect's rate read off 200,000 forced scenes. The number of objects is the model's one confounder, driving both the causes and the effect, and the scenes list their small objects first, so a column that addresses an object by position is systematically misleading. A count's extreme values are rare in the data and rarer still within every stratum of the confounder, so no estimator recovers their interventional probability well; the questions whose cause or effect lives on one object are where the pipelines differ. Every pipeline was fitted on 400 scenes per setting and asked the questions; *mean* and *max absolute error* are over every supported cause region of every answered question, the *support-weighted* error weighs each region by the training rows it holds, *worst ordering* is the mean absolute error under the reordering of the parts the pipeline did worst on, and *rank correlation* is Spearman's between the answered and the true probabilities over a question's regions.

| pipeline | questions answered | mean abs. error | support-weighted abs. error | max abs. error | mean abs. error, worst ordering | rank correlation with truth |
|---|---|---|---|---|---|---|
| relational circuit | 100.0% | 0.068 | 0.026 | 0.444 | 0.068 | 0.58 |
| hybrid circuit | 100.0% | 0.072 | 0.029 | 0.444 | 0.072 | 0.56 |
| propositional tree | 57.1% | 0.087 | 0.058 | 0.444 | 0.087 | 0.40 |
| unrolled tree | 100.0% | 0.083 | 0.064 | 0.444 | 0.083 | 0.26 |
| scalars-only tree | 0.0% | - | - | - | - | - |
| regression adjustment | 57.1% | 0.048 | 0.033 | 0.489 | 0.048 | 0.53 |
| neural adjustment | 100.0% | 0.068 | 0.027 | 0.440 | 0.068 | 0.36 |

Mean absolute error per setting of the model:

| objects per scene | confounding strength | relational circuit | hybrid circuit | propositional tree | unrolled tree | scalars-only tree | regression adjustment | neural adjustment |
|---|---|---|---|---|---|---|---|---|
| 5 | 0.6 | 0.053 | 0.073 | 0.060 | 0.065 | - | 0.083 | 0.069 |
| 10 | 0.0 | 0.062 | 0.067 | 0.082 | 0.081 | - | 0.018 | 0.065 |
| 10 | 0.3 | 0.063 | 0.070 | 0.082 | 0.085 | - | 0.031 | 0.038 |
| 10 | 0.6 | 0.071 | 0.069 | 0.090 | 0.083 | - | 0.043 | 0.057 |
| 20 | 0.6 | 0.084 | 0.080 | 0.108 | 0.093 | - | 0.066 | 0.099 |

Mean absolute error per question, over every setting:

| question | relational circuit | hybrid circuit | propositional tree | unrolled tree | scalars-only tree | regression adjustment | neural adjustment |
|---|---|---|---|---|---|---|---|
| small_object_count_causes_graspability_adjusting_object_count | 0.125 | 0.128 | 0.125 | 0.125 | - | 0.078 | 0.131 |
| occluded_object_count_causes_graspability_adjusting_object_count | 0.052 | 0.049 | 0.052 | 0.053 | - | 0.030 | 0.048 |
| clear_viewpoint_count_causes_graspability_adjusting_object_count | 0.075 | 0.092 | 0.075 | 0.081 | - | 0.024 | 0.040 |
| catalogue_causes_graspability_adjusting_object_count | 0.020 | 0.032 | 0.020 | 0.018 | - | 0.021 | 0.046 |
| catalogue_causes_occluded_object_0 | 0.033 | 0.033 | - | 0.133 | - | - | 0.026 |
| occluded_object_count_causes_blocked_object_0 | 0.009 | 0.009 | - | 0.054 | - | - | 0.045 |
| size_causes_blocked_object_0 | 0.007 | 0.007 | - | 0.044 | - | - | 0.019 |
## How many grounding samples it takes

Inference on a grounded circuit is exact; grounding itself draws Monte-Carlo samples for every count the query leaves open and mixes one copy of the part templates per sampled value, so marginalising the open counts is a consistent estimate, not an exact sum. The relational circuit was fitted once and asked the same two questions with grounding drawing more and more samples; *deviation* is the largest difference, over the cause regions, from the answer at 32,000 samples, and *settled from* is the smallest number of samples from which every larger one stays within 0.01 of it.

### small_object_count_causes_graspability_adjusting_object_count

Settled from 8,000 samples.

| samples | answered | deviation from reference | seconds |
|---|---|---|---|
| 50 | answered | 0.171 | 164.6 |
| 200 | answered | 0.078 | 84.5 |
| 1,000 | answered | 0.041 | 120.4 |
| 2,000 | answered | 0.024 | 167.0 |
| 8,000 | answered | 0.006 | 230.6 |
| 32,000 | answered | 0.000 | 247.3 |

### occluded_object_count_causes_blocked_object_0

Settled from 50 samples.

| samples | answered | deviation from reference | seconds |
|---|---|---|---|
| 50 | answered | 0.006 | 155.1 |
| 200 | answered | 0.005 | 61.3 |
| 1,000 | answered | 0.005 | 121.2 |
| 2,000 | answered | 0.003 | 167.9 |
| 8,000 | answered | 0.000 | 247.9 |
| 32,000 | answered | 0.000 | 260.1 |


## Cost against the number of objects

The pipelines that model the parts, fitted on synthetic scenes of growing size (400 scenes each) and asked one question about a part. The relational circuit's part templates pool every part of every scene into one circuit, so their size follows the number of distinct part attributes, not the number of parts; the unrolled table carries one block of columns per position, so its tree grows with the widest scene. *First ask* includes fitting the cause-specific model, *asked again* is grounding and adjustment alone.

| parts per scene | relational circuit, fit seconds | relational circuit, nodes | relational circuit, first ask | relational circuit, asked again | hybrid circuit, fit seconds | hybrid circuit, nodes | hybrid circuit, first ask | hybrid circuit, asked again | unrolled tree, fit seconds | unrolled tree, nodes | unrolled tree, first ask | unrolled tree, asked again |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 15.9 | 2330 | 13.1 | 5.2 | 10.3 | 1260 | 13.2 | 5.2 | 5.9 | 7422 | 3.9 | 0.5 |
| 10 | 19.5 | 3553 | 25.9 | 16.1 | 12.2 | 1622 | 26.0 | 15.8 | 9.5 | 16953 | 8.0 | 1.3 |
| 20 | 26.9 | 5250 | 60.5 | 47.1 | 16.2 | 2270 | 59.6 | 47.8 | 26.9 | 36905 | 24.9 | 3.6 |
| 50 | 44.2 | 10016 | 120.0 | 97.4 | 27.8 | 4230 | 118.3 | 99.0 | 276.0 | 116486 | 268.0 | 10.9 |

## What the results show

- The relational circuit answered 14 of 14 questions.
- The hybrid circuit answered 14 of 14 questions.
- The propositional tree answered 11 of 14 questions, refusing `catalogue_causes_occluded_object_0` because the fitted table has no column for the queried variables; `occluded_object_count_causes_blocked_object_0` because the fitted table has no column for the queried variables; `size_causes_blocked_object_0` because the fitted table has no column for the queried variables.
- The unrolled tree answered 14 of 14 questions.
- The scalars-only tree answered 1 of 14 questions, refusing `small_object_count_causes_graspability_adjusting_extent` because the fitted table has no column for the queried variables; `small_object_count_causes_graspability_adjusting_object_count` because the fitted table has no column for the queried variables; `small_object_count_causes_graspability_adjusting_extent_and_object_count` because the fitted table has no column for the queried variables; `occluded_object_count_causes_graspability_adjusting_extent` because the fitted table has no column for the queried variables; `occluded_object_count_causes_graspability_adjusting_object_count` because the fitted table has no column for the queried variables; `occluded_object_count_causes_graspability_adjusting_extent_and_object_count` because the fitted table has no column for the queried variables; `clear_viewpoint_count_causes_graspability_adjusting_extent` because the fitted table has no column for the queried variables; `clear_viewpoint_count_causes_graspability_adjusting_object_count` because the fitted table has no column for the queried variables; `clear_viewpoint_count_causes_graspability_adjusting_extent_and_object_count` because the fitted table has no column for the queried variables; `catalogue_causes_graspability_adjusting_small_object_count` because the fitted table has no column for the queried variables; `catalogue_causes_occluded_object_0` because the fitted table has no column for the queried variables; `occluded_object_count_causes_blocked_object_0` because the fitted table has no column for the queried variables; `size_causes_blocked_object_0` because the fitted table has no column for the queried variables.
- The regression adjustment answered 11 of 14 questions, refusing `catalogue_causes_occluded_object_0` because the fitted table has no column for the queried variables; `occluded_object_count_causes_blocked_object_0` because the fitted table has no column for the queried variables; `size_causes_blocked_object_0` because the fitted table has no column for the queried variables.
- The neural adjustment answered 14 of 14 questions.
- On `small_object_count_causes_graspability_adjusting_extent`, the pipelines disagree on the most effective setting: the relational circuit, the hybrid circuit, the propositional tree and the unrolled tree say 3 small objects (0.49, 0.49, 0.49, 0.49); the regression adjustment says 0 small objects (0.42); the neural adjustment says 13 small objects (0.39).
- On `small_object_count_causes_graspability_adjusting_object_count`, the pipelines disagree on the most effective setting: the relational circuit, the hybrid circuit, the propositional tree and the unrolled tree say 14 small objects (0.51, 0.51, 0.51, 0.51); the regression adjustment says 0 small objects (0.42); the neural adjustment says 11 small objects (0.36).
- On `small_object_count_causes_graspability_adjusting_extent_and_object_count`, the pipelines disagree on the most effective setting: the relational circuit, the hybrid circuit, the propositional tree, the unrolled tree and the neural adjustment say 14 small objects (0.51, 0.51, 0.51, 0.51, 0.42); the regression adjustment says 0 small objects (0.38).
- On `occluded_object_count_causes_graspability_adjusting_extent`, the pipelines disagree on the most effective setting: the relational circuit, the hybrid circuit, the propositional tree, the unrolled tree and the regression adjustment say 16 occluded objects (0.52, 0.52, 0.52, 0.52, 0.47); the neural adjustment says 2 occluded objects (0.47).
- On `occluded_object_count_causes_graspability_adjusting_object_count`, the pipelines disagree on the most effective setting: the relational circuit and the propositional tree say 12 occluded objects (0.47, 0.47); the hybrid circuit, the unrolled tree, the regression adjustment and the neural adjustment say 16 occluded objects (0.61, 0.55, 0.71, 0.68).
- On `occluded_object_count_causes_graspability_adjusting_extent_and_object_count`, the pipelines disagree on the most effective setting: the relational circuit and the propositional tree say 12 occluded objects (0.45, 0.46); the hybrid circuit, the unrolled tree, the regression adjustment and the neural adjustment say 16 occluded objects (0.61, 0.55, 0.70, 0.68).
- On `clear_viewpoint_count_causes_graspability_adjusting_extent`, the pipelines disagree on the most effective setting: the relational circuit, the hybrid circuit, the propositional tree and the unrolled tree say 2 clear viewpoints (0.59, 0.59, 0.59, 0.59); the regression adjustment says 0 clear viewpoints (0.53); the neural adjustment says 18 clear viewpoints (0.38).
- On `clear_viewpoint_count_causes_graspability_adjusting_object_count`, the pipelines disagree on the most effective setting: the relational circuit, the hybrid circuit, the propositional tree and the unrolled tree say 15 clear viewpoints (0.60, 0.60, 0.60, 0.60); the regression adjustment says 0 clear viewpoints (0.55); the neural adjustment says 37 clear viewpoints (0.38).
- On `clear_viewpoint_count_causes_graspability_adjusting_extent_and_object_count`, the pipelines disagree on the most effective setting: the relational circuit says 15 clear viewpoints (0.52); the hybrid circuit, the propositional tree and the unrolled tree say 1 clear viewpoints (0.60, 0.53, 0.63); the regression adjustment says 0 clear viewpoints (0.55); the neural adjustment says 18 clear viewpoints (0.38).
- On `catalogue_causes_graspability_adjusting_extent`, every pipeline that answered finds the grasp catalogue the most effective setting (adjusted probabilities: relational circuit 0.41, hybrid circuit 0.41, propositional tree 0.41, unrolled tree 0.41, scalars-only tree 0.41, regression adjustment 0.41, neural adjustment 0.38).
- On `catalogue_causes_graspability_adjusting_small_object_count`, the pipelines disagree on the most effective setting: the relational circuit, the hybrid circuit, the propositional tree, the unrolled tree and the regression adjustment say the grasp catalogue (0.39, 0.40, 0.40, 0.40, 0.40); the neural adjustment says the ycb-video catalogue (0.38).
- On `catalogue_causes_occluded_object_0`, the pipelines disagree on the most effective setting: the relational circuit and the hybrid circuit say the grasp catalogue (0.36, 0.36); the unrolled tree says the ycb-video catalogue (0.29); the neural adjustment says the mixed catalogue (0.35).
- On `occluded_object_count_causes_blocked_object_0`, the pipelines disagree on the most effective setting: the relational circuit and the hybrid circuit say 10 occluded objects (0.13, 0.13); the unrolled tree says 3 occluded objects (0.18); the neural adjustment says 0 occluded objects (0.25).
- On `size_causes_blocked_object_0`, the pipelines disagree on the most effective setting: the relational circuit, the hybrid circuit and the neural adjustment say object 0 being small (0.12, 0.12, 0.17); the unrolled tree says object 0 being large (0.15).
- On the scalars, the scalars-only tree assigns the highest mean log-likelihood (0.20, against relational circuit 0.10, hybrid circuit -0.06, propositional tree 0.10, unrolled tree -0.02) to the held-out scenes every pipeline covers; coverage: relational circuit 100.0%, hybrid circuit 100.0%, propositional tree 100.0%, unrolled tree 100.0%, scalars-only tree 95.8%.
- On the scalars and counts, the relational circuit assigns the highest mean log-likelihood (-12.76, against hybrid circuit -13.26, propositional tree -12.76, unrolled tree -13.09) to the held-out scenes every pipeline covers; coverage: relational circuit 91.1%, hybrid circuit 94.2%, propositional tree 91.1%, unrolled tree 95.8%.
- On the whole scene, the hybrid circuit assigns the highest mean log-likelihood (83.33, against relational circuit -17.22, unrolled tree 26.91) to the held-out scenes every pipeline covers; coverage: relational circuit 87.4%, hybrid circuit 84.3%, unrolled tree 67.0%.
- The relational circuit takes 311.59 seconds per answered question on average once its models are fitted.
- The hybrid circuit takes 189.33 seconds per answered question on average once its models are fitted.
- The propositional tree takes 156.63 seconds per answered question on average once its models are fitted.
- The unrolled tree takes 120.56 seconds per answered question on average once its models are fitted.
- The scalars-only tree takes 0.23 seconds per answered question on average once its models are fitted.

## How many small objects cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.446 | [0.37, 0.53] |
| 1 | 35 | 0.046 | 0.429 | 0.429 | [0.28, 0.59] |
| 2 | 48 | 0.063 | 0.333 | 0.333 | [0.22, 0.47] |
| 3 | 55 | 0.072 | 0.491 | 0.491 | [0.36, 0.62] |
| 4 | 66 | 0.087 | 0.333 | 0.333 | [0.23, 0.45] |
| 5 | 66 | 0.087 | 0.409 | 0.409 | [0.30, 0.53] |
| 6 | 67 | 0.088 | 0.313 | 0.313 | [0.22, 0.43] |
| 7 | 78 | 0.102 | 0.359 | 0.359 | [0.26, 0.47] |
| 8 | 52 | 0.068 | 0.327 | 0.327 | [0.22, 0.46] |
| 9 | 42 | 0.055 | 0.381 | 0.381 | [0.25, 0.53] |
| 10 | 29 | 0.038 | 0.414 | 0.414 | [0.26, 0.59] |
| 11 | 25 | 0.033 | 0.240 | 0.240 | [0.11, 0.43] |
| 12 | 15 | 0.020 | 0.200 | 0.200 | [0.07, 0.45] |
| 13 | 12 | 0.016 | 0.333 | 0.333 | [0.14, 0.61] |
| 14 | 14 | 0.018 | 0.357 | 0.357 | [0.16, 0.61] |
| 15 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 16 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 17 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 20 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### hybrid circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.446 | [0.37, 0.53] |
| 1 | 35 | 0.046 | 0.429 | 0.429 | [0.28, 0.59] |
| 2 | 48 | 0.063 | 0.333 | 0.333 | [0.22, 0.47] |
| 3 | 55 | 0.072 | 0.491 | 0.491 | [0.36, 0.62] |
| 4 | 66 | 0.087 | 0.333 | 0.333 | [0.23, 0.45] |
| 5 | 66 | 0.087 | 0.409 | 0.409 | [0.30, 0.53] |
| 6 | 67 | 0.088 | 0.313 | 0.313 | [0.22, 0.43] |
| 7 | 78 | 0.102 | 0.359 | 0.359 | [0.26, 0.47] |
| 8 | 52 | 0.068 | 0.327 | 0.327 | [0.22, 0.46] |
| 9 | 42 | 0.055 | 0.381 | 0.381 | [0.25, 0.53] |
| 10 | 29 | 0.038 | 0.414 | 0.414 | [0.26, 0.59] |
| 11 | 25 | 0.033 | 0.240 | 0.240 | [0.11, 0.43] |
| 12 | 15 | 0.020 | 0.200 | 0.200 | [0.07, 0.45] |
| 13 | 12 | 0.016 | 0.333 | 0.333 | [0.14, 0.61] |
| 14 | 14 | 0.018 | 0.357 | 0.357 | [0.16, 0.61] |
| 15 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 16 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 17 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 20 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.446 | [0.37, 0.53] |
| 1 | 35 | 0.046 | 0.429 | 0.429 | [0.28, 0.59] |
| 2 | 48 | 0.063 | 0.333 | 0.333 | [0.22, 0.47] |
| 3 | 55 | 0.072 | 0.491 | 0.491 | [0.36, 0.62] |
| 4 | 66 | 0.087 | 0.333 | 0.333 | [0.23, 0.45] |
| 5 | 66 | 0.087 | 0.409 | 0.409 | [0.30, 0.53] |
| 6 | 67 | 0.088 | 0.313 | 0.313 | [0.22, 0.43] |
| 7 | 78 | 0.102 | 0.359 | 0.359 | [0.26, 0.47] |
| 8 | 52 | 0.068 | 0.327 | 0.327 | [0.22, 0.46] |
| 9 | 42 | 0.055 | 0.381 | 0.381 | [0.25, 0.53] |
| 10 | 29 | 0.038 | 0.414 | 0.414 | [0.26, 0.59] |
| 11 | 25 | 0.033 | 0.240 | 0.240 | [0.11, 0.43] |
| 12 | 15 | 0.020 | 0.200 | 0.200 | [0.07, 0.45] |
| 13 | 12 | 0.016 | 0.333 | 0.333 | [0.14, 0.61] |
| 14 | 14 | 0.018 | 0.357 | 0.357 | [0.16, 0.61] |
| 15 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 16 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 17 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 20 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.446 | [0.37, 0.53] |
| 1 | 35 | 0.046 | 0.429 | 0.429 | [0.28, 0.59] |
| 2 | 48 | 0.063 | 0.333 | 0.333 | [0.22, 0.47] |
| 3 | 55 | 0.072 | 0.491 | 0.491 | [0.36, 0.62] |
| 4 | 66 | 0.087 | 0.333 | 0.333 | [0.23, 0.45] |
| 5 | 66 | 0.087 | 0.409 | 0.409 | [0.30, 0.53] |
| 6 | 67 | 0.088 | 0.313 | 0.313 | [0.22, 0.43] |
| 7 | 78 | 0.102 | 0.359 | 0.359 | [0.26, 0.47] |
| 8 | 52 | 0.068 | 0.327 | 0.327 | [0.22, 0.46] |
| 9 | 42 | 0.055 | 0.381 | 0.381 | [0.25, 0.53] |
| 10 | 29 | 0.038 | 0.414 | 0.414 | [0.26, 0.59] |
| 11 | 25 | 0.033 | 0.240 | 0.240 | [0.11, 0.43] |
| 12 | 15 | 0.020 | 0.200 | 0.200 | [0.07, 0.45] |
| 13 | 12 | 0.016 | 0.333 | 0.333 | [0.14, 0.61] |
| 14 | 14 | 0.018 | 0.357 | 0.357 | [0.16, 0.61] |
| 15 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 16 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 17 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 20 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.419 | [0.34, 0.50] |
| 1 | 35 | 0.046 | 0.429 | 0.411 | [0.26, 0.57] |
| 2 | 48 | 0.063 | 0.333 | 0.402 | [0.28, 0.54] |
| 3 | 55 | 0.072 | 0.491 | 0.394 | [0.28, 0.53] |
| 4 | 66 | 0.087 | 0.333 | 0.386 | [0.28, 0.51] |
| 5 | 66 | 0.087 | 0.409 | 0.378 | [0.27, 0.50] |
| 6 | 67 | 0.088 | 0.313 | 0.369 | [0.26, 0.49] |
| 7 | 78 | 0.102 | 0.359 | 0.361 | [0.26, 0.47] |
| 8 | 52 | 0.068 | 0.327 | 0.353 | [0.24, 0.49] |
| 9 | 42 | 0.055 | 0.381 | 0.345 | [0.22, 0.50] |
| 10 | 29 | 0.038 | 0.414 | 0.337 | [0.19, 0.52] |
| 11 | 25 | 0.033 | 0.240 | 0.330 | [0.18, 0.53] |
| 12 | 15 | 0.020 | 0.200 | 0.322 | [0.14, 0.57] |
| 13 | 12 | 0.016 | 0.333 | 0.314 | [0.13, 0.59] |
| 14 | 14 | 0.018 | 0.357 | 0.307 | [0.13, 0.57] |
| 15 † | 4 | 0.005 | 0.500 | 0.300 | [0.06, 0.73] |
| 16 † | 3 | 0.004 | 0.333 | 0.292 | [0.05, 0.77] |
| 17 † | 2 | 0.003 | 0.000 | 0.285 | [0.03, 0.82] |
| 20 † | 2 | 0.003 | 0.000 | 0.264 | [0.03, 0.81] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.42).

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

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.441 | [0.36, 0.52] |
| 1 | 35 | 0.046 | 0.429 | 0.432 | [0.28, 0.59] |
| 2 | 48 | 0.063 | 0.333 | 0.389 | [0.26, 0.53] |
| 3 | 55 | 0.072 | 0.491 | 0.452 | [0.33, 0.58] |
| 4 | 66 | 0.087 | 0.333 | 0.336 | [0.23, 0.46] |
| 5 | 66 | 0.087 | 0.409 | 0.403 | [0.29, 0.52] |
| 6 | 67 | 0.088 | 0.313 | 0.330 | [0.23, 0.45] |
| 7 | 78 | 0.102 | 0.359 | 0.354 | [0.26, 0.47] |
| 8 | 52 | 0.068 | 0.327 | 0.313 | [0.20, 0.45] |
| 9 | 42 | 0.055 | 0.381 | 0.381 | [0.25, 0.53] |
| 10 | 29 | 0.038 | 0.414 | 0.417 | [0.26, 0.60] |
| 11 | 25 | 0.033 | 0.240 | 0.266 | [0.13, 0.46] |
| 12 | 15 | 0.020 | 0.200 | 0.108 | [0.03, 0.35] |
| 13 | 12 | 0.016 | 0.333 | 0.291 | [0.11, 0.57] |
| 14 | 14 | 0.018 | 0.357 | 0.512 | [0.28, 0.74] |
| 15 † | 4 | 0.005 | 0.500 | 0.498 | [0.15, 0.85] |
| 16 † | 3 | 0.004 | 0.333 | 0.572 | [0.16, 0.90] |
| 17 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 20 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.44).

### hybrid circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.446 | [0.37, 0.53] |
| 1 | 35 | 0.046 | 0.429 | 0.428 | [0.28, 0.59] |
| 2 | 48 | 0.063 | 0.333 | 0.311 | [0.20, 0.45] |
| 3 | 55 | 0.072 | 0.491 | 0.487 | [0.36, 0.62] |
| 4 | 66 | 0.087 | 0.333 | 0.318 | [0.22, 0.44] |
| 5 | 66 | 0.087 | 0.409 | 0.406 | [0.30, 0.53] |
| 6 | 67 | 0.088 | 0.313 | 0.313 | [0.21, 0.43] |
| 7 | 78 | 0.102 | 0.359 | 0.354 | [0.26, 0.46] |
| 8 | 52 | 0.068 | 0.327 | 0.304 | [0.20, 0.44] |
| 9 | 42 | 0.055 | 0.381 | 0.399 | [0.27, 0.55] |
| 10 | 29 | 0.038 | 0.414 | 0.403 | [0.25, 0.58] |
| 11 | 25 | 0.033 | 0.240 | 0.263 | [0.13, 0.46] |
| 12 | 15 | 0.020 | 0.200 | 0.106 | [0.03, 0.35] |
| 13 | 12 | 0.016 | 0.333 | 0.287 | [0.11, 0.57] |
| 14 | 14 | 0.018 | 0.357 | 0.506 | [0.27, 0.74] |
| 15 † | 4 | 0.005 | 0.500 | 0.494 | [0.15, 0.85] |
| 16 † | 3 | 0.004 | 0.333 | 0.563 | [0.15, 0.90] |
| 17 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 20 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.441 | [0.36, 0.52] |
| 1 | 35 | 0.046 | 0.429 | 0.419 | [0.27, 0.58] |
| 2 | 48 | 0.063 | 0.333 | 0.365 | [0.24, 0.51] |
| 3 | 55 | 0.072 | 0.491 | 0.457 | [0.33, 0.59] |
| 4 | 66 | 0.087 | 0.333 | 0.319 | [0.22, 0.44] |
| 5 | 66 | 0.087 | 0.409 | 0.413 | [0.30, 0.53] |
| 6 | 67 | 0.088 | 0.313 | 0.314 | [0.22, 0.43] |
| 7 | 78 | 0.102 | 0.359 | 0.358 | [0.26, 0.47] |
| 8 | 52 | 0.068 | 0.327 | 0.316 | [0.21, 0.45] |
| 9 | 42 | 0.055 | 0.381 | 0.385 | [0.25, 0.54] |
| 10 | 29 | 0.038 | 0.414 | 0.431 | [0.27, 0.61] |
| 11 | 25 | 0.033 | 0.240 | 0.263 | [0.13, 0.46] |
| 12 | 15 | 0.020 | 0.200 | 0.106 | [0.03, 0.35] |
| 13 | 12 | 0.016 | 0.333 | 0.287 | [0.11, 0.57] |
| 14 | 14 | 0.018 | 0.357 | 0.506 | [0.27, 0.74] |
| 15 † | 4 | 0.005 | 0.500 | 0.494 | [0.15, 0.85] |
| 16 † | 3 | 0.004 | 0.333 | 0.563 | [0.15, 0.90] |
| 17 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 20 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.44).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.446 | [0.37, 0.53] |
| 1 | 35 | 0.046 | 0.429 | 0.425 | [0.28, 0.59] |
| 2 | 48 | 0.063 | 0.333 | 0.318 | [0.20, 0.46] |
| 3 | 55 | 0.072 | 0.491 | 0.494 | [0.37, 0.62] |
| 4 | 66 | 0.087 | 0.333 | 0.327 | [0.23, 0.45] |
| 5 | 66 | 0.087 | 0.409 | 0.411 | [0.30, 0.53] |
| 6 | 67 | 0.088 | 0.313 | 0.313 | [0.21, 0.43] |
| 7 | 78 | 0.102 | 0.359 | 0.358 | [0.26, 0.47] |
| 8 | 52 | 0.068 | 0.327 | 0.311 | [0.20, 0.45] |
| 9 | 42 | 0.055 | 0.381 | 0.384 | [0.25, 0.54] |
| 10 | 29 | 0.038 | 0.414 | 0.381 | [0.23, 0.56] |
| 11 | 25 | 0.033 | 0.240 | 0.233 | [0.11, 0.43] |
| 12 | 15 | 0.020 | 0.200 | 0.106 | [0.03, 0.35] |
| 13 | 12 | 0.016 | 0.333 | 0.287 | [0.11, 0.57] |
| 14 | 14 | 0.018 | 0.357 | 0.506 | [0.27, 0.74] |
| 15 † | 4 | 0.005 | 0.500 | 0.494 | [0.15, 0.85] |
| 16 † | 3 | 0.004 | 0.333 | 0.563 | [0.15, 0.90] |
| 17 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 20 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.418 | [0.34, 0.50] |
| 1 | 35 | 0.046 | 0.429 | 0.410 | [0.26, 0.57] |
| 2 | 48 | 0.063 | 0.333 | 0.402 | [0.28, 0.54] |
| 3 | 55 | 0.072 | 0.491 | 0.394 | [0.28, 0.53] |
| 4 | 66 | 0.087 | 0.333 | 0.386 | [0.28, 0.51] |
| 5 | 66 | 0.087 | 0.409 | 0.377 | [0.27, 0.50] |
| 6 | 67 | 0.088 | 0.313 | 0.369 | [0.26, 0.49] |
| 7 | 78 | 0.102 | 0.359 | 0.362 | [0.26, 0.47] |
| 8 | 52 | 0.068 | 0.327 | 0.354 | [0.24, 0.49] |
| 9 | 42 | 0.055 | 0.381 | 0.346 | [0.22, 0.50] |
| 10 | 29 | 0.038 | 0.414 | 0.338 | [0.19, 0.52] |
| 11 | 25 | 0.033 | 0.240 | 0.331 | [0.18, 0.53] |
| 12 | 15 | 0.020 | 0.200 | 0.323 | [0.14, 0.57] |
| 13 | 12 | 0.016 | 0.333 | 0.316 | [0.13, 0.59] |
| 14 | 14 | 0.018 | 0.357 | 0.308 | [0.13, 0.57] |
| 15 † | 4 | 0.005 | 0.500 | 0.301 | [0.06, 0.73] |
| 16 † | 3 | 0.004 | 0.333 | 0.294 | [0.05, 0.77] |
| 17 † | 2 | 0.003 | 0.000 | 0.287 | [0.03, 0.82] |
| 20 † | 2 | 0.003 | 0.000 | 0.266 | [0.03, 0.81] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.42).

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

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.433 | [0.36, 0.51] |
| 1 | 35 | 0.046 | 0.429 | 0.432 | [0.28, 0.59] |
| 2 | 48 | 0.063 | 0.333 | 0.389 | [0.26, 0.53] |
| 3 | 55 | 0.072 | 0.491 | 0.441 | [0.32, 0.57] |
| 4 | 66 | 0.087 | 0.333 | 0.337 | [0.23, 0.46] |
| 5 | 66 | 0.087 | 0.409 | 0.404 | [0.29, 0.52] |
| 6 | 67 | 0.088 | 0.313 | 0.325 | [0.22, 0.44] |
| 7 | 78 | 0.102 | 0.359 | 0.357 | [0.26, 0.47] |
| 8 | 52 | 0.068 | 0.327 | 0.302 | [0.19, 0.44] |
| 9 | 42 | 0.055 | 0.381 | 0.372 | [0.24, 0.52] |
| 10 | 29 | 0.038 | 0.414 | 0.416 | [0.26, 0.59] |
| 11 | 25 | 0.033 | 0.240 | 0.268 | [0.13, 0.46] |
| 12 | 15 | 0.020 | 0.200 | 0.109 | [0.03, 0.35] |
| 13 | 12 | 0.016 | 0.333 | 0.293 | [0.11, 0.57] |
| 14 | 14 | 0.018 | 0.357 | 0.514 | [0.28, 0.74] |
| 15 † | 4 | 0.005 | 0.500 | 0.496 | [0.15, 0.85] |
| 16 † | 3 | 0.004 | 0.333 | 0.015 | [0.00, 0.57] |
| 17 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 20 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.43).

### hybrid circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.444 | [0.37, 0.52] |
| 1 | 35 | 0.046 | 0.429 | 0.427 | [0.28, 0.59] |
| 2 | 48 | 0.063 | 0.333 | 0.312 | [0.20, 0.45] |
| 3 | 55 | 0.072 | 0.491 | 0.489 | [0.36, 0.62] |
| 4 | 66 | 0.087 | 0.333 | 0.320 | [0.22, 0.44] |
| 5 | 66 | 0.087 | 0.409 | 0.407 | [0.30, 0.53] |
| 6 | 67 | 0.088 | 0.313 | 0.315 | [0.22, 0.43] |
| 7 | 78 | 0.102 | 0.359 | 0.354 | [0.26, 0.47] |
| 8 | 52 | 0.068 | 0.327 | 0.306 | [0.20, 0.44] |
| 9 | 42 | 0.055 | 0.381 | 0.398 | [0.26, 0.55] |
| 10 | 29 | 0.038 | 0.414 | 0.405 | [0.25, 0.58] |
| 11 | 25 | 0.033 | 0.240 | 0.262 | [0.13, 0.46] |
| 12 | 15 | 0.020 | 0.200 | 0.107 | [0.03, 0.35] |
| 13 | 12 | 0.016 | 0.333 | 0.287 | [0.11, 0.57] |
| 14 | 14 | 0.018 | 0.357 | 0.505 | [0.27, 0.74] |
| 15 † | 4 | 0.005 | 0.500 | 0.499 | [0.15, 0.85] |
| 16 † | 3 | 0.004 | 0.333 | 0.567 | [0.16, 0.90] |
| 17 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 20 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.44).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.433 | [0.36, 0.51] |
| 1 | 35 | 0.046 | 0.429 | 0.425 | [0.28, 0.59] |
| 2 | 48 | 0.063 | 0.333 | 0.368 | [0.25, 0.51] |
| 3 | 55 | 0.072 | 0.491 | 0.448 | [0.32, 0.58] |
| 4 | 66 | 0.087 | 0.333 | 0.319 | [0.22, 0.44] |
| 5 | 66 | 0.087 | 0.409 | 0.413 | [0.30, 0.53] |
| 6 | 67 | 0.088 | 0.313 | 0.306 | [0.21, 0.42] |
| 7 | 78 | 0.102 | 0.359 | 0.354 | [0.26, 0.47] |
| 8 | 52 | 0.068 | 0.327 | 0.313 | [0.20, 0.45] |
| 9 | 42 | 0.055 | 0.381 | 0.383 | [0.25, 0.53] |
| 10 | 29 | 0.038 | 0.414 | 0.430 | [0.27, 0.61] |
| 11 | 25 | 0.033 | 0.240 | 0.265 | [0.13, 0.46] |
| 12 | 15 | 0.020 | 0.200 | 0.107 | [0.03, 0.35] |
| 13 | 12 | 0.016 | 0.333 | 0.290 | [0.11, 0.57] |
| 14 | 14 | 0.018 | 0.357 | 0.508 | [0.27, 0.74] |
| 15 † | 4 | 0.005 | 0.500 | 0.492 | [0.15, 0.85] |
| 16 † | 3 | 0.004 | 0.333 | 0.014 | [0.00, 0.57] |
| 17 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 20 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.43).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.445 | [0.37, 0.53] |
| 1 | 35 | 0.046 | 0.429 | 0.425 | [0.28, 0.59] |
| 2 | 48 | 0.063 | 0.333 | 0.318 | [0.20, 0.46] |
| 3 | 55 | 0.072 | 0.491 | 0.489 | [0.36, 0.62] |
| 4 | 66 | 0.087 | 0.333 | 0.326 | [0.23, 0.45] |
| 5 | 66 | 0.087 | 0.409 | 0.411 | [0.30, 0.53] |
| 6 | 67 | 0.088 | 0.313 | 0.314 | [0.22, 0.43] |
| 7 | 78 | 0.102 | 0.359 | 0.354 | [0.26, 0.47] |
| 8 | 52 | 0.068 | 0.327 | 0.311 | [0.20, 0.45] |
| 9 | 42 | 0.055 | 0.381 | 0.385 | [0.25, 0.54] |
| 10 | 29 | 0.038 | 0.414 | 0.383 | [0.23, 0.56] |
| 11 | 25 | 0.033 | 0.240 | 0.232 | [0.11, 0.43] |
| 12 | 15 | 0.020 | 0.200 | 0.106 | [0.03, 0.35] |
| 13 | 12 | 0.016 | 0.333 | 0.287 | [0.11, 0.57] |
| 14 | 14 | 0.018 | 0.357 | 0.507 | [0.27, 0.74] |
| 15 † | 4 | 0.005 | 0.500 | 0.494 | [0.15, 0.85] |
| 16 † | 3 | 0.004 | 0.333 | 0.015 | [0.00, 0.57] |
| 17 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 20 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 148 | 0.194 | 0.446 | 0.378 | [0.30, 0.46] |
| 1 | 35 | 0.046 | 0.429 | 0.378 | [0.24, 0.54] |
| 2 | 48 | 0.063 | 0.333 | 0.378 | [0.25, 0.52] |
| 3 | 55 | 0.072 | 0.491 | 0.378 | [0.26, 0.51] |
| 4 | 66 | 0.087 | 0.333 | 0.378 | [0.27, 0.50] |
| 5 | 66 | 0.087 | 0.409 | 0.377 | [0.27, 0.50] |
| 6 | 67 | 0.088 | 0.313 | 0.377 | [0.27, 0.50] |
| 7 | 78 | 0.102 | 0.359 | 0.377 | [0.28, 0.49] |
| 8 | 52 | 0.068 | 0.327 | 0.377 | [0.26, 0.51] |
| 9 | 42 | 0.055 | 0.381 | 0.377 | [0.25, 0.53] |
| 10 | 29 | 0.038 | 0.414 | 0.377 | [0.22, 0.56] |
| 11 | 25 | 0.033 | 0.240 | 0.377 | [0.22, 0.57] |
| 12 | 15 | 0.020 | 0.200 | 0.377 | [0.18, 0.62] |
| 13 | 12 | 0.016 | 0.333 | 0.377 | [0.17, 0.65] |
| 14 | 14 | 0.018 | 0.357 | 0.377 | [0.18, 0.63] |
| 15 † | 4 | 0.005 | 0.500 | 0.376 | [0.09, 0.78] |
| 16 † | 3 | 0.004 | 0.333 | 0.376 | [0.08, 0.82] |
| 17 † | 2 | 0.003 | 0.000 | 0.376 | [0.06, 0.86] |
| 20 † | 2 | 0.003 | 0.000 | 0.376 | [0.06, 0.86] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.38).

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

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 1 † | 4 | 0.005 | 0.250 | 0.250 | [0.05, 0.70] |
| 2 | 12 | 0.016 | 0.167 | 0.182 | [0.05, 0.46] |
| 3 | 17 | 0.022 | 0.353 | 0.312 | [0.15, 0.55] |
| 4 | 25 | 0.033 | 0.160 | 0.160 | [0.06, 0.35] |
| 5 | 25 | 0.033 | 0.400 | 0.400 | [0.23, 0.59] |
| 6 | 42 | 0.055 | 0.190 | 0.190 | [0.10, 0.33] |
| 7 | 71 | 0.093 | 0.324 | 0.324 | [0.23, 0.44] |
| 8 | 72 | 0.094 | 0.431 | 0.431 | [0.32, 0.55] |
| 9 | 87 | 0.114 | 0.345 | 0.345 | [0.25, 0.45] |
| 10 | 95 | 0.125 | 0.453 | 0.453 | [0.36, 0.55] |
| 11 | 93 | 0.122 | 0.409 | 0.409 | [0.31, 0.51] |
| 12 | 61 | 0.080 | 0.426 | 0.426 | [0.31, 0.55] |
| 13 | 51 | 0.067 | 0.333 | 0.333 | [0.22, 0.47] |
| 14 | 43 | 0.056 | 0.372 | 0.372 | [0.24, 0.52] |
| 15 | 25 | 0.033 | 0.400 | 0.400 | [0.23, 0.59] |
| 16 | 21 | 0.028 | 0.524 | 0.524 | [0.32, 0.72] |
| 17 † | 8 | 0.010 | 0.625 | 0.625 | [0.31, 0.86] |
| 18 † | 6 | 0.008 | 0.667 | 0.667 | [0.30, 0.90] |
| 19 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |
| 20 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### hybrid circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 1 † | 4 | 0.005 | 0.250 | 0.250 | [0.05, 0.70] |
| 2 | 12 | 0.016 | 0.167 | 0.182 | [0.05, 0.46] |
| 3 | 17 | 0.022 | 0.353 | 0.312 | [0.15, 0.55] |
| 4 | 25 | 0.033 | 0.160 | 0.160 | [0.06, 0.35] |
| 5 | 25 | 0.033 | 0.400 | 0.400 | [0.23, 0.59] |
| 6 | 42 | 0.055 | 0.190 | 0.190 | [0.10, 0.33] |
| 7 | 71 | 0.093 | 0.324 | 0.324 | [0.23, 0.44] |
| 8 | 72 | 0.094 | 0.431 | 0.431 | [0.32, 0.55] |
| 9 | 87 | 0.114 | 0.345 | 0.345 | [0.25, 0.45] |
| 10 | 95 | 0.125 | 0.453 | 0.453 | [0.36, 0.55] |
| 11 | 93 | 0.122 | 0.409 | 0.409 | [0.31, 0.51] |
| 12 | 61 | 0.080 | 0.426 | 0.426 | [0.31, 0.55] |
| 13 | 51 | 0.067 | 0.333 | 0.333 | [0.22, 0.47] |
| 14 | 43 | 0.056 | 0.372 | 0.372 | [0.24, 0.52] |
| 15 | 25 | 0.033 | 0.400 | 0.400 | [0.23, 0.59] |
| 16 | 21 | 0.028 | 0.524 | 0.524 | [0.32, 0.72] |
| 17 † | 8 | 0.010 | 0.625 | 0.625 | [0.31, 0.86] |
| 18 † | 6 | 0.008 | 0.667 | 0.667 | [0.30, 0.90] |
| 19 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |
| 20 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 1 † | 4 | 0.005 | 0.250 | 0.250 | [0.05, 0.70] |
| 2 | 12 | 0.016 | 0.167 | 0.182 | [0.05, 0.46] |
| 3 | 17 | 0.022 | 0.353 | 0.312 | [0.15, 0.55] |
| 4 | 25 | 0.033 | 0.160 | 0.160 | [0.06, 0.35] |
| 5 | 25 | 0.033 | 0.400 | 0.400 | [0.23, 0.59] |
| 6 | 42 | 0.055 | 0.190 | 0.190 | [0.10, 0.33] |
| 7 | 71 | 0.093 | 0.324 | 0.324 | [0.23, 0.44] |
| 8 | 72 | 0.094 | 0.431 | 0.431 | [0.32, 0.55] |
| 9 | 87 | 0.114 | 0.345 | 0.345 | [0.25, 0.45] |
| 10 | 95 | 0.125 | 0.453 | 0.453 | [0.36, 0.55] |
| 11 | 93 | 0.122 | 0.409 | 0.409 | [0.31, 0.51] |
| 12 | 61 | 0.080 | 0.426 | 0.426 | [0.31, 0.55] |
| 13 | 51 | 0.067 | 0.333 | 0.333 | [0.22, 0.47] |
| 14 | 43 | 0.056 | 0.372 | 0.372 | [0.24, 0.52] |
| 15 | 25 | 0.033 | 0.400 | 0.400 | [0.23, 0.59] |
| 16 | 21 | 0.028 | 0.524 | 0.524 | [0.32, 0.72] |
| 17 † | 8 | 0.010 | 0.625 | 0.625 | [0.31, 0.86] |
| 18 † | 6 | 0.008 | 0.667 | 0.667 | [0.30, 0.90] |
| 19 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |
| 20 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 1 † | 4 | 0.005 | 0.250 | 0.250 | [0.05, 0.70] |
| 2 | 12 | 0.016 | 0.167 | 0.182 | [0.05, 0.46] |
| 3 | 17 | 0.022 | 0.353 | 0.312 | [0.15, 0.55] |
| 4 | 25 | 0.033 | 0.160 | 0.160 | [0.06, 0.35] |
| 5 | 25 | 0.033 | 0.400 | 0.400 | [0.23, 0.59] |
| 6 | 42 | 0.055 | 0.190 | 0.190 | [0.10, 0.33] |
| 7 | 71 | 0.093 | 0.324 | 0.324 | [0.23, 0.44] |
| 8 | 72 | 0.094 | 0.431 | 0.431 | [0.32, 0.55] |
| 9 | 87 | 0.114 | 0.345 | 0.345 | [0.25, 0.45] |
| 10 | 95 | 0.125 | 0.453 | 0.453 | [0.36, 0.55] |
| 11 | 93 | 0.122 | 0.409 | 0.409 | [0.31, 0.51] |
| 12 | 61 | 0.080 | 0.426 | 0.426 | [0.31, 0.55] |
| 13 | 51 | 0.067 | 0.333 | 0.333 | [0.22, 0.47] |
| 14 | 43 | 0.056 | 0.372 | 0.372 | [0.24, 0.52] |
| 15 | 25 | 0.033 | 0.400 | 0.400 | [0.23, 0.59] |
| 16 | 21 | 0.028 | 0.524 | 0.524 | [0.32, 0.72] |
| 17 † | 8 | 0.010 | 0.625 | 0.625 | [0.31, 0.86] |
| 18 † | 6 | 0.008 | 0.667 | 0.667 | [0.30, 0.90] |
| 19 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |
| 20 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.246 | [0.04, 0.74] |
| 1 † | 4 | 0.005 | 0.250 | 0.258 | [0.05, 0.71] |
| 2 | 12 | 0.016 | 0.167 | 0.271 | [0.10, 0.55] |
| 3 | 17 | 0.022 | 0.353 | 0.283 | [0.13, 0.52] |
| 4 | 25 | 0.033 | 0.160 | 0.296 | [0.15, 0.49] |
| 5 | 25 | 0.033 | 0.400 | 0.310 | [0.16, 0.51] |
| 6 | 42 | 0.055 | 0.190 | 0.323 | [0.20, 0.47] |
| 7 | 71 | 0.093 | 0.324 | 0.337 | [0.24, 0.45] |
| 8 | 72 | 0.094 | 0.431 | 0.351 | [0.25, 0.47] |
| 9 | 87 | 0.114 | 0.345 | 0.366 | [0.27, 0.47] |
| 10 | 95 | 0.125 | 0.453 | 0.381 | [0.29, 0.48] |
| 11 | 93 | 0.122 | 0.409 | 0.395 | [0.30, 0.50] |
| 12 | 61 | 0.080 | 0.426 | 0.411 | [0.30, 0.54] |
| 13 | 51 | 0.067 | 0.333 | 0.426 | [0.30, 0.56] |
| 14 | 43 | 0.056 | 0.372 | 0.441 | [0.30, 0.59] |
| 15 | 25 | 0.033 | 0.400 | 0.457 | [0.28, 0.64] |
| 16 | 21 | 0.028 | 0.524 | 0.473 | [0.28, 0.67] |
| 17 † | 8 | 0.010 | 0.625 | 0.488 | [0.21, 0.78] |
| 18 † | 6 | 0.008 | 0.667 | 0.504 | [0.19, 0.81] |
| 19 † | 1 | 0.001 | 1.000 | 0.520 | [0.06, 0.95] |
| 20 † | 1 | 0.001 | 1.000 | 0.536 | [0.06, 0.95] |

EQL's own `cause` search settles on 16: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.47).

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

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.342 | [0.06, 0.80] |
| 1 † | 4 | 0.005 | 0.250 | 0.143 | [0.02, 0.62] |
| 2 | 12 | 0.016 | 0.167 | 0.180 | [0.05, 0.46] |
| 3 | 17 | 0.022 | 0.353 | 0.241 | [0.10, 0.48] |
| 4 | 25 | 0.033 | 0.160 | 0.070 | [0.02, 0.24] |
| 5 | 25 | 0.033 | 0.400 | 0.328 | [0.18, 0.52] |
| 6 | 42 | 0.055 | 0.190 | 0.197 | [0.10, 0.34] |
| 7 | 71 | 0.093 | 0.324 | 0.285 | [0.19, 0.40] |
| 8 | 72 | 0.094 | 0.431 | 0.390 | [0.29, 0.51] |
| 9 | 87 | 0.114 | 0.345 | 0.297 | [0.21, 0.40] |
| 10 | 95 | 0.125 | 0.453 | 0.435 | [0.34, 0.54] |
| 11 | 93 | 0.122 | 0.409 | 0.404 | [0.31, 0.51] |
| 12 | 61 | 0.080 | 0.426 | 0.468 | [0.35, 0.59] |
| 13 | 51 | 0.067 | 0.333 | 0.369 | [0.25, 0.51] |
| 14 | 43 | 0.056 | 0.372 | 0.408 | [0.27, 0.56] |
| 15 | 25 | 0.033 | 0.400 | 0.414 | [0.25, 0.61] |
| 16 | 21 | 0.028 | 0.524 | 0.379 | [0.21, 0.59] |
| 17 † | 8 | 0.010 | 0.625 | 0.764 | [0.42, 0.94] |
| 18 † | 6 | 0.008 | 0.667 | 0.503 | [0.19, 0.81] |
| 19 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |
| 20 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.44).

### hybrid circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.359 | [0.07, 0.81] |
| 1 † | 4 | 0.005 | 0.250 | 0.136 | [0.02, 0.61] |
| 2 | 12 | 0.016 | 0.167 | 0.184 | [0.06, 0.47] |
| 3 | 17 | 0.022 | 0.353 | 0.237 | [0.10, 0.47] |
| 4 | 25 | 0.033 | 0.160 | 0.095 | [0.03, 0.27] |
| 5 | 25 | 0.033 | 0.400 | 0.383 | [0.22, 0.58] |
| 6 | 42 | 0.055 | 0.190 | 0.176 | [0.09, 0.32] |
| 7 | 71 | 0.093 | 0.324 | 0.299 | [0.21, 0.41] |
| 8 | 72 | 0.094 | 0.431 | 0.380 | [0.28, 0.50] |
| 9 | 87 | 0.114 | 0.345 | 0.318 | [0.23, 0.42] |
| 10 | 95 | 0.125 | 0.453 | 0.453 | [0.36, 0.55] |
| 11 | 93 | 0.122 | 0.409 | 0.387 | [0.29, 0.49] |
| 12 | 61 | 0.080 | 0.426 | 0.413 | [0.30, 0.54] |
| 13 | 51 | 0.067 | 0.333 | 0.343 | [0.23, 0.48] |
| 14 | 43 | 0.056 | 0.372 | 0.339 | [0.22, 0.49] |
| 15 | 25 | 0.033 | 0.400 | 0.455 | [0.28, 0.64] |
| 16 | 21 | 0.028 | 0.524 | 0.607 | [0.40, 0.78] |
| 17 † | 8 | 0.010 | 0.625 | 0.761 | [0.42, 0.93] |
| 18 † | 6 | 0.008 | 0.667 | 0.519 | [0.20, 0.82] |
| 19 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |
| 20 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.359 | [0.07, 0.81] |
| 1 † | 4 | 0.005 | 0.250 | 0.136 | [0.02, 0.61] |
| 2 | 12 | 0.016 | 0.167 | 0.184 | [0.06, 0.47] |
| 3 | 17 | 0.022 | 0.353 | 0.237 | [0.10, 0.47] |
| 4 | 25 | 0.033 | 0.160 | 0.070 | [0.02, 0.24] |
| 5 | 25 | 0.033 | 0.400 | 0.333 | [0.18, 0.53] |
| 6 | 42 | 0.055 | 0.190 | 0.197 | [0.10, 0.34] |
| 7 | 71 | 0.093 | 0.324 | 0.287 | [0.20, 0.40] |
| 8 | 72 | 0.094 | 0.431 | 0.389 | [0.28, 0.50] |
| 9 | 87 | 0.114 | 0.345 | 0.295 | [0.21, 0.40] |
| 10 | 95 | 0.125 | 0.453 | 0.443 | [0.35, 0.54] |
| 11 | 93 | 0.122 | 0.409 | 0.403 | [0.31, 0.50] |
| 12 | 61 | 0.080 | 0.426 | 0.475 | [0.35, 0.60] |
| 13 | 51 | 0.067 | 0.333 | 0.365 | [0.25, 0.50] |
| 14 | 43 | 0.056 | 0.372 | 0.396 | [0.26, 0.54] |
| 15 | 25 | 0.033 | 0.400 | 0.434 | [0.26, 0.62] |
| 16 | 21 | 0.028 | 0.524 | 0.391 | [0.22, 0.60] |
| 17 † | 8 | 0.010 | 0.625 | 0.761 | [0.42, 0.93] |
| 18 † | 6 | 0.008 | 0.667 | 0.519 | [0.20, 0.82] |
| 19 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |
| 20 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.44).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.359 | [0.07, 0.81] |
| 1 † | 4 | 0.005 | 0.250 | 0.136 | [0.02, 0.61] |
| 2 | 12 | 0.016 | 0.167 | 0.184 | [0.06, 0.47] |
| 3 | 17 | 0.022 | 0.353 | 0.237 | [0.10, 0.47] |
| 4 | 25 | 0.033 | 0.160 | 0.093 | [0.03, 0.27] |
| 5 | 25 | 0.033 | 0.400 | 0.373 | [0.21, 0.57] |
| 6 | 42 | 0.055 | 0.190 | 0.157 | [0.08, 0.30] |
| 7 | 71 | 0.093 | 0.324 | 0.316 | [0.22, 0.43] |
| 8 | 72 | 0.094 | 0.431 | 0.365 | [0.26, 0.48] |
| 9 | 87 | 0.114 | 0.345 | 0.304 | [0.22, 0.41] |
| 10 | 95 | 0.125 | 0.453 | 0.439 | [0.34, 0.54] |
| 11 | 93 | 0.122 | 0.409 | 0.407 | [0.31, 0.51] |
| 12 | 61 | 0.080 | 0.426 | 0.410 | [0.30, 0.54] |
| 13 | 51 | 0.067 | 0.333 | 0.362 | [0.24, 0.50] |
| 14 | 43 | 0.056 | 0.372 | 0.388 | [0.26, 0.54] |
| 15 | 25 | 0.033 | 0.400 | 0.475 | [0.30, 0.66] |
| 16 | 21 | 0.028 | 0.524 | 0.549 | [0.35, 0.74] |
| 17 † | 8 | 0.010 | 0.625 | 0.761 | [0.42, 0.93] |
| 18 † | 6 | 0.008 | 0.667 | 0.519 | [0.20, 0.82] |
| 19 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |
| 20 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.44).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.063 | [0.00, 0.61] |
| 1 † | 4 | 0.005 | 0.250 | 0.078 | [0.01, 0.56] |
| 2 | 12 | 0.016 | 0.167 | 0.097 | [0.02, 0.37] |
| 3 | 17 | 0.022 | 0.353 | 0.120 | [0.03, 0.35] |
| 4 | 25 | 0.033 | 0.160 | 0.147 | [0.06, 0.33] |
| 5 | 25 | 0.033 | 0.400 | 0.178 | [0.07, 0.37] |
| 6 | 42 | 0.055 | 0.190 | 0.214 | [0.12, 0.36] |
| 7 | 71 | 0.093 | 0.324 | 0.254 | [0.17, 0.37] |
| 8 | 72 | 0.094 | 0.431 | 0.298 | [0.21, 0.41] |
| 9 | 87 | 0.114 | 0.345 | 0.346 | [0.25, 0.45] |
| 10 | 95 | 0.125 | 0.453 | 0.397 | [0.30, 0.50] |
| 11 | 93 | 0.122 | 0.409 | 0.450 | [0.35, 0.55] |
| 12 | 61 | 0.080 | 0.426 | 0.503 | [0.38, 0.62] |
| 13 | 51 | 0.067 | 0.333 | 0.557 | [0.42, 0.68] |
| 14 | 43 | 0.056 | 0.372 | 0.609 | [0.46, 0.74] |
| 15 | 25 | 0.033 | 0.400 | 0.660 | [0.46, 0.81] |
| 16 | 21 | 0.028 | 0.524 | 0.707 | [0.49, 0.86] |
| 17 † | 8 | 0.010 | 0.625 | 0.750 | [0.41, 0.93] |
| 18 † | 6 | 0.008 | 0.667 | 0.790 | [0.40, 0.96] |
| 19 † | 1 | 0.001 | 1.000 | 0.825 | [0.14, 0.99] |
| 20 † | 1 | 0.001 | 1.000 | 0.856 | [0.15, 0.99] |

EQL's own `cause` search settles on 16: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.71).

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

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.268 | [0.04, 0.75] |
| 1 † | 4 | 0.005 | 0.250 | 0.124 | [0.01, 0.60] |
| 2 | 12 | 0.016 | 0.167 | 0.213 | [0.07, 0.50] |
| 3 | 17 | 0.022 | 0.353 | 0.210 | [0.08, 0.45] |
| 4 | 25 | 0.033 | 0.160 | 0.095 | [0.03, 0.27] |
| 5 | 25 | 0.033 | 0.400 | 0.315 | [0.17, 0.51] |
| 6 | 42 | 0.055 | 0.190 | 0.197 | [0.10, 0.34] |
| 7 | 71 | 0.093 | 0.324 | 0.282 | [0.19, 0.40] |
| 8 | 72 | 0.094 | 0.431 | 0.372 | [0.27, 0.49] |
| 9 | 87 | 0.114 | 0.345 | 0.285 | [0.20, 0.39] |
| 10 | 95 | 0.125 | 0.453 | 0.436 | [0.34, 0.54] |
| 11 | 93 | 0.122 | 0.409 | 0.404 | [0.31, 0.51] |
| 12 | 61 | 0.080 | 0.426 | 0.453 | [0.33, 0.58] |
| 13 | 51 | 0.067 | 0.333 | 0.373 | [0.25, 0.51] |
| 14 | 43 | 0.056 | 0.372 | 0.394 | [0.26, 0.54] |
| 15 | 25 | 0.033 | 0.400 | 0.414 | [0.25, 0.61] |
| 16 | 21 | 0.028 | 0.524 | 0.380 | [0.21, 0.59] |
| 17 † | 8 | 0.010 | 0.625 | 0.761 | [0.42, 0.93] |
| 18 † | 6 | 0.008 | 0.667 | 0.504 | [0.19, 0.82] |
| 19 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |
| 20 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.44).

### hybrid circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.358 | [0.07, 0.81] |
| 1 † | 4 | 0.005 | 0.250 | 0.137 | [0.02, 0.61] |
| 2 | 12 | 0.016 | 0.167 | 0.184 | [0.06, 0.47] |
| 3 | 17 | 0.022 | 0.353 | 0.236 | [0.10, 0.47] |
| 4 | 25 | 0.033 | 0.160 | 0.096 | [0.03, 0.27] |
| 5 | 25 | 0.033 | 0.400 | 0.387 | [0.22, 0.58] |
| 6 | 42 | 0.055 | 0.190 | 0.176 | [0.09, 0.32] |
| 7 | 71 | 0.093 | 0.324 | 0.299 | [0.20, 0.41] |
| 8 | 72 | 0.094 | 0.431 | 0.378 | [0.28, 0.49] |
| 9 | 87 | 0.114 | 0.345 | 0.318 | [0.23, 0.42] |
| 10 | 95 | 0.125 | 0.453 | 0.452 | [0.36, 0.55] |
| 11 | 93 | 0.122 | 0.409 | 0.386 | [0.29, 0.49] |
| 12 | 61 | 0.080 | 0.426 | 0.413 | [0.30, 0.54] |
| 13 | 51 | 0.067 | 0.333 | 0.343 | [0.23, 0.48] |
| 14 | 43 | 0.056 | 0.372 | 0.340 | [0.22, 0.49] |
| 15 | 25 | 0.033 | 0.400 | 0.456 | [0.28, 0.64] |
| 16 | 21 | 0.028 | 0.524 | 0.607 | [0.40, 0.78] |
| 17 † | 8 | 0.010 | 0.625 | 0.761 | [0.42, 0.93] |
| 18 † | 6 | 0.008 | 0.667 | 0.519 | [0.20, 0.82] |
| 19 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |
| 20 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.45).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.286 | [0.05, 0.77] |
| 1 † | 4 | 0.005 | 0.250 | 0.119 | [0.01, 0.60] |
| 2 | 12 | 0.016 | 0.167 | 0.217 | [0.07, 0.50] |
| 3 | 17 | 0.022 | 0.353 | 0.207 | [0.08, 0.44] |
| 4 | 25 | 0.033 | 0.160 | 0.084 | [0.02, 0.26] |
| 5 | 25 | 0.033 | 0.400 | 0.324 | [0.17, 0.52] |
| 6 | 42 | 0.055 | 0.190 | 0.196 | [0.10, 0.34] |
| 7 | 71 | 0.093 | 0.324 | 0.284 | [0.19, 0.40] |
| 8 | 72 | 0.094 | 0.431 | 0.372 | [0.27, 0.49] |
| 9 | 87 | 0.114 | 0.345 | 0.284 | [0.20, 0.39] |
| 10 | 95 | 0.125 | 0.453 | 0.444 | [0.35, 0.54] |
| 11 | 93 | 0.122 | 0.409 | 0.403 | [0.31, 0.50] |
| 12 | 61 | 0.080 | 0.426 | 0.458 | [0.34, 0.58] |
| 13 | 51 | 0.067 | 0.333 | 0.369 | [0.25, 0.51] |
| 14 | 43 | 0.056 | 0.372 | 0.379 | [0.25, 0.53] |
| 15 | 25 | 0.033 | 0.400 | 0.434 | [0.26, 0.62] |
| 16 | 21 | 0.028 | 0.524 | 0.392 | [0.22, 0.60] |
| 17 † | 8 | 0.010 | 0.625 | 0.757 | [0.42, 0.93] |
| 18 † | 6 | 0.008 | 0.667 | 0.522 | [0.20, 0.83] |
| 19 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |
| 20 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.44).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.358 | [0.07, 0.81] |
| 1 † | 4 | 0.005 | 0.250 | 0.136 | [0.02, 0.61] |
| 2 | 12 | 0.016 | 0.167 | 0.184 | [0.06, 0.47] |
| 3 | 17 | 0.022 | 0.353 | 0.231 | [0.09, 0.47] |
| 4 | 25 | 0.033 | 0.160 | 0.090 | [0.03, 0.26] |
| 5 | 25 | 0.033 | 0.400 | 0.373 | [0.21, 0.57] |
| 6 | 42 | 0.055 | 0.190 | 0.157 | [0.08, 0.30] |
| 7 | 71 | 0.093 | 0.324 | 0.315 | [0.22, 0.43] |
| 8 | 72 | 0.094 | 0.431 | 0.354 | [0.25, 0.47] |
| 9 | 87 | 0.114 | 0.345 | 0.303 | [0.22, 0.41] |
| 10 | 95 | 0.125 | 0.453 | 0.439 | [0.34, 0.54] |
| 11 | 93 | 0.122 | 0.409 | 0.407 | [0.31, 0.51] |
| 12 | 61 | 0.080 | 0.426 | 0.408 | [0.29, 0.53] |
| 13 | 51 | 0.067 | 0.333 | 0.363 | [0.24, 0.50] |
| 14 | 43 | 0.056 | 0.372 | 0.387 | [0.26, 0.54] |
| 15 | 25 | 0.033 | 0.400 | 0.474 | [0.29, 0.66] |
| 16 | 21 | 0.028 | 0.524 | 0.549 | [0.35, 0.74] |
| 17 † | 8 | 0.010 | 0.625 | 0.761 | [0.42, 0.93] |
| 18 † | 6 | 0.008 | 0.667 | 0.519 | [0.20, 0.82] |
| 19 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |
| 20 † | 1 | 0.001 | 1.000 | 1.000 | [0.21, 1.00] |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.44).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.064 | [0.00, 0.61] |
| 1 † | 4 | 0.005 | 0.250 | 0.080 | [0.01, 0.57] |
| 2 | 12 | 0.016 | 0.167 | 0.099 | [0.02, 0.37] |
| 3 | 17 | 0.022 | 0.353 | 0.122 | [0.03, 0.35] |
| 4 | 25 | 0.033 | 0.160 | 0.149 | [0.06, 0.33] |
| 5 | 25 | 0.033 | 0.400 | 0.180 | [0.08, 0.37] |
| 6 | 42 | 0.055 | 0.190 | 0.216 | [0.12, 0.36] |
| 7 | 71 | 0.093 | 0.324 | 0.256 | [0.17, 0.37] |
| 8 | 72 | 0.094 | 0.431 | 0.300 | [0.21, 0.41] |
| 9 | 87 | 0.114 | 0.345 | 0.347 | [0.26, 0.45] |
| 10 | 95 | 0.125 | 0.453 | 0.397 | [0.30, 0.50] |
| 11 | 93 | 0.122 | 0.409 | 0.449 | [0.35, 0.55] |
| 12 | 61 | 0.080 | 0.426 | 0.502 | [0.38, 0.62] |
| 13 | 51 | 0.067 | 0.333 | 0.555 | [0.42, 0.68] |
| 14 | 43 | 0.056 | 0.372 | 0.607 | [0.46, 0.74] |
| 15 | 25 | 0.033 | 0.400 | 0.656 | [0.46, 0.81] |
| 16 | 21 | 0.028 | 0.524 | 0.703 | [0.49, 0.85] |
| 17 † | 8 | 0.010 | 0.625 | 0.746 | [0.41, 0.93] |
| 18 † | 6 | 0.008 | 0.667 | 0.786 | [0.39, 0.95] |
| 19 † | 1 | 0.001 | 1.000 | 0.821 | [0.14, 0.99] |
| 20 † | 1 | 0.001 | 1.000 | 0.852 | [0.15, 0.99] |

EQL's own `cause` search settles on 16: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.70).

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

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.481 | [0.41, 0.55] |
| 1 | 11 | 0.014 | 0.545 | 0.545 | [0.28, 0.79] |
| 2 | 18 | 0.024 | 0.556 | 0.588 | [0.37, 0.78] |
| 3 † | 8 | 0.010 | 0.625 | 0.500 | [0.22, 0.78] |
| 4 † | 3 | 0.004 | 0.667 | 0.667 | [0.21, 0.94] |
| 5 † | 7 | 0.009 | 0.429 | 0.429 | [0.16, 0.75] |
| 6 † | 5 | 0.007 | 0.600 | 0.600 | [0.23, 0.88] |
| 7 | 10 | 0.013 | 0.500 | 0.500 | [0.24, 0.76] |
| 8 † | 5 | 0.007 | 0.400 | 0.400 | [0.12, 0.77] |
| 9 † | 6 | 0.008 | 0.500 | 0.500 | [0.19, 0.81] |
| 10 | 10 | 0.013 | 0.400 | 0.400 | [0.17, 0.69] |
| 11 | 10 | 0.013 | 0.100 | 0.100 | [0.02, 0.40] |
| 12 † | 5 | 0.007 | 0.800 | 1.000 | [0.57, 1.00] |
| 13 † | 9 | 0.012 | 0.222 | 0.222 | [0.06, 0.55] |
| 14 † | 3 | 0.004 | 1.000 | 1.000 | [0.44, 1.00] |
| 15 | 10 | 0.013 | 0.500 | 0.444 | [0.20, 0.72] |
| 16 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 17 † | 4 | 0.005 | 0.500 | 0.667 | [0.25, 0.92] |
| 18 | 11 | 0.014 | 0.273 | 0.200 | [0.06, 0.50] |
| 19 † | 5 | 0.007 | 0.800 | 0.667 | [0.28, 0.91] |
| 20 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 21 † | 7 | 0.009 | 0.714 | 0.714 | [0.36, 0.92] |
| 22 † | 5 | 0.007 | 0.600 | 0.600 | [0.23, 0.88] |
| 23 † | 9 | 0.012 | 0.778 | 0.750 | [0.43, 0.92] |
| 24 † | 3 | 0.004 | 0.667 | 0.667 | [0.21, 0.94] |
| 25 † | 7 | 0.009 | 0.571 | 0.571 | [0.25, 0.84] |
| 26 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 27 † | 8 | 0.010 | 0.750 | 0.714 | [0.38, 0.91] |
| 28 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 29 † | 3 | 0.004 | 0.667 | 0.667 | [0.21, 0.94] |
| 30 † | 6 | 0.008 | 0.667 | 0.600 | [0.25, 0.87] |
| 31 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 32 † | 5 | 0.007 | 0.400 | 0.400 | [0.12, 0.77] |
| 33 † | 5 | 0.007 | 0.400 | 0.250 | [0.05, 0.66] |
| 34 † | 9 | 0.012 | 0.111 | 0.111 | [0.02, 0.44] |
| 35 † | 5 | 0.007 | 1.000 | 1.000 | [0.57, 1.00] |
| 36 † | 6 | 0.008 | 0.500 | 0.400 | [0.13, 0.75] |
| 37 | 10 | 0.013 | 0.400 | 0.333 | [0.13, 0.63] |
| 38 † | 6 | 0.008 | 0.500 | 0.500 | [0.19, 0.81] |
| 39 † | 7 | 0.009 | 0.286 | 0.286 | [0.08, 0.64] |
| 40 | 11 | 0.014 | 0.273 | 0.250 | [0.09, 0.54] |
| 41 † | 7 | 0.009 | 0.286 | 0.286 | [0.08, 0.64] |
| 42 | 10 | 0.013 | 0.500 | 0.500 | [0.24, 0.76] |
| 43 | 14 | 0.018 | 0.357 | 0.357 | [0.16, 0.61] |
| 44 | 13 | 0.017 | 0.308 | 0.308 | [0.13, 0.58] |
| 45 | 15 | 0.020 | 0.400 | 0.385 | [0.19, 0.63] |
| 46 | 22 | 0.029 | 0.136 | 0.136 | [0.05, 0.33] |
| 47 | 20 | 0.026 | 0.300 | 0.300 | [0.15, 0.52] |
| 48 | 29 | 0.038 | 0.172 | 0.172 | [0.08, 0.35] |
| 49 | 32 | 0.042 | 0.312 | 0.312 | [0.18, 0.49] |
| 50 | 45 | 0.059 | 0.111 | 0.111 | [0.05, 0.23] |
| 51 | 36 | 0.047 | 0.139 | 0.139 | [0.06, 0.29] |
| 52 | 71 | 0.093 | 0.197 | 0.197 | [0.12, 0.30] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.48).

### hybrid circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.481 | [0.41, 0.55] |
| 1 | 11 | 0.014 | 0.545 | 0.545 | [0.28, 0.79] |
| 2 | 18 | 0.024 | 0.556 | 0.588 | [0.37, 0.78] |
| 3 † | 8 | 0.010 | 0.625 | 0.500 | [0.22, 0.78] |
| 4 † | 3 | 0.004 | 0.667 | 0.667 | [0.21, 0.94] |
| 5 † | 7 | 0.009 | 0.429 | 0.429 | [0.16, 0.75] |
| 6 † | 5 | 0.007 | 0.600 | 0.600 | [0.23, 0.88] |
| 7 | 10 | 0.013 | 0.500 | 0.500 | [0.24, 0.76] |
| 8 † | 5 | 0.007 | 0.400 | 0.400 | [0.12, 0.77] |
| 9 † | 6 | 0.008 | 0.500 | 0.500 | [0.19, 0.81] |
| 10 | 10 | 0.013 | 0.400 | 0.400 | [0.17, 0.69] |
| 11 | 10 | 0.013 | 0.100 | 0.100 | [0.02, 0.40] |
| 12 † | 5 | 0.007 | 0.800 | 1.000 | [0.57, 1.00] |
| 13 † | 9 | 0.012 | 0.222 | 0.222 | [0.06, 0.55] |
| 14 † | 3 | 0.004 | 1.000 | 1.000 | [0.44, 1.00] |
| 15 | 10 | 0.013 | 0.500 | 0.444 | [0.20, 0.72] |
| 16 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 17 † | 4 | 0.005 | 0.500 | 0.667 | [0.25, 0.92] |
| 18 | 11 | 0.014 | 0.273 | 0.200 | [0.06, 0.50] |
| 19 † | 5 | 0.007 | 0.800 | 0.667 | [0.28, 0.91] |
| 20 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 21 † | 7 | 0.009 | 0.714 | 0.714 | [0.36, 0.92] |
| 22 † | 5 | 0.007 | 0.600 | 0.600 | [0.23, 0.88] |
| 23 † | 9 | 0.012 | 0.778 | 0.750 | [0.43, 0.92] |
| 24 † | 3 | 0.004 | 0.667 | 0.667 | [0.21, 0.94] |
| 25 † | 7 | 0.009 | 0.571 | 0.571 | [0.25, 0.84] |
| 26 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 27 † | 8 | 0.010 | 0.750 | 0.714 | [0.38, 0.91] |
| 28 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 29 † | 3 | 0.004 | 0.667 | 0.667 | [0.21, 0.94] |
| 30 † | 6 | 0.008 | 0.667 | 0.600 | [0.25, 0.87] |
| 31 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 32 † | 5 | 0.007 | 0.400 | 0.400 | [0.12, 0.77] |
| 33 † | 5 | 0.007 | 0.400 | 0.250 | [0.05, 0.66] |
| 34 † | 9 | 0.012 | 0.111 | 0.111 | [0.02, 0.44] |
| 35 † | 5 | 0.007 | 1.000 | 1.000 | [0.57, 1.00] |
| 36 † | 6 | 0.008 | 0.500 | 0.400 | [0.13, 0.75] |
| 37 | 10 | 0.013 | 0.400 | 0.333 | [0.13, 0.63] |
| 38 † | 6 | 0.008 | 0.500 | 0.500 | [0.19, 0.81] |
| 39 † | 7 | 0.009 | 0.286 | 0.286 | [0.08, 0.64] |
| 40 | 11 | 0.014 | 0.273 | 0.250 | [0.09, 0.54] |
| 41 † | 7 | 0.009 | 0.286 | 0.286 | [0.08, 0.64] |
| 42 | 10 | 0.013 | 0.500 | 0.500 | [0.24, 0.76] |
| 43 | 14 | 0.018 | 0.357 | 0.357 | [0.16, 0.61] |
| 44 | 13 | 0.017 | 0.308 | 0.308 | [0.13, 0.58] |
| 45 | 15 | 0.020 | 0.400 | 0.385 | [0.19, 0.63] |
| 46 | 22 | 0.029 | 0.136 | 0.136 | [0.05, 0.33] |
| 47 | 20 | 0.026 | 0.300 | 0.300 | [0.15, 0.52] |
| 48 | 29 | 0.038 | 0.172 | 0.172 | [0.08, 0.35] |
| 49 | 32 | 0.042 | 0.312 | 0.312 | [0.18, 0.49] |
| 50 | 45 | 0.059 | 0.111 | 0.111 | [0.05, 0.23] |
| 51 | 36 | 0.047 | 0.139 | 0.139 | [0.06, 0.29] |
| 52 | 71 | 0.093 | 0.197 | 0.197 | [0.12, 0.30] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.48).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.481 | [0.41, 0.55] |
| 1 | 11 | 0.014 | 0.545 | 0.545 | [0.28, 0.79] |
| 2 | 18 | 0.024 | 0.556 | 0.588 | [0.37, 0.78] |
| 3 † | 8 | 0.010 | 0.625 | 0.500 | [0.22, 0.78] |
| 4 † | 3 | 0.004 | 0.667 | 0.667 | [0.21, 0.94] |
| 5 † | 7 | 0.009 | 0.429 | 0.429 | [0.16, 0.75] |
| 6 † | 5 | 0.007 | 0.600 | 0.600 | [0.23, 0.88] |
| 7 | 10 | 0.013 | 0.500 | 0.500 | [0.24, 0.76] |
| 8 † | 5 | 0.007 | 0.400 | 0.400 | [0.12, 0.77] |
| 9 † | 6 | 0.008 | 0.500 | 0.500 | [0.19, 0.81] |
| 10 | 10 | 0.013 | 0.400 | 0.400 | [0.17, 0.69] |
| 11 | 10 | 0.013 | 0.100 | 0.100 | [0.02, 0.40] |
| 12 † | 5 | 0.007 | 0.800 | 1.000 | [0.57, 1.00] |
| 13 † | 9 | 0.012 | 0.222 | 0.222 | [0.06, 0.55] |
| 14 † | 3 | 0.004 | 1.000 | 1.000 | [0.44, 1.00] |
| 15 | 10 | 0.013 | 0.500 | 0.444 | [0.20, 0.72] |
| 16 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 17 † | 4 | 0.005 | 0.500 | 0.667 | [0.25, 0.92] |
| 18 | 11 | 0.014 | 0.273 | 0.200 | [0.06, 0.50] |
| 19 † | 5 | 0.007 | 0.800 | 0.667 | [0.28, 0.91] |
| 20 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 21 † | 7 | 0.009 | 0.714 | 0.714 | [0.36, 0.92] |
| 22 † | 5 | 0.007 | 0.600 | 0.600 | [0.23, 0.88] |
| 23 † | 9 | 0.012 | 0.778 | 0.750 | [0.43, 0.92] |
| 24 † | 3 | 0.004 | 0.667 | 0.667 | [0.21, 0.94] |
| 25 † | 7 | 0.009 | 0.571 | 0.571 | [0.25, 0.84] |
| 26 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 27 † | 8 | 0.010 | 0.750 | 0.714 | [0.38, 0.91] |
| 28 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 29 † | 3 | 0.004 | 0.667 | 0.667 | [0.21, 0.94] |
| 30 † | 6 | 0.008 | 0.667 | 0.600 | [0.25, 0.87] |
| 31 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 32 † | 5 | 0.007 | 0.400 | 0.400 | [0.12, 0.77] |
| 33 † | 5 | 0.007 | 0.400 | 0.250 | [0.05, 0.66] |
| 34 † | 9 | 0.012 | 0.111 | 0.111 | [0.02, 0.44] |
| 35 † | 5 | 0.007 | 1.000 | 1.000 | [0.57, 1.00] |
| 36 † | 6 | 0.008 | 0.500 | 0.400 | [0.13, 0.75] |
| 37 | 10 | 0.013 | 0.400 | 0.333 | [0.13, 0.63] |
| 38 † | 6 | 0.008 | 0.500 | 0.500 | [0.19, 0.81] |
| 39 † | 7 | 0.009 | 0.286 | 0.286 | [0.08, 0.64] |
| 40 | 11 | 0.014 | 0.273 | 0.250 | [0.09, 0.54] |
| 41 † | 7 | 0.009 | 0.286 | 0.286 | [0.08, 0.64] |
| 42 | 10 | 0.013 | 0.500 | 0.500 | [0.24, 0.76] |
| 43 | 14 | 0.018 | 0.357 | 0.357 | [0.16, 0.61] |
| 44 | 13 | 0.017 | 0.308 | 0.308 | [0.13, 0.58] |
| 45 | 15 | 0.020 | 0.400 | 0.385 | [0.19, 0.63] |
| 46 | 22 | 0.029 | 0.136 | 0.136 | [0.05, 0.33] |
| 47 | 20 | 0.026 | 0.300 | 0.300 | [0.15, 0.52] |
| 48 | 29 | 0.038 | 0.172 | 0.172 | [0.08, 0.35] |
| 49 | 32 | 0.042 | 0.312 | 0.312 | [0.18, 0.49] |
| 50 | 45 | 0.059 | 0.111 | 0.111 | [0.05, 0.23] |
| 51 | 36 | 0.047 | 0.139 | 0.139 | [0.06, 0.29] |
| 52 | 71 | 0.093 | 0.197 | 0.197 | [0.12, 0.30] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.48).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.481 | [0.41, 0.55] |
| 1 | 11 | 0.014 | 0.545 | 0.545 | [0.28, 0.79] |
| 2 | 18 | 0.024 | 0.556 | 0.588 | [0.37, 0.78] |
| 3 † | 8 | 0.010 | 0.625 | 0.500 | [0.22, 0.78] |
| 4 † | 3 | 0.004 | 0.667 | 0.667 | [0.21, 0.94] |
| 5 † | 7 | 0.009 | 0.429 | 0.429 | [0.16, 0.75] |
| 6 † | 5 | 0.007 | 0.600 | 0.600 | [0.23, 0.88] |
| 7 | 10 | 0.013 | 0.500 | 0.500 | [0.24, 0.76] |
| 8 † | 5 | 0.007 | 0.400 | 0.400 | [0.12, 0.77] |
| 9 † | 6 | 0.008 | 0.500 | 0.500 | [0.19, 0.81] |
| 10 | 10 | 0.013 | 0.400 | 0.400 | [0.17, 0.69] |
| 11 | 10 | 0.013 | 0.100 | 0.100 | [0.02, 0.40] |
| 12 † | 5 | 0.007 | 0.800 | 1.000 | [0.57, 1.00] |
| 13 † | 9 | 0.012 | 0.222 | 0.222 | [0.06, 0.55] |
| 14 † | 3 | 0.004 | 1.000 | 1.000 | [0.44, 1.00] |
| 15 | 10 | 0.013 | 0.500 | 0.444 | [0.20, 0.72] |
| 16 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 17 † | 4 | 0.005 | 0.500 | 0.667 | [0.25, 0.92] |
| 18 | 11 | 0.014 | 0.273 | 0.200 | [0.06, 0.50] |
| 19 † | 5 | 0.007 | 0.800 | 0.667 | [0.28, 0.91] |
| 20 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 21 † | 7 | 0.009 | 0.714 | 0.714 | [0.36, 0.92] |
| 22 † | 5 | 0.007 | 0.600 | 0.600 | [0.23, 0.88] |
| 23 † | 9 | 0.012 | 0.778 | 0.750 | [0.43, 0.92] |
| 24 † | 3 | 0.004 | 0.667 | 0.667 | [0.21, 0.94] |
| 25 † | 7 | 0.009 | 0.571 | 0.571 | [0.25, 0.84] |
| 26 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 27 † | 8 | 0.010 | 0.750 | 0.714 | [0.38, 0.91] |
| 28 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 29 † | 3 | 0.004 | 0.667 | 0.667 | [0.21, 0.94] |
| 30 † | 6 | 0.008 | 0.667 | 0.600 | [0.25, 0.87] |
| 31 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 32 † | 5 | 0.007 | 0.400 | 0.400 | [0.12, 0.77] |
| 33 † | 5 | 0.007 | 0.400 | 0.250 | [0.05, 0.66] |
| 34 † | 9 | 0.012 | 0.111 | 0.111 | [0.02, 0.44] |
| 35 † | 5 | 0.007 | 1.000 | 1.000 | [0.57, 1.00] |
| 36 † | 6 | 0.008 | 0.500 | 0.400 | [0.13, 0.75] |
| 37 | 10 | 0.013 | 0.400 | 0.333 | [0.13, 0.63] |
| 38 † | 6 | 0.008 | 0.500 | 0.500 | [0.19, 0.81] |
| 39 † | 7 | 0.009 | 0.286 | 0.286 | [0.08, 0.64] |
| 40 | 11 | 0.014 | 0.273 | 0.250 | [0.09, 0.54] |
| 41 † | 7 | 0.009 | 0.286 | 0.286 | [0.08, 0.64] |
| 42 | 10 | 0.013 | 0.500 | 0.500 | [0.24, 0.76] |
| 43 | 14 | 0.018 | 0.357 | 0.357 | [0.16, 0.61] |
| 44 | 13 | 0.017 | 0.308 | 0.308 | [0.13, 0.58] |
| 45 | 15 | 0.020 | 0.400 | 0.385 | [0.19, 0.63] |
| 46 | 22 | 0.029 | 0.136 | 0.136 | [0.05, 0.33] |
| 47 | 20 | 0.026 | 0.300 | 0.300 | [0.15, 0.52] |
| 48 | 29 | 0.038 | 0.172 | 0.172 | [0.08, 0.35] |
| 49 | 32 | 0.042 | 0.312 | 0.312 | [0.18, 0.49] |
| 50 | 45 | 0.059 | 0.111 | 0.111 | [0.05, 0.23] |
| 51 | 36 | 0.047 | 0.139 | 0.139 | [0.06, 0.29] |
| 52 | 71 | 0.093 | 0.197 | 0.197 | [0.12, 0.30] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.48).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.530 | [0.46, 0.60] |
| 1 | 11 | 0.014 | 0.545 | 0.524 | [0.26, 0.77] |
| 2 | 18 | 0.024 | 0.556 | 0.518 | [0.30, 0.72] |
| 3 † | 8 | 0.010 | 0.625 | 0.511 | [0.22, 0.79] |
| 4 † | 3 | 0.004 | 0.667 | 0.505 | [0.13, 0.88] |
| 5 † | 7 | 0.009 | 0.429 | 0.499 | [0.20, 0.80] |
| 6 † | 5 | 0.007 | 0.600 | 0.493 | [0.17, 0.83] |
| 7 | 10 | 0.013 | 0.500 | 0.487 | [0.23, 0.75] |
| 8 † | 5 | 0.007 | 0.400 | 0.481 | [0.16, 0.82] |
| 9 † | 6 | 0.008 | 0.500 | 0.475 | [0.17, 0.80] |
| 10 | 10 | 0.013 | 0.400 | 0.469 | [0.21, 0.74] |
| 11 | 10 | 0.013 | 0.100 | 0.463 | [0.21, 0.74] |
| 12 † | 5 | 0.007 | 0.800 | 0.457 | [0.15, 0.80] |
| 13 † | 9 | 0.012 | 0.222 | 0.451 | [0.19, 0.74] |
| 14 † | 3 | 0.004 | 1.000 | 0.445 | [0.10, 0.85] |
| 15 | 10 | 0.013 | 0.500 | 0.439 | [0.19, 0.72] |
| 16 † | 2 | 0.003 | 0.000 | 0.433 | [0.07, 0.88] |
| 17 † | 4 | 0.005 | 0.500 | 0.427 | [0.11, 0.81] |
| 18 | 11 | 0.014 | 0.273 | 0.421 | [0.19, 0.69] |
| 19 † | 5 | 0.007 | 0.800 | 0.415 | [0.12, 0.78] |
| 20 † | 3 | 0.004 | 0.333 | 0.409 | [0.09, 0.83] |
| 21 † | 7 | 0.009 | 0.714 | 0.403 | [0.14, 0.73] |
| 22 † | 5 | 0.007 | 0.600 | 0.397 | [0.12, 0.77] |
| 23 † | 9 | 0.012 | 0.778 | 0.391 | [0.15, 0.69] |
| 24 † | 3 | 0.004 | 0.667 | 0.386 | [0.08, 0.82] |
| 25 † | 7 | 0.009 | 0.571 | 0.380 | [0.13, 0.71] |
| 26 † | 4 | 0.005 | 0.500 | 0.374 | [0.09, 0.78] |
| 27 † | 8 | 0.010 | 0.750 | 0.368 | [0.13, 0.69] |
| 28 † | 4 | 0.005 | 0.500 | 0.363 | [0.09, 0.77] |
| 29 † | 3 | 0.004 | 0.667 | 0.357 | [0.07, 0.81] |
| 30 † | 6 | 0.008 | 0.667 | 0.352 | [0.11, 0.71] |
| 31 † | 3 | 0.004 | 0.333 | 0.346 | [0.07, 0.80] |
| 32 † | 5 | 0.007 | 0.400 | 0.341 | [0.09, 0.73] |
| 33 † | 5 | 0.007 | 0.400 | 0.335 | [0.09, 0.73] |
| 34 † | 9 | 0.012 | 0.111 | 0.330 | [0.12, 0.64] |
| 35 † | 5 | 0.007 | 1.000 | 0.324 | [0.08, 0.72] |
| 36 † | 6 | 0.008 | 0.500 | 0.319 | [0.09, 0.69] |
| 37 | 10 | 0.013 | 0.400 | 0.314 | [0.12, 0.62] |
| 38 † | 6 | 0.008 | 0.500 | 0.308 | [0.09, 0.68] |
| 39 † | 7 | 0.009 | 0.286 | 0.303 | [0.09, 0.66] |
| 40 | 11 | 0.014 | 0.273 | 0.298 | [0.11, 0.59] |
| 41 † | 7 | 0.009 | 0.286 | 0.293 | [0.09, 0.65] |
| 42 | 10 | 0.013 | 0.500 | 0.288 | [0.10, 0.59] |
| 43 | 14 | 0.018 | 0.357 | 0.283 | [0.12, 0.54] |
| 44 | 13 | 0.017 | 0.308 | 0.278 | [0.11, 0.55] |
| 45 | 15 | 0.020 | 0.400 | 0.273 | [0.11, 0.53] |
| 46 | 22 | 0.029 | 0.136 | 0.268 | [0.13, 0.48] |
| 47 | 20 | 0.026 | 0.300 | 0.264 | [0.12, 0.48] |
| 48 | 29 | 0.038 | 0.172 | 0.259 | [0.13, 0.44] |
| 49 | 32 | 0.042 | 0.312 | 0.254 | [0.14, 0.43] |
| 50 | 45 | 0.059 | 0.111 | 0.250 | [0.15, 0.39] |
| 51 | 36 | 0.047 | 0.139 | 0.245 | [0.13, 0.41] |
| 52 | 71 | 0.093 | 0.197 | 0.241 | [0.16, 0.35] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.53).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.378 | [0.31, 0.45] |
| 1 | 11 | 0.014 | 0.545 | 0.378 | [0.16, 0.66] |
| 2 | 18 | 0.024 | 0.556 | 0.377 | [0.19, 0.60] |
| 3 † | 8 | 0.010 | 0.625 | 0.377 | [0.14, 0.70] |
| 4 † | 3 | 0.004 | 0.667 | 0.377 | [0.08, 0.82] |
| 5 † | 7 | 0.009 | 0.429 | 0.377 | [0.13, 0.71] |
| 6 † | 5 | 0.007 | 0.600 | 0.377 | [0.11, 0.75] |
| 7 | 10 | 0.013 | 0.500 | 0.377 | [0.15, 0.67] |
| 8 † | 5 | 0.007 | 0.400 | 0.377 | [0.11, 0.75] |
| 9 † | 6 | 0.008 | 0.500 | 0.377 | [0.12, 0.73] |
| 10 | 10 | 0.013 | 0.400 | 0.377 | [0.15, 0.67] |
| 11 | 10 | 0.013 | 0.100 | 0.377 | [0.15, 0.67] |
| 12 † | 5 | 0.007 | 0.800 | 0.377 | [0.11, 0.75] |
| 13 † | 9 | 0.012 | 0.222 | 0.377 | [0.15, 0.68] |
| 14 † | 3 | 0.004 | 1.000 | 0.378 | [0.08, 0.82] |
| 15 | 10 | 0.013 | 0.500 | 0.378 | [0.15, 0.67] |
| 16 † | 2 | 0.003 | 0.000 | 0.378 | [0.06, 0.86] |
| 17 † | 4 | 0.005 | 0.500 | 0.378 | [0.09, 0.78] |
| 18 | 11 | 0.014 | 0.273 | 0.378 | [0.16, 0.66] |
| 19 † | 5 | 0.007 | 0.800 | 0.378 | [0.11, 0.75] |
| 20 † | 3 | 0.004 | 0.333 | 0.378 | [0.08, 0.82] |
| 21 † | 7 | 0.009 | 0.714 | 0.378 | [0.13, 0.71] |
| 22 † | 5 | 0.007 | 0.600 | 0.378 | [0.11, 0.75] |
| 23 † | 9 | 0.012 | 0.778 | 0.378 | [0.15, 0.68] |
| 24 † | 3 | 0.004 | 0.667 | 0.378 | [0.08, 0.82] |
| 25 † | 7 | 0.009 | 0.571 | 0.378 | [0.13, 0.71] |
| 26 † | 4 | 0.005 | 0.500 | 0.378 | [0.09, 0.78] |
| 27 † | 8 | 0.010 | 0.750 | 0.378 | [0.14, 0.70] |
| 28 † | 4 | 0.005 | 0.500 | 0.378 | [0.09, 0.78] |
| 29 † | 3 | 0.004 | 0.667 | 0.378 | [0.08, 0.82] |
| 30 † | 6 | 0.008 | 0.667 | 0.378 | [0.12, 0.73] |
| 31 † | 3 | 0.004 | 0.333 | 0.378 | [0.08, 0.82] |
| 32 † | 5 | 0.007 | 0.400 | 0.378 | [0.11, 0.75] |
| 33 † | 5 | 0.007 | 0.400 | 0.378 | [0.11, 0.75] |
| 34 † | 9 | 0.012 | 0.111 | 0.378 | [0.15, 0.68] |
| 35 † | 5 | 0.007 | 1.000 | 0.378 | [0.11, 0.75] |
| 36 † | 6 | 0.008 | 0.500 | 0.378 | [0.12, 0.73] |
| 37 | 10 | 0.013 | 0.400 | 0.378 | [0.15, 0.67] |
| 38 † | 6 | 0.008 | 0.500 | 0.378 | [0.12, 0.73] |
| 39 † | 7 | 0.009 | 0.286 | 0.378 | [0.13, 0.71] |
| 40 | 11 | 0.014 | 0.273 | 0.378 | [0.16, 0.66] |
| 41 † | 7 | 0.009 | 0.286 | 0.378 | [0.13, 0.71] |
| 42 | 10 | 0.013 | 0.500 | 0.378 | [0.15, 0.67] |
| 43 | 14 | 0.018 | 0.357 | 0.378 | [0.18, 0.63] |
| 44 | 13 | 0.017 | 0.308 | 0.378 | [0.17, 0.64] |
| 45 | 15 | 0.020 | 0.400 | 0.378 | [0.18, 0.62] |
| 46 | 22 | 0.029 | 0.136 | 0.377 | [0.21, 0.58] |
| 47 | 20 | 0.026 | 0.300 | 0.377 | [0.20, 0.59] |
| 48 | 29 | 0.038 | 0.172 | 0.377 | [0.23, 0.56] |
| 49 | 32 | 0.042 | 0.312 | 0.377 | [0.23, 0.55] |
| 50 | 45 | 0.059 | 0.111 | 0.377 | [0.25, 0.52] |
| 51 | 36 | 0.047 | 0.139 | 0.377 | [0.24, 0.54] |
| 52 | 71 | 0.093 | 0.197 | 0.377 | [0.27, 0.49] |

EQL's own `cause` search settles on 18: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.38).


## How many clear viewpoints cause every object of a scene to stay graspable, adjusting for the number of objects?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.492 | [0.42, 0.56] |
| 1 | 11 | 0.014 | 0.545 | 0.526 | [0.26, 0.77] |
| 2 | 18 | 0.024 | 0.556 | 0.501 | [0.29, 0.71] |
| 3 † | 8 | 0.010 | 0.625 | 0.744 | [0.40, 0.93] |
| 4 † | 3 | 0.004 | 0.667 | 0.756 | [0.26, 0.96] |
| 5 † | 7 | 0.009 | 0.429 | 0.181 | [0.04, 0.55] |
| 6 † | 5 | 0.007 | 0.600 | 0.532 | [0.19, 0.85] |
| 7 | 10 | 0.013 | 0.500 | 0.433 | [0.19, 0.71] |
| 8 † | 5 | 0.007 | 0.400 | 0.441 | [0.14, 0.80] |
| 9 † | 6 | 0.008 | 0.500 | 0.583 | [0.24, 0.86] |
| 10 | 10 | 0.013 | 0.400 | 0.383 | [0.16, 0.67] |
| 11 | 10 | 0.013 | 0.100 | 0.077 | [0.01, 0.38] |
| 12 † | 5 | 0.007 | 0.800 | 0.842 | [0.41, 0.98] |
| 13 † | 9 | 0.012 | 0.222 | 0.263 | [0.08, 0.59] |
| 14 † | 3 | 0.004 | 1.000 | 1.000 | [0.44, 1.00] |
| 15 | 10 | 0.013 | 0.500 | 0.603 | [0.32, 0.83] |
| 16 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 17 † | 4 | 0.005 | 0.500 | 0.510 | [0.16, 0.86] |
| 18 | 11 | 0.014 | 0.273 | 0.316 | [0.12, 0.61] |
| 19 † | 5 | 0.007 | 0.800 | 0.815 | [0.39, 0.97] |
| 20 † | 3 | 0.004 | 0.333 | 0.327 | [0.06, 0.79] |
| 21 † | 7 | 0.009 | 0.714 | 0.709 | [0.35, 0.92] |
| 22 † | 5 | 0.007 | 0.600 | 0.681 | [0.29, 0.92] |
| 23 † | 9 | 0.012 | 0.778 | 0.790 | [0.46, 0.94] |
| 24 † | 3 | 0.004 | 0.667 | 0.670 | [0.21, 0.94] |
| 25 † | 7 | 0.009 | 0.571 | 0.383 | [0.13, 0.72] |
| 26 † | 4 | 0.005 | 0.500 | 0.458 | [0.13, 0.83] |
| 27 † | 8 | 0.010 | 0.750 | 0.791 | [0.45, 0.95] |
| 28 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 29 † | 3 | 0.004 | 0.667 | 0.827 | [0.31, 0.98] |
| 30 † | 6 | 0.008 | 0.667 | 0.625 | [0.27, 0.88] |
| 31 † | 3 | 0.004 | 0.333 | 0.324 | [0.06, 0.79] |
| 32 † | 5 | 0.007 | 0.400 | 0.258 | [0.06, 0.67] |
| 33 † | 5 | 0.007 | 0.400 | 0.615 | [0.24, 0.89] |
| 34 † | 9 | 0.012 | 0.111 | 0.149 | [0.03, 0.48] |
| 35 † | 5 | 0.007 | 1.000 | 1.000 | [0.57, 1.00] |
| 36 † | 6 | 0.008 | 0.500 | 0.490 | [0.18, 0.81] |
| 37 | 10 | 0.013 | 0.400 | 0.417 | [0.18, 0.70] |
| 38 † | 6 | 0.008 | 0.500 | 0.635 | [0.28, 0.89] |
| 39 † | 7 | 0.009 | 0.286 | 0.306 | [0.09, 0.66] |
| 40 | 11 | 0.014 | 0.273 | 0.334 | [0.13, 0.62] |
| 41 † | 7 | 0.009 | 0.286 | 0.147 | [0.03, 0.52] |
| 42 | 10 | 0.013 | 0.500 | 0.401 | [0.17, 0.69] |
| 43 | 14 | 0.018 | 0.357 | 0.427 | [0.21, 0.67] |
| 44 | 13 | 0.017 | 0.308 | 0.261 | [0.10, 0.53] |
| 45 | 15 | 0.020 | 0.400 | 0.366 | [0.17, 0.61] |
| 46 | 22 | 0.029 | 0.136 | 0.130 | [0.04, 0.33] |
| 47 | 20 | 0.026 | 0.300 | 0.335 | [0.17, 0.55] |
| 48 | 29 | 0.038 | 0.172 | 0.175 | [0.08, 0.35] |
| 49 | 32 | 0.042 | 0.312 | 0.272 | [0.15, 0.44] |
| 50 | 45 | 0.059 | 0.111 | 0.103 | [0.04, 0.23] |
| 51 | 36 | 0.047 | 0.139 | 0.128 | [0.05, 0.27] |
| 52 | 71 | 0.093 | 0.197 | 0.171 | [0.10, 0.28] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.49).

### hybrid circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.480 | [0.41, 0.55] |
| 1 | 11 | 0.014 | 0.545 | 0.532 | [0.27, 0.78] |
| 2 | 18 | 0.024 | 0.556 | 0.501 | [0.29, 0.71] |
| 3 † | 8 | 0.010 | 0.625 | 0.743 | [0.40, 0.93] |
| 4 † | 3 | 0.004 | 0.667 | 0.756 | [0.26, 0.96] |
| 5 † | 7 | 0.009 | 0.429 | 0.184 | [0.04, 0.55] |
| 6 † | 5 | 0.007 | 0.600 | 0.526 | [0.19, 0.84] |
| 7 | 10 | 0.013 | 0.500 | 0.432 | [0.19, 0.71] |
| 8 † | 5 | 0.007 | 0.400 | 0.439 | [0.14, 0.79] |
| 9 † | 6 | 0.008 | 0.500 | 0.581 | [0.24, 0.86] |
| 10 | 10 | 0.013 | 0.400 | 0.389 | [0.16, 0.68] |
| 11 | 10 | 0.013 | 0.100 | 0.076 | [0.01, 0.38] |
| 12 † | 5 | 0.007 | 0.800 | 0.835 | [0.40, 0.97] |
| 13 † | 9 | 0.012 | 0.222 | 0.263 | [0.08, 0.58] |
| 14 † | 3 | 0.004 | 1.000 | 1.000 | [0.44, 1.00] |
| 15 | 10 | 0.013 | 0.500 | 0.600 | [0.31, 0.83] |
| 16 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 17 † | 4 | 0.005 | 0.500 | 0.510 | [0.16, 0.86] |
| 18 | 11 | 0.014 | 0.273 | 0.319 | [0.12, 0.61] |
| 19 † | 5 | 0.007 | 0.800 | 0.823 | [0.39, 0.97] |
| 20 † | 3 | 0.004 | 0.333 | 0.338 | [0.06, 0.79] |
| 21 † | 7 | 0.009 | 0.714 | 0.707 | [0.35, 0.91] |
| 22 † | 5 | 0.007 | 0.600 | 0.677 | [0.28, 0.92] |
| 23 † | 9 | 0.012 | 0.778 | 0.789 | [0.46, 0.94] |
| 24 † | 3 | 0.004 | 0.667 | 0.672 | [0.21, 0.94] |
| 25 † | 7 | 0.009 | 0.571 | 0.380 | [0.13, 0.71] |
| 26 † | 4 | 0.005 | 0.500 | 0.453 | [0.13, 0.83] |
| 27 † | 8 | 0.010 | 0.750 | 0.795 | [0.45, 0.95] |
| 28 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 29 † | 3 | 0.004 | 0.667 | 0.819 | [0.30, 0.98] |
| 30 † | 6 | 0.008 | 0.667 | 0.616 | [0.26, 0.88] |
| 31 † | 3 | 0.004 | 0.333 | 0.315 | [0.06, 0.78] |
| 32 † | 5 | 0.007 | 0.400 | 0.258 | [0.06, 0.67] |
| 33 † | 5 | 0.007 | 0.400 | 0.602 | [0.23, 0.88] |
| 34 † | 9 | 0.012 | 0.111 | 0.148 | [0.03, 0.47] |
| 35 † | 5 | 0.007 | 1.000 | 1.000 | [0.57, 1.00] |
| 36 † | 6 | 0.008 | 0.500 | 0.482 | [0.18, 0.80] |
| 37 | 10 | 0.013 | 0.400 | 0.427 | [0.19, 0.71] |
| 38 † | 6 | 0.008 | 0.500 | 0.627 | [0.27, 0.88] |
| 39 † | 7 | 0.009 | 0.286 | 0.302 | [0.09, 0.65] |
| 40 | 11 | 0.014 | 0.273 | 0.336 | [0.13, 0.62] |
| 41 † | 7 | 0.009 | 0.286 | 0.144 | [0.03, 0.51] |
| 42 | 10 | 0.013 | 0.500 | 0.404 | [0.17, 0.69] |
| 43 | 14 | 0.018 | 0.357 | 0.427 | [0.21, 0.67] |
| 44 | 13 | 0.017 | 0.308 | 0.254 | [0.10, 0.53] |
| 45 | 15 | 0.020 | 0.400 | 0.365 | [0.17, 0.61] |
| 46 | 22 | 0.029 | 0.136 | 0.106 | [0.03, 0.30] |
| 47 | 20 | 0.026 | 0.300 | 0.330 | [0.17, 0.55] |
| 48 | 29 | 0.038 | 0.172 | 0.142 | [0.06, 0.31] |
| 49 | 32 | 0.042 | 0.312 | 0.318 | [0.18, 0.49] |
| 50 | 45 | 0.059 | 0.111 | 0.117 | [0.05, 0.24] |
| 51 | 36 | 0.047 | 0.139 | 0.128 | [0.05, 0.27] |
| 52 | 71 | 0.093 | 0.197 | 0.184 | [0.11, 0.29] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.48).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.492 | [0.42, 0.56] |
| 1 | 11 | 0.014 | 0.545 | 0.532 | [0.27, 0.78] |
| 2 | 18 | 0.024 | 0.556 | 0.501 | [0.29, 0.71] |
| 3 † | 8 | 0.010 | 0.625 | 0.743 | [0.40, 0.93] |
| 4 † | 3 | 0.004 | 0.667 | 0.756 | [0.26, 0.96] |
| 5 † | 7 | 0.009 | 0.429 | 0.184 | [0.04, 0.55] |
| 6 † | 5 | 0.007 | 0.600 | 0.526 | [0.19, 0.84] |
| 7 | 10 | 0.013 | 0.500 | 0.432 | [0.19, 0.71] |
| 8 † | 5 | 0.007 | 0.400 | 0.439 | [0.14, 0.79] |
| 9 † | 6 | 0.008 | 0.500 | 0.581 | [0.24, 0.86] |
| 10 | 10 | 0.013 | 0.400 | 0.389 | [0.16, 0.68] |
| 11 | 10 | 0.013 | 0.100 | 0.076 | [0.01, 0.38] |
| 12 † | 5 | 0.007 | 0.800 | 0.835 | [0.40, 0.97] |
| 13 † | 9 | 0.012 | 0.222 | 0.263 | [0.08, 0.58] |
| 14 † | 3 | 0.004 | 1.000 | 1.000 | [0.44, 1.00] |
| 15 | 10 | 0.013 | 0.500 | 0.600 | [0.31, 0.83] |
| 16 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 17 † | 4 | 0.005 | 0.500 | 0.510 | [0.16, 0.86] |
| 18 | 11 | 0.014 | 0.273 | 0.319 | [0.12, 0.61] |
| 19 † | 5 | 0.007 | 0.800 | 0.823 | [0.39, 0.97] |
| 20 † | 3 | 0.004 | 0.333 | 0.338 | [0.06, 0.79] |
| 21 † | 7 | 0.009 | 0.714 | 0.707 | [0.35, 0.91] |
| 22 † | 5 | 0.007 | 0.600 | 0.677 | [0.28, 0.92] |
| 23 † | 9 | 0.012 | 0.778 | 0.789 | [0.46, 0.94] |
| 24 † | 3 | 0.004 | 0.667 | 0.672 | [0.21, 0.94] |
| 25 † | 7 | 0.009 | 0.571 | 0.380 | [0.13, 0.71] |
| 26 † | 4 | 0.005 | 0.500 | 0.453 | [0.13, 0.83] |
| 27 † | 8 | 0.010 | 0.750 | 0.795 | [0.45, 0.95] |
| 28 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 29 † | 3 | 0.004 | 0.667 | 0.819 | [0.30, 0.98] |
| 30 † | 6 | 0.008 | 0.667 | 0.616 | [0.26, 0.88] |
| 31 † | 3 | 0.004 | 0.333 | 0.315 | [0.06, 0.78] |
| 32 † | 5 | 0.007 | 0.400 | 0.258 | [0.06, 0.67] |
| 33 † | 5 | 0.007 | 0.400 | 0.602 | [0.23, 0.88] |
| 34 † | 9 | 0.012 | 0.111 | 0.148 | [0.03, 0.47] |
| 35 † | 5 | 0.007 | 1.000 | 1.000 | [0.57, 1.00] |
| 36 † | 6 | 0.008 | 0.500 | 0.482 | [0.18, 0.80] |
| 37 | 10 | 0.013 | 0.400 | 0.427 | [0.19, 0.71] |
| 38 † | 6 | 0.008 | 0.500 | 0.627 | [0.27, 0.88] |
| 39 † | 7 | 0.009 | 0.286 | 0.302 | [0.09, 0.65] |
| 40 | 11 | 0.014 | 0.273 | 0.336 | [0.13, 0.62] |
| 41 † | 7 | 0.009 | 0.286 | 0.144 | [0.03, 0.51] |
| 42 | 10 | 0.013 | 0.500 | 0.404 | [0.17, 0.69] |
| 43 | 14 | 0.018 | 0.357 | 0.427 | [0.21, 0.67] |
| 44 | 13 | 0.017 | 0.308 | 0.254 | [0.10, 0.53] |
| 45 | 15 | 0.020 | 0.400 | 0.365 | [0.17, 0.61] |
| 46 | 22 | 0.029 | 0.136 | 0.128 | [0.04, 0.32] |
| 47 | 20 | 0.026 | 0.300 | 0.330 | [0.17, 0.55] |
| 48 | 29 | 0.038 | 0.172 | 0.178 | [0.08, 0.35] |
| 49 | 32 | 0.042 | 0.312 | 0.277 | [0.15, 0.45] |
| 50 | 45 | 0.059 | 0.111 | 0.104 | [0.04, 0.23] |
| 51 | 36 | 0.047 | 0.139 | 0.131 | [0.06, 0.28] |
| 52 | 71 | 0.093 | 0.197 | 0.171 | [0.10, 0.27] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.49).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.482 | [0.41, 0.55] |
| 1 | 11 | 0.014 | 0.545 | 0.532 | [0.27, 0.78] |
| 2 | 18 | 0.024 | 0.556 | 0.501 | [0.29, 0.71] |
| 3 † | 8 | 0.010 | 0.625 | 0.743 | [0.40, 0.93] |
| 4 † | 3 | 0.004 | 0.667 | 0.756 | [0.26, 0.96] |
| 5 † | 7 | 0.009 | 0.429 | 0.184 | [0.04, 0.55] |
| 6 † | 5 | 0.007 | 0.600 | 0.526 | [0.19, 0.84] |
| 7 | 10 | 0.013 | 0.500 | 0.432 | [0.19, 0.71] |
| 8 † | 5 | 0.007 | 0.400 | 0.439 | [0.14, 0.79] |
| 9 † | 6 | 0.008 | 0.500 | 0.581 | [0.24, 0.86] |
| 10 | 10 | 0.013 | 0.400 | 0.389 | [0.16, 0.68] |
| 11 | 10 | 0.013 | 0.100 | 0.076 | [0.01, 0.38] |
| 12 † | 5 | 0.007 | 0.800 | 0.835 | [0.40, 0.97] |
| 13 † | 9 | 0.012 | 0.222 | 0.263 | [0.08, 0.58] |
| 14 † | 3 | 0.004 | 1.000 | 1.000 | [0.44, 1.00] |
| 15 | 10 | 0.013 | 0.500 | 0.600 | [0.31, 0.83] |
| 16 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 17 † | 4 | 0.005 | 0.500 | 0.510 | [0.16, 0.86] |
| 18 | 11 | 0.014 | 0.273 | 0.319 | [0.12, 0.61] |
| 19 † | 5 | 0.007 | 0.800 | 0.823 | [0.39, 0.97] |
| 20 † | 3 | 0.004 | 0.333 | 0.338 | [0.06, 0.79] |
| 21 † | 7 | 0.009 | 0.714 | 0.707 | [0.35, 0.91] |
| 22 † | 5 | 0.007 | 0.600 | 0.677 | [0.28, 0.92] |
| 23 † | 9 | 0.012 | 0.778 | 0.789 | [0.46, 0.94] |
| 24 † | 3 | 0.004 | 0.667 | 0.672 | [0.21, 0.94] |
| 25 † | 7 | 0.009 | 0.571 | 0.380 | [0.13, 0.71] |
| 26 † | 4 | 0.005 | 0.500 | 0.453 | [0.13, 0.83] |
| 27 † | 8 | 0.010 | 0.750 | 0.795 | [0.45, 0.95] |
| 28 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 29 † | 3 | 0.004 | 0.667 | 0.819 | [0.30, 0.98] |
| 30 † | 6 | 0.008 | 0.667 | 0.616 | [0.26, 0.88] |
| 31 † | 3 | 0.004 | 0.333 | 0.315 | [0.06, 0.78] |
| 32 † | 5 | 0.007 | 0.400 | 0.258 | [0.06, 0.67] |
| 33 † | 5 | 0.007 | 0.400 | 0.602 | [0.23, 0.88] |
| 34 † | 9 | 0.012 | 0.111 | 0.148 | [0.03, 0.47] |
| 35 † | 5 | 0.007 | 1.000 | 1.000 | [0.57, 1.00] |
| 36 † | 6 | 0.008 | 0.500 | 0.482 | [0.18, 0.80] |
| 37 | 10 | 0.013 | 0.400 | 0.427 | [0.19, 0.71] |
| 38 † | 6 | 0.008 | 0.500 | 0.627 | [0.27, 0.88] |
| 39 † | 7 | 0.009 | 0.286 | 0.302 | [0.09, 0.65] |
| 40 | 11 | 0.014 | 0.273 | 0.336 | [0.13, 0.62] |
| 41 † | 7 | 0.009 | 0.286 | 0.144 | [0.03, 0.51] |
| 42 | 10 | 0.013 | 0.500 | 0.404 | [0.17, 0.69] |
| 43 | 14 | 0.018 | 0.357 | 0.427 | [0.21, 0.67] |
| 44 | 13 | 0.017 | 0.308 | 0.254 | [0.10, 0.53] |
| 45 | 15 | 0.020 | 0.400 | 0.365 | [0.17, 0.61] |
| 46 | 22 | 0.029 | 0.136 | 0.125 | [0.04, 0.32] |
| 47 | 20 | 0.026 | 0.300 | 0.330 | [0.17, 0.55] |
| 48 | 29 | 0.038 | 0.172 | 0.148 | [0.06, 0.32] |
| 49 | 32 | 0.042 | 0.312 | 0.313 | [0.18, 0.49] |
| 50 | 45 | 0.059 | 0.111 | 0.116 | [0.05, 0.24] |
| 51 | 36 | 0.047 | 0.139 | 0.129 | [0.05, 0.27] |
| 52 | 71 | 0.093 | 0.197 | 0.170 | [0.10, 0.27] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.48).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.552 | [0.48, 0.62] |
| 1 | 11 | 0.014 | 0.545 | 0.545 | [0.28, 0.79] |
| 2 | 18 | 0.024 | 0.556 | 0.538 | [0.32, 0.74] |
| 3 † | 8 | 0.010 | 0.625 | 0.531 | [0.24, 0.81] |
| 4 † | 3 | 0.004 | 0.667 | 0.525 | [0.14, 0.89] |
| 5 † | 7 | 0.009 | 0.429 | 0.518 | [0.21, 0.81] |
| 6 † | 5 | 0.007 | 0.600 | 0.511 | [0.18, 0.84] |
| 7 | 10 | 0.013 | 0.500 | 0.504 | [0.24, 0.77] |
| 8 † | 5 | 0.007 | 0.400 | 0.497 | [0.17, 0.83] |
| 9 † | 6 | 0.008 | 0.500 | 0.490 | [0.18, 0.81] |
| 10 | 10 | 0.013 | 0.400 | 0.483 | [0.22, 0.75] |
| 11 | 10 | 0.013 | 0.100 | 0.476 | [0.22, 0.75] |
| 12 † | 5 | 0.007 | 0.800 | 0.469 | [0.15, 0.81] |
| 13 † | 9 | 0.012 | 0.222 | 0.462 | [0.20, 0.75] |
| 14 † | 3 | 0.004 | 1.000 | 0.456 | [0.11, 0.85] |
| 15 | 10 | 0.013 | 0.500 | 0.449 | [0.20, 0.73] |
| 16 † | 2 | 0.003 | 0.000 | 0.442 | [0.08, 0.88] |
| 17 † | 4 | 0.005 | 0.500 | 0.435 | [0.12, 0.82] |
| 18 | 11 | 0.014 | 0.273 | 0.428 | [0.19, 0.70] |
| 19 † | 5 | 0.007 | 0.800 | 0.422 | [0.13, 0.78] |
| 20 † | 3 | 0.004 | 0.333 | 0.415 | [0.09, 0.83] |
| 21 † | 7 | 0.009 | 0.714 | 0.408 | [0.15, 0.74] |
| 22 † | 5 | 0.007 | 0.600 | 0.401 | [0.12, 0.77] |
| 23 † | 9 | 0.012 | 0.778 | 0.395 | [0.16, 0.70] |
| 24 † | 3 | 0.004 | 0.667 | 0.388 | [0.08, 0.82] |
| 25 † | 7 | 0.009 | 0.571 | 0.382 | [0.13, 0.72] |
| 26 † | 4 | 0.005 | 0.500 | 0.375 | [0.09, 0.78] |
| 27 † | 8 | 0.010 | 0.750 | 0.369 | [0.13, 0.69] |
| 28 † | 4 | 0.005 | 0.500 | 0.362 | [0.09, 0.77] |
| 29 † | 3 | 0.004 | 0.667 | 0.356 | [0.07, 0.80] |
| 30 † | 6 | 0.008 | 0.667 | 0.350 | [0.10, 0.71] |
| 31 † | 3 | 0.004 | 0.333 | 0.343 | [0.06, 0.80] |
| 32 † | 5 | 0.007 | 0.400 | 0.337 | [0.09, 0.73] |
| 33 † | 5 | 0.007 | 0.400 | 0.331 | [0.09, 0.72] |
| 34 † | 9 | 0.012 | 0.111 | 0.325 | [0.12, 0.64] |
| 35 † | 5 | 0.007 | 1.000 | 0.319 | [0.08, 0.71] |
| 36 † | 6 | 0.008 | 0.500 | 0.313 | [0.09, 0.68] |
| 37 | 10 | 0.013 | 0.400 | 0.307 | [0.11, 0.61] |
| 38 † | 6 | 0.008 | 0.500 | 0.301 | [0.08, 0.68] |
| 39 † | 7 | 0.009 | 0.286 | 0.295 | [0.09, 0.65] |
| 40 | 11 | 0.014 | 0.273 | 0.289 | [0.11, 0.58] |
| 41 † | 7 | 0.009 | 0.286 | 0.284 | [0.08, 0.64] |
| 42 | 10 | 0.013 | 0.500 | 0.278 | [0.10, 0.58] |
| 43 | 14 | 0.018 | 0.357 | 0.272 | [0.11, 0.53] |
| 44 | 13 | 0.017 | 0.308 | 0.267 | [0.10, 0.54] |
| 45 | 15 | 0.020 | 0.400 | 0.262 | [0.11, 0.51] |
| 46 | 22 | 0.029 | 0.136 | 0.256 | [0.12, 0.46] |
| 47 | 20 | 0.026 | 0.300 | 0.251 | [0.11, 0.47] |
| 48 | 29 | 0.038 | 0.172 | 0.246 | [0.13, 0.43] |
| 49 | 32 | 0.042 | 0.312 | 0.241 | [0.13, 0.41] |
| 50 | 45 | 0.059 | 0.111 | 0.236 | [0.14, 0.38] |
| 51 | 36 | 0.047 | 0.139 | 0.231 | [0.12, 0.39] |
| 52 | 71 | 0.093 | 0.197 | 0.226 | [0.14, 0.34] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.55).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.378 | [0.31, 0.45] |
| 1 | 11 | 0.014 | 0.545 | 0.378 | [0.16, 0.66] |
| 2 | 18 | 0.024 | 0.556 | 0.378 | [0.19, 0.60] |
| 3 † | 8 | 0.010 | 0.625 | 0.377 | [0.14, 0.70] |
| 4 † | 3 | 0.004 | 0.667 | 0.377 | [0.08, 0.82] |
| 5 † | 7 | 0.009 | 0.429 | 0.377 | [0.13, 0.71] |
| 6 † | 5 | 0.007 | 0.600 | 0.377 | [0.11, 0.75] |
| 7 | 10 | 0.013 | 0.500 | 0.377 | [0.15, 0.67] |
| 8 † | 5 | 0.007 | 0.400 | 0.377 | [0.11, 0.75] |
| 9 † | 6 | 0.008 | 0.500 | 0.377 | [0.12, 0.73] |
| 10 | 10 | 0.013 | 0.400 | 0.377 | [0.15, 0.67] |
| 11 | 10 | 0.013 | 0.100 | 0.377 | [0.15, 0.67] |
| 12 † | 5 | 0.007 | 0.800 | 0.377 | [0.11, 0.75] |
| 13 † | 9 | 0.012 | 0.222 | 0.378 | [0.15, 0.68] |
| 14 † | 3 | 0.004 | 1.000 | 0.378 | [0.08, 0.82] |
| 15 | 10 | 0.013 | 0.500 | 0.378 | [0.15, 0.67] |
| 16 † | 2 | 0.003 | 0.000 | 0.378 | [0.06, 0.86] |
| 17 † | 4 | 0.005 | 0.500 | 0.378 | [0.09, 0.78] |
| 18 | 11 | 0.014 | 0.273 | 0.378 | [0.16, 0.66] |
| 19 † | 5 | 0.007 | 0.800 | 0.378 | [0.11, 0.75] |
| 20 † | 3 | 0.004 | 0.333 | 0.378 | [0.08, 0.82] |
| 21 † | 7 | 0.009 | 0.714 | 0.378 | [0.13, 0.71] |
| 22 † | 5 | 0.007 | 0.600 | 0.378 | [0.11, 0.75] |
| 23 † | 9 | 0.012 | 0.778 | 0.378 | [0.15, 0.68] |
| 24 † | 3 | 0.004 | 0.667 | 0.378 | [0.08, 0.82] |
| 25 † | 7 | 0.009 | 0.571 | 0.378 | [0.13, 0.71] |
| 26 † | 4 | 0.005 | 0.500 | 0.378 | [0.09, 0.78] |
| 27 † | 8 | 0.010 | 0.750 | 0.378 | [0.14, 0.70] |
| 28 † | 4 | 0.005 | 0.500 | 0.378 | [0.09, 0.78] |
| 29 † | 3 | 0.004 | 0.667 | 0.378 | [0.08, 0.82] |
| 30 † | 6 | 0.008 | 0.667 | 0.378 | [0.12, 0.73] |
| 31 † | 3 | 0.004 | 0.333 | 0.378 | [0.08, 0.82] |
| 32 † | 5 | 0.007 | 0.400 | 0.378 | [0.11, 0.75] |
| 33 † | 5 | 0.007 | 0.400 | 0.378 | [0.11, 0.75] |
| 34 † | 9 | 0.012 | 0.111 | 0.378 | [0.15, 0.68] |
| 35 † | 5 | 0.007 | 1.000 | 0.378 | [0.11, 0.75] |
| 36 † | 6 | 0.008 | 0.500 | 0.378 | [0.12, 0.73] |
| 37 | 10 | 0.013 | 0.400 | 0.378 | [0.15, 0.67] |
| 38 † | 6 | 0.008 | 0.500 | 0.378 | [0.12, 0.73] |
| 39 † | 7 | 0.009 | 0.286 | 0.378 | [0.13, 0.71] |
| 40 | 11 | 0.014 | 0.273 | 0.378 | [0.16, 0.66] |
| 41 † | 7 | 0.009 | 0.286 | 0.378 | [0.13, 0.71] |
| 42 | 10 | 0.013 | 0.500 | 0.378 | [0.15, 0.67] |
| 43 | 14 | 0.018 | 0.357 | 0.378 | [0.18, 0.63] |
| 44 | 13 | 0.017 | 0.308 | 0.378 | [0.17, 0.64] |
| 45 | 15 | 0.020 | 0.400 | 0.378 | [0.18, 0.62] |
| 46 | 22 | 0.029 | 0.136 | 0.378 | [0.21, 0.58] |
| 47 | 20 | 0.026 | 0.300 | 0.378 | [0.20, 0.59] |
| 48 | 29 | 0.038 | 0.172 | 0.378 | [0.23, 0.56] |
| 49 | 32 | 0.042 | 0.312 | 0.378 | [0.23, 0.55] |
| 50 | 45 | 0.059 | 0.111 | 0.377 | [0.25, 0.52] |
| 51 | 36 | 0.047 | 0.139 | 0.377 | [0.24, 0.54] |
| 52 | 71 | 0.093 | 0.197 | 0.377 | [0.27, 0.49] |

EQL's own `cause` search settles on 37: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.38).


## How many clear viewpoints cause every object of a scene to stay graspable, adjusting for how far the clutter is spread out and the number of objects?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.487 | [0.42, 0.56] |
| 1 | 11 | 0.014 | 0.545 | 0.521 | [0.26, 0.77] |
| 2 | 18 | 0.024 | 0.556 | 0.494 | [0.29, 0.70] |
| 3 † | 8 | 0.010 | 0.625 | 0.593 | [0.28, 0.84] |
| 4 † | 3 | 0.004 | 0.667 | 0.764 | [0.26, 0.97] |
| 5 † | 7 | 0.009 | 0.429 | 0.131 | [0.02, 0.50] |
| 6 † | 5 | 0.007 | 0.600 | 0.672 | [0.28, 0.92] |
| 7 | 10 | 0.013 | 0.500 | 0.370 | [0.15, 0.66] |
| 8 † | 5 | 0.007 | 0.400 | 0.317 | [0.08, 0.71] |
| 9 † | 6 | 0.008 | 0.500 | 0.612 | [0.26, 0.88] |
| 10 | 10 | 0.013 | 0.400 | 0.298 | [0.11, 0.60] |
| 11 | 10 | 0.013 | 0.100 | 0.138 | [0.03, 0.45] |
| 12 † | 5 | 0.007 | 0.800 | 1.000 | [0.57, 1.00] |
| 13 † | 9 | 0.012 | 0.222 | 0.191 | [0.05, 0.52] |
| 14 † | 3 | 0.004 | 1.000 | 1.000 | [0.44, 1.00] |
| 15 | 10 | 0.013 | 0.500 | 0.525 | [0.25, 0.78] |
| 16 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 17 † | 4 | 0.005 | 0.500 | 0.606 | [0.21, 0.90] |
| 18 | 11 | 0.014 | 0.273 | 0.186 | [0.05, 0.48] |
| 19 † | 5 | 0.007 | 0.800 | 0.259 | [0.06, 0.67] |
| 20 † | 3 | 0.004 | 0.333 | 0.323 | [0.06, 0.79] |
| 21 † | 7 | 0.009 | 0.714 | 0.544 | [0.23, 0.83] |
| 22 † | 5 | 0.007 | 0.600 | 0.787 | [0.36, 0.96] |
| 23 † | 9 | 0.012 | 0.778 | 0.816 | [0.49, 0.95] |
| 24 † | 3 | 0.004 | 0.667 | 0.662 | [0.20, 0.94] |
| 25 † | 7 | 0.009 | 0.571 | 0.372 | [0.13, 0.71] |
| 26 † | 4 | 0.005 | 0.500 | 0.559 | [0.18, 0.88] |
| 27 † | 8 | 0.010 | 0.750 | 0.739 | [0.40, 0.92] |
| 28 † | 4 | 0.005 | 0.500 | 0.567 | [0.19, 0.88] |
| 29 † | 3 | 0.004 | 0.667 | 0.748 | [0.25, 0.96] |
| 30 † | 6 | 0.008 | 0.667 | 0.603 | [0.25, 0.87] |
| 31 † | 3 | 0.004 | 0.333 | 0.415 | [0.09, 0.83] |
| 32 † | 5 | 0.007 | 0.400 | 0.275 | [0.06, 0.68] |
| 33 † | 5 | 0.007 | 0.400 | 0.574 | [0.21, 0.87] |
| 34 † | 9 | 0.012 | 0.111 | 0.164 | [0.04, 0.49] |
| 35 † | 5 | 0.007 | 1.000 | 1.000 | [0.57, 1.00] |
| 36 † | 6 | 0.008 | 0.500 | 0.214 | [0.05, 0.60] |
| 37 | 10 | 0.013 | 0.400 | 0.378 | [0.15, 0.67] |
| 38 † | 6 | 0.008 | 0.500 | 0.743 | [0.36, 0.94] |
| 39 † | 7 | 0.009 | 0.286 | 0.310 | [0.09, 0.66] |
| 40 | 11 | 0.014 | 0.273 | 0.290 | [0.11, 0.58] |
| 41 † | 7 | 0.009 | 0.286 | 0.229 | [0.06, 0.59] |
| 42 | 10 | 0.013 | 0.500 | 0.186 | [0.05, 0.50] |
| 43 | 14 | 0.018 | 0.357 | 0.380 | [0.18, 0.63] |
| 44 | 13 | 0.017 | 0.308 | 0.319 | [0.13, 0.59] |
| 45 | 15 | 0.020 | 0.400 | 0.313 | [0.14, 0.56] |
| 46 | 22 | 0.029 | 0.136 | 0.152 | [0.06, 0.35] |
| 47 | 20 | 0.026 | 0.300 | 0.414 | [0.23, 0.63] |
| 48 | 29 | 0.038 | 0.172 | 0.203 | [0.10, 0.38] |
| 49 | 32 | 0.042 | 0.312 | 0.273 | [0.15, 0.44] |
| 50 | 45 | 0.059 | 0.111 | 0.114 | [0.05, 0.24] |
| 51 | 36 | 0.047 | 0.139 | 0.128 | [0.05, 0.27] |
| 52 | 71 | 0.093 | 0.197 | 0.195 | [0.12, 0.30] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.49).

### hybrid circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.478 | [0.41, 0.55] |
| 1 | 11 | 0.014 | 0.545 | 0.602 | [0.32, 0.83] |
| 2 | 18 | 0.024 | 0.556 | 0.528 | [0.31, 0.73] |
| 3 † | 8 | 0.010 | 0.625 | 0.600 | [0.29, 0.85] |
| 4 † | 3 | 0.004 | 0.667 | 0.752 | [0.26, 0.96] |
| 5 † | 7 | 0.009 | 0.429 | 0.163 | [0.03, 0.53] |
| 6 † | 5 | 0.007 | 0.600 | 0.520 | [0.18, 0.84] |
| 7 | 10 | 0.013 | 0.500 | 0.434 | [0.19, 0.71] |
| 8 † | 5 | 0.007 | 0.400 | 0.444 | [0.14, 0.80] |
| 9 † | 6 | 0.008 | 0.500 | 0.578 | [0.24, 0.86] |
| 10 | 10 | 0.013 | 0.400 | 0.390 | [0.16, 0.68] |
| 11 | 10 | 0.013 | 0.100 | 0.091 | [0.02, 0.39] |
| 12 † | 5 | 0.007 | 0.800 | 1.000 | [0.57, 1.00] |
| 13 † | 9 | 0.012 | 0.222 | 0.266 | [0.08, 0.59] |
| 14 † | 3 | 0.004 | 1.000 | 1.000 | [0.44, 1.00] |
| 15 | 10 | 0.013 | 0.500 | 0.515 | [0.25, 0.77] |
| 16 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 17 † | 4 | 0.005 | 0.500 | 0.600 | [0.20, 0.90] |
| 18 | 11 | 0.014 | 0.273 | 0.256 | [0.09, 0.55] |
| 19 † | 5 | 0.007 | 0.800 | 0.637 | [0.26, 0.90] |
| 20 † | 3 | 0.004 | 0.333 | 0.323 | [0.06, 0.79] |
| 21 † | 7 | 0.009 | 0.714 | 0.554 | [0.24, 0.83] |
| 22 † | 5 | 0.007 | 0.600 | 0.671 | [0.28, 0.92] |
| 23 † | 9 | 0.012 | 0.778 | 0.748 | [0.43, 0.92] |
| 24 † | 3 | 0.004 | 0.667 | 0.666 | [0.21, 0.94] |
| 25 † | 7 | 0.009 | 0.571 | 0.419 | [0.15, 0.74] |
| 26 † | 4 | 0.005 | 0.500 | 0.624 | [0.22, 0.91] |
| 27 † | 8 | 0.010 | 0.750 | 0.748 | [0.41, 0.93] |
| 28 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 29 † | 3 | 0.004 | 0.667 | 0.814 | [0.30, 0.98] |
| 30 † | 6 | 0.008 | 0.667 | 0.532 | [0.21, 0.83] |
| 31 † | 3 | 0.004 | 0.333 | 0.311 | [0.05, 0.78] |
| 32 † | 5 | 0.007 | 0.400 | 0.260 | [0.06, 0.67] |
| 33 † | 5 | 0.007 | 0.400 | 0.441 | [0.14, 0.79] |
| 34 † | 9 | 0.012 | 0.111 | 0.144 | [0.03, 0.47] |
| 35 † | 5 | 0.007 | 1.000 | 1.000 | [0.57, 1.00] |
| 36 † | 6 | 0.008 | 0.500 | 0.312 | [0.09, 0.68] |
| 37 | 10 | 0.013 | 0.400 | 0.418 | [0.18, 0.70] |
| 38 † | 6 | 0.008 | 0.500 | 0.629 | [0.27, 0.88] |
| 39 † | 7 | 0.009 | 0.286 | 0.307 | [0.09, 0.66] |
| 40 | 11 | 0.014 | 0.273 | 0.289 | [0.11, 0.58] |
| 41 † | 7 | 0.009 | 0.286 | 0.182 | [0.04, 0.55] |
| 42 | 10 | 0.013 | 0.500 | 0.255 | [0.08, 0.56] |
| 43 | 14 | 0.018 | 0.357 | 0.431 | [0.22, 0.68] |
| 44 | 13 | 0.017 | 0.308 | 0.253 | [0.09, 0.52] |
| 45 | 15 | 0.020 | 0.400 | 0.405 | [0.20, 0.65] |
| 46 | 22 | 0.029 | 0.136 | 0.109 | [0.03, 0.30] |
| 47 | 20 | 0.026 | 0.300 | 0.343 | [0.18, 0.56] |
| 48 | 29 | 0.038 | 0.172 | 0.143 | [0.06, 0.31] |
| 49 | 32 | 0.042 | 0.312 | 0.314 | [0.18, 0.49] |
| 50 | 45 | 0.059 | 0.111 | 0.121 | [0.05, 0.25] |
| 51 | 36 | 0.047 | 0.139 | 0.130 | [0.06, 0.28] |
| 52 | 71 | 0.093 | 0.197 | 0.188 | [0.11, 0.29] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.48).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.486 | [0.41, 0.56] |
| 1 | 11 | 0.014 | 0.545 | 0.528 | [0.27, 0.77] |
| 2 | 18 | 0.024 | 0.556 | 0.493 | [0.28, 0.70] |
| 3 † | 8 | 0.010 | 0.625 | 0.589 | [0.28, 0.84] |
| 4 † | 3 | 0.004 | 0.667 | 0.764 | [0.26, 0.97] |
| 5 † | 7 | 0.009 | 0.429 | 0.132 | [0.02, 0.50] |
| 6 † | 5 | 0.007 | 0.600 | 0.670 | [0.28, 0.91] |
| 7 | 10 | 0.013 | 0.500 | 0.365 | [0.15, 0.66] |
| 8 † | 5 | 0.007 | 0.400 | 0.319 | [0.08, 0.71] |
| 9 † | 6 | 0.008 | 0.500 | 0.613 | [0.26, 0.88] |
| 10 | 10 | 0.013 | 0.400 | 0.302 | [0.11, 0.60] |
| 11 | 10 | 0.013 | 0.100 | 0.138 | [0.03, 0.45] |
| 12 † | 5 | 0.007 | 0.800 | 1.000 | [0.57, 1.00] |
| 13 † | 9 | 0.012 | 0.222 | 0.193 | [0.05, 0.52] |
| 14 † | 3 | 0.004 | 1.000 | 1.000 | [0.44, 1.00] |
| 15 | 10 | 0.013 | 0.500 | 0.518 | [0.25, 0.78] |
| 16 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 17 † | 4 | 0.005 | 0.500 | 0.613 | [0.21, 0.90] |
| 18 | 11 | 0.014 | 0.273 | 0.186 | [0.05, 0.48] |
| 19 † | 5 | 0.007 | 0.800 | 0.280 | [0.06, 0.69] |
| 20 † | 3 | 0.004 | 0.333 | 0.335 | [0.06, 0.79] |
| 21 † | 7 | 0.009 | 0.714 | 0.535 | [0.23, 0.82] |
| 22 † | 5 | 0.007 | 0.600 | 0.785 | [0.36, 0.96] |
| 23 † | 9 | 0.012 | 0.778 | 0.815 | [0.49, 0.95] |
| 24 † | 3 | 0.004 | 0.667 | 0.664 | [0.21, 0.94] |
| 25 † | 7 | 0.009 | 0.571 | 0.369 | [0.12, 0.71] |
| 26 † | 4 | 0.005 | 0.500 | 0.556 | [0.18, 0.88] |
| 27 † | 8 | 0.010 | 0.750 | 0.744 | [0.40, 0.93] |
| 28 † | 4 | 0.005 | 0.500 | 0.564 | [0.18, 0.88] |
| 29 † | 3 | 0.004 | 0.667 | 0.738 | [0.25, 0.96] |
| 30 † | 6 | 0.008 | 0.667 | 0.591 | [0.25, 0.86] |
| 31 † | 3 | 0.004 | 0.333 | 0.403 | [0.09, 0.83] |
| 32 † | 5 | 0.007 | 0.400 | 0.277 | [0.06, 0.68] |
| 33 † | 5 | 0.007 | 0.400 | 0.558 | [0.20, 0.86] |
| 34 † | 9 | 0.012 | 0.111 | 0.163 | [0.04, 0.49] |
| 35 † | 5 | 0.007 | 1.000 | 1.000 | [0.57, 1.00] |
| 36 † | 6 | 0.008 | 0.500 | 0.212 | [0.05, 0.60] |
| 37 | 10 | 0.013 | 0.400 | 0.388 | [0.16, 0.68] |
| 38 † | 6 | 0.008 | 0.500 | 0.735 | [0.35, 0.93] |
| 39 † | 7 | 0.009 | 0.286 | 0.304 | [0.09, 0.66] |
| 40 | 11 | 0.014 | 0.273 | 0.294 | [0.11, 0.59] |
| 41 † | 7 | 0.009 | 0.286 | 0.230 | [0.06, 0.59] |
| 42 | 10 | 0.013 | 0.500 | 0.184 | [0.05, 0.49] |
| 43 | 14 | 0.018 | 0.357 | 0.377 | [0.18, 0.63] |
| 44 | 13 | 0.017 | 0.308 | 0.313 | [0.13, 0.58] |
| 45 | 15 | 0.020 | 0.400 | 0.312 | [0.14, 0.56] |
| 46 | 22 | 0.029 | 0.136 | 0.145 | [0.05, 0.34] |
| 47 | 20 | 0.026 | 0.300 | 0.406 | [0.22, 0.62] |
| 48 | 29 | 0.038 | 0.172 | 0.207 | [0.10, 0.38] |
| 49 | 32 | 0.042 | 0.312 | 0.276 | [0.15, 0.45] |
| 50 | 45 | 0.059 | 0.111 | 0.116 | [0.05, 0.24] |
| 51 | 36 | 0.047 | 0.139 | 0.132 | [0.06, 0.28] |
| 52 | 71 | 0.093 | 0.197 | 0.194 | [0.12, 0.30] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.49).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.482 | [0.41, 0.55] |
| 1 | 11 | 0.014 | 0.545 | 0.634 | [0.35, 0.85] |
| 2 | 18 | 0.024 | 0.556 | 0.530 | [0.32, 0.73] |
| 3 † | 8 | 0.010 | 0.625 | 0.600 | [0.29, 0.85] |
| 4 † | 3 | 0.004 | 0.667 | 0.758 | [0.26, 0.97] |
| 5 † | 7 | 0.009 | 0.429 | 0.164 | [0.03, 0.53] |
| 6 † | 5 | 0.007 | 0.600 | 0.520 | [0.18, 0.84] |
| 7 | 10 | 0.013 | 0.500 | 0.434 | [0.19, 0.71] |
| 8 † | 5 | 0.007 | 0.400 | 0.442 | [0.14, 0.80] |
| 9 † | 6 | 0.008 | 0.500 | 0.576 | [0.24, 0.86] |
| 10 | 10 | 0.013 | 0.400 | 0.391 | [0.16, 0.68] |
| 11 | 10 | 0.013 | 0.100 | 0.092 | [0.02, 0.40] |
| 12 † | 5 | 0.007 | 0.800 | 1.000 | [0.57, 1.00] |
| 13 † | 9 | 0.012 | 0.222 | 0.265 | [0.08, 0.59] |
| 14 † | 3 | 0.004 | 1.000 | 1.000 | [0.44, 1.00] |
| 15 | 10 | 0.013 | 0.500 | 0.513 | [0.25, 0.77] |
| 16 † | 2 | 0.003 | 0.000 | 0.000 | [0.00, 0.66] |
| 17 † | 4 | 0.005 | 0.500 | 0.603 | [0.21, 0.90] |
| 18 | 11 | 0.014 | 0.273 | 0.256 | [0.09, 0.55] |
| 19 † | 5 | 0.007 | 0.800 | 0.641 | [0.26, 0.90] |
| 20 † | 3 | 0.004 | 0.333 | 0.327 | [0.06, 0.79] |
| 21 † | 7 | 0.009 | 0.714 | 0.547 | [0.23, 0.83] |
| 22 † | 5 | 0.007 | 0.600 | 0.669 | [0.28, 0.91] |
| 23 † | 9 | 0.012 | 0.778 | 0.750 | [0.43, 0.92] |
| 24 † | 3 | 0.004 | 0.667 | 0.666 | [0.21, 0.94] |
| 25 † | 7 | 0.009 | 0.571 | 0.376 | [0.13, 0.71] |
| 26 † | 4 | 0.005 | 0.500 | 0.625 | [0.22, 0.91] |
| 27 † | 8 | 0.010 | 0.750 | 0.748 | [0.41, 0.93] |
| 28 † | 4 | 0.005 | 0.500 | 0.500 | [0.15, 0.85] |
| 29 † | 3 | 0.004 | 0.667 | 0.809 | [0.29, 0.98] |
| 30 † | 6 | 0.008 | 0.667 | 0.531 | [0.21, 0.83] |
| 31 † | 3 | 0.004 | 0.333 | 0.310 | [0.05, 0.78] |
| 32 † | 5 | 0.007 | 0.400 | 0.260 | [0.06, 0.67] |
| 33 † | 5 | 0.007 | 0.400 | 0.438 | [0.14, 0.79] |
| 34 † | 9 | 0.012 | 0.111 | 0.143 | [0.03, 0.47] |
| 35 † | 5 | 0.007 | 1.000 | 1.000 | [0.57, 1.00] |
| 36 † | 6 | 0.008 | 0.500 | 0.313 | [0.09, 0.68] |
| 37 | 10 | 0.013 | 0.400 | 0.421 | [0.18, 0.70] |
| 38 † | 6 | 0.008 | 0.500 | 0.626 | [0.27, 0.88] |
| 39 † | 7 | 0.009 | 0.286 | 0.308 | [0.09, 0.66] |
| 40 | 11 | 0.014 | 0.273 | 0.288 | [0.11, 0.58] |
| 41 † | 7 | 0.009 | 0.286 | 0.181 | [0.04, 0.55] |
| 42 | 10 | 0.013 | 0.500 | 0.260 | [0.09, 0.57] |
| 43 | 14 | 0.018 | 0.357 | 0.429 | [0.21, 0.67] |
| 44 | 13 | 0.017 | 0.308 | 0.257 | [0.10, 0.53] |
| 45 | 15 | 0.020 | 0.400 | 0.407 | [0.20, 0.65] |
| 46 | 22 | 0.029 | 0.136 | 0.127 | [0.04, 0.32] |
| 47 | 20 | 0.026 | 0.300 | 0.337 | [0.17, 0.56] |
| 48 | 29 | 0.038 | 0.172 | 0.150 | [0.06, 0.32] |
| 49 | 32 | 0.042 | 0.312 | 0.314 | [0.18, 0.49] |
| 50 | 45 | 0.059 | 0.111 | 0.115 | [0.05, 0.24] |
| 51 | 36 | 0.047 | 0.139 | 0.130 | [0.06, 0.28] |
| 52 | 71 | 0.093 | 0.197 | 0.171 | [0.10, 0.27] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.48).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.554 | [0.48, 0.62] |
| 1 | 11 | 0.014 | 0.545 | 0.547 | [0.28, 0.79] |
| 2 | 18 | 0.024 | 0.556 | 0.540 | [0.32, 0.74] |
| 3 † | 8 | 0.010 | 0.625 | 0.533 | [0.24, 0.81] |
| 4 † | 3 | 0.004 | 0.667 | 0.526 | [0.14, 0.89] |
| 5 † | 7 | 0.009 | 0.429 | 0.519 | [0.21, 0.81] |
| 6 † | 5 | 0.007 | 0.600 | 0.512 | [0.18, 0.84] |
| 7 | 10 | 0.013 | 0.500 | 0.505 | [0.24, 0.77] |
| 8 † | 5 | 0.007 | 0.400 | 0.498 | [0.17, 0.83] |
| 9 † | 6 | 0.008 | 0.500 | 0.491 | [0.18, 0.81] |
| 10 | 10 | 0.013 | 0.400 | 0.484 | [0.23, 0.75] |
| 11 | 10 | 0.013 | 0.100 | 0.477 | [0.22, 0.75] |
| 12 † | 5 | 0.007 | 0.800 | 0.470 | [0.15, 0.81] |
| 13 † | 9 | 0.012 | 0.222 | 0.463 | [0.20, 0.75] |
| 14 † | 3 | 0.004 | 1.000 | 0.456 | [0.11, 0.85] |
| 15 | 10 | 0.013 | 0.500 | 0.449 | [0.20, 0.73] |
| 16 † | 2 | 0.003 | 0.000 | 0.443 | [0.08, 0.88] |
| 17 † | 4 | 0.005 | 0.500 | 0.436 | [0.12, 0.82] |
| 18 | 11 | 0.014 | 0.273 | 0.429 | [0.19, 0.70] |
| 19 † | 5 | 0.007 | 0.800 | 0.422 | [0.13, 0.78] |
| 20 † | 3 | 0.004 | 0.333 | 0.415 | [0.09, 0.84] |
| 21 † | 7 | 0.009 | 0.714 | 0.408 | [0.15, 0.74] |
| 22 † | 5 | 0.007 | 0.600 | 0.402 | [0.12, 0.77] |
| 23 † | 9 | 0.012 | 0.778 | 0.395 | [0.16, 0.70] |
| 24 † | 3 | 0.004 | 0.667 | 0.388 | [0.08, 0.82] |
| 25 † | 7 | 0.009 | 0.571 | 0.382 | [0.13, 0.72] |
| 26 † | 4 | 0.005 | 0.500 | 0.375 | [0.09, 0.78] |
| 27 † | 8 | 0.010 | 0.750 | 0.368 | [0.13, 0.69] |
| 28 † | 4 | 0.005 | 0.500 | 0.362 | [0.09, 0.77] |
| 29 † | 3 | 0.004 | 0.667 | 0.355 | [0.07, 0.80] |
| 30 † | 6 | 0.008 | 0.667 | 0.349 | [0.10, 0.71] |
| 31 † | 3 | 0.004 | 0.333 | 0.343 | [0.06, 0.80] |
| 32 † | 5 | 0.007 | 0.400 | 0.336 | [0.09, 0.73] |
| 33 † | 5 | 0.007 | 0.400 | 0.330 | [0.09, 0.72] |
| 34 † | 9 | 0.012 | 0.111 | 0.324 | [0.12, 0.64] |
| 35 † | 5 | 0.007 | 1.000 | 0.318 | [0.08, 0.71] |
| 36 † | 6 | 0.008 | 0.500 | 0.312 | [0.09, 0.68] |
| 37 | 10 | 0.013 | 0.400 | 0.306 | [0.11, 0.61] |
| 38 † | 6 | 0.008 | 0.500 | 0.300 | [0.08, 0.67] |
| 39 † | 7 | 0.009 | 0.286 | 0.294 | [0.09, 0.65] |
| 40 | 11 | 0.014 | 0.273 | 0.288 | [0.11, 0.58] |
| 41 † | 7 | 0.009 | 0.286 | 0.282 | [0.08, 0.64] |
| 42 | 10 | 0.013 | 0.500 | 0.277 | [0.10, 0.58] |
| 43 | 14 | 0.018 | 0.357 | 0.271 | [0.11, 0.53] |
| 44 | 13 | 0.017 | 0.308 | 0.266 | [0.10, 0.54] |
| 45 | 15 | 0.020 | 0.400 | 0.260 | [0.10, 0.51] |
| 46 | 22 | 0.029 | 0.136 | 0.255 | [0.12, 0.46] |
| 47 | 20 | 0.026 | 0.300 | 0.249 | [0.11, 0.47] |
| 48 | 29 | 0.038 | 0.172 | 0.244 | [0.12, 0.42] |
| 49 | 32 | 0.042 | 0.312 | 0.239 | [0.12, 0.41] |
| 50 | 45 | 0.059 | 0.111 | 0.234 | [0.13, 0.38] |
| 51 | 36 | 0.047 | 0.139 | 0.229 | [0.12, 0.39] |
| 52 | 71 | 0.093 | 0.197 | 0.224 | [0.14, 0.33] |

EQL's own `cause` search settles on 0: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.55).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 | 181 | 0.237 | 0.481 | 0.378 | [0.31, 0.45] |
| 1 | 11 | 0.014 | 0.545 | 0.378 | [0.16, 0.66] |
| 2 | 18 | 0.024 | 0.556 | 0.378 | [0.19, 0.60] |
| 3 † | 8 | 0.010 | 0.625 | 0.377 | [0.14, 0.70] |
| 4 † | 3 | 0.004 | 0.667 | 0.377 | [0.08, 0.82] |
| 5 † | 7 | 0.009 | 0.429 | 0.377 | [0.13, 0.71] |
| 6 † | 5 | 0.007 | 0.600 | 0.377 | [0.11, 0.75] |
| 7 | 10 | 0.013 | 0.500 | 0.377 | [0.15, 0.67] |
| 8 † | 5 | 0.007 | 0.400 | 0.377 | [0.11, 0.75] |
| 9 † | 6 | 0.008 | 0.500 | 0.377 | [0.12, 0.73] |
| 10 | 10 | 0.013 | 0.400 | 0.377 | [0.15, 0.67] |
| 11 | 10 | 0.013 | 0.100 | 0.378 | [0.15, 0.67] |
| 12 † | 5 | 0.007 | 0.800 | 0.378 | [0.11, 0.75] |
| 13 † | 9 | 0.012 | 0.222 | 0.378 | [0.15, 0.68] |
| 14 † | 3 | 0.004 | 1.000 | 0.378 | [0.08, 0.82] |
| 15 | 10 | 0.013 | 0.500 | 0.378 | [0.15, 0.67] |
| 16 † | 2 | 0.003 | 0.000 | 0.378 | [0.06, 0.86] |
| 17 † | 4 | 0.005 | 0.500 | 0.378 | [0.09, 0.78] |
| 18 | 11 | 0.014 | 0.273 | 0.378 | [0.16, 0.66] |
| 19 † | 5 | 0.007 | 0.800 | 0.378 | [0.11, 0.76] |
| 20 † | 3 | 0.004 | 0.333 | 0.378 | [0.08, 0.82] |
| 21 † | 7 | 0.009 | 0.714 | 0.378 | [0.13, 0.71] |
| 22 † | 5 | 0.007 | 0.600 | 0.378 | [0.11, 0.76] |
| 23 † | 9 | 0.012 | 0.778 | 0.378 | [0.15, 0.68] |
| 24 † | 3 | 0.004 | 0.667 | 0.378 | [0.08, 0.82] |
| 25 † | 7 | 0.009 | 0.571 | 0.378 | [0.13, 0.71] |
| 26 † | 4 | 0.005 | 0.500 | 0.378 | [0.09, 0.78] |
| 27 † | 8 | 0.010 | 0.750 | 0.378 | [0.14, 0.70] |
| 28 † | 4 | 0.005 | 0.500 | 0.378 | [0.09, 0.78] |
| 29 † | 3 | 0.004 | 0.667 | 0.378 | [0.08, 0.82] |
| 30 † | 6 | 0.008 | 0.667 | 0.378 | [0.12, 0.73] |
| 31 † | 3 | 0.004 | 0.333 | 0.378 | [0.08, 0.82] |
| 32 † | 5 | 0.007 | 0.400 | 0.378 | [0.11, 0.75] |
| 33 † | 5 | 0.007 | 0.400 | 0.378 | [0.11, 0.75] |
| 34 † | 9 | 0.012 | 0.111 | 0.378 | [0.15, 0.68] |
| 35 † | 5 | 0.007 | 1.000 | 0.378 | [0.11, 0.75] |
| 36 † | 6 | 0.008 | 0.500 | 0.377 | [0.12, 0.73] |
| 37 | 10 | 0.013 | 0.400 | 0.377 | [0.15, 0.67] |
| 38 † | 6 | 0.008 | 0.500 | 0.377 | [0.12, 0.73] |
| 39 † | 7 | 0.009 | 0.286 | 0.377 | [0.13, 0.71] |
| 40 | 11 | 0.014 | 0.273 | 0.377 | [0.16, 0.66] |
| 41 † | 7 | 0.009 | 0.286 | 0.377 | [0.13, 0.71] |
| 42 | 10 | 0.013 | 0.500 | 0.377 | [0.15, 0.67] |
| 43 | 14 | 0.018 | 0.357 | 0.377 | [0.18, 0.63] |
| 44 | 13 | 0.017 | 0.308 | 0.377 | [0.17, 0.64] |
| 45 | 15 | 0.020 | 0.400 | 0.377 | [0.18, 0.62] |
| 46 | 22 | 0.029 | 0.136 | 0.377 | [0.21, 0.58] |
| 47 | 20 | 0.026 | 0.300 | 0.377 | [0.20, 0.59] |
| 48 | 29 | 0.038 | 0.172 | 0.377 | [0.23, 0.56] |
| 49 | 32 | 0.042 | 0.312 | 0.377 | [0.23, 0.55] |
| 50 | 45 | 0.059 | 0.111 | 0.377 | [0.25, 0.52] |
| 51 | 36 | 0.047 | 0.139 | 0.377 | [0.24, 0.54] |
| 52 | 71 | 0.093 | 0.197 | 0.377 | [0.27, 0.49] |

EQL's own `cause` search settles on 18: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.38).


## Does the object catalogue a scene is built from cause every object of it to stay graspable, adjusting for its spread (extent)?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.407 | 0.407 | [0.36, 0.46] |
| mixed | 248 | 0.169 | 0.333 | 0.333 | [0.28, 0.39] |
| ycb-video | 129 | 0.325 | 0.355 | 0.355 | [0.28, 0.44] |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.41).

### hybrid circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.407 | 0.407 | [0.36, 0.46] |
| mixed | 129 | 0.169 | 0.333 | 0.333 | [0.26, 0.42] |
| ycb-video | 248 | 0.325 | 0.355 | 0.355 | [0.30, 0.42] |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.41).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.407 | 0.407 | [0.36, 0.46] |
| mixed | 129 | 0.169 | 0.333 | 0.333 | [0.26, 0.42] |
| ycb-video | 248 | 0.325 | 0.355 | 0.355 | [0.30, 0.42] |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.41).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.407 | 0.407 | [0.36, 0.46] |
| mixed | 129 | 0.169 | 0.333 | 0.333 | [0.26, 0.42] |
| ycb-video | 248 | 0.325 | 0.355 | 0.355 | [0.30, 0.42] |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.41).

### scalars-only tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.407 | 0.407 | [0.36, 0.46] |
| mixed | 129 | 0.169 | 0.333 | 0.333 | [0.26, 0.42] |
| ycb-video | 248 | 0.325 | 0.355 | 0.355 | [0.30, 0.42] |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.41).

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.407 | 0.407 | [0.36, 0.46] |
| mixed | 129 | 0.169 | 0.333 | 0.340 | [0.26, 0.43] |
| ycb-video | 248 | 0.325 | 0.355 | 0.352 | [0.30, 0.41] |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.41).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.407 | 0.377 | [0.33, 0.43] |
| mixed | 129 | 0.169 | 0.333 | 0.377 | [0.30, 0.46] |
| ycb-video | 248 | 0.325 | 0.355 | 0.377 | [0.32, 0.44] |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.38).


## Does the object catalogue a scene is built from cause every object of it to stay graspable, adjusting for its small-object count?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.407 | 0.394 | [0.35, 0.44] |
| mixed | 248 | 0.169 | 0.333 | 0.269 | [0.22, 0.33] |
| ycb-video | 129 | 0.325 | 0.355 | 0.339 | [0.26, 0.42] |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.39).

### hybrid circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.407 | 0.398 | [0.35, 0.45] |
| mixed | 129 | 0.169 | 0.333 | 0.305 | [0.23, 0.39] |
| ycb-video | 248 | 0.325 | 0.355 | 0.357 | [0.30, 0.42] |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.40).

### propositional tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.407 | 0.398 | [0.35, 0.45] |
| mixed | 129 | 0.169 | 0.333 | 0.281 | [0.21, 0.36] |
| ycb-video | 248 | 0.325 | 0.355 | 0.340 | [0.28, 0.40] |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.40).

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.407 | 0.398 | [0.35, 0.45] |
| mixed | 129 | 0.169 | 0.333 | 0.304 | [0.23, 0.39] |
| ycb-video | 248 | 0.325 | 0.355 | 0.357 | [0.30, 0.42] |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.40).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.407 | 0.401 | [0.35, 0.45] |
| mixed | 129 | 0.169 | 0.333 | 0.340 | [0.26, 0.43] |
| ycb-video | 248 | 0.325 | 0.355 | 0.359 | [0.30, 0.42] |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.40).

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.407 | 0.377 | [0.33, 0.43] |
| mixed | 129 | 0.169 | 0.333 | 0.377 | [0.30, 0.46] |
| ycb-video | 248 | 0.325 | 0.355 | 0.377 | [0.32, 0.44] |

EQL's own `cause` search settles on ycb-video: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.38).


## Does the object catalogue a scene is built from cause object 0 of it to be heavily occluded?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.363 | 0.363 | [0.32, 0.41] |
| mixed | 248 | 0.169 | 0.335 | 0.335 | [0.28, 0.40] |
| ycb-video | 129 | 0.325 | 0.319 | 0.319 | [0.24, 0.40] |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.36).

### hybrid circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.363 | 0.363 | [0.32, 0.41] |
| mixed | 248 | 0.169 | 0.335 | 0.335 | [0.28, 0.40] |
| ycb-video | 129 | 0.325 | 0.319 | 0.319 | [0.24, 0.40] |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.36).

### propositional tree

Refused: the fitted table has no column for the queried variables.

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 386 | 0.506 | 0.277 | 0.277 | [0.23, 0.32] |
| mixed | 129 | 0.169 | 0.233 | 0.233 | [0.17, 0.31] |
| ycb-video | 248 | 0.325 | 0.286 | 0.286 | [0.23, 0.35] |

EQL's own `cause` search settles on grasp: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.28).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

Refused: the fitted table has no column for the queried variables.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| grasp | 5536 | 0.499 | 0.337 | 0.344 | [0.33, 0.36] |
| mixed | 1905 | 0.172 | 0.349 | 0.347 | [0.33, 0.37] |
| ycb-video | 3650 | 0.329 | 0.301 | 0.343 | [0.33, 0.36] |

EQL's own `cause` search settles on mixed: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.35).


## How many occluded objects cause object 0 of a scene to lose every grasp?

One row per region of the cause the model distinguishes. *n* is how many training rows the region holds, and † marks a region below the support threshold; *P(region)* is how much of the fitted population it holds; *naive P(effect)* is the effect's probability simply conditioned on the region; *adjusted* is the interventional probability after summing out the question's confounders, which is what the question asks for, and the *interval* is its Wilson interval over the region's n. Where naive and adjusted agree, the confounders carried no further information within that region.

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.067 | 0.067 | [0.00, 0.62] |
| 1 † | 4 | 0.005 | 0.073 | 0.073 | [0.00, 0.56] |
| 2 | 12 | 0.016 | 0.075 | 0.075 | [0.01, 0.34] |
| 3 | 17 | 0.022 | 0.080 | 0.080 | [0.02, 0.30] |
| 4 | 25 | 0.033 | 0.092 | 0.092 | [0.03, 0.27] |
| 5 | 25 | 0.033 | 0.098 | 0.098 | [0.03, 0.27] |
| 6 | 42 | 0.055 | 0.104 | 0.104 | [0.04, 0.23] |
| 7 | 71 | 0.093 | 0.111 | 0.111 | [0.06, 0.21] |
| 8 | 72 | 0.094 | 0.121 | 0.121 | [0.06, 0.22] |
| 9 | 87 | 0.114 | 0.124 | 0.124 | [0.07, 0.21] |
| 10 | 95 | 0.125 | 0.127 | 0.127 | [0.07, 0.21] |
| 11 | 93 | 0.122 | 0.119 | 0.119 | [0.07, 0.20] |
| 12 | 61 | 0.080 | 0.113 | 0.113 | [0.06, 0.22] |
| 13 | 51 | 0.067 | 0.097 | 0.097 | [0.04, 0.21] |
| 14 | 43 | 0.056 | 0.102 | 0.102 | [0.04, 0.23] |
| 15 | 25 | 0.033 | 0.099 | 0.099 | [0.03, 0.27] |
| 16 | 21 | 0.028 | 0.098 | 0.098 | [0.03, 0.29] |
| 17 † | 8 | 0.010 | 0.094 | 0.094 | [0.01, 0.44] |
| 18 † | 6 | 0.008 | 0.091 | 0.091 | [0.01, 0.49] |
| 19 † | 1 | 0.001 | 0.083 | 0.083 | [0.00, 0.83] |
| 20 † | 1 | 0.001 | 0.149 | 0.149 | [0.01, 0.85] |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.13).

### hybrid circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.067 | 0.067 | [0.00, 0.62] |
| 1 † | 4 | 0.005 | 0.073 | 0.073 | [0.00, 0.56] |
| 2 | 12 | 0.016 | 0.075 | 0.075 | [0.01, 0.34] |
| 3 | 17 | 0.022 | 0.080 | 0.080 | [0.02, 0.30] |
| 4 | 25 | 0.033 | 0.092 | 0.092 | [0.03, 0.27] |
| 5 | 25 | 0.033 | 0.098 | 0.098 | [0.03, 0.27] |
| 6 | 42 | 0.055 | 0.104 | 0.104 | [0.04, 0.23] |
| 7 | 71 | 0.093 | 0.111 | 0.111 | [0.06, 0.21] |
| 8 | 72 | 0.094 | 0.121 | 0.121 | [0.06, 0.22] |
| 9 | 87 | 0.114 | 0.124 | 0.124 | [0.07, 0.21] |
| 10 | 95 | 0.125 | 0.127 | 0.127 | [0.07, 0.21] |
| 11 | 93 | 0.122 | 0.119 | 0.119 | [0.07, 0.20] |
| 12 | 61 | 0.080 | 0.113 | 0.113 | [0.06, 0.22] |
| 13 | 51 | 0.067 | 0.097 | 0.097 | [0.04, 0.21] |
| 14 | 43 | 0.056 | 0.102 | 0.102 | [0.04, 0.23] |
| 15 | 25 | 0.033 | 0.099 | 0.099 | [0.03, 0.27] |
| 16 | 21 | 0.028 | 0.098 | 0.098 | [0.03, 0.29] |
| 17 † | 8 | 0.010 | 0.094 | 0.094 | [0.01, 0.44] |
| 18 † | 6 | 0.008 | 0.091 | 0.091 | [0.01, 0.49] |
| 19 † | 1 | 0.001 | 0.083 | 0.083 | [0.00, 0.83] |
| 20 † | 1 | 0.001 | 0.149 | 0.149 | [0.01, 0.85] |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.13).

### propositional tree

Refused: the fitted table has no column for the queried variables.

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| 0 † | 3 | 0.004 | 0.333 | 0.333 | [0.06, 0.79] |
| 1 † | 4 | 0.005 | 0.250 | 0.250 | [0.05, 0.70] |
| 2 | 12 | 0.016 | 0.083 | 0.083 | [0.01, 0.35] |
| 3 | 17 | 0.022 | 0.176 | 0.176 | [0.06, 0.41] |
| 4 | 25 | 0.033 | 0.160 | 0.160 | [0.06, 0.35] |
| 5 | 25 | 0.033 | 0.080 | 0.080 | [0.02, 0.25] |
| 6 | 42 | 0.055 | 0.119 | 0.119 | [0.05, 0.25] |
| 7 | 71 | 0.093 | 0.085 | 0.085 | [0.04, 0.17] |
| 8 | 72 | 0.094 | 0.125 | 0.125 | [0.07, 0.22] |
| 9 | 87 | 0.114 | 0.092 | 0.092 | [0.05, 0.17] |
| 10 | 95 | 0.125 | 0.126 | 0.126 | [0.07, 0.21] |
| 11 | 93 | 0.122 | 0.097 | 0.097 | [0.05, 0.17] |
| 12 | 61 | 0.080 | 0.098 | 0.098 | [0.05, 0.20] |
| 13 | 51 | 0.067 | 0.098 | 0.098 | [0.04, 0.21] |
| 14 | 43 | 0.056 | 0.070 | 0.070 | [0.02, 0.19] |
| 15 | 25 | 0.033 | 0.040 | 0.040 | [0.01, 0.20] |
| 16 | 21 | 0.028 | 0.048 | 0.048 | [0.01, 0.23] |
| 17 † | 8 | 0.010 | 0.000 | 0.000 | [0.00, 0.32] |
| 18 † | 6 | 0.008 | 0.000 | 0.000 | [0.00, 0.39] |
| 19 † | 1 | 0.001 | 0.000 | 0.000 | [0.00, 0.79] |
| 20 † | 1 | 0.001 | 0.000 | 0.000 | [0.00, 0.79] |

EQL's own `cause` search settles on 10: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.13).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

Refused: the fitted table has no column for the queried variables.

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

### relational circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| large | 3602 | 0.325 | 0.103 | 0.103 | [0.09, 0.11] |
| medium | 3578 | 0.323 | 0.108 | 0.108 | [0.10, 0.12] |
| small | 3911 | 0.353 | 0.122 | 0.122 | [0.11, 0.13] |

EQL's own `cause` search settles on small: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.12).

### hybrid circuit

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| large | 3602 | 0.325 | 0.103 | 0.103 | [0.09, 0.11] |
| medium | 3578 | 0.323 | 0.108 | 0.108 | [0.10, 0.12] |
| small | 3911 | 0.353 | 0.122 | 0.122 | [0.11, 0.13] |

EQL's own `cause` search settles on small: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.12).

### propositional tree

Refused: the fitted table has no column for the queried variables.

### unrolled tree

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| large | 178 | 0.233 | 0.152 | 0.152 | [0.11, 0.21] |
| medium | 307 | 0.402 | 0.094 | 0.094 | [0.07, 0.13] |
| small | 278 | 0.364 | 0.076 | 0.076 | [0.05, 0.11] |

EQL's own `cause` search settles on medium: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.09).

### scalars-only tree

Refused: the fitted table has no column for the queried variables.

### regression adjustment

Refused: the fitted table has no column for the queried variables.

### neural adjustment

| cause region | n | P(region) | naive P(effect) | adjusted P(effect | do(cause)) | 95% interval |
|---|---|---|---|---|---|
| large | 3602 | 0.325 | 0.103 | 0.156 | [0.14, 0.17] |
| medium | 3578 | 0.323 | 0.108 | 0.158 | [0.15, 0.17] |
| small | 3911 | 0.353 | 0.122 | 0.173 | [0.16, 0.18] |

EQL's own `cause` search settles on small: the region most probable once the effect is required to hold, from which the query's samples are drawn (P(effect | do) = 0.17).

