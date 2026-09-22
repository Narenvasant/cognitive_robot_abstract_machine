# Three datasets, one pipeline

The same comparison, run the same way, on three relational datasets: the CTU
Mutagenesis molecules (188 molecules, atoms and bonds as parts), Tracy's clutter-picking
attempts (300 recorded attempts, neighbouring cartons as parts) and GraspClutter6D (954
real cluttered scenes, object instances and camera frames as parts). Every number below
is read off the three `results.md` files, each written by
`experiments.causal_reasoning.comparison` from the same pipelines, the same studies and
the same report; only the domain, the data and the questions are each dataset's own.

The pipelines are the same everywhere: a **relational circuit** (an RSPN, grounded per
query and registered as a causal circuit), a **propositional tree** (a JPT on the
example's scalars and its aggregation counts), an **unrolled tree** (the same tree with
every part's attributes under the part's position), a **scalars-only tree**, and
**regression adjustment** (a logistic regression on the propositional table, the
backdoor estimator that is not a circuit). GraspClutter6D adds a **hybrid circuit**,
which holds the camera frames by position, since the recording rig fixes their order,
and the objects as exchangeable parts. Every circuit answers a `cause`/`causes_effect`
EQL query by backdoor adjustment on a model stratified to be support-deterministic over
the cause.

## The datasets

| | Mutagenesis | Tracy clutter picking | GraspClutter6D |
|---|---|---|---|
| examples (fit / held out) | 188 (150 / 38) | 300 (240 / 60) | 954 (763 / 191) |
| parts per example | 14 to 40 atoms, bonds | 9 neighbours | 5 to 20 objects, 52 frames |
| effect | mutagenic, 66.5% | target lifted, 65.3% | every object graspable, 37.6% |
| questions | 14 | 7 | 14 |
| about one part | 3 | 2 | 3 |
| adjustments compared | ind1 / atom count / both | environment / none | extent / object count / both |
| known truth | – | – | synthetic model, 5 settings |

## Which questions each pipeline can answer

| pipeline | Mutagenesis | Tracy | GraspClutter6D |
|---|---|---|---|
| relational circuit | 14 of 14 | 7 of 7 | 14 of 14 |
| hybrid circuit | – | – | 14 of 14 |
| propositional tree | 11 of 14 | 5 of 7 | 11 of 14 |
| unrolled tree | 12 of 14 | 6 of 7 | 14 of 14 |
| scalars-only tree | 1 of 14 | 2 of 7 | 1 of 14 |
| regression adjustment | 11 of 14 | 5 of 7 | 11 of 14 |

The pattern is the same on every dataset. A flat table refuses a question that
constrains a column it does not have: the scalars-only tree has no counts, so it can
pose only the questions whose cause and confounder are both scalars; the propositional
tree and the regression baseline have no parts, so they refuse every question whose
cause or effect lives on one; the unrolled tree has the parts by position, so it answers
those about whatever part the example lists there, and refuses the ones the listing
order makes impossible (no molecule's first atom is terminal, so both terminal-atom
questions; no column for the twelfth neighbour of a nine-neighbour recording). The
relational circuit grounds itself for whatever the query names and answers everything,
including the atom-element question the earlier version of the pipeline refused.

## Where the columns are shared, the answers are the same

On every question whose cause and confounders are scalars or counts, the relational
circuit and the propositional tree give the same numbers to the third decimal, on all
three datasets, because the relational circuit's class circuit is that tree fitted on
the same rows. The unrolled and hybrid models are within a few hundredths of them.
Contrasts (adjusted probability at the highest supported region minus at the lowest,
with Newcombe's interval) read off the relational circuit:

| question | contrast |
|---|---|
| Mutagenesis: branching atoms → mutagenic, adjusting for atom count | 0.87 [0.49, 0.97] (10 → 21) |
| Mutagenesis: aromatic bonds → mutagenic, adjusting for atom count | 0.73 [0.47, 0.86] (6 → 19) |
| Mutagenesis: ind1 → mutagenic, adjusting for logp | 0.64 [0.51, 0.74] |
| Tracy: friction → lifted, adjusting for environment | 1.00 [0.81, 1.00] (0.125 → 0.75) |
| Tracy: adjacent neighbours → lifted, adjusting for environment | -0.65 [-0.81, -0.37] (0 → 4) |
| GraspClutter6D: catalogue → every object graspable, adjusting for small objects | 0.13 [0.05, 0.20] (mixed → grasp) |
| GraspClutter6D: occluded objects → every object graspable, adjusting for object count | 0.20 [-0.13, 0.44] (2 → 16) |
| GraspClutter6D: clear viewpoints → every object graspable, adjusting for object count | -0.32 [-0.42, -0.19] (0 → 52) |

Regression adjustment agrees on the direction of every trend and disagrees on its
size, in both directions: it flattens the step relations (Tracy's friction contrast
0.31 for 1.00, Mutagenesis' aromatic-bond contrast 0.05 for 0.73 under both
confounders) and inflates the non-monotone one (GraspClutter6D's occluded-object
contrast 0.61 for 0.20). A logistic model is the wrong shape for both, and a circuit
reads the effect off each region without assuming one.

**What adjusting changes.** Every dataset asks its count questions under more than one
set of confounders. Adjusting for the example's size (the atom count, the object count)
moves single regions by up to 0.3 on Mutagenesis and 0.15 on GraspClutter6D and leaves
the trend within 0.1 and the contrast within 0.05 on Mutagenesis, while on
GraspClutter6D it is the adjustment that turns the small-object contrast from -0.09 to
+0.07: the number of objects is the confounder there, and extent is not. On Tracy,
adjusting for the environment changes no region by more than 0.01; the crowding count
is not a stand-in for the environment.

## The answers about one part

| question | relational circuit | unrolled tree |
|---|---|---|
| Mutagenesis: ind1 → atom is carbon | 0.55 vs 0.43 | 1.00 vs 1.00 |
| Mutagenesis: element → atom is terminal | h, cl 1.00; o 0.94; c, n 0.00 | refused (no terminal atom 0) |
| Tracy: neighbour along closing axis → disturbed | 0.18 vs 0.03, [0.12, 0.17] over 2,160 neighbours | 0.22 vs 0.03, [0.11, 0.28] over 240 rows |
| GraspClutter6D: catalogue → object heavily occluded | grasp 0.36 vs ycb-video 0.32 | ycb-video 0.29 highest |
| GraspClutter6D: object size → loses every grasp | small 0.12 vs large 0.10, [0.00, 0.03] | large 0.15 highest, [0.02, 0.14] |

The relational circuit's answers are the part statistics (the valence table on
Mutagenesis, the neighbour counts on Tracy, the object counts on GraspClutter6D), read
over every part of every training example. The unrolled tree's are answers about the
part the dataset happens to list first: the carbon the CTU listing puts first, the
first-drawn neighbour, the first-labelled object.

## What reordering the parts does

Twenty random orderings of every example's parts, the pipelines that model the parts
refitted each time, the part questions asked again.

| | Mutagenesis | Tracy | GraspClutter6D |
|---|---|---|---|
| relational circuit: widest range of an answer | 0.00 | 0.00 | 0.00 |
| relational circuit: most effective region moved | 0% | 0% | 0% |
| unrolled tree: widest range of an answer | 1.00 | 0.20 | 0.50 |
| unrolled tree: most effective region moved | 95% | 0% | 88% |
| unrolled tree: whole-example likelihood drop | 56.5 nats | 33.3 nats | 120.4 nats |
| unrolled tree: coverage, dataset order → reordered | 44.7% → 4.9% | 43.3% → 24.8% | 67.0% → 45.2% |

Nothing about the relational circuit depends on the order the parts are written in; its
answers and its likelihood do not move by 1e-9. The unrolled tree's answers about a
part move with the listing, and its held-out likelihood and coverage fall with every
reordering. On GraspClutter6D the hybrid circuit loses 124 nats too, as it should: it
holds the camera frames by position because the rig's order is real, and shuffling the
frames destroys that. On Tracy the unrolled tree's most effective region never moves
because the closing-axis effect is large enough to survive any listing; its size still
ranges by 0.20.

## Explaining whole examples

Mean held-out log-likelihood over the examples every pipeline modelling the view
covers, and the coverage (the share of held-out examples inside the plain model's
support at all).

| | Mutagenesis | Tracy | GraspClutter6D |
|---|---|---|---|
| relational circuit | -40.8 (73.7%) | 154.3 (78.3%) | -17.2 (87.4%) |
| hybrid circuit | – | – | 83.3 (84.3%) |
| unrolled tree | -91.3 (44.7%) | 88.9 (43.3%) | 26.9 (67.0%) |

On every dataset the relational circuit covers many more held-out examples than the
unrolled tree, and on the two whose parts have no order it explains them far better.
On GraspClutter6D the unrolled tree's 52 camera frames per scene are genuinely
positional, and there the hybrid circuit, exchangeable over the objects and positional
over the frames, is the best model by 56 nats over the unrolled tree and 100 over the
fully exchangeable circuit: the right answer is to treat as exchangeable exactly the
parts that are. The learning curves say the same thing from the data side: the
relational circuit's templates pool every part of every training example, so at a fifth
of the data it already covers 30 to 36% of the held-out examples where the unrolled
tree, one row per example, covers 2 to 13%.

## Against a known truth

Only GraspClutter6D has a synthetic model of its domain whose interventional
probabilities are known by construction (the cause forced in the mechanism, the effect's
rate read off 200,000 forced scenes), under five settings of object count and
confounding strength.

| pipeline | questions answered | support-weighted abs. error | rank correlation with truth |
|---|---|---|---|
| relational circuit | 100% | 0.026 | 0.58 |
| hybrid circuit | 100% | 0.029 | 0.56 |
| propositional tree | 57% | 0.058 | 0.40 |
| unrolled tree | 100% | 0.064 | 0.26 |
| scalars-only tree | 0% | – | – |

On the count questions every pipeline that answers has the same error, since they are
the same tree on the same columns; on the three questions about an object the
relational circuit's error is 0.033, 0.009 and 0.007 against the unrolled tree's 0.133,
0.054 and 0.044, and the error of every pipeline grows with the number of objects in a
scene (0.05 at five, 0.08 at twenty for the relational circuit) as the counts' extremes
grow rarer.

## Cost

| | Mutagenesis | Tracy | GraspClutter6D |
|---|---|---|---|
| relational circuit: nodes / fit seconds | 4,663 / 44 | 7,266 / 8 | 37,013 / 190 |
| propositional tree: nodes / fit seconds | 3,009 / 10 | 2,003 / 1 | 10,429 / 43 |
| unrolled tree: nodes / fit seconds | 72,243 / 83 | 24,312 / 5 | 364,486 / 523 |
| relational circuit: seconds per question, fitted | 7.5 | 4.3 | 322 |
| propositional tree: seconds per question, fitted | 7.1 | 0.6 | 158 |
| unrolled tree: seconds per question, fitted | 8.9 | 0.8 | 122 |
| parts per example in the scaling study | – | 4 to 25 | 5 to 50 |
| relational circuit: nodes at the largest size | – | 7,148 | 10,016 |
| unrolled tree: nodes at the largest size | – | 14,709 | 116,486 |

The relational circuit is between a third and a fifteenth of the unrolled tree's size,
because its templates pool every part into one circuit whose size follows the number of
distinct part attributes, not the number of parts, and the scaling studies show the gap
widening with the size of the example. What a question costs once the models are fitted
is the backdoor adjustment: cheap where the cause and the confounders have few regions
(Tracy), and minutes for every circuit alike where a two-confounder adjustment sums over
hundreds of thousands of leaf regions (GraspClutter6D); grounding adds two to four
minutes to an object question on GraspClutter6D and under five seconds elsewhere.
Grounding itself settles from fifty Monte-Carlo samples wherever the open counts take
few values, and from 8,000 on the GraspClutter6D count question whose open counts run
to fifty-two.

## What the three runs add up to

- Every estimator that has the columns gives the same causal answer, circuit or not:
  the relational circuit is not a better model of the scalars and counts than a flat
  tree, it is that tree, and the comparison is honest about it.
- The questions a flat learner can pose depend on how the data was flattened, and its
  answers about a part depend on the order the parts were written down; the relational
  circuit's do not, on any of the three datasets.
- Where the truth is known, the relational circuit's answers about a part are three to
  five times closer to it than the unrolled tree's, and its whole-example likelihood is
  the best on every dataset whose parts have no order, and second only to its own
  hybrid on the one whose frames do.
- The price is grounding time on large examples and the same backdoor adjustment every
  circuit pays; the model itself is a fraction of the size of the tree that holds the
  parts by position.

## For the paper

Text and tables in the manuscript's own conventions (formal, third person, no hyphens,
no colons inside sentences, every number from `results.md`), written to drop into
`paper/main.tex` of the AISTATS submission. The rows for the other two datasets are
already in `tab:comparison` there; the block below adds the third.

### Rows for `tab:comparison`

```latex
\midrule
\multicolumn{6}{l}{\emph{GraspClutter6D, 954 scenes, 5 to 20 objects, 52 frames}} \\
Relational    & 14/14 & 87.4\% & $-17.2$ & 0.00 & 0.0 \\
Hybrid        & 14/14 & 84.3\% &  83.3   & 0.00 & 123.6 \\
Propositional & 11/14 & ---    & ---     & ---  & --- \\
Unrolled      & 14/14 & 67.0\% &  26.9   & 0.50 & 120.4 \\
Scalars only  &  1/14 & ---    & ---     & ---  & --- \\
Regression    & 11/14 & ---    & ---     & ---  & --- \\
```

### The `Real scenes` paragraph of the experiments section

```latex
\paragraph{Real scenes.}
The third dataset is GraspClutter6D \citep{graspclutter6d}, 954 real bin, shelf and
table scenes photographed from 52 poses, with the pose and visibility of every object
instance in every frame and analytic grasps checked for collision against every scene.
A scene is the queried class, its value fields the object catalogue it is built from,
the spread and the stacking of its clutter and whether every object in it keeps a
grasp, which 37.6\% do, and its two collection fields hold one object per instance
and one viewpoint per frame. The frames are numbered by the recording rig, so a
position means the same camera in every scene, and a fourth circuit, the hybrid,
holds the viewpoints by position and the objects as exchangeable parts. The fourteen
queries name a cause on the scene, on one of its counts under three sets of
confounders, and on one object. Every result of the first two datasets repeats
(\Cref{tab:comparison}). The relational, hybrid and unrolled models answer all
fourteen queries, the propositional tree and the regression eleven and the scalars
only tree one. Where the columns are shared the answers agree to three decimals, and
what they say is that the number of objects is the confounder that matters, since
adjusting for it moves single regions by up to 0.15 and turns the contrast of the
small object count from $-0.09$ to $0.07$, where adjusting for the spread of the
clutter changes nothing to three decimals. The regression agrees on every direction
and inflates the occluded object contrast to 0.61 against 0.20 for the circuits,
having assumed a monotone relation the data does not have. On the three queries about
an object the unrolled tree takes the sign of two effects the other way from the
relational circuit, and over twenty reorderings its most effective region moves in
88\% of them while the relational and hybrid answers do not move. The whole scene log
likelihood places the three models where the order of their parts puts them. The
hybrid circuit scores held out scenes at 83.3 against 26.9 for the unrolled tree and
$-17.2$ for the fully exchangeable circuit, and reordering the parts costs the hybrid
124 nats and the unrolled tree 120 while the relational circuit is unchanged, so the
model that treats as exchangeable exactly the parts that are exchangeable is the one
that explains the scenes.
```

### A paragraph on the known truth, for the experiments section or the appendix

```latex
\paragraph{Against a known truth.}
Reordering shows that an answer is stable, not that it is right. A structural causal
model over the GraspClutter6D domain, whose interventional probabilities are read
off 200000 scenes sampled with the cause forced in the mechanism, gives every
pipeline a truth to be scored against, under five settings of the number of objects
and of the strength with which that number drives both the causes and the effect. The
model lists the small objects first, so a column that addresses an object by position
is misleading by construction. Fitted on 400 scenes per setting and asked seven
queries, the relational circuit answers every one with a support weighted absolute
error of 0.026 and a rank correlation with the truth of 0.58, the hybrid 0.029 and
0.56, the propositional tree 0.058 and 0.40 on the four queries it can pose and the
unrolled tree 0.064 and 0.26 (\Cref{tab:truth}). On the count queries every pipeline
that answers has the same error, since on those columns they are the same tree, and
the extremes of a count are rare within every stratum of the confounder. On the three
queries about an object the relational circuit's error is 0.033, 0.009 and 0.007
against 0.133, 0.054 and 0.044 for the unrolled tree, and the relational error grows
from 0.05 to 0.08 as the scenes grow from five to twenty objects.
```

```latex
\begin{table}[t]
\centering
\caption{Every pipeline against the synthetic model's interventional
probabilities, over five settings and seven queries. \emph{Weighted} is the mean
absolute error with each cause region weighted by the training rows it holds,
\emph{Rank} the mean Spearman correlation between answered and true probabilities
over a query's regions, and the last three columns the mean absolute error on the
three queries whose cause or effect lives on one object.}
\label{tab:truth}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcrrrrr}
\toprule
Pipeline & Answ. & Weighted & Rank & Catalogue & Occluded & Size \\
\midrule
Relational    & 100\% & 0.026 & 0.58 & 0.033 & 0.009 & 0.007 \\
Hybrid        & 100\% & 0.029 & 0.56 & 0.033 & 0.009 & 0.007 \\
Propositional &  57\% & 0.058 & 0.40 & ---   & ---   & ---   \\
Unrolled      & 100\% & 0.064 & 0.26 & 0.133 & 0.054 & 0.044 \\
Scalars only  &   0\% & ---   & ---  & ---   & ---   & ---   \\
\bottomrule
\end{tabular}
\end{table}
```

### Lines for the `Additional Experiments` appendix

```latex
\paragraph{What adjusting changes.}
Every count query is asked under more than one set of confounders. On the molecules,
adjusting for the number of atoms in place of the structural indicator moves single
regions by up to 0.3 and leaves every trend within 0.1 and every contrast within 0.05
of its value under the indicator. On the attempts, adjusting for the environment
changes no region of the crowding query by more than 0.01. On the scenes, adjusting
for the spread of the clutter leaves the adjusted probabilities equal to the naive
ones to three decimals in almost every region, and adjusting for the number of
objects is what moves them.

\paragraph{Learning curves.}
Fitted on a fifth of the data and scored on the same held out fifth, the relational
circuit already covers 30\% of the molecules, 30\% of the attempts and 36\% of the
scenes, against 10\%, 2\% and 13\% for the unrolled tree, since the relational
template pools every part of every training example where the unrolled table sees
one row per example. At four fifths the coverages are 75\%, 76\% and 84\% against
48\%, 43\% and 64\%.

\paragraph{Cost against the size of the example.}
On synthetic scenes of 5 to 50 objects, 400 per size, the relational circuit grows
from 2330 to 10016 nodes and its fit from 16 to 44 seconds, the hybrid from 1260 to
4230 nodes and 10 to 28 seconds, and the unrolled tree from 7422 to 116486 nodes and
6 to 276 seconds. What a query costs once the models are fitted is the backdoor
adjustment, which is cheap where the cause and the confounders have few regions and
takes every circuit alike between two and thirty minutes on the scenes when two
confounders with several hundred regions each are summed over, and the grounding,
which adds two to four minutes to a query about an object on the scenes and under
five seconds elsewhere. Grounding integrates over the distinct values it has sampled
for the counts a query leaves open, so it settles from fifty samples wherever those
counts take few values and from 8000 on the scene query whose open count runs to
fifty two.
```
