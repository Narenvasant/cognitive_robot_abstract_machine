# Mutagenesis: relational circuit against flat-table tree

## What this experiment is for

The Tracy clutter-picking experiment compares a relational causal circuit with a flat
joint probability tree on a robot's own recorded attempts. This experiment puts the
same two pipelines, and the same `cause`/`causes_effect` EQL queries, to a standard
relational-learning benchmark instead: the CTU Mutagenesis dataset
(https://relational.fel.cvut.cz/dataset/Mutagenesis), 188 nitroaromatic molecules with
their atoms and bonds and whether each tested mutagenic. It shows that the argument
does not depend on the robot: the relational circuit answers every molecule-level
question exactly as the flat tree does, answers questions about individual atoms the
flat tree cannot even pose, and scores whole molecules the flat tree has no columns
for.

- The **relational circuit** is an RSPN fitted on the molecules' relational structure:
  one circuit over the molecule's own attributes and its aggregation counts, one
  template over an atom, one over a bond. For every query it grounds itself into a
  circuit over the queried molecule, atoms and bonds and registers that circuit as a
  `CausalCircuit`.
- The **flat-table tree** is a joint probability tree (JPT) fitted on the same
  molecules flattened into one table of the molecule's own attributes and the same
  four aggregation counts. It is registered as a `CausalCircuit` in the same way.

`results.md` holds the comparison: which questions each pipeline answers and what it
answers, how many models each fitted, how long that took, how big the models are, how
well each explains held-out molecules, how fast each answers, and how often a molecule
is mutagenic in the first place. `causal_query_results.md` is the earlier, single-model
run of the branching-atom question on the relational circuit alone.

## The domain

`domain.py` holds the classes every other module works on.

| class | what it is |
|---|---|
| `MutagenesisMolecule` | one molecule: the `ind1` structural indicator, `logp`, `lumo`, whether it is `mutagenic`, and its `atoms` and `bonds` |
| `MutagenesisAtom` | one atom as an exchangeable part: its element, atom-type code, partial charge and how many bonds it takes part in |
| `MutagenesisBond` | one bond as an exchangeable part: its type |
| `MutagenesisMoleculeAggregations` | the counts the relational model derives over the parts: chlorine atoms and branching atoms over the atoms, double bonds and aromatic bonds over the bonds |

`dataset.py` fetches the molecules from the CTU database, holds them as a
`MutagenesisDataset` with a train/test split and mutagenic rates grouped by any key,
and generates synthetic molecules of the same shape for tests that must run without
network access.

## Why the flat table is propositional

A recorded clutter has a fixed number of neighbours, so the Tracy flat table unrolls
one block of columns per neighbour. A molecule has between 14 and 40 atoms, the most
common exact atom-and-bond count covers 13 of the 188 molecules, and the atoms have no
canonical order, so there is no fixed-width unrolling in which one column means the
same thing in two molecules. The flat table is therefore the classic propositional
summary of a relational example: the molecule's scalars plus its four counts, and no
atoms or bonds. That is also exactly the table the relational circuit's class-level
circuit is fitted on, which is what makes the comparison sharp: on a molecule's own
attributes and counts the two pipelines fit the same tree, and everything the
relational circuit does beyond that comes from modelling the parts.

Every query lists one atom and one bond with all their attributes open. For the
relational circuit that is what makes grounding retain the molecule's counts as
variables (a query with an empty atom list is a molecule with no atoms, whose counts
are zero). The flat table ignores a part a query merely lists and refuses a query that
constrains one: sets one of its attributes, or marks it as cause, confounder or effect.

## The pipelines

| file | what it holds |
|---|---|
| `flat_table.py` | `MoleculeSchema`, how EQL names every attribute, and `FlatTable`, the molecules as one row of scalars and counts each |
| `pipelines.py` | `CausalQueryPipeline` and its two implementations, `RelationalPipeline` and `FlatTablePipeline` |
| `queries.py` | the question catalogue, each question one EQL query that reads the same for either pipeline |
| `evaluation.py` | asking every question to every pipeline and recording what came of it |
| `report.py` | rendering the comparison as Markdown |
| `run_pipeline.py` | the whole comparison end to end |
| `causal_query.py` | the earlier single-model run of the branching-atom question |

**One model per cause.** Backdoor adjustment needs the circuit to be
support-deterministic over the cause: no sum unit may mix branches that overlap on it.
A fit guarantees that by stratifying its training rows on the cause's exact value.
Each pipeline keeps one plain model for everything that is not a causal query, such as
scoring held-out molecules, and fits one further model per cause variable the first
time it is asked about that cause. A cause on an atom attribute stratifies the atom
template in the relational pipeline; the flat pipeline has no column for it.

**The questions.**

1. *Branching atoms, aromatic bonds and double bonds cause mutagenicity*, each
   adjusting for the `ind1` indicator, which marks the fused-ring molecules that are
   both large and mostly mutagenic. The cause is a count over the exchangeable parts.
2. *The indicator causes mutagenicity*, adjusting for the branching-atom count. The
   cause is an attribute of the molecule itself.
3. *The indicator causes an atom to be carbon*, and *the branching-atom count causes
   an atom to be terminal* (to have a single bond). The cause is molecule-level, the
   effect one atom's own attribute.
4. *An atom's element causes it to be terminal*. Cause and effect both live on one
   atom.

Both pipelines answer 1 and 2, and answer them identically. Only the relational
circuit answers 3. Neither answers 4, for two different reasons, see below.

## What the results show

- On every molecule-level question the two pipelines agree to the third decimal:
  same regions, same naive and adjusted probabilities. The relational circuit's
  class circuit and the flat tree are the same tree, and Monte-Carlo grounding with
  2000 samples reproduces its regions exactly. Mutagenicity rises from 0 at seven
  branching atoms to 1 at seventeen and above, from 0.24 at six aromatic bonds to 1
  at fourteen and above, and the indicator raises it from 0.37 to 0.94 once the
  branching-atom count is adjusted for. Adjustment moves little: within a fixed count
  the indicator carries almost no further information about mutagenicity.
- The two atom-level effects are real and only the relational circuit reads them: a
  molecule with the indicator set has 0.52 carbon atoms per atom against 0.44 without,
  and an atom of a molecule with 25 branching atoms is terminal with probability 0.34
  against 0.51 at 8. Both match the raw atom counts.
- The atom-level cause is refused by both, for different reasons. Grounding with the
  counts left open mixes one copy of the atom template per sampled count, and those
  copies overlap on the element without being identical (a copy for a molecule with
  no chlorine has no chlorine atom, a copy for one with some has), so the grounded
  circuit is not support-deterministic over the element and verification rejects
  it. Verification used to let such a mixture through, because it only inspected a
  sum whose children were disjoint somewhere; the regions then read off it carried
  more probability together than one, with the carbon region counted twice. It now
  treats any sum whose children differ on the cause as a split, and the experiment
  still checks that the regions it reports partition the cause. The flat table
  refuses the same question because it has no column for an atom's element.
- On a molecule's own attributes and counts both pipelines cover 89.5% of the
  held-out molecules at a mean log-likelihood of -8.24. Only the relational circuit
  scores whole molecules, atoms and bonds included; there it covers 60.5%, because one
  atom of a rare element or one charge outside the fitted ranges puts the whole
  molecule outside the support.
- Once fitted, the flat tree answers in 0.2 seconds and the relational circuit in 3,
  the difference being the Monte-Carlo grounding. The relational circuit is eleven
  times larger, most of it the atom template over 4000 training atoms.

## Running it

```bash
# fit, score and question both pipelines; needs network access to the CTU database
# and the experiments ORM interface
python scripts/regenerate_all_orm.py
python -m experiments.causal_reasoning.mutagenesis.run_pipeline
```

The tests under `test/causal_reasoning_test/test_mutagenesis` run the pipelines on the
synthetic molecules, so they need no network access.
