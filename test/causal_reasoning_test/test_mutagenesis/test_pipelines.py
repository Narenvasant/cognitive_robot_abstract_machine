"""
Tests for the two causal-query pipelines on synthetic molecules.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.causal_reasoning.mutagenesis.dataset import (
    synthetic_mutagenesis_molecules,
)
from experiments.causal_reasoning.mutagenesis.domain import MutagenesisElement
from experiments.causal_reasoning.mutagenesis.evaluation import (
    QuestionAsker,
    Refusal,
)
from experiments.causal_reasoning.mutagenesis.exceptions import (
    FlatTableSchemaMismatchError,
    OneCausePerQueryError,
    PipelineNotFittedError,
)
from experiments.causal_reasoning.mutagenesis.flat_table import (
    FlatTable,
    MoleculeSchema,
    PartAttribute,
)
from experiments.causal_reasoning.mutagenesis.pipelines import (
    CauseStratification,
    FlatTablePipeline,
    RelationalPipeline,
)
from experiments.causal_reasoning.mutagenesis.queries import (
    BranchingAtomsCauseTerminalAtom,
    CountCausesMutagenicity,
    ElementCausesTerminalAtom,
    IndicatorCausesElement,
    IndicatorCausesMutagenicity,
    atom_query,
    bond_query,
    molecule_query,
)
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import cause


@pytest.fixture(scope="module")
def atom_count() -> int:
    """
    How many atoms each synthetic molecule has.
    """
    return 3


@pytest.fixture(scope="module")
def molecules(atom_count):
    return synthetic_mutagenesis_molecules(
        np.random.default_rng(0),
        molecule_count=150,
        atom_count=atom_count,
        bond_count=4,
    )


@pytest.fixture(scope="module")
def relational_pipeline(molecules):
    pipeline = RelationalPipeline(min_samples_per_leaf=15)
    pipeline.fit(molecules[:120])
    return pipeline


@pytest.fixture(scope="module")
def flat_table_pipeline(molecules):
    pipeline = FlatTablePipeline(min_samples_per_leaf=15)
    pipeline.fit(molecules[:120])
    return pipeline


# %% flat table


@pytest.fixture(scope="module")
def schema():
    return MoleculeSchema()


def test_flat_table_columns_are_named_like_eql_variables(schema):
    assert schema.scalar_column("mutagenic") == "MutagenesisMolecule.mutagenic"
    assert (
        schema.part_column(PartAttribute("atoms", 2, "element"))
        == "MutagenesisMolecule.atoms[2].element"
    )
    assert schema.aggregation_columns == (
        "MutagenesisMoleculeAggregations.chlorine_count()",
        "MutagenesisMoleculeAggregations.branching_atom_count()",
        "MutagenesisMoleculeAggregations.double_bond_count()",
        "MutagenesisMoleculeAggregations.aromatic_bond_count()",
    )


def test_flat_table_row_holds_scalars_and_counts_only(molecules, schema, atom_count):
    molecule = molecules[0]
    row = FlatTable().row(molecule)
    assert row[schema.scalar_column("logp")] == molecule.logp
    assert row[schema.aggregation_column("chlorine_count")] == atom_count
    assert set(row) == set(FlatTable().columns)
    assert not any(schema.part_attribute(column) for column in row)


def test_schema_tells_a_part_attribute_from_a_molecule_variable(schema):
    assert schema.part_attribute(
        "MutagenesisMolecule.bonds[3].bond_type"
    ) == PartAttribute("bonds", 3, "bond_type")
    assert schema.part_attribute(schema.scalar_column("lumo")) is None
    assert schema.part_attribute(schema.aggregation_column("chlorine_count")) is None


# %% stratification per cause


def test_a_molecule_level_cause_stratifies_the_class_circuit(schema):
    stratification = CauseStratification.for_variable(
        schema.aggregation_column("branching_atom_count")
    )
    assert stratification.class_columns == [
        schema.aggregation_column("branching_atom_count")
    ]
    assert stratification.part_attributes == {}


def test_an_atom_cause_stratifies_the_atom_template(schema):
    stratification = CauseStratification.for_variable(
        schema.part_column(PartAttribute("atoms", 1, "element"))
    )
    assert stratification.class_columns is None
    assert stratification.part_attributes == {"atoms": ["element"]}


def test_unfitted_pipeline_refuses_to_serve_a_model():
    with pytest.raises(PipelineNotFittedError):
        RelationalPipeline().registry_for(None)


def test_two_causes_in_one_query_are_rejected(relational_pipeline):
    query = molecule_query(
        [atom_query()], [bond_query()], indicator_1=cause, chlorine_count=cause
    )
    query.causes_effect(query.variable.mutagenic == True)
    with pytest.raises(OneCausePerQueryError):
        ProbabilisticBackend(model_registry=relational_pipeline.registry).rank_causes(
            query
        )


def test_flat_table_pipeline_cannot_fit_a_model_for_an_atom_cause(
    flat_table_pipeline, schema
):
    with pytest.raises(FlatTableSchemaMismatchError):
        flat_table_pipeline.registry_for(
            schema.part_column(PartAttribute("atoms", 0, "element"))
        )


# %% answering the questions


@pytest.fixture(scope="module")
def asker():
    return QuestionAsker(random_seed=0)


@pytest.fixture(scope="module")
def chlorine_causes_mutagenicity():
    return CountCausesMutagenicity(
        statistic_name="chlorine_count", count_noun="chlorine atoms"
    )


def _adjusted_by_region(outcome):
    return {
        effect.cause_region: effect.adjusted_probability for effect in outcome.effects
    }


@pytest.mark.parametrize(
    "pipeline_name", ["relational_pipeline", "flat_table_pipeline"]
)
def test_both_pipelines_find_chlorine_the_cause_of_mutagenicity(
    request, asker, pipeline_name, atom_count, chlorine_causes_mutagenicity
):
    """
    The synthetic molecules are mutagenic exactly when every atom is chlorine, so the
    adjusted effect must be certain at the full chlorine count and impossible at none
    for either pipeline.
    """
    pipeline = request.getfixturevalue(pipeline_name)
    outcome = asker.ask(pipeline, chlorine_causes_mutagenicity)

    assert outcome.answered
    adjusted = _adjusted_by_region(outcome)
    assert set(adjusted) == {str(atom_count), "0"}
    assert adjusted[str(atom_count)] == pytest.approx(1.0)
    assert adjusted["0"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "pipeline_name", ["relational_pipeline", "flat_table_pipeline"]
)
def test_both_pipelines_answer_about_the_indicator(request, asker, pipeline_name):
    pipeline = request.getfixturevalue(pipeline_name)
    outcome = asker.ask(pipeline, IndicatorCausesMutagenicity())

    assert outcome.answered
    assert set(_adjusted_by_region(outcome)) == {"True", "False"}


def test_relational_pipeline_answers_about_one_atom(relational_pipeline, asker):
    outcome = asker.ask(
        relational_pipeline,
        IndicatorCausesElement(element=MutagenesisElement.CHLORINE),
    )
    assert outcome.answered
    assert set(_adjusted_by_region(outcome)) == {"True", "False"}


def test_relational_pipeline_answers_a_count_cause_of_an_atom_effect(
    relational_pipeline, asker, atom_count
):
    outcome = asker.ask(relational_pipeline, BranchingAtomsCauseTerminalAtom())
    assert outcome.answered
    assert set(_adjusted_by_region(outcome)) == {
        str(count) for count in range(atom_count + 1)
    }


def test_flat_table_pipeline_refuses_a_question_about_one_atom(
    flat_table_pipeline, asker
):
    outcome = asker.ask(flat_table_pipeline, BranchingAtomsCauseTerminalAtom())
    assert not outcome.answered
    assert outcome.refusal == Refusal.SCHEMA_MISMATCH


def test_flat_table_pipeline_refuses_an_atom_cause(flat_table_pipeline, asker):
    outcome = asker.ask(flat_table_pipeline, ElementCausesTerminalAtom())
    assert not outcome.answered
    assert outcome.refusal == Refusal.SCHEMA_MISMATCH


def test_an_effect_that_never_occurs_is_reported(relational_pipeline, asker):
    """
    No synthetic atom is carbon, so no region of the cause gives the effect any
    probability.
    """
    outcome = asker.ask(
        relational_pipeline, IndicatorCausesElement(element=MutagenesisElement.CARBON)
    )
    assert outcome.refusal == Refusal.EFFECT_NEVER_OCCURS


def test_each_cause_gets_its_own_model(
    relational_pipeline, schema, asker, chlorine_causes_mutagenicity
):
    """
    Every distinct cause asked about fitted one further model, on top of the plain one.
    """
    asker.ask(relational_pipeline, chlorine_causes_mutagenicity)
    asker.ask(relational_pipeline, IndicatorCausesMutagenicity())

    assert relational_pipeline.fit_report.model_count == 1 + len(
        relational_pipeline.cause_models
    )
    assert set(relational_pipeline.cause_models) >= {
        schema.aggregation_column("chlorine_count"),
        schema.scalar_column("indicator_1"),
    }


# %% likelihood


@pytest.mark.parametrize(
    "pipeline_name", ["relational_pipeline", "flat_table_pipeline"]
)
def test_held_out_likelihood_covers_some_molecules(request, molecules, pipeline_name):
    pipeline = request.getfixturevalue(pipeline_name)
    report = pipeline.log_likelihood(molecules[120:])
    assert report.molecule_count == 30
    assert 0 < report.covered_molecule_count <= report.molecule_count
    assert np.isfinite(report.mean_log_likelihood)


def test_both_plain_models_are_the_same_tree(
    relational_pipeline, flat_table_pipeline, molecules
):
    """
    The relational circuit's class-level circuit and the flat-table tree are fitted on
    the same rows with the same settings, so they score held-out molecules alike.
    """
    relational = relational_pipeline.log_likelihood(molecules[120:])
    flat = flat_table_pipeline.log_likelihood(molecules[120:])
    assert np.allclose(relational.log_likelihoods, flat.log_likelihoods)


def test_only_the_relational_pipeline_scores_whole_molecules(
    relational_pipeline, flat_table_pipeline, molecules
):
    whole = relational_pipeline.whole_molecule_log_likelihood(molecules[120:])
    scalars_and_counts = relational_pipeline.log_likelihood(molecules[120:])
    assert flat_table_pipeline.whole_molecule_log_likelihood(molecules[120:]) is None
    assert whole.molecule_count == 30
    assert 0 < whole.covered_molecule_count <= scalars_and_counts.covered_molecule_count
    assert np.isfinite(whole.mean_log_likelihood)
