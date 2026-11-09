"""
Experiment 3 of the pairwise-relations prototype: a scratch copy of the Mutagenesis
domain whose bonds refer to their two atoms, with each atom's bond count computed as an
exposure instead of filled in by the loader.

Usage: python -m prototype_studies.mutagenesis_with_bonds <output.json>
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import create_engine

from experiments.causal_reasoning.mutagenesis.dataset import MutagenesisDatabaseConnection
from experiments.causal_reasoning.mutagenesis.domain import (
    MutagenesisBondType,
    MutagenesisElement,
)
from krrood.entity_query_language.factories import variable
from krrood.parametrization.feature_extraction.exposures import (
    ExposureStatistic,
    exposure_statistic,
)
from krrood.parametrization.feature_extraction.part_reference import PartReference

# %% the scratch domain


@dataclass
class BondedAtom:
    element: MutagenesisElement
    atom_type: int
    charge: float
    loaded_bond_count: int
    """
    The bond count the Mutagenesis loader fills in, kept to compare with.
    """


@dataclass
class ReferencingBond:
    first_atom: BondedAtom = field(metadata=PartReference().as_dict())
    second_atom: BondedAtom = field(metadata=PartReference().as_dict())
    bond_type: MutagenesisBondType


@dataclass
class BondedMolecule:
    mutagenic: bool
    atoms: list[BondedAtom]
    bonds: list[ReferencingBond]


@dataclass(eq=False)
class BondedAtomExposures(ExposureStatistic[BondedAtom]):
    @exposure_statistic(variable(BondedMolecule).atoms)
    def bond_count(self) -> int:
        return sum(
            bond.first_atom is self.role_taker or bond.second_atom is self.role_taker
            for bond in self.instance.bonds
        )

    @exposure_statistic(variable(BondedMolecule).atoms)
    def bonded_chlorine_count(self) -> int:
        return sum(
            (bond.second_atom if bond.first_atom is self.role_taker else bond.first_atom).element
            is MutagenesisElement.CHLORINE
            for bond in self.instance.bonds
            if bond.first_atom is self.role_taker or bond.second_atom is self.role_taker
        )


def fetch_bonded_molecules() -> list[BondedMolecule]:
    engine = create_engine(MutagenesisDatabaseConnection().url, connect_args={"connect_timeout": 5})
    with engine.connect():
        drugs = pd.read_sql_table("drugs", engine)
        atoms = pd.read_sql_table("atoms", engine)
        bonds = pd.read_sql_table("bonds", engine)
    loaded_bond_count = (
        pd.concat([bonds["atom1_id"], bonds["atom2_id"]]).value_counts().to_dict()
    )
    atom_of_id = {
        row.id: BondedAtom(
            element=MutagenesisElement(row.element),
            atom_type=int(row.atom_type),
            charge=float(row.charge),
            loaded_bond_count=int(loaded_bond_count.get(row.id, 0)),
        )
        for row in atoms.itertuples()
    }
    atoms_of_drug = atoms.groupby("drug_id")["id"].apply(list).to_dict()
    bonds_of_drug = {
        drug_id: [
            ReferencingBond(
                atom_of_id[row.atom1_id], atom_of_id[row.atom2_id], MutagenesisBondType(row.bond_type)
            )
            for row in group.itertuples()
        ]
        for drug_id, group in bonds.groupby("drug_id")
    }
    return [
        BondedMolecule(
            mutagenic=bool(drug.active),
            atoms=[atom_of_id[atom_id] for atom_id in atoms_of_drug.get(drug.id, [])],
            bonds=bonds_of_drug.get(drug.id, []),
        )
        for drug in drugs.itertuples()
    ]


def main():
    molecules = fetch_bonded_molecules()
    mismatches, atoms_checked, terminal, terminal_with_one_bond = 0, 0, 0, 0
    chlorine_bonded = 0
    for molecule in molecules:
        for atom in molecule.atoms:
            exposures = BondedAtomExposures(role_taker=atom, instance=molecule).values("atoms")
            atoms_checked += 1
            mismatches += exposures["bond_count"] != atom.loaded_bond_count
            chlorine_bonded += exposures["bonded_chlorine_count"] > 0
            if atom.element is MutagenesisElement.HYDROGEN or atom.element is MutagenesisElement.CHLORINE:
                terminal += 1
                terminal_with_one_bond += exposures["bond_count"] == 1
    result = {
        "molecules": len(molecules),
        "atoms_checked": atoms_checked,
        "bond_count_mismatches": int(mismatches),
        "hydrogen_and_chlorine_atoms": terminal,
        "of_them_with_exactly_one_bond": terminal_with_one_bond,
        "atoms_bonded_to_a_chlorine": int(chlorine_bonded),
    }
    print(json.dumps(result))
    json.dump(result, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
