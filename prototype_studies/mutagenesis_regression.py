"""
Answer the Mutagenesis question catalogue and the branching-atom causal query on the
live dataset, and write every number to a JSON file, so two checkouts can be compared.
Usage: mutagenesis_regression.py <output.json>
"""
import dataclasses
import json
import sys
import time

import numpy as np

import experiments.orm.ormatic_interface  # noqa: F401  # maps the domain classes

from experiments.causal_reasoning.mutagenesis.causal_query import BranchingAtomCountCausalQuery
from experiments.causal_reasoning.mutagenesis.dataset import fetch_mutagenesis_molecules
from experiments.causal_reasoning.mutagenesis.do_query import MutagenesisDoQuery, question_catalogue

molecules = fetch_mutagenesis_molecules()
result = {"molecules": len(molecules), "questions": {}}
for question in question_catalogue():
    start = time.time()
    answer = MutagenesisDoQuery(question).run(molecules)
    result["questions"][question.name] = {
        "regions": [dataclasses.asdict(region) for region in answer.regions],
        "seconds": time.time() - start,
    }
    print(question.name, f"{time.time() - start:.0f} s", flush=True)
np.random.seed(0)
branching = BranchingAtomCountCausalQuery().run(molecules, atom_count=2)
result["branching_atom_count"] = {
    "effects": [dataclasses.asdict(effect) for effect in branching.effects],
    "support_determinism_verified": branching.support_determinism_verified,
}
json.dump(result, open(sys.argv[1], "w"), indent=1, default=str)
print("written", sys.argv[1])
