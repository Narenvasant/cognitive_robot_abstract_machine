# Relational causal queries on probabilistic circuits

Code for the experiments on interventional queries in relational domains. Three
experiments fit the same set of models on three datasets, ask each model the same
causal queries, and write a report of the answers.

The three experiments live under
`experiments/src/experiments/causal_reasoning/`, and they share the pipeline in
`causal_reasoning/comparison/`.

## Installation

Python 3.12 is required.

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install the workspace and the test dependencies:

   ```bash
   uv sync --extra dev --active
   ```

3. Build the object relational mapping the experiments store their examples
   through. This writes one `ormatic_interface.py` per package and takes a few
   seconds:

   ```bash
   python scripts/regenerate_all_orm.py
   ```

4. Check the installation by running the tests of the three experiments. They use
   synthetic data and a trimmed sample of the scene dataset, so they need no
   network access:

   ```bash
   python -m pytest test/experiments_test/causal_reasoning_test/
   ```

## Running the experiments

Each experiment writes a `results.md` next to its own code. Every number in that
report is produced by the run, so a rerun overwrites it.

### Mutagenesis

The molecules are fetched from the CTU Prague relational learning repository on
first use, so this run needs network access.

```bash
python -m experiments.causal_reasoning.mutagenesis.run_pipeline
```

Output: `experiments/src/experiments/causal_reasoning/mutagenesis/results.md`

### Clutter picking

The 300 recorded attempts ship with the code, in
`experiments/src/experiments/causal_reasoning/tracy_clutter_picking/data/milk_clutter_attempts.json`.
The run reads that file and needs no network access.

```bash
python -m experiments.causal_reasoning.tracy_clutter_picking.run_pipeline
```

Output: `experiments/src/experiments/causal_reasoning/tracy_clutter_picking/results.md`

Recording new attempts needs the simulator and the motion framework, which are not
part of this release. The recorded attempts are included so the comparison can be
rerun without them.

### GraspClutter6D

The scenes come from the public GraspClutter6D dataset. The grasp and collision
labels have to be on the machine before the first run. They are two of the
dataset's own archives, about 5 GB to download and more once extracted:

```bash
python - <<'EOF'
import huggingface_hub, pathlib, py7zr
directory = pathlib.Path.home() / "graspclutter6d-dataset"
for name in ("split_info.7z", "models_eval.7z", "collision_label.7z", "grasp_label.7z"):
    archive = huggingface_hub.hf_hub_download(
        "GraspClutter6D/GraspClutter6D", name, repo_type="dataset",
        local_dir=str(directory),
    )
    with py7zr.SevenZipFile(archive, mode="r") as opened:
        opened.extractall(path=str(directory))
EOF
```

Point the loader at that directory and run:

```bash
export SEMANTIC_DIGITAL_TWIN_DATASET_ROOT="$HOME/graspclutter6d-dataset"
python -m experiments.causal_reasoning.graspclutter6d.run_pipeline
```

Output: `experiments/src/experiments/causal_reasoning/graspclutter6d/results.md`

This is the longest of the three. One split of the full comparison takes about
eight hours on a desktop machine.

## Reading the results

Each experiment's README describes its domain, the queries it asks and the models
it fits. `graspclutter6d/comparison.md` compares the three datasets.

Every run takes the same options, including `--orderings` for how many times the
parts of an example are shuffled and `--output` for where the report goes. Pass
`--help` to any `run_pipeline` for the full list.
