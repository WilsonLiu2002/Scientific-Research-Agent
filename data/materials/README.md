# Local Materials Data

`matbench_expt_gap.json.gz` is the Matbench experimental band-gap regression
dataset. It contains 4,604 composition-level records and is used here for
candidate screening, retrieval context, and a transparent composition-neighbor
baseline.

- Dataset task: `matbench_expt_gap`
- Property: experimentally measured band gap in eV
- Download: <https://ml.materialsproject.org/projects/matbench_expt_gap.json.gz>
- Project: <https://github.com/materialsproject/matbench>
- Dataset notes: <https://hackingmaterials.lbl.gov/matminer/dataset_summary.html>

The database values are labeled `measured`. Values returned by the local kNN
adapter are predictions and must not be cited as measurements. The screen is a
hypothesis-ranking aid, not evidence of synthesizability, phase stability,
toxicity, or experimental validation.
