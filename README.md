# Readiness-Aware Multi-Region Infrastructure Provisioning

Code, synthetic data and experiment results for **Readiness-Aware Scheduling for Multi-Region Infrastructure Provisioning Under API Rate Limits**.

The method uses Differential Evolution (DE) to tune task order, prerequisite readiness gates, regional API budgets, global concurrency and optional start deferral. An event-driven digital twin evaluates schedule candidates. The included baselines are FIFO, parallel-max and a tuned rate-limit-aware greedy scheduler.

## Repository contents

- `code/run_all.py`: full experiment pipeline, statistics and figures.
- `code/twin.py`: data cleaning and calibration of throttle, readiness-time and failure models.
- `code/sim.py`: dependency-graph construction, simulator, baseline policies and DE search.
- `code/fig1_flowchart.dot`: Graphviz source for the methodology figure.
- `data/A5P3_multiregion_provisioning.csv`: 16,000 synthetic tasks with 22 columns.
- `data/A5P3_README.md`: generation assumptions and the full data dictionary.
- `results/`: supplied result tables, JPG figures, flowchart source and fitted metrics.
- `requirements.txt`: exact Python dependency pins copied from the experiment bundle.

The paper documents and presentation files are not included in this code-and-data package.

## Setup

Use Python 3.13, the version reported for the experiments, and install Graphviz so the `dot` executable is available on `PATH`. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
dot -V
```

On Windows, activate the environment with `.venv\Scripts\activate` instead of the `source` command.

## Reproduce the experiments

From the repository root:

```bash
python3 code/run_all.py
```

The script resolves inputs relative to its own file location, reads `data/A5P3_multiregion_provisioning.csv`, and writes tables, metrics, JPG figures and a console log to `results/`. It creates a `figures/` directory for 300 dpi PNG figures. A run overwrites the corresponding bundled results. Allow several minutes for the complete experiment; runtime depends on the machine and software environment.

The pipeline fits the twin on 2025 records and evaluates 2026 records. Scheduling experiments use ten held-out plans, five seeds and 30 Monte-Carlo executions per policy and plan. DE uses 20 population members, 50 generations, mutation scale 0.6 and crossover rate 0.9. Fixed seeds support repeatability, while measured runtime and floating-point results can vary across environments.

## Outputs

The main summaries are `results/table3_main_comparison.csv`, `results/table4_ablation.csv`, `results/twin_fidelity.csv` and `results/metrics.json`. Per-plan/per-seed results, sensitivity data and paired Wilcoxon tests are also included. See `code/README.md` for the complete output list.

The supplied `metrics.json` follows Python's JSON encoder convention and includes `NaN` for unavailable baseline metrics. Strict JSON consumers must treat those entries as missing values before parsing.

## Data provenance and interpretation

All tasks, regions and identifiers are synthetic. The data README reports generator seed 2079. The original `gen_author5.py` dataset generator is not part of this bundle, so the included CSV is the experiment input rather than a dataset that this repository regenerates.

The twin shares functional forms with the synthetic generator, which favors its measured fidelity. Task graphs follow a fixed dependency template, and carbon estimates use assumed resource-class powers. Results are simulation evidence rather than measurements of a live cloud deployment. Execution makespan excludes optional start deferral; end-to-end latency includes it.

The supplied files have undergone syntax, input-path and file-size checks. The full experiments have not been rerun as part of this release preparation.

## Author and license

Author: Akhil Venkata Sai Madineni. This package is distributed under the [MIT license](LICENSE).

## Citation and archive

Source repository: [https://github.com/Akhilmadineni/readiness-aware-provisioning](https://github.com/Akhilmadineni/readiness-aware-provisioning).

Use [`CITATION.cff`](CITATION.cff) for the software citation. The matching [`.zenodo.json`](.zenodo.json) supplies the archive metadata for release `v1.0.0`. The version DOI will be added after Zenodo completes archiving.
