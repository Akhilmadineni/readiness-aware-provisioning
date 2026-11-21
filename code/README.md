# A5P3 code: readiness-aware DE scheduling for multi-region IaC provisioning

## How to run

```
pip install -r requirements.txt      # plus Graphviz (`dot`) on the PATH
python3 run_all.py                   # about 8-10 min on one CPU core
```

The script reads `../data/A5P3_multiregion_provisioning.csv` and writes everything to `../results/` and `../figures/`.
All random generators are seeded, so repeated runs give identical numbers (only the CPU-time figures change).

## Files

| File | Content |
|---|---|
| `run_all.py` | Full pipeline: cleaning, twin calibration + fidelity check, plan construction, greedy tuning, DE and baselines (5 seeds x 10 plans x 30 Monte-Carlo runs), Wilcoxon tests, ablation, carbon-weight sweep, figures, metrics.json |
| `twin.py` | Pre-processing (winsorising, median imputation, outlier flags) and the digital twin: Poisson-MLE throttle model (Eq. 2), Huber readiness-time model (Eq. 3), logistic failure model (Eq. 4), carbon profile by region x hour |
| `sim.py` | Plan DAG construction, event-driven list-scheduling simulator, FIFO / parallel-max / rate-limit-aware greedy policies, genome decoding (Eq. 7), objective J (Eq. 6) and DE/rand/1/bin (Eqs. 8-10) |
| `fig1_flowchart.dot` | Graphviz source of the methodology flowchart (Fig. 1) |

## Outputs (`../results/`)

| File | Content |
|---|---|
| `table1_dataset_summary.csv`, `region_summary.csv` | Dataset summary (Table I) and per-region statistics |
| `table2_setup.csv` | Hyper-parameters and setup (Table II) |
| `twin_fidelity.csv` | R2/MAE of log time-to-ready, throttle AUC/MAE, failure PR-AUC/ROC-AUC on 2026 rows: type median vs twin vs LightGBM |
| `greedy_tuning_train_plans.csv` | Grid search of the greedy baseline (theta, global cap) on 2025 plans |
| `test_plans.csv` | Size, regions and submit hour of the 10 test plans |
| `main_per_plan_seed.csv` | Monte-Carlo means of every policy per plan and seed |
| `table3_main_comparison.csv` | Mean +- std over seeds (Table III) |
| `wilcoxon_de_vs_baselines.csv` | Paired Wilcoxon tests, DE vs each baseline |
| `ablation_per_plan_seed.csv`, `table4_ablation.csv` | Ablation study (Table IV) |
| `gamma_sensitivity*.csv` | Carbon-weight sweep (Fig. 6) |
| `de_best_genomes.csv` | Best DE genome per plan and seed |
| `metrics.json` | Fitted twin parameters and all headline numbers |
| `fig*.jpg`, `fig1_flowchart.dot` | JPG copies of all figures (PNG 300 dpi in `../figures/`) |
| `run_log.txt` | Console log |
