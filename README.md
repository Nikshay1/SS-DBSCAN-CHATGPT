# SS-DBSCAN DWDM Lab Project

This repository contains a Python implementation of:

- **Simple DBSCAN**
- **SS-DBSCAN** from the IEEE Access 2024 paper

The project now runs the comparison on **three datasets** in one command:

- `dataset/lettersPreProc.csv` from the reference `TibaZaki/SS_DBSCAN` repo
- `dataset/iris.arff` from Weka
- a generated synthetic varied-density dataset saved as `dataset/synthetic_varied_density.csv`

## Requirements

- Python `3.10+`
- `numpy`
- `matplotlib`

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install numpy matplotlib
```

## Run the project

```bash
python run_project.py
```

That single command:

- loads `dataset/lettersPreProc.csv`
- loads `dataset/iris.arff`
- generates the synthetic dataset and saves it to `dataset/synthetic_varied_density.csv`
- runs DBSCAN and SS-DBSCAN on all three datasets
- writes combined summaries and per-dataset outputs

## Dataset-specific settings

- `lettersPreProc.csv`: `eps = 8.0`, `min_pts = 17`
- `dataset/iris.arff`: `eps = 0.9`, `min_pts = 4`
- `synthetic_varied_density.csv`: `eps = 0.45`, `min_pts = 5`

## Verified current run

From the latest verified run:

- `letters`: DBSCAN found `1` cluster, SS-DBSCAN found `60`
- `dataset/iris.arff`: DBSCAN found `2` clusters, SS-DBSCAN found `4`
- `synthetic`: DBSCAN found `1` cluster, SS-DBSCAN found the intended `2` clusters

## Outputs

Combined output files:

- `outputs/metrics_summary.csv`
- `outputs/runtime_summary.csv`
- `outputs/cluster_assignments.csv`

Per-dataset output folders:

- `outputs/letters/`
- `outputs/iris/`
- `outputs/synthetic/`

Per-dataset figures:

- `outputs/figures/letters_dbscan_vs_ssdbscan.png`
- `outputs/figures/letters_runtime_comparison.png`
- `outputs/figures/iris_dbscan_vs_ssdbscan.png`
- `outputs/figures/iris_runtime_comparison.png`
- `outputs/figures/synthetic_dbscan_vs_ssdbscan.png`
- `outputs/figures/synthetic_runtime_comparison.png`

## Project structure

- `run_project.py` - runs all three experiments and writes summaries
- `src/ssdbscan_lab/algorithms.py` - DBSCAN and SS-DBSCAN implementations
- `src/ssdbscan_lab/datasets.py` - letters loader, iris ARFF loader, and synthetic dataset generator
- `src/ssdbscan_lab/benchmark.py` - runtime benchmarking helpers
- `src/ssdbscan_lab/visualization.py` - cluster and runtime plots
- `docs/SS_DBSCAN_Lab_Report.md` - project report

## Notes

- Letters clustering uses all `16` feature columns for distance.
- Iris clustering uses the `4` numeric flower measurements from the ARFF file.
- The synthetic dataset uses `(x, y)` for distance and `importance_radius` only for the SS-DBSCAN importance rule.
- For the synthetic dataset, SS-DBSCAN is configured so only important points can seed expansion-capable clusters; this avoids the tiny bridge fragments that previously showed up as extra clusters.
- Letters and iris figures use PCA for 2D visualization; the synthetic figure uses the original `x` and `y` coordinates.

## Reference paper

T. Zaki Abdulhameed, S. A. Yousif, V. W. Samawi, and H. Imad Al-Shaikhli, "SS-DBSCAN: Semi-Supervised Density-Based Spatial Clustering of Applications With Noise for Meaningful Clustering in Diverse Density Data," IEEE Access, vol. 12, pp. 131507-131520, 2024, doi: 10.1109/ACCESS.2024.3457587.
