# SS-DBSCAN DWDM Lab Project

This repository contains a Python implementation of:

- **Simple DBSCAN**
- **SS-DBSCAN** from the IEEE Access 2024 paper

The project now runs against the reference dataset from the public SS-DBSCAN repo:

- `dataset/lettersPreProc.csv`
- 16 feature columns
- 1 class column in the final position

## Run the project

```bash
python run_project.py
```

The default run uses the reference repo parameters:

- `eps = 8`
- `min_pts = 17`

## Outputs

Running the project generates:

- `outputs/metrics_summary.csv`
- `outputs/runtime_summary.csv`
- `outputs/cluster_assignments.csv`
- `outputs/figures/dbscan_vs_ssdbscan.png`
- `outputs/figures/runtime_comparison.png`

## Main files

- `run_project.py` - one-command runner for `dataset/lettersPreProc.csv`
- `src/ssdbscan_lab/algorithms.py` - DBSCAN and SS-DBSCAN implementations
- `src/ssdbscan_lab/datasets.py` - letters dataset loader and importance rule metadata
- `src/ssdbscan_lab/benchmark.py` - runtime benchmark on dataset subsets
- `src/ssdbscan_lab/visualization.py` - comparison plots
- `docs/SS_DBSCAN_Lab_Report.md` - project write-up
- `outputs/` - generated graphs and CSV summaries

## Paper used

T. Zaki Abdulhameed, S. A. Yousif, V. W. Samawi, and H. Imad Al-Shaikhli, "SS-DBSCAN: Semi-Supervised Density-Based Spatial Clustering of Applications With Noise for Meaningful Clustering in Diverse Density Data," IEEE Access, vol. 12, pp. 131507-131520, 2024, doi: 10.1109/ACCESS.2024.3457587.
