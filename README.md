# SS-DBSCAN DWDM Lab Project

This repository contains a Python implementation of:

- **Simple DBSCAN**
- **SS-DBSCAN** from the IEEE Access 2024 paper

The project is configured to run on the reference dataset from the public `TibaZaki/SS_DBSCAN` repository:

- `dataset/lettersPreProc.csv`
- `20,000` rows
- `16` feature columns
- `1` final class-label column

The dataset is used directly from `dataset/`; this project does not generate a replacement dataset.

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

The default run uses the same main parameters shown in the reference repo example:

- `eps = 8`
- `min_pts = 17`

## Outputs

Running the project generates:

- `outputs/metrics_summary.csv`
- `outputs/runtime_summary.csv`
- `outputs/cluster_assignments.csv`
- `outputs/figures/dbscan_vs_ssdbscan.png`
- `outputs/figures/runtime_comparison.png`

## Project structure

- `run_project.py` - one-command runner for `dataset/lettersPreProc.csv`
- `src/ssdbscan_lab/algorithms.py` - DBSCAN and SS-DBSCAN implementations
- `src/ssdbscan_lab/datasets.py` - letters dataset loader and importance rule metadata
- `src/ssdbscan_lab/benchmark.py` - runtime benchmark on dataset subsets
- `src/ssdbscan_lab/visualization.py` - comparison plots
- `docs/SS_DBSCAN_Lab_Report.md` - project report and discussion of results
- `outputs/` - generated CSV summaries and figures

## Notes

- The clustering run uses all `16` feature columns for distance calculations.
- The SS-DBSCAN importance rule follows the letters-dataset logic from the reference repo.
- The 2D comparison plot is a PCA projection for visualization only; clustering is still performed in the original 16-dimensional feature space.

## Reference paper

T. Zaki Abdulhameed, S. A. Yousif, V. W. Samawi, and H. Imad Al-Shaikhli, "SS-DBSCAN: Semi-Supervised Density-Based Spatial Clustering of Applications With Noise for Meaningful Clustering in Diverse Density Data," IEEE Access, vol. 12, pp. 131507-131520, 2024, doi: 10.1109/ACCESS.2024.3457587.
