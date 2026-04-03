# SS-DBSCAN DWDM Lab Project

This repository contains a Python implementation of:

- **Simple DBSCAN**
- **SS-DBSCAN** from the IEEE Access 2024 paper

It also generates the minimum lab outputs requested in your checklist:

- Simple algorithm description
- Research paper algorithm description
- Parameter-difference table
- Graph comparing simple DBSCAN vs SS-DBSCAN
- Python runnable code
- Theoretical difference between both algorithms
- Runtime/time-analysis graph

## Run the project

```bash
python run_project.py
```

## Main files

- `run_project.py` - one-command runner
- `src/ssdbscan_lab/algorithms.py` - DBSCAN and SS-DBSCAN implementations
- `src/ssdbscan_lab/datasets.py` - synthetic varied-density dataset
- `src/ssdbscan_lab/benchmark.py` - runtime benchmark
- `src/ssdbscan_lab/visualization.py` - comparison plots
- `docs/SS_DBSCAN_Lab_Report.md` - theory + comparison report
- `outputs/` - generated graphs and CSV summaries

## Paper used

T. Zaki Abdulhameed, S. A. Yousif, V. W. Samawi, and H. Imad Al-Shaikhli, "SS-DBSCAN: Semi-Supervised Density-Based Spatial Clustering of Applications With Noise for Meaningful Clustering in Diverse Density Data," IEEE Access, vol. 12, pp. 131507-131520, 2024, doi: 10.1109/ACCESS.2024.3457587.
