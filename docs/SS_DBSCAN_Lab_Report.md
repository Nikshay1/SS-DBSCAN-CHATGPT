# SS-DBSCAN DWDM Lab Project Report

## 1. Project Objective

This project compares **DBSCAN** with **SS-DBSCAN** on three datasets:

1. `dataset/lettersPreProc.csv`
2. `dataset/iris.arff`
3. a generated synthetic varied-density dataset saved as `dataset/synthetic_varied_density.csv`

The main goal is to show how SS-DBSCAN changes clustering behavior by adding a user-defined **importance condition** to cluster expansion, while keeping the overall density-based structure of DBSCAN.

## 2. Simple Algorithm Description: DBSCAN

DBSCAN is an unsupervised density-based clustering algorithm.

**Main idea**

- For each point, find neighbors within distance `eps`.
- If the point has at least `min_pts` neighbors, it is dense enough to support a cluster.
- Start a cluster from dense points and expand through neighboring dense points.
- Points that do not belong to any cluster are labeled as noise.

**Decision rule**

```text
neighbor_count(point) >= min_pts
```

**Advantages**

- No need to provide the number of clusters in advance.
- Can find irregular cluster shapes.
- Naturally detects noise.

**Weaknesses**

- Sensitive to the choice of `eps` and `min_pts`.
- A single global density threshold may merge structures that should remain separate.

## 3. Research Paper Algorithm Description: SS-DBSCAN

SS-DBSCAN extends DBSCAN by adding a user-defined function:

```text
Is_important(point)
```

In this project's implementation:

- A dense seed point can start a cluster.
- A point can continue cluster expansion only if it is both dense and important.

So the SS-DBSCAN expansion condition becomes:

```text
neighbor_count(point) >= min_pts AND Is_important(point) == True
```

This is why SS-DBSCAN is considered **semi-supervised**: the user injects domain knowledge through the importance rule.

## 4. Parameter Difference Table

| Parameter / Concept | DBSCAN | SS-DBSCAN |
|---|---|---|
| `eps` | Required | Required |
| `min_pts` | Required | Required |
| `Is_important(point)` | Not used | Required extra rule |
| Cluster growth | Any dense expansion point can continue growth | Only dense and important expansion points continue growth |
| Supervision type | Unsupervised | Semi-supervised |
| Main control idea | Density only | Density + user-defined importance |
| Worst-case time complexity | `O(n^2)` | `O(n^2)` |

## 5. Datasets Used

### Dataset 1: `lettersPreProc.csv`

- Source: reference `TibaZaki/SS_DBSCAN` repository
- Rows: `20,000`
- Feature columns: `16`
- Final column: class label
- Number of classes: `26`

### Dataset 2: `dataset/iris.arff`

- Source: Weka sample dataset
- Rows: `150`
- Feature columns: `4`
- Classes: `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`

### Dataset 3: `synthetic_varied_density.csv`

- Source: generated inside this project from the provided synthetic-data code
- Rows: `440`
- Feature columns: `3`
- Features: `x`, `y`, `importance_radius`
- Labels: two cluster labels plus noise labels

## 6. Importance Rules Used

### Letters dataset

To match the reference repo, a point is important if at least one of these feature columns is close to its maximum:

- `6, 7, 8, 9, 11, 12, 13, 14, 15, 16`

Rule:

```text
feature_j(point) >= max(feature_j) - 2
for at least one selected feature j
```

### Iris dataset

For iris, the project uses the petal measurements as the importance cue.

Selected features:

- feature `3` = petal length
- feature `4` = petal width

Rule:

```text
petal length and petal width both stay inside the 15th-85th percentile band
```

This marks points with representative petal shapes as important expansion candidates.

### Synthetic dataset

For the generated synthetic dataset, the importance rule follows the provided code idea:

```text
importance_radius > 2 * mean(importance_radius)
```

This makes central blob points important while bridge points remain unimportant.

## 7. Experimental Setup

| Dataset | Distance features used | `eps` | `min_pts` | Importance rule |
|---|---|---:|---:|---|
| `lettersPreProc.csv` | all `16` features | `8.0` | `17` | selected features near max |
| `dataset/iris.arff` | all `4` numeric features | `0.9` | `4` | petal features inside percentile band |
| `synthetic_varied_density.csv` | `x`, `y` only | `0.45` | `5` | high `importance_radius` |

### Implementation note

The project uses chunked epsilon-neighborhood computation instead of storing one full distance matrix for the largest dataset. This keeps the implementation feasible for `20,000` rows while preserving the expected quadratic worst-case time behavior.

## 8. Graphs Produced by the Project

After running:

```bash
python run_project.py
```

the project creates these cluster comparison figures:

- `outputs/figures/letters_dbscan_vs_ssdbscan.png`
- `outputs/figures/iris_dbscan_vs_ssdbscan.png`
- `outputs/figures/synthetic_dbscan_vs_ssdbscan.png`

and these runtime figures:

- `outputs/figures/letters_runtime_comparison.png`
- `outputs/figures/iris_runtime_comparison.png`
- `outputs/figures/synthetic_runtime_comparison.png`

### How to read the cluster figures

- Left plot = simple DBSCAN
- Right plot = SS-DBSCAN
- Black edge = point satisfies `Is_important`
- Black ring = point is used as a core expansion point

For letters and iris, the plots use a PCA projection for visualization. For the synthetic dataset, the plot uses the original `x` and `y` coordinates.

## 9. Measured Results from the Current Run

The verified project run produced the following metrics:

| Dataset | Algorithm | Clusters found | Noise points | Core points | V-measure | ARI | Runtime (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| letters | DBSCAN | 1 | 0 | 19,989 | 0.0000 | 0.0000 | 20.563980 |
| letters | SS-DBSCAN | 60 | 0 | 2,216 | 0.1488 | 0.0042 | 13.936038 |
| iris | DBSCAN | 2 | 0 | 149 | 0.7337 | 0.5681 | 0.001998 |
| iris | SS-DBSCAN | 4 | 0 | 96 | 0.6958 | 0.5592 | 0.001525 |
| synthetic | DBSCAN | 1 | 23 | 415 | 0.1294 | 0.0525 | 0.012078 |
| synthetic | SS-DBSCAN | 2 | 128 | 152 | 0.7605 | 0.7977 | 0.009416 |

## 10. Interpretation of Results

### Letters dataset

- DBSCAN collapses the whole dataset into one cluster with these reference parameters.
- SS-DBSCAN becomes much more selective and creates many more clusters.
- The letters run mainly shows that the importance rule drastically changes expansion behavior, even though both clustering-quality metrics remain low.

### Iris dataset

- DBSCAN produces `2` clusters on the raw four-dimensional iris measurements.
- SS-DBSCAN produces `4` clusters with nearly the same ARI and V-measure.
- On this dataset, SS-DBSCAN changes the cluster structure but does not clearly outperform DBSCAN.

### Synthetic dataset

- DBSCAN merges the two main blobs into a single cluster.
- SS-DBSCAN recovers the intended two-cluster structure because bridge points are not important enough to seed or continue expansion.
- This is the clearest case where the importance rule improves clustering quality.

## 11. Runtime Analysis

Runtime was benchmarked separately for each dataset.

### Letters runtime summary

| Points | DBSCAN (s) | SS-DBSCAN (s) |
|---:|---:|---:|
| 1,000 | 0.055356 | 0.038859 |
| 2,000 | 0.280042 | 0.215297 |
| 4,000 | 1.059117 | 0.806675 |
| 8,000 | 3.685767 | 2.698778 |

### Iris runtime summary

| Points | DBSCAN (s) | SS-DBSCAN (s) |
|---:|---:|---:|
| 50 | 0.000463 | 0.000400 |
| 75 | 0.000791 | 0.000661 |
| 100 | 0.002118 | 0.001923 |
| 150 | 0.003286 | 0.002753 |

### Synthetic runtime summary

| Points | DBSCAN (s) | SS-DBSCAN (s) |
|---:|---:|---:|
| 180 | 0.003312 | 0.002072 |
| 300 | 0.005201 | 0.003982 |
| 420 | 0.008467 | 0.006299 |
| 540 | 0.013655 | 0.010939 |

### Runtime observation

- All three experiments follow the expected upward growth as dataset size increases.
- SS-DBSCAN is slightly faster in these runs because the importance rule reduces the number of points that continue expansion.

## 12. Output Files Produced by the Project

Combined output files:

- `outputs/metrics_summary.csv`
- `outputs/runtime_summary.csv`
- `outputs/cluster_assignments.csv`

Per-dataset output folders:

- `outputs/letters/`
- `outputs/iris/`
- `outputs/synthetic/`

Each per-dataset folder contains:

- `metrics_summary.csv`
- `runtime_summary.csv`
- `cluster_assignments.csv`

The combined `cluster_assignments.csv` file stores:

- dataset id
- original class label
- original display label
- DBSCAN cluster label
- SS-DBSCAN cluster label
- importance flag
- importance score

## 13. How to Run on Python

From the project root:

```bash
python run_project.py
```

If you want to import the algorithms directly:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from ssdbscan_lab.algorithms import dbscan, ss_dbscan
from ssdbscan_lab.datasets import load_letters_dataset, load_iris_dataset, make_synthetic_experiment_dataset
```

## 14. Short Conclusion

This project now compares DBSCAN and SS-DBSCAN on three datasets instead of one. The results show three different behaviors:

- On `lettersPreProc.csv`, SS-DBSCAN changes the expansion structure strongly, but the clustering quality is still weak with the reference parameters.
- On `dataset/iris.arff`, SS-DBSCAN changes the cluster structure while staying close to DBSCAN in ARI and V-measure.
- On the generated synthetic dataset, SS-DBSCAN clearly improves separation because the importance rule blocks expansion through bridge points.

Overall, the project demonstrates that SS-DBSCAN can preserve more meaningful structure when the importance rule captures useful domain knowledge, while still keeping the same worst-case `O(n^2)` time complexity.

## 15. References

1. T. Zaki Abdulhameed, S. A. Yousif, V. W. Samawi, and H. Imad Al-Shaikhli, "SS-DBSCAN: Semi-Supervised Density-Based Spatial Clustering of Applications With Noise for Meaningful Clustering in Diverse Density Data," IEEE Access, vol. 12, pp. 131507-131520, 2024, doi: 10.1109/ACCESS.2024.3457587.
2. Tiba Zaki, `TibaZaki/SS_DBSCAN`, GitHub repository containing `lettersPreProc.csv` and the reference example for letters data.
3. Weka sample dataset `dataset/iris.arff`.
