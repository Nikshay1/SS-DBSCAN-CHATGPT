# SS-DBSCAN DWDM Lab Project Report

## 1. Project Objective

This project compares **DBSCAN** with the **SS-DBSCAN** algorithm proposed in the paper *SS-DBSCAN: Semi-Supervised Density-Based Spatial Clustering of Applications With Noise for Meaningful Clustering in Diverse Density Data* by Abdulhameed et al., IEEE Access, 2024.

The aim is to show how adding a user-defined **importance condition** to core-point selection can reduce wrong cluster merging when the dataset contains **different densities**.

## 2. Simple Algorithm Description: DBSCAN

DBSCAN is an unsupervised density-based clustering algorithm.

**Main idea**

- For every point, find all neighbors within distance `eps`.
- If the number of neighbors is at least `min_pts`, that point is a **core point**.
- Start a cluster from each core point and expand it through neighboring core points.
- Points reachable from a core point but not core themselves become **border points**.
- Points not assigned to any cluster are labeled **noise**.

**Core-point rule**

```text
neighbor_count(point) >= min_pts  =>  core point
```

**Strengths**

- Number of clusters is not required in advance.
- Can detect arbitrary-shaped clusters.
- Can identify noise points.

**Weaknesses**

- Sensitive to `eps` and `min_pts`.
- Can merge separate clusters through dense bridge points when densities vary.

## 3. Research Paper Algorithm Description: SS-DBSCAN

The paper modifies DBSCAN by adding an extra condition for core-point selection.

Instead of allowing every dense point to expand a cluster, SS-DBSCAN expands only through points that satisfy a user-defined function:

```text
Is_important(point)
```

So the core-point rule becomes:

```text
neighbor_count(point) >= min_pts AND Is_important(point) == True
```

This makes SS-DBSCAN **semi-supervised**, because domain knowledge is injected through the importance rule.

### Paper-inspired `Is_important` rule used in this project

The paper gives a marble example where a point is important when its radius is greater than twice the mean radius. This project uses the same style of rule on a synthetic 2D dataset with an extra `importance_radius` feature:

```text
importance_radius(point) > 2 * mean(importance_radius of all points)
```

Bridge points are generated with low radius, while central points in real clusters are generated with high radius. Therefore, plain DBSCAN tends to merge clusters through the bridge, while SS-DBSCAN blocks expansion through low-importance bridge points.

## 4. Parameter Difference Table

| Parameter / Concept | DBSCAN | SS-DBSCAN |
|---|---|---|
| `eps` | Required | Required |
| `min_pts` | Required | Required |
| `Is_important(point)` | Not used | Required extra condition |
| Core-point rule | `neighbor_count >= min_pts` | `neighbor_count >= min_pts AND Is_important(point)` |
| Supervision type | Unsupervised | Semi-supervised |
| Main control idea | Density only | Density + user-defined importance |
| Main risk | Can merge different-density clusters through dense bridges | Depends on a meaningful importance rule |
| Worst-case time complexity | `O(n^2)` with full distance matrix | `O(n^2)` with full distance matrix |

## 5. Theoretical Difference Between Algorithm 1 and Algorithm 2

Here, **Algorithm 1** means simple DBSCAN and **Algorithm 2** means SS-DBSCAN from the research paper.

| Point of difference | Simple DBSCAN | SS-DBSCAN |
|---|---|---|
| Decision basis | Density neighborhood count only | Density neighborhood count + importance condition |
| Human knowledge | No user knowledge injected | User provides a dataset-specific condition |
| Cluster expansion | Any core point can expand the cluster | Only important core points expand the cluster |
| Effect on varied-density data | May over-merge clusters | Can preserve more meaningful clusters |
| Core-point count | Usually higher | Usually lower due to the extra filter |
| Asymptotic complexity | `O(n^2)` | `O(n^2)` |

## 6. Experimental Setup Used in This Project

Because the original paper datasets are not included in this repository, this project uses a **synthetic varied-density dataset** designed to demonstrate the exact issue discussed in the paper.

### Dataset design

- Cluster 1: dense Gaussian blob on the left
- Cluster 2: dense Gaussian blob on the right
- Bridge points: low-importance points between the blobs
- Random noise points: scattered outliers
- Extra feature: `importance_radius`, used only by `Is_important`

### Parameters used

- `eps = 0.45`
- `min_pts = 5`
- Distance features = `(x, y)`
- Importance rule = `importance_radius > 2 * mean(importance_radius)`

## 7. Graph: Simple Algorithm vs Research Paper Algorithm

Run the project:

```bash
python run_project.py
```

Then open this graph:

![DBSCAN vs SS-DBSCAN](../outputs/figures/dbscan_vs_ssdbscan.png)

**How to read the graph**

- Left plot = simple DBSCAN
- Right plot = SS-DBSCAN
- Marker size = importance radius
- Black edge = point satisfies `Is_important`
- Black ring = point is used as a core point

## 8. Time Analysis / Computation Runtime

The paper states that adding `Is_important(point)` does **not** change worst-case complexity. Both algorithms remain `O(n^2)` in the worst case when a full pairwise distance matrix is used.

This project verifies runtime experimentally by running both algorithms on increasing dataset sizes and saving this bar chart:

![Runtime Comparison](../outputs/figures/runtime_comparison.png)

Numeric timing values are written to:

```text
outputs/runtime_summary.csv
```

## 9. Output Files Produced by This Project

After running `python run_project.py`, these files are generated:

- `outputs/metrics_summary.csv`
- `outputs/runtime_summary.csv`
- `outputs/figures/dbscan_vs_ssdbscan.png`
- `outputs/figures/runtime_comparison.png`

## 10. How to Run on Python

From the project root:

```bash
python run_project.py
```

If you want to import the algorithms in your own script:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from ssdbscan_lab.algorithms import dbscan, ss_dbscan
```

## 11. Short Conclusion

Simple DBSCAN uses only density to decide core points, so it can merge clusters when bridge points are dense enough. SS-DBSCAN adds a user-defined importance condition and can therefore produce more meaningful clusters for varied-density data, while keeping the same `O(n^2)` worst-case complexity reported in the paper.

## 12. Reference

T. Zaki Abdulhameed, S. A. Yousif, V. W. Samawi, and H. Imad Al-Shaikhli, "SS-DBSCAN: Semi-Supervised Density-Based Spatial Clustering of Applications With Noise for Meaningful Clustering in Diverse Density Data," IEEE Access, vol. 12, pp. 131507-131520, 2024, doi: 10.1109/ACCESS.2024.3457587.
