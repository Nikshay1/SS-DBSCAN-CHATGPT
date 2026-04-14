"""Matplotlib plots for algorithm comparison and runtime analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .algorithms import ClusteringResult
from .benchmark import RuntimeRow
from .datasets import ExperimentDataset


def _cluster_colors(labels: np.ndarray) -> list[str]:
    palette = plt.get_cmap("tab20")
    colors = []
    for label in labels:
        if label == -1:
            colors.append("#9E9E9E")
        else:
            colors.append(palette((int(label) - 1) % 20))
    return colors


def _plot_one_result(
    ax: plt.Axes,
    *,
    dataset: ExperimentDataset,
    result: ClusteringResult,
    title: str,
) -> None:
    coords = dataset.visualization_points
    importance_profile = dataset.importance_profile
    marker_sizes = 16 + (7 * np.clip(importance_profile.scores, 0, 6))
    colors = _cluster_colors(result.labels)

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=marker_sizes,
        c=colors,
        alpha=0.82,
        linewidths=0.45,
        edgecolors=np.where(importance_profile.important_mask, "#111111", "#FFFFFF"),
    )

    core_coords = coords[result.core_mask]
    ax.scatter(
        core_coords[:, 0],
        core_coords[:, 1],
        s=28,
        facecolors="none",
        edgecolors="#000000",
        linewidths=0.7,
    )

    noise_count = int(np.sum(result.labels == -1))
    ax.set_title(
        f"{title}\nclusters={result.n_clusters}, noise={noise_count}, core={int(result.core_mask.sum())}"
    )
    ax.set_xlabel(dataset.visualization_axis_labels[0])
    ax.set_ylabel(dataset.visualization_axis_labels[1])
    ax.grid(alpha=0.15)


def plot_cluster_comparison(
    *,
    dataset: ExperimentDataset,
    dbscan_result: ClusteringResult,
    ss_dbscan_result: ClusteringResult,
    output_path: Path,
) -> None:
    """Save a side-by-side comparison plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True, sharey=True)
    _plot_one_result(
        axes[0],
        dataset=dataset,
        result=dbscan_result,
        title="Simple Algorithm: DBSCAN",
    )
    _plot_one_result(
        axes[1],
        dataset=dataset,
        result=ss_dbscan_result,
        title="Research Paper Algorithm: SS-DBSCAN",
    )
    fig.suptitle(
        f"DBSCAN vs SS-DBSCAN on {dataset.display_name}\n"
        "Projection uses dataset metadata; black edge = important point, black ring = core point"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_comparison(
    *,
    rows: list[RuntimeRow],
    dataset_name: str,
    output_path: Path,
) -> None:
    """Save runtime bars for DBSCAN and SS-DBSCAN."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_positions = np.arange(len(rows))
    width = 0.36
    dbscan_times = [row.dbscan_seconds for row in rows]
    ss_dbscan_times = [row.ss_dbscan_seconds for row in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_positions - width / 2, dbscan_times, width, label="DBSCAN")
    ax.bar(x_positions + width / 2, ss_dbscan_times, width, label="SS-DBSCAN")

    ax.set_title(f"Runtime Analysis on {dataset_name}")
    ax.set_xlabel("Number of points")
    ax.set_ylabel("Average runtime (seconds)")
    ax.set_xticks(x_positions, [str(row.n_points) for row in rows])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for x_pos, db_time, ss_time in zip(x_positions, dbscan_times, ss_dbscan_times):
        ax.text(x_pos - width / 2, db_time, f"{db_time:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(x_pos + width / 2, ss_time, f"{ss_time:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
