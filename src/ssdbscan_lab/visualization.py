"""Matplotlib plots for algorithm comparison and runtime analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .algorithms import ClusteringResult
from .benchmark import RuntimeRow
from .datasets import SyntheticDataset


def _cluster_colors(labels: np.ndarray) -> list[str]:
    palette = plt.get_cmap("tab10")
    colors = []
    for label in labels:
        if label == -1:
            colors.append("#9E9E9E")
        else:
            colors.append(palette((int(label) - 1) % 10))
    return colors


def _plot_one_result(
    ax: plt.Axes,
    *,
    dataset: SyntheticDataset,
    result: ClusteringResult,
    title: str,
    importance_threshold: float,
) -> None:
    coords = dataset.points[:, :2]
    marker_sizes = 16 + 8 * dataset.points[:, 2]
    colors = _cluster_colors(result.labels)
    important_mask = dataset.points[:, 2] > importance_threshold

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=marker_sizes,
        c=colors,
        alpha=0.88,
        linewidths=0.5,
        edgecolors=np.where(important_mask, "#111111", "#FFFFFF"),
    )

    core_coords = coords[result.core_mask]
    ax.scatter(
        core_coords[:, 0],
        core_coords[:, 1],
        s=32,
        facecolors="none",
        edgecolors="#000000",
        linewidths=0.8,
    )

    noise_count = int(np.sum(result.labels == -1))
    ax.set_title(
        f"{title}\nclusters={result.n_clusters}, noise={noise_count}, core={int(result.core_mask.sum())}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)


def plot_cluster_comparison(
    *,
    dataset: SyntheticDataset,
    dbscan_result: ClusteringResult,
    ss_dbscan_result: ClusteringResult,
    importance_threshold: float,
    output_path: Path,
) -> None:
    """Save a side-by-side comparison plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    _plot_one_result(
        axes[0],
        dataset=dataset,
        result=dbscan_result,
        title="Simple Algorithm: DBSCAN",
        importance_threshold=importance_threshold,
    )
    _plot_one_result(
        axes[1],
        dataset=dataset,
        result=ss_dbscan_result,
        title="Research Paper Algorithm: SS-DBSCAN",
        importance_threshold=importance_threshold,
    )
    fig.suptitle(
        "DBSCAN vs SS-DBSCAN on a Varied-Density Dataset\n"
        "Marker size = importance radius, black edge = important point, black ring = core point"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_comparison(*, rows: list[RuntimeRow], output_path: Path) -> None:
    """Save runtime bars for DBSCAN and SS-DBSCAN."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_positions = np.arange(len(rows))
    width = 0.36
    dbscan_times = [row.dbscan_seconds for row in rows]
    ss_dbscan_times = [row.ss_dbscan_seconds for row in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_positions - width / 2, dbscan_times, width, label="DBSCAN")
    ax.bar(x_positions + width / 2, ss_dbscan_times, width, label="SS-DBSCAN")

    ax.set_title("Runtime Analysis: Simple DBSCAN vs SS-DBSCAN")
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
