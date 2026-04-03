"""Clustering quality metrics implemented with NumPy only."""

from __future__ import annotations

import math

import numpy as np


def _contingency_matrix(true_labels: np.ndarray, predicted_labels: np.ndarray) -> np.ndarray:
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    if true_labels.shape[0] != predicted_labels.shape[0]:
        raise ValueError("true_labels and predicted_labels must have the same length.")

    _, true_inverse = np.unique(true_labels, return_inverse=True)
    _, predicted_inverse = np.unique(predicted_labels, return_inverse=True)

    matrix = np.zeros(
        (true_inverse.max() + 1, predicted_inverse.max() + 1),
        dtype=np.int64,
    )
    np.add.at(matrix, (true_inverse, predicted_inverse), 1)
    return matrix


def _entropy(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log(probs)))


def _conditional_entropy(contingency: np.ndarray, axis: int) -> float:
    total = contingency.sum()
    if total == 0:
        return 0.0

    outer_counts = contingency.sum(axis=axis)
    conditional_entropy = 0.0

    if axis == 0:
        for cluster_index, cluster_total in enumerate(outer_counts):
            if cluster_total == 0:
                continue
            probs = contingency[:, cluster_index] / cluster_total
            probs = probs[probs > 0]
            conditional_entropy += (cluster_total / total) * float(
                -np.sum(probs * np.log(probs))
            )
    else:
        for class_index, class_total in enumerate(outer_counts):
            if class_total == 0:
                continue
            probs = contingency[class_index, :] / class_total
            probs = probs[probs > 0]
            conditional_entropy += (class_total / total) * float(
                -np.sum(probs * np.log(probs))
            )

    return conditional_entropy


def v_measure_score(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """Compute V-measure from homogeneity and completeness."""
    contingency = _contingency_matrix(true_labels, predicted_labels)

    class_entropy = _entropy(contingency.sum(axis=1))
    cluster_entropy = _entropy(contingency.sum(axis=0))

    homogeneity = 1.0
    if class_entropy > 0:
        homogeneity = 1.0 - _conditional_entropy(contingency, axis=0) / class_entropy

    completeness = 1.0
    if cluster_entropy > 0:
        completeness = 1.0 - _conditional_entropy(contingency, axis=1) / cluster_entropy

    if homogeneity + completeness == 0:
        return 0.0
    return 2.0 * homogeneity * completeness / (homogeneity + completeness)


def adjusted_rand_index(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """Compute the Adjusted Rand Index."""
    contingency = _contingency_matrix(true_labels, predicted_labels)
    n_samples = contingency.sum()
    if n_samples < 2:
        return 1.0

    sum_nij = sum(math.comb(int(value), 2) for value in contingency.ravel())
    row_sums = contingency.sum(axis=1)
    col_sums = contingency.sum(axis=0)
    sum_ai = sum(math.comb(int(value), 2) for value in row_sums)
    sum_bj = sum(math.comb(int(value), 2) for value in col_sums)
    total_pairs = math.comb(int(n_samples), 2)

    expected_index = (sum_ai * sum_bj) / total_pairs if total_pairs else 0.0
    max_index = 0.5 * (sum_ai + sum_bj)
    denominator = max_index - expected_index

    if denominator == 0:
        return 1.0 if sum_nij == expected_index else 0.0
    return (sum_nij - expected_index) / denominator
