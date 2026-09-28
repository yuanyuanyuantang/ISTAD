#!/usr/bin/env python3
"""Permutation-invariant metrics for the ISTAD Stage C stability audit."""

import math

import numpy as np


EPS = np.finfo(np.float64).eps


def _same_partition(labels_a, labels_b):
    labels_a = np.asarray(labels_a).reshape(-1)
    labels_b = np.asarray(labels_b).reshape(-1)
    return bool(np.array_equal(
        labels_a[:, None] == labels_a[None, :],
        labels_b[:, None] == labels_b[None, :],
    ))


def _contingency(labels_a, labels_b):
    labels_a = np.asarray(labels_a).reshape(-1)
    labels_b = np.asarray(labels_b).reshape(-1)
    if labels_a.shape != labels_b.shape:
        raise ValueError("partition label vectors must have the same shape")
    _, inverse_a = np.unique(labels_a, return_inverse=True)
    _, inverse_b = np.unique(labels_b, return_inverse=True)
    table = np.zeros(
        (int(inverse_a.max()) + 1, int(inverse_b.max()) + 1),
        dtype=np.int64,
    )
    np.add.at(table, (inverse_a, inverse_b), 1)
    return table


def partition_scores(labels_a, labels_b):
    """Return adjusted Rand index and arithmetic-mean normalized MI."""
    labels_a = np.asarray(labels_a).reshape(-1)
    labels_b = np.asarray(labels_b).reshape(-1)
    table = _contingency(labels_a, labels_b)
    n = int(table.sum())
    if n < 2:
        return 1.0, 1.0

    row = table.sum(axis=1).astype(np.float64)
    col = table.sum(axis=0).astype(np.float64)
    values = table.astype(np.float64)

    choose2 = lambda x: x * (x - 1.0) / 2.0
    index = float(choose2(values).sum())
    row_index = float(choose2(row).sum())
    col_index = float(choose2(col).sum())
    total_pairs = choose2(float(n))
    expected = row_index * col_index / total_pairs
    maximum = 0.5 * (row_index + col_index)
    denominator = maximum - expected
    if abs(denominator) <= EPS:
        ari = 1.0 if _same_partition(labels_a, labels_b) else 0.0
    else:
        ari = (index - expected) / denominator
    ari = float(np.clip(ari, -1.0, 1.0))

    probabilities = values / float(n)
    row_p = row / float(n)
    col_p = col / float(n)
    nz_i, nz_j = np.nonzero(values)
    mutual_information = 0.0
    for i, j in zip(nz_i.tolist(), nz_j.tolist()):
        p_ij = probabilities[i, j]
        mutual_information += p_ij * math.log(p_ij / (row_p[i] * col_p[j]))
    entropy_a = -float(np.sum(row_p[row_p > 0] * np.log(row_p[row_p > 0])))
    entropy_b = -float(np.sum(col_p[col_p > 0] * np.log(col_p[col_p > 0])))
    entropy_mean = 0.5 * (entropy_a + entropy_b)
    if entropy_mean <= EPS:
        nmi = 1.0 if _same_partition(labels_a, labels_b) else 0.0
    else:
        nmi = mutual_information / entropy_mean
    return ari, float(np.clip(nmi, 0.0, 1.0))


def summarize(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("metric summaries require at least one finite value")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _validate_pair(incidence_a, incidence_b):
    if incidence_a.shape != incidence_b.shape:
        raise ValueError(
            f"incidence shapes differ: {incidence_a.shape} versus {incidence_b.shape}"
        )
    if len(incidence_a.shape) != 3:
        raise ValueError("incidence must have shape (time, variables, hyperedges)")
    if incidence_a.shape[0] < 1 or incidence_a.shape[1] < 2:
        raise ValueError("incidence needs at least one point and two variables")


def _row_normalize(incidence):
    incidence = np.asarray(incidence, dtype=np.float64)
    denominator = incidence.sum(axis=2, keepdims=True)
    return incidence / np.maximum(denominator, EPS)


def relation_metrics(incidence_a, incidence_b, neighbor_k=5, chunk_size=256):
    """Compare induced variable relations without using hyperedge identities."""
    _validate_pair(incidence_a, incidence_b)
    n_points, n_variables, _ = incidence_a.shape
    k = min(max(1, int(neighbor_k)), n_variables - 1)
    upper = np.triu_indices(n_variables, k=1)

    ari_values = []
    nmi_values = []
    cosine_values = []
    relative_frobenius_values = []
    neighbor_jaccard_values = []

    for low in range(0, n_points, int(chunk_size)):
        high = min(n_points, low + int(chunk_size))
        block_a = np.asarray(incidence_a[low:high], dtype=np.float64)
        block_b = np.asarray(incidence_b[low:high], dtype=np.float64)
        if not np.all(np.isfinite(block_a)) or not np.all(np.isfinite(block_b)):
            raise ValueError("incidence contains a non-finite value")
        if np.any(block_a < 0) or np.any(block_b < 0):
            raise ValueError("incidence contains a negative value")

        assignments_a = np.argmax(block_a, axis=2)
        assignments_b = np.argmax(block_b, axis=2)
        for labels_a, labels_b in zip(assignments_a, assignments_b):
            ari, nmi = partition_scores(labels_a, labels_b)
            ari_values.append(ari)
            nmi_values.append(nmi)

        memberships_a = _row_normalize(block_a)
        memberships_b = _row_normalize(block_b)
        relation_a = np.matmul(memberships_a, np.swapaxes(memberships_a, 1, 2))
        relation_b = np.matmul(memberships_b, np.swapaxes(memberships_b, 1, 2))

        flat_a = relation_a[:, upper[0], upper[1]]
        flat_b = relation_b[:, upper[0], upper[1]]
        norm_a = np.linalg.norm(flat_a, axis=1)
        norm_b = np.linalg.norm(flat_b, axis=1)
        product = norm_a * norm_b
        cosine = np.divide(
            np.sum(flat_a * flat_b, axis=1),
            product,
            out=np.zeros_like(product),
            where=product > EPS,
        )
        both_zero = (norm_a <= EPS) & (norm_b <= EPS)
        cosine[both_zero] = 1.0
        cosine_values.extend(np.clip(cosine, 0.0, 1.0).tolist())

        difference = np.linalg.norm(flat_a - flat_b, axis=1)
        scale = 0.5 * (norm_a + norm_b)
        relative = np.divide(
            difference,
            scale,
            out=np.zeros_like(scale),
            where=scale > EPS,
        )
        relative_frobenius_values.extend(relative.tolist())

        diagonal = np.arange(n_variables)
        relation_a[:, diagonal, diagonal] = -np.inf
        relation_b[:, diagonal, diagonal] = -np.inf
        neighbors_a = np.argsort(-relation_a, axis=2, kind="stable")[:, :, :k]
        neighbors_b = np.argsort(-relation_b, axis=2, kind="stable")[:, :, :k]
        intersection = (
            neighbors_a[:, :, :, None] == neighbors_b[:, :, None, :]
        ).any(axis=3).sum(axis=2)
        jaccard = intersection / (2.0 * k - intersection)
        neighbor_jaccard_values.extend(jaccard.reshape(-1).tolist())

    return {
        "top1_partition_ari": summarize(ari_values),
        "top1_partition_nmi": summarize(nmi_values),
        "co_membership_cosine": summarize(cosine_values),
        "co_membership_relative_frobenius": summarize(relative_frobenius_values),
        f"top{int(k)}_neighbor_jaccard": summarize(neighbor_jaccard_values),
        "neighbor_k": int(k),
    }


def hyperedge_column_cosine(incidence_a, incidence_b, chunk_size=256):
    """Event-wide cosine matrix between every pair of hyperedge columns."""
    _validate_pair(incidence_a, incidence_b)
    n_points, _, n_edges = incidence_a.shape
    dot = np.zeros((n_edges, n_edges), dtype=np.float64)
    norm_a = np.zeros(n_edges, dtype=np.float64)
    norm_b = np.zeros(n_edges, dtype=np.float64)
    for low in range(0, n_points, int(chunk_size)):
        high = min(n_points, low + int(chunk_size))
        block_a = np.asarray(incidence_a[low:high], dtype=np.float64)
        block_b = np.asarray(incidence_b[low:high], dtype=np.float64)
        dot += np.einsum("tcm,tcn->mn", block_a, block_b, optimize=True)
        norm_a += np.einsum("tcm,tcm->m", block_a, block_a, optimize=True)
        norm_b += np.einsum("tcm,tcm->m", block_b, block_b, optimize=True)
    denominator = np.sqrt(norm_a)[:, None] * np.sqrt(norm_b)[None, :]
    similarity = np.divide(
        dot,
        denominator,
        out=np.zeros_like(dot),
        where=denominator > EPS,
    )
    both_zero = (norm_a[:, None] <= EPS) & (norm_b[None, :] <= EPS)
    similarity[both_zero] = 1.0
    return np.clip(similarity, 0.0, 1.0)


def hungarian_maximize(similarity):
    """Return the maximum-weight square assignment as A-index -> B-index."""
    similarity = np.asarray(similarity, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("Hungarian matching requires a square similarity matrix")
    if not np.all(np.isfinite(similarity)):
        raise ValueError("similarity matrix contains a non-finite value")

    n = similarity.shape[0]
    cost = float(similarity.max()) - similarity
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        min_value = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if current < min_value[j]:
                    min_value[j] = current
                    way[j] = j0
                if min_value[j] < delta:
                    delta = min_value[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    min_value[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = np.empty(n, dtype=np.int64)
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    return assignment


def kendall_tau_b(values_a, values_b):
    values_a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    values_b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    if values_a.shape != values_b.shape:
        raise ValueError("Kendall vectors must have the same shape")
    concordant = 0
    discordant = 0
    tied_a = 0
    tied_b = 0
    for i in range(len(values_a) - 1):
        for j in range(i + 1, len(values_a)):
            delta_a = values_a[i] - values_a[j]
            delta_b = values_b[i] - values_b[j]
            if delta_a == 0 and delta_b == 0:
                continue
            if delta_a == 0:
                tied_a += 1
            elif delta_b == 0:
                tied_b += 1
            elif delta_a * delta_b > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + tied_a)
        * (concordant + discordant + tied_b)
    )
    if denominator <= EPS:
        return None
    return float((concordant - discordant) / denominator)


def _cosine(vector_a, vector_b):
    vector_a = np.asarray(vector_a, dtype=np.float64).reshape(-1)
    vector_b = np.asarray(vector_b, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator <= EPS:
        return 1.0 if np.linalg.norm(vector_a - vector_b) <= EPS else 0.0
    return float(np.clip(np.dot(vector_a, vector_b) / denominator, -1.0, 1.0))


def aligned_hyperedge_metrics(
    incidence_a,
    incidence_b,
    edge_evidence_a,
    edge_evidence_b,
    chunk_size=256,
):
    """Match latent edges event-wide, then evaluate aligned edge quantities."""
    _validate_pair(incidence_a, incidence_b)
    n_points, n_variables, n_edges = incidence_a.shape
    edge_evidence_a = np.asarray(edge_evidence_a, dtype=np.float64).reshape(-1)
    edge_evidence_b = np.asarray(edge_evidence_b, dtype=np.float64).reshape(-1)
    if edge_evidence_a.shape != (n_edges,) or edge_evidence_b.shape != (n_edges,):
        raise ValueError("peak hyperedge evidence does not match incidence")

    similarity = hyperedge_column_cosine(incidence_a, incidence_b, chunk_size)
    a_to_b = hungarian_maximize(similarity)
    b_to_a = np.empty(n_edges, dtype=np.int64)
    b_to_a[a_to_b] = np.arange(n_edges, dtype=np.int64)

    agreement_values = []
    legacy_jaccard_values = []
    for low in range(0, n_points, int(chunk_size)):
        high = min(n_points, low + int(chunk_size))
        labels_a = np.argmax(incidence_a[low:high], axis=2)
        labels_b = np.argmax(incidence_b[low:high], axis=2)
        labels_b_aligned = b_to_a[labels_b]
        agree = (labels_a == labels_b_aligned).sum(axis=1).astype(np.float64)
        agreement_values.extend((agree / n_variables).tolist())
        legacy_jaccard_values.extend((agree / (2.0 * n_variables - agree)).tolist())

    evidence_b_aligned = edge_evidence_b[a_to_b]
    matched_cosines = similarity[np.arange(n_edges), a_to_b]
    return {
        "a_to_b_hyperedge_match": [int(value) for value in a_to_b],
        "matched_incidence_column_cosine": summarize(matched_cosines),
        "aligned_top1_assignment_agreement": summarize(agreement_values),
        "aligned_top1_assignment_jaccard": summarize(legacy_jaccard_values),
        "aligned_peak_edge_evidence_cosine": _cosine(
            edge_evidence_a, evidence_b_aligned
        ),
        "aligned_peak_edge_evidence_kendall_tau_b": kendall_tau_b(
            edge_evidence_a, evidence_b_aligned
        ),
    }
