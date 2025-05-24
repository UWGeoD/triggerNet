"""
utils.py

Utility functions for earthquake/fracture catalog analysis:

- Gutenberg-Richter b-value estimation
- Spatial fractal (correlation) dimension estimation

These functions are used for parameter estimation in the nearest-neighbor clustering pipeline.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

def estimate_b_value(mags, m_min=None, bin_width=None):
    """
    Maximum-likelihood estimation of Gutenberg-Richter b-value.

    The b-value quantifies the relative occurrence of large and small events:
        b = log10(e) / (mean(mags) - (m_min - bin_width/2))

    Args:
        mags (array-like): Sequence of event magnitudes.
        m_min (float, optional): Minimum magnitude cutoff. If None, use minimum of mags.
        bin_width (float, optional): Magnitude bin width. If None, estimated from data.

    Returns:
        float: Estimated b-value.

    Raises:
        ValueError: If mags is empty or has insufficient unique values for bin width estimation.

    Reference:
        Aki, K. (1965). Maximum likelihood estimate of b in the formula log N = a - bM.
    """
    mags = np.asarray(mags)
    if mags.size == 0:
        raise ValueError("Empty magnitude array passed to estimate_b_value.")
    if m_min is None:
        m_min = mags.min()
    if bin_width is None:
        unique_mags = np.unique(np.round(mags, 3))
        if len(unique_mags) < 2:
            bin_width = 0.1
        else:
            bin_width = np.min(np.diff(unique_mags))
    mean_mag = mags.mean()
    b_est = np.log10(np.e) / (mean_mag - (m_min - bin_width / 2))
    return b_est


def estimate_fractal_dimension(coords, r_vals=None):
    """
    Estimate spatial fractal (correlation) dimension using the pair-count method.

    The correlation dimension Df is the slope in log-log space:
        C(r) ~ r^Df
    where C(r) is the fraction of pairs within distance r.

    Args:
        coords (array-like): (N, d) array of event positions (x/y or x/y/z).
        r_vals (array-like, optional): Radii at which to compute C(r). If None, auto-chosen.

    Returns:
        float: Estimated fractal (correlation) dimension.

    Reference:
        Grassberger, P., & Procaccia, I. (1983). Characterization of strange attractors.
    """
    pts = np.asarray(coords)
    if pts.ndim != 2:
        raise ValueError(f"coords must be 2D array of shape (N, d), got {pts.shape}")

    N = pts.shape[0]
    if N < 2:
        raise ValueError("At least two points required for fractal dimension estimation.")

    # Build k-d tree for fast neighbor counting
    tree = cKDTree(pts)

    # Choose radii for correlation sum
    if r_vals is None:
        # Compute all pairwise distances, ignore zeros on diagonal
        dmat = tree.sparse_distance_matrix(tree, max_distance=np.inf)
        dmat = coo_matrix(dmat)
        if dmat.data.size == 0:
            raise ValueError("All points are identical; cannot estimate fractal dimension.")
        dmax = np.max(dmat.data)
        r_vals = np.logspace(np.log10(dmax * 0.01), np.log10(dmax * 0.25), num=20)

    C = []
    for r in r_vals:
        # cKDTree.count_neighbors counts all pairs (including i=j); subtract N for self-pairs
        count = tree.count_neighbors(tree, r) - N
        # Normalize by total number of pairs (N choose 2)
        C.append(count / (N * (N - 1)))

    # Fit slope in linear region of log(C) vs log(r)
    log_r = np.log(r_vals)
    log_C = np.log(C)
    slope, _ = np.polyfit(log_r, log_C, 1)
    return float(slope)
