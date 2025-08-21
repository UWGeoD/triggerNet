"""
utils.py

Utility functions for earthquake/fracture catalog analysis:

- Gutenberg-Richter b-value estimation

These functions are used for parameter estimation in the nearest-neighbor clustering pipeline.
"""

import numpy as np

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
        Aki, K. (1965). Aki, K. Maximum likelihood estimate of b in the formula log N = a - bM and its confidence limits.
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