#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:24:49 2025

@author: ellie

utils.py
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

def estimate_b_value(mags, m_min=None, Δm=None):
    """
    Maximum-likelihood estimate of Gutenberg-Richter b-value.
    b = log10(e) / (mean(mags) - (m_min - Δm/2))

    Args:
        mags : 1D array of magnitudes.
        m_min: minimum magnitude cutoff (if None, use mags.min()).
        Δm   : magnitude bin width (if None, assume 0.1).
    Returns:
        b_est : estimated b-value.
    """
    mags = np.asarray(mags)
    if m_min is None:
        m_min = mags.min()
    if Δm is None:
        # typical magnitude bin (catalog resolution)
        Δm = np.min(np.diff(np.unique(np.round(mags, 3))))  
    mean_mag = mags.mean()
    b_est = np.log10(np.e) / (mean_mag - (m_min - Δm/2))
    return b_est

# Using a section of the distribution which obeys the power law distribution and
# fits the AE fractal dimension definition (𝜇(𝑅 < 𝑟𝑖) = 𝐴ℎ𝑟𝑖𝐷𝑓), 𝐷𝑓 can be 
# obtained with 𝑙𝑜𝑔𝜇 =𝑐𝑜𝑛𝑠𝑡𝑎𝑛𝑡 + 𝐷𝑓 ∗ 𝑙𝑜𝑔𝑟 𝑖.
def estimate_fractal_dimension(coords, r_vals=None):  ###### just use 1.6
    """
    Estimate spatial fractal (correlation) dimension using the
    pair-count method: C(r) ~ r^D₂.

    Args:
        coords: (N×d) array of spatial positions (e.g. lat/lon or x/y/z).
        r_vals: radii at which to compute C(r). If None, auto-pick log-spaced.
    Returns:
        D2 : estimated fractal (correlation) dimension (slope of log C vs. log r).
    """
    pts = np.asarray(coords)
    N, d = pts.shape

    # build fast tree
    tree = cKDTree(pts)

    # choose radii
    if r_vals is None:
        # from 1% to 25% of max pairwise distance
        dmat = tree.sparse_distance_matrix(tree, max_distance=np.inf)
        dmat = coo_matrix(dmat)  # Convert to COO format
        dmax = np.max(dmat.data)
        r_vals = np.logspace(np.log10(dmax*0.01), np.log10(dmax*0.25), num=20)

    C = []
    for r in r_vals:
        # count all pairs (i<j) with distance <= r
        # cKDTree.count_neighbors counts all pairs, including i=j; subtract N
        count = tree.count_neighbors(tree, r) - N
        # normalize by N*(N-1)
        C.append(count / (N*(N-1)))

    # fit slope in the linear region of log(C) vs. log(r)
    log_r = np.log(r_vals)
    log_C = np.log(C)
    # do a simple linear fit
    slope, _ = np.polyfit(log_r, log_C, 1)
    # D2 ≈ slope
    return float(slope)