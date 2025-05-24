#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis.py

Exploratory analysis and diagnostic tools for nearest-neighbor clustering.
Includes:
- Histograms and heatmaps of nearest-neighbor metrics (η, T, R)
- Automatic threshold estimation (η₀) via Gaussian Mixture Model (GMM)
- Shuffling-based null model for event catalogs
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from clustering import compute_nnd

def find_nthresh(df, **gmm_kwargs):
    """
    Estimate the η threshold (η₀) using a Gaussian Mixture Model (GMM)
    fit to the log10(η) distribution from original and shuffled catalogs.

    Args:
        df (pd.DataFrame): Catalog with computed 'nnd' column.
        **gmm_kwargs: Extra keyword args for GaussianMixture.

    Returns:
        float: Estimated η₀ (threshold separating clusters) in original scale (not log10).

    Notes:
        - shuffle_and_compute_nnd is used for the null/random distribution.
        - Typically used to set config.ETA0.
    """
    nnd_rand = shuffle_and_compute_nnd(df)
    real_eta = df['nnd'].values
    rand_eta = nnd_rand['nnd']
    mask_r = np.isfinite(real_eta) & (real_eta > 0)
    mask_s = np.isfinite(rand_eta) & (rand_eta > 0)
    X = np.log10(np.concatenate([real_eta[mask_r], rand_eta[mask_s]]))[:, None]

    gmm = GaussianMixture(n_components=2, **gmm_kwargs).fit(X)
    mu, cov, w = gmm.means_.flatten(), gmm.covariances_.flatten(), gmm.weights_.flatten()
    idx = np.argsort(mu)
    mu0, mu1 = mu[idx[0]], mu[idx[1]]
    s0, s1 = np.sqrt(cov[idx[0]]), np.sqrt(cov[idx[1]])
    w0, w1 = w[idx[0]], w[idx[1]]

    A = 1 / (2 * s1**2) - 1 / (2 * s0**2)
    B = mu0 / (s0**2) - mu1 / (s1**2)
    C = (mu1**2) / (2 * s1**2) - (mu0**2) / (2 * s0**2) - np.log((w1 * s0) / (w0 * s1))
    disc = B * B - 4 * A * C
    if disc < 0:
        x_star = (mu0 + mu1) / 2
    else:
        r1 = (-B + np.sqrt(disc)) / (2 * A)
        r2 = (-B - np.sqrt(disc)) / (2 * A)
        # Select intersection between means
        x_star = r1 if mu0 < r1 < mu1 else r2

    eta_star = 10 ** x_star
    return eta_star


def shuffle_and_compute_nnd(df, seed=None):
    """
    Shuffle columns of the input catalog independently, then compute nearest-neighbor distances.

    Args:
        df (pd.DataFrame): Catalog with columns ['time','x','y','mag'] (optionally 'z').
        seed (int, optional): Seed for reproducibility.

    Returns:
        dict: Output from compute_nnd() on the shuffled catalog.

    Notes:
        Used to generate null model for threshold estimation or for diagnostics.
    """
    rng = np.random.RandomState(seed)
    shuffled = df.copy()
    shuffled['x'] = df['x'].sample(frac=1, random_state=rng).values
    shuffled['y'] = df['y'].sample(frac=1, random_state=rng.randint(0, 2**32)).values
    shuffled['time'] = df['time'].sample(frac=1, random_state=rng.randint(0, 2**32)).values
    shuffled['mag'] = df['mag'].sample(frac=1, random_state=rng.randint(0, 2**32)).values
    if 'z' in df:
        shuffled['z'] = df['z'].sample(frac=1, random_state=rng.randint(0, 2**32)).values

    nnd_rand = compute_nnd(shuffled)
    return nnd_rand


def plot_log_eta_hist(df, bins=50, seed=None):
    """
    Plot histogram of log10(η) for original vs. shuffled catalogs.

    Args:
        df (pd.DataFrame): Must have a 'nnd' column of original nearest-neighbor distances.
        bins (int): Number of bins in histogram.
        seed (int or None): Random seed for reproducible shuffling.

    Returns:
        fig, ax: Matplotlib Figure and Axes objects.
    """

    # Check if nnd column exists
    if 'nnd' not in df.columns:
        raise ValueError("Input DataFrame must include a 'nnd' column.")

    orig = np.log10(df['nnd'][np.isfinite(df['nnd']) & (df['nnd'] > 0)].to_numpy())

    # Compute shuffled nnd
    nnd_rand_dict = shuffle_and_compute_nnd(df, seed=seed)
    rand_nnd = nnd_rand_dict['nnd']
    rand = np.log10(rand_nnd[np.isfinite(rand_nnd) & (rand_nnd > 0)])

    # Shared bin edges for fair comparison
    min_bin = min(orig.min(), rand.min())
    max_bin = max(orig.max(), rand.max())
    bin_edges = np.linspace(min_bin, max_bin, bins + 1)

    fig, ax = plt.subplots()
    ax.hist(orig, bins=bin_edges, alpha=0.6, label='Original', color='C0')
    ax.hist(rand, bins=bin_edges, alpha=0.6, label='Shuffled', color='C1')
    ax.set_xlabel(r'$\log_{10}(\eta)$')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of log10(η): Original vs Shuffled')
    ax.legend()
    return fig, ax


def plot_logTR_contours(df, levels=8, seed=None, grid_n=200, 
                        cmap_orig='Blues', cmap_rand='Oranges', cmap_combined='Greens'):
    """
    Plot smooth contour lines of log10(T) vs log10(R) for original and shuffled catalogs
    (with color differentiation), and also save a combined (original+shuffled) contour plot.

    Args:
        df (pd.DataFrame): DataFrame with 'T' and 'R' columns.
        levels (int): Number of contour levels.
        seed (int or None): Random seed for shuffling.
        output_path (str or None): If provided, saves <output_path>_overlaid.png and <output_path>_combined.png.
        grid_n (int): Grid size for the mesh.
        cmap_orig, cmap_rand, cmap_combined: Colormaps for original, shuffled, and combined contours.

    Returns:
        (fig_overlaid, ax_overlaid), (fig_combined, ax_combined)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    from analysis import shuffle_and_compute_nnd

    # --- Prepare data ---
    # Original
    T = np.asarray(df['T'])
    R = np.asarray(df['R'])
    mask = np.isfinite(T) & np.isfinite(R) & (T > 0) & (R > 0)
    logT = np.log10(T[mask])
    logR = np.log10(R[mask])
    data_orig = np.vstack([logT, logR])

    # Shuffled
    nnd_rand = shuffle_and_compute_nnd(df, seed=seed)
    T_rand = nnd_rand['T']
    R_rand = nnd_rand['R']
    mask_rand = np.isfinite(T_rand) & np.isfinite(R_rand) & (T_rand > 0) & (R_rand > 0)
    logT_rand = np.log10(T_rand[mask_rand])
    logR_rand = np.log10(R_rand[mask_rand])
    data_rand = np.vstack([logT_rand, logR_rand])

    # Combined
    logT_all = np.concatenate([logT, logT_rand])
    logR_all = np.concatenate([logR, logR_rand])
    data_all = np.vstack([logT_all, logR_all])

    # Grid
    xmin = min(logT.min(), logT_rand.min())
    xmax = max(logT.max(), logT_rand.max())
    ymin = min(logR.min(), logR_rand.min())
    ymax = max(logR.max(), logR_rand.max())
    range_min = min(xmin, ymin)
    range_max = max(xmax, ymax)
    X, Y = np.meshgrid(
        np.linspace(range_min, range_max, grid_n),
        np.linspace(range_min, range_max, grid_n)
    )
    positions = np.vstack([X.ravel(), Y.ravel()])

    # KDEs
    kde_orig = gaussian_kde(data_orig)
    kde_rand = gaussian_kde(data_rand)
    kde_all = gaussian_kde(data_all, bw_method=0.2)
    Z_orig = np.reshape(kde_orig(positions), X.shape)
    Z_rand = np.reshape(kde_rand(positions), X.shape)
    Z_all = np.reshape(kde_all(positions), X.shape)

    # --- Overlaid plot ---
    fig_overlaid, ax_overlaid = plt.subplots(figsize=(7, 6))
    cs_orig = ax_overlaid.contour(X, Y, Z_orig, levels=levels, cmap=cmap_orig, linewidths=2)
    cs_rand = ax_overlaid.contour(X, Y, Z_rand, levels=levels, cmap=cmap_rand, linewidths=2)
    solid_proxy = plt.Line2D([], [], color=plt.get_cmap(cmap_orig)(0.7), linestyle='-', linewidth=2)
    dashed_proxy = plt.Line2D([], [], color=plt.get_cmap(cmap_rand)(0.7), linestyle='-', linewidth=2)
    ax_overlaid.legend([solid_proxy, dashed_proxy], ['Original', 'Shuffled'])
    ax_overlaid.set_xlabel(r'$\log_{10}(T)$')
    ax_overlaid.set_ylabel(r'$\log_{10}(R)$')
    ax_overlaid.set_title('Contours of log10(T) vs log10(R): Original (solid) & Shuffled (dashed)')
    ax_overlaid.grid(True, linestyle=':', alpha=0.5)

    # --- Combined plot ---
    fig_combined, ax_combined = plt.subplots(figsize=(7, 6))
    cs_combined = ax_combined.contour(X, Y, Z_all, levels=levels, cmap=cmap_combined, linewidths=2)
    ax_combined.set_xlabel(r'$\log_{10}(T)$')
    ax_combined.set_ylabel(r'$\log_{10}(R)$')
    ax_combined.set_title('Contours of log10(T) vs log10(R): Combined Original + Shuffled')
    ax_combined.grid(True, linestyle=':', alpha=0.5)
    
    # Enforce symmetry
    for ax in [ax_overlaid, ax_combined]:  # add other axes as needed
        ax.set_xlim(range_min, range_max)
        ax.set_ylim(range_min, range_max)
        ax.set_aspect('equal', adjustable='box')

    return (fig_overlaid, ax_overlaid), (fig_combined, ax_combined)
