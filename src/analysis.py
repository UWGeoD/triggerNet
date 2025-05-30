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
from scipy.stats import gaussian_kde
import config

def find_nthresh(df, runs=10, **gmm_kwargs):
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
    # nnd_rand = shuffle_and_compute_nnd(df)
    real_eta = df['nnd'].values
    # rand_eta = nnd_rand['nnd']
    mask_r = np.isfinite(real_eta) & (real_eta > 0)
    # mask_s = np.isfinite(rand_eta) & (rand_eta > 0)
    X = np.log10(np.concatenate([real_eta[mask_r]]))[:, None]
    
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
    return eta_star, None

    # thresh_seed_pairs = []
    # for i in range(runs):
    #     # --- Shuffled ---
    #     nnd_rand_dict = shuffle_and_compute_nnd(df, seed=i)
    #     nnd_rand = nnd_rand_dict['nnd']
    #     mask_rand = np.isfinite(nnd_rand)
    #     log_eta_rand = np.log10(nnd_rand[mask_rand].astype(np.complex128))
    #     log_eta_rand = np.real(log_eta_rand)
    #     mask_log_rand = np.isfinite(log_eta_rand)
    #     log_eta_rand = log_eta_rand[mask_log_rand]
    #     quantile = 0.0005  # 5th percentile; adjust as needed (e.g., 0.01 for 1%)
    #     thresh = np.quantile(log_eta_rand, quantile)
    #     thresh_seed_pairs.append((thresh, i))

    # # Find the median threshold and its associated seed
    # thresh_arr = np.array([pair[0] for pair in thresh_seed_pairs])
    # seeds_arr = np.array([pair[1] for pair in thresh_seed_pairs])
    # median_idx = np.argsort(thresh_arr)[len(thresh_arr)//2]
    # median_thresh = thresh_arr[median_idx]
    # median_seed = seeds_arr[median_idx]
    # return 10**median_thresh, median_seed


def shuffle_and_compute_nnd(df, seed=42):
    """
    Shuffle rows of the (x,y) location together, but shuffle time and mag separately,
    then compute nearest‑neighbor distances.

    Args:
        df (pd.DataFrame): Catalog with columns ['time','x','y','mag'] (optionally 'z').
        seed (int, optional): Seed for reproducibility.

    Returns:
        dict: Output from compute_nnd() on the shuffled catalog.
    """
    rng = np.random.RandomState(seed)
    N = len(df)

    # Independent random permutations (like randperm in MATLAB)
    perm1 = rng.permutation(N)
    perm2 = rng.permutation(N)
    perm3 = rng.permutation(N)

    shuffled = df.copy()

    # Shuffle (x, y) together using perm1
    shuffled[['x', 'y']] = df.loc[perm1, ['x', 'y']].values

    # Shuffle time using perm2
    shuffled['time'] = df.loc[perm2, 'time'].values

    # Shuffle mag using perm3
    shuffled['mag'] = df.loc[perm3, 'mag'].values

    # If 'z' exists, shuffle it with (x, y)
    if 'z' in df.columns:
        shuffled['z'] = df.loc[perm1, 'z'].values

    # Now pass to your NND function
    nnd_rand = compute_nnd(shuffled)
    return nnd_rand

def plot_log_eta_hist(df, nnd_rand_dict, bins=50, seed=42):
    """
    Plot histogram of log10(η) for original vs. shuffled catalogs.

    Args:
        df (pd.DataFrame): Must have a 'nnd' column of original nearest-neighbor distances.
        bins (int): Number of bins in histogram.
        seed (int or None): Random seed for reproducible shuffling.

    Returns:
        fig, ax: Matplotlib Figure and Axes objects.
    """
    
    if 'nnd' not in df.columns:
        raise ValueError("Input DataFrame must include a 'nnd' column.")

    # --- Original ---
    nnd_orig = df['nnd'].to_numpy()
    mask_orig = np.isfinite(nnd_orig)
    log_eta_orig = np.log10(nnd_orig[mask_orig].astype(np.complex128))
    log_eta_orig = np.real(log_eta_orig)
    mask_log_orig = np.isfinite(log_eta_orig)
    log_eta_orig = log_eta_orig[mask_log_orig]

    # --- Shuffled ---
    # nnd_rand_dict = shuffle_and_compute_nnd(df, seed=seed)
    nnd_rand = nnd_rand_dict['nnd']
    mask_rand = np.isfinite(nnd_rand)
    log_eta_rand = np.log10(nnd_rand[mask_rand].astype(np.complex128))
    log_eta_rand = np.real(log_eta_rand)
    mask_log_rand = np.isfinite(log_eta_rand)
    log_eta_rand = log_eta_rand[mask_log_rand]

    # Shared bin edges for fair comparison
    min_bin = min(log_eta_orig.min(), log_eta_rand.min())
    max_bin = max(log_eta_orig.max(), log_eta_rand.max())
    bin_edges = np.linspace(min_bin, max_bin, bins + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Subplot 1: Original only
    axs[0].hist(log_eta_orig, bins=bin_edges, alpha=0.8, color='C0', label='Original')
    axs[0].set_xlabel(r'$\log_{10}(\eta)$')
    axs[0].set_ylabel('Count')
    axs[0].set_title('Original Catalog')
    thresh = np.log10(config.ETA0)
    axs[0].axvline(thresh, color='k', linestyle='--', linewidth=2, label=f'Separating line: $x={thresh:.2f}$')
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.5)

    # Subplot 2: Overlayed
    axs[1].hist(log_eta_orig, bins=bin_edges, alpha=0.6, color='C0', label='Original')
    axs[1].hist(log_eta_rand, bins=bin_edges, alpha=0.6, color='C1', label='Shuffled', histtype='step', linewidth=2)
    axs[1].set_xlabel(r'$\log_{10}(\eta)$')
    axs[1].set_title('Original with Shuffled Overlay')
    axs[1].axvline(thresh, color='k', linestyle='--', linewidth=2, label=f'Separating line: $x={thresh:.2f}$')
    axs[1].legend()
    axs[1].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    return fig, axs


def plot_logTR_contours(df, nnd_rand_dict, levels=8, seed=42, grid_n=200, 
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

    # --- Prepare data ---
    T = np.asarray(df['T'])
    R = np.asarray(df['R'])
    logT = np.log10(T.astype(np.complex128))
    logR = np.log10(R.astype(np.complex128))
    logT = np.real(logT)
    logR = np.real(logR)
    mask = np.isfinite(logT) & np.isfinite(logR)
    logT = logT[mask]
    logR = logR[mask]
    data_orig = np.vstack([logT, logR])
    
    # Shuffled
    # nnd_rand = shuffle_and_compute_nnd(df, seed=seed)
    T_rand = nnd_rand_dict['T']
    R_rand = nnd_rand_dict['R']
    logT_rand = np.log10(T_rand.astype(np.complex128))
    logR_rand = np.log10(R_rand.astype(np.complex128))
    logT_rand = np.real(logT_rand)
    logR_rand = np.real(logR_rand)
    mask_rand = np.isfinite(logT_rand) & np.isfinite(logR_rand)
    logT_rand = logT_rand[mask_rand]
    logR_rand = logR_rand[mask_rand]
    data_rand = np.vstack([logT_rand, logR_rand])

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
    Z_orig = np.reshape(kde_orig(positions), X.shape)
    Z_rand = np.reshape(kde_rand(positions), X.shape)

    fig, axs = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    # Subplot 1: Original only
    axs[0].scatter(logT, logR, color=plt.get_cmap(cmap_orig)(0.6), s=15, alpha=0.35, label='Original (scatter)')
    cs_orig = axs[0].contour(X, Y, Z_orig, levels=levels, cmap=cmap_orig, linewidths=2)
    axs[0].set_xlabel(r'$\log_{10}(T)$', fontsize=14)
    axs[0].set_ylabel(r'$\log_{10}(R)$', fontsize=14)
    axs[0].set_title('Original Catalog', fontsize=16)
    c = np.log10(config.ETA0)
    xlim = axs[0].get_xlim()
    ylim = axs[0].get_ylim()
    # Draw separating line (slope -1)
    points = []
    for x in xlim:
        y = -x + c
        if ylim[0] <= y <= ylim[1]:
            points.append((x, y))
    for y in ylim:
        x = -y + c
        if xlim[0] <= x <= xlim[1]:
            points.append((x, y))
    points = list(set(points))
    if len(points) >= 2:
        points = sorted(points, key=lambda p: p[0])
        x_vals, y_vals = zip(*points)
        axs[0].plot(x_vals, y_vals, 'k--', linewidth=2.5, label=f'Separating line: y=-x{c:+.2f}')
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.5)
    axs[0].set_aspect('equal', adjustable='box')

    # Subplot 2: Overlayed
    axs[1].scatter(logT, logR, color=plt.get_cmap(cmap_orig)(0.6), s=15, alpha=0.35, label='Original (scatter)')
    axs[1].scatter(logT_rand, logR_rand, color=plt.get_cmap(cmap_rand)(0.7), s=15, alpha=0.35, label='Shuffled (scatter)')
    cs_orig = axs[1].contour(X, Y, Z_orig, levels=levels, cmap=cmap_orig, linewidths=2)
    cs_rand = axs[1].contour(X, Y, Z_rand, levels=levels, cmap=cmap_rand, linewidths=2)
    axs[1].set_xlabel(r'$\log_{10}(T)$', fontsize=14)
    axs[1].set_title('Original with Shuffled Overlay', fontsize=16)
    # Draw separating line (slope -1)
    xlim = axs[1].get_xlim()
    ylim = axs[1].get_ylim()
    points = []
    for x in xlim:
        y = -x + c
        if ylim[0] <= y <= ylim[1]:
            points.append((x, y))
    for y in ylim:
        x = -y + c
        if xlim[0] <= x <= xlim[1]:
            points.append((x, y))
    points = list(set(points))
    if len(points) >= 2:
        points = sorted(points, key=lambda p: p[0])
        x_vals, y_vals = zip(*points)
        axs[1].plot(x_vals, y_vals, 'k--', linewidth=2.5, label=f'Separating line: y=-x{c:+.2f}')
    axs[1].legend()
    axs[1].grid(True, linestyle=':', alpha=0.5)
    axs[1].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    return fig, axs