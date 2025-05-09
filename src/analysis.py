#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 12:23:55 2025

@author: ellie

analysis.py

Exploratory analysis: histogram of log10 η, Weibull fit, and GMM-based threshold estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from sklearn.mixture import GaussianMixture
from clustering import compute_nnd
from matplotlib.lines import Line2D


def plot_log_eta_hist(nnd, bins=100):
    """
    Plot histogram of log10 η values to inspect unimodal/bimodal distribution.

    Args:
        nnd: array of nearest-neighbor distances (η), may contain np.nan or np.inf.
        bins: number of histogram bins.

    Returns:
        Matplotlib Figure and Axes.
    """
    eta = nnd[np.isfinite(nnd)]
    log_eta = np.log10(eta)
    fig, ax = plt.subplots()
    ax.hist(log_eta, bins=bins, density=True)
    ax.set_xlabel('log10 η')
    ax.set_ylabel('Density')
    ax.set_title('Histogram of log10 η')
    return fig, ax


def find_nthresh(df, **gmm_kwargs):
    """
    Estimate eta threshold via analytic GMM intersection, then
    plot log10(T) vs log10(R) contours for original and shuffled.

    Returns:
      eta_star: threshold on original η scale
    """
    # prepare data
    nnd_rand = shuffle_and_compute_nnd(df)
    real_eta = df['nnd']
    rand_eta = nnd_rand['nnd']
    mask_r = np.isfinite(real_eta) & (real_eta > 0)
    mask_s = np.isfinite(rand_eta) & (rand_eta > 0)
    X = np.log10(np.concatenate([real_eta[mask_r], rand_eta[mask_s]]))[:,None]
    
    gmm = GaussianMixture(n_components=2, **gmm_kwargs).fit(X)
    mu, cov, w = gmm.means_.flatten(), gmm.covariances_.flatten(), gmm.weights_.flatten()
    idx = np.argsort(mu)
    mu0, mu1 = mu[idx[0]], mu[idx[1]]
    s0, s1   = np.sqrt(cov[idx[0]]), np.sqrt(cov[idx[1]])
    w0, w1   = w[idx[0]], w[idx[1]]

    A = 1/(2*s1*s1) - 1/(2*s0*s0)
    B = mu0/(s0*s0)       - mu1/(s1*s1)
    C = (mu1*mu1)/(2*s1*s1) - (mu0*mu0)/(2*s0*s0) - np.log((w1*s0)/(w0*s1))
    disc = B*B - 4*A*C
    if disc < 0:
        x_star = (mu0+mu1)/2
    else:
        r1 = (-B + np.sqrt(disc)) / (2*A)
        r2 = (-B - np.sqrt(disc)) / (2*A)
        x_star = r1 if mu0 < r1 < mu1 else r2

    eta_star = 10**x_star
    
    return eta_star


def plot_logTR_heatmap(T, R, output_path=None, bins=100):
    """
    Plot a heatmap of log10(T) vs log10(R) as a 2D histogram.

    Args:
        T (array‑like): normalized time components (η’s T part).
        R (array‑like): normalized space components (η’s R part).
        output_path (str, optional): if given, save the figure to this path.
        bins (int or [int,int]): number of bins in each dimension.
    Returns:
        fig, ax: the Matplotlib figure and axes objects.
    """
    # Remove NaNs
    mask = np.isfinite(T) & np.isfinite(R)
    logT = np.log10(T[mask])
    logR = np.log10(R[mask])

    fig, ax = plt.subplots(figsize=(6,5))
    h = ax.hist2d(logT, logR, bins=bins, cmap='viridis')
    cb = fig.colorbar(h[3], ax=ax)
    cb.set_label('Counts')

    ax.set_xlabel(r'$\log_{10}T$')
    ax.set_ylabel(r'$\log_{10}R$')
    ax.set_title('Heatmap of log10(T) vs log10(R)')
    ax.grid(True, linestyle=':', alpha=0.5)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig, ax

def shuffle_and_compute_nnd(df, seed=None):
    """
    Shuffle of x, y, z, time, and mag independently,
    then compute nearest‐neighbor on the shuffled catalog.

    Args:
        df: DataFrame with columns ['time','x','y','mag'] and optional 'z'.
        seed: Base random seed for reproducibility.

    Returns:
        dict from compute_nnd on the shuffled catalog.
    """
    rng = np.random.RandomState(seed)

    # Use pandas.sample to shuffle each series independently
    shuffled = df.assign(
        x    = df['x'].sample(frac=1, random_state=rng).values,
        y    = df['y'].sample(frac=1, random_state=rng.randint(0, 2**32)).values,
        time = df['time'].sample(frac=1, random_state=rng.randint(0, 2**32)).values,
        mag  = df['mag'].sample(frac=1, random_state=rng.randint(0, 2**32)).values,
        **({'z': df['z'].sample(frac=1, random_state=rng.randint(0, 2**32)).values}
           if 'z' in df else {})
    )
    nnd_rand = compute_nnd(shuffled)
    
    # # Histogram of log10(eta): original vs shuffled
    # orig = np.log10(df['nnd'].dropna())
    # arr = nnd_rand['nnd']            # a NumPy array
    # rand = np.log10(arr[~np.isnan(arr)])
    
    # bins = np.linspace(min(orig.min(), rand.min()),
    #                    max(orig.max(), rand.max()), 50)
    
    # plt.figure(figsize=(8, 5))
    # plt.hist(orig, bins=bins, alpha=0.6, label='Original', color='C0')
    # plt.hist(rand, bins=bins, alpha=0.6, label='Shuffled', color='C1')
    # plt.xlabel('log10(η)')
    # plt.ylabel('Count')
    # plt.legend()
    # plt.title('Comparison of original vs shuffled η distributions')
    # plt.savefig("results_hist_comparison.png", dpi=300)
    
    return nnd_rand