#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 12:23:43 2025

@author: ellie

clustering.py

Functions to extract nearest-neighbor distances, build spanning tree and cluster forest.
"""
import numpy as np
import networkx as nx
import config
from distances import compute_eta, compute_TR
from scipy.spatial.distance import pdist


def compute_nnd(df):
    n = len(df)
    times = df['time'].to_numpy()
    mags  = df['mag'].to_numpy()
    x     = df['x'].to_numpy()
    y     = df['y'].to_numpy()
    z     = df['z'].to_numpy() if 'z' in df.columns else np.zeros(n)

    # 1) Precompute medians via pdist (only half‐matrix)
    coords = np.column_stack((x,y,z))
    all_dists = pdist(coords, metric='euclidean')
    medD = np.median(all_dists)

    # For time differences, we only want positive lags, but pdist on times gives abs differences.
    # Because abs doesn’t change the median, we can use it:
    all_dt = pdist(times.reshape(-1,1), metric='euclidean')
    medT = np.median(all_dt)

    # 2) Precompute moment‐weights
    M      = 10 ** (-config.B * mags)
    sqrtM  = np.sqrt(M)

    # Prepare outputs
    nnd    = np.full(n, np.nan)
    parent = np.full(n, -1, dtype=int)
    T_arr  = np.full(n, np.nan)
    R_arr  = np.full(n, np.nan)

    for j in range(1, n):
        # raw to all earlier
        dx = x[:j] - x[j]
        dy = y[:j] - y[j]
        dz = z[:j] - z[j]
        r  = np.sqrt(dx*dx + dy*dy + dz*dz)

        dt = times[j] - times[:j]
        valid = dt > 0
        if not np.any(valid):
            continue

        # median‐normalize
        Dn = r[valid]  / medD
        Tn = dt[valid] / medT
        W  = sqrtM[:j][valid]

        # η-metric
        eta_vals = (Dn**config.DF) * Tn * (W**2)   # W = sqrt(M), so W^2=M

        best     = np.argmin(eta_vals)
        i_star   = np.flatnonzero(valid)[best]
        nnd[j]   = eta_vals[best]
        parent[j]= i_star

        # compute T,R exactly as TopoMat: R=Dn^Df*W, T=Tn*W
        R_arr[j] = (Dn[best]**config.DF) * W[best]
        T_arr[j] =  Tn[best]          * W[best]

    return {'nnd': nnd, 'parent': parent, 'T': T_arr, 'R': R_arr}


def build_spanning_tree(nnd_dict, nthresh=config.ETA0):
    """
    Build an undirected spanning tree from NND outputs.

    Args:
        nnd_dict: output dict from compute_nnd()

    Returns:
        NetworkX Graph with nodes [0..n-1] and edges (j,parent[j]) labeled with
        'eta': η_min and 'strong': η_min < ETA0.
    """
    nnd = nnd_dict['nnd']
    parent = nnd_dict['parent']
    G = nx.Graph()
    n = len(nnd)
    G.add_nodes_from(range(n))
    for j in range(1, n):
        i = parent[j]
        eta_val = nnd[j]
        strong = eta_val < nthresh
        G.add_edge(i, j, eta=eta_val, strong=strong)
    return G


def extract_forest(tree):
    """
    From a spanning tree, remove weak edges to produce a forest of clusters.

    Args:
        tree: NetworkX Graph with 'strong' edge attribute.

    Returns:
        forest: NetworkX Graph containing only strong edges, same node set.
    """
    forest = nx.Graph()
    forest.add_nodes_from(tree.nodes())
    # add only strong edges
    for u, v, attrs in tree.edges(data=True):
        if attrs.get('strong', False):
            forest.add_edge(u, v, **attrs)
    return forest
