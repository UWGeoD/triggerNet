
"""
clustering.py

Core clustering functions for nearest-neighbor earthquake/AE event analysis:

- compute_nnd: compute nearest-neighbor distances, parents, and normalized T/R
- build_spanning_tree: build a labeled spanning tree based on NND/parent relationships
- extract_forest: extract cluster forest by removing weak edges
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import config

def compute_nnd(df):
    """
    Compute nearest-neighbor metric η for each event in a catalog.

    For each event j, finds the prior event i* minimizing η, and records:
        - η (nearest-neighbor metric)
        - parent (index of i*)
        - normalized T, R components

    Args:
        df (pd.DataFrame): Must have columns ['time','x','y','mag'] (optionally 'z').

    Returns:
        dict:
            'nnd'    : np.ndarray, η_min for each event (np.nan for event 0)
            'parent' : np.ndarray, parent index for each event
            'T'      : np.ndarray, normalized time components
            'R'      : np.ndarray, normalized space components

    Raises:
        ValueError: if required columns are missing or config is incomplete.
    """
    # Validate inputs and config
    required_cols = ['time', 'x', 'y', 'mag']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' required in input DataFrame.")
    if getattr(config, "B", None) is None or getattr(config, "DF", None) is None:
        raise ValueError("Config parameters 'B' and 'DF' must be set before computing NND.")
    
    n = len(df)
    times = df['time'].to_numpy()
    mags  = df['mag'].to_numpy()
    x     = df['x'].to_numpy()
    y     = df['y'].to_numpy()
    z     = df['z'].to_numpy() if 'z' in df.columns else np.zeros(n)
    
    # Compute medians via pairwise distances
    coords = np.column_stack((x, y, z))
    all_dists = pdist(coords, metric='euclidean')
    medD = np.median(all_dists)
    all_dt = pdist(times.reshape(-1, 1), metric='euclidean')
    medT = np.median(all_dt)
    
    # Precompute moment weights
    M = 10 ** (-config.B * mags)
    sqrtM = np.sqrt(M)
    
    # Outputs
    nnd = np.full(n, np.nan)
    parent = np.full(n, -1, dtype=int)
    T_arr = np.full(n, np.nan)
    R_arr = np.full(n, np.nan)
    
    for j in range(1, n):
        # Compute distances/times to all prior events
        dx = x[:j] - x[j]
        dy = y[:j] - y[j]
        dz = z[:j] - z[j]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        dt = times[j] - times[:j]
        valid = dt > 0
        if not np.any(valid):
            continue
    
        # Normalize
        Dn = r[valid] / medD
        Tn = dt[valid] / medT
        W = sqrtM[:j][valid]
    
        # η-metric as defined in clustering literature
        eta_vals = (Dn ** config.DF) * Tn * (W ** 2)
        best = np.argmin(eta_vals)
        i_star = np.flatnonzero(valid)[best]
        nnd[j] = eta_vals[best]
        parent[j] = i_star
    
        # Also record T, R as per TopoMat
        R_arr[j] = (Dn[best] ** config.DF) * W[best]
        T_arr[j] = Tn[best] * W[best]
    
    return {'nnd': nnd, 'parent': parent, 'T': T_arr, 'R': R_arr}

    # # unpack
    # times      = df['time'].to_numpy()
    # coords = df[['x','y']].to_numpy()
    # if 'z' in df:
    #     coords = np.column_stack((coords, df['z'].to_numpy()))
    # mags   = df['mag'].to_numpy()
    # n      = len(df)

    # # build full matrices
    # all_dists  = squareform(pdist(coords, metric='euclidean'))
    # all_dt = times.reshape(1, n) - times.reshape(n, 1)

    # # weights
    # M     = 10.0 ** (-config.B * mags)
    # sqrtM = np.sqrt(M)

    # # outputs
    # nnd       = np.full(n, np.nan)
    # parent    = np.full(n, -1, dtype=int)
    # R_arr = np.zeros(n)
    # T_arr = np.zeros(n)

    # # loop “i = 1…N-1” → j = N-i
    # for i in range(1, n):
    #     j = n - i
    #     Dn  = all_dists[j, :j]
    #     Tn = all_dt[j, :j]

    #     # η-vector exactly as MATLAB: (Dis^Df) * (–Time) * M
    #     eta_vals = (Dn ** config.DF) * (-Tn) * M[:j]

    #     # find minimal η
    #     k = np.argmin(eta_vals)
    #     minval = eta_vals[k]

    #     # record
    #     nnd[j]       = minval
    #     parent[j]    = k
    #     R_arr[j] = (Dn[k] ** config.DF) * sqrtM[k]
    #     T_arr[j] = (-Tn[k])    * sqrtM[k]
    

    # return {'nnd': nnd, 'parent': parent, 'T': T_arr, 'R': R_arr}


def build_spanning_tree(nnd_dict, nthresh=None):
    """
    Build an undirected spanning tree from NND output.

    Each edge (j, parent[j]) is labeled with:
        - 'eta': nearest-neighbor metric value
        - 'strong': whether eta < nthresh (for cluster splitting)

    Args:
        nnd_dict (dict): Output from compute_nnd
        nthresh (float, optional): Threshold for strong links (default: config.ETA0)

    Returns:
        networkx.Graph: Spanning tree with edge labels.

    Raises:
        ValueError: if parent or nnd arrays are missing.
    """
    if nthresh is None:
        nthresh = getattr(config, "ETA0", None)
    if nthresh is None:
        raise ValueError("No eta threshold provided and config.ETA0 is not set.")

    nnd = nnd_dict['nnd']
    parent = nnd_dict['parent']
    n = len(nnd)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for j in range(1, n):
        i = parent[j]
        eta_val = nnd[j]
        strong = eta_val < nthresh
        G.add_edge(i, j, eta=eta_val, strong=strong)
    return G


def extract_forest(tree):
    """
    Extract cluster forest from a spanning tree by keeping only 'strong' edges.

    Args:
        tree (networkx.Graph): Output from build_spanning_tree with 'strong' edge attributes.

    Returns:
        networkx.Graph: Forest containing only strong edges (clusters).

    Notes:
        - Node set is unchanged; only edge set is filtered.
    """
    forest = nx.Graph()
    forest.add_nodes_from(tree.nodes())
    for u, v, attrs in tree.edges(data=True):
        if attrs.get('strong', False):
            forest.add_edge(u, v, **attrs)
    return forest
