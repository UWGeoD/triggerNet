#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 12:26:04 2025

@author: ellie

main.py

Orchestrates the nearest-neighbor clustering pipeline:
1. Load & filter catalog
2. Compute NND and parents
3. Build spanning tree and extract cluster forest
4. Save results and generate plots
"""

import argparse
import pandas as pd
import networkx as nx
import os, sys
import scipy
from utils import estimate_b_value, estimate_fractal_dimension
import config
import numpy as np
import matplotlib.pyplot as plt

# ROOT = os.path.dirname(__file__)
# SRC  = os.path.join(ROOT, "src")
# sys.path.insert(0, SRC)

from data_io import load_catalog
from clustering import compute_nnd, build_spanning_tree, extract_forest
from analysis import (
    plot_log_eta_hist,
    find_nthresh,
    plot_logTR_heatmap,
    shuffle_and_compute_nnd
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Nearest-neighbor earthquake clustering"
    )
    parser.add_argument('-i','--input', required=True,
                        help='Input CSV catalog')
    parser.add_argument('-o','--output_prefix', default='results',
                        help='Output prefix for CSV/PNG files')
    return parser.parse_args()

def set_config_param(name, value, fallback_fn=None, default=None):
    """
    Sets a config variable in the config module.
    - If `value` is not None, use it.
    - Else if `fallback_fn` is provided, call it to compute the value.
    - Else if `default` is provided, use that.
    """
    if value is not None:
        final_value = value
    elif fallback_fn is not None:
        final_value = fallback_fn()
        print(f"Estimated {name} = {final_value:.3f}")
    elif default is not None:
        final_value = default
        print(f"Defaulted {name} = {final_value}")
    else:
        final_value = None
        print(f"{name} remains unset")

    config.__dict__[name] = final_value

# def main():
#     args = parse_args()

#     # 1. load & filter
#     df = load_catalog(
#         file_path=args.input,
#         time_col=args.time_col,
#         x_col= args.x_col,
#         y_col= args.y_col,
#         mag_col= args.mag_col,
#         z_col=args.z_col,
#         time_format=args.time_format
#     )

#     # 2. NND & parents
#     nnd_dict = compute_nnd(df)
#     for k in ('nnd','parent','T','R'):
#         df[k] = nnd_dict[k]
#     df.to_csv(f"{args.output_prefix}_nnd.csv", index=False)

#     # 3. spanning tree & forest
#     tree   = build_spanning_tree(nnd_dict)
#     forest = extract_forest(tree)

#     edges = pd.DataFrame([
#         {'u':u,'v':v,'eta':a['eta'],'strong':a['strong']}
#         for u,v,a in tree.edges(data=True)
#     ])
#     edges.to_csv(f"{args.output_prefix}_tree.csv", index=False)

#     with open(f"{args.output_prefix}_clusters.txt","w") as f:
#         for i, comp in enumerate(nx.connected_components(forest),1):
#             f.write(f"Cluster {i}: {sorted(comp)}\n")

#     # 4a. histogram of log10 η
#     fig1, _ = plot_log_eta_hist(nnd_dict['nnd'])
#     fig1.savefig(f"{args.output_prefix}_hist.png", dpi=300)

#     # 4b. Weibull fit
#     params = fit_weibull(nnd_dict['nnd'])
#     fig2, _ = plot_weibull_fit(nnd_dict['nnd'], params)
#     fig2.savefig(f"{args.output_prefix}_weibull.png", dpi=300)

#     # 4c. optional GMM
#     if args.estimate_eta0:
#         eta0_est, _ = estimate_eta0_gmm(nnd_dict['nnd'])
#         print(f"Estimated η₀ (GMM): {eta0_est:.2e}")

#     print("Done. Outputs prefixed with", args.output_prefix)

def main():
    
    set_config_param('MAG_CUTOFF', config.MAG_CUTOFF)
    set_config_param('Q', config.Q, default=0.5)

    # 1. load & filter
    df = load_catalog(
        file_path="data/mixed_mode_orig.mat",
        time_col="time",
        x_col= "x",
        y_col= "y",
        mag_col= "mag",
        z_col=None,
        time_format=None
    )

    set_config_param('B', config.B, fallback_fn=lambda: estimate_b_value(df['mag']))
    set_config_param('DF', config.DF, fallback_fn=lambda: estimate_fractal_dimension(
        df[['x', 'y', 'z']] if 'z' in df.columns else df[['x', 'y']]
    ))

    # 2. NND & parents
    nnd_dict = compute_nnd(df)
    for k in ('nnd','parent','T','R'):
        df[k] = nnd_dict[k]
    df.to_csv(f"results_nnd.csv", index=False)
    
    set_config_param('ETA0', config.ETA0, fallback_fn=lambda: find_nthresh(df, n_init=50, random_state=42))

    # 3. spanning tree & forest
    tree   = build_spanning_tree(nnd_dict)
    forest = extract_forest(tree)
    edges = pd.DataFrame([
        {'u':u,'v':v,'eta':a['eta'],'strong':a['strong']}
        for u,v,a in tree.edges(data=True)
    ])
    edges.to_csv(f"results_tree.csv", index=False)
    
    directed_forest = nx.DiGraph()
    directed_forest.add_nodes_from(forest.nodes())
    strong_edges = edges[edges['strong']]
    for _, row in strong_edges.iterrows():
        directed_forest.add_edge(int(row['u']), int(row['v']))
    nodes = sorted(directed_forest.nodes())
    A = nx.to_numpy_array(directed_forest, nodelist=nodes, dtype=int)
    adj_df = pd.DataFrame(A, index=nodes, columns=nodes)
    adj_df.to_csv(f"results_adjacency.csv")

    with open(f"results_clusters.txt","w") as f:
        for i, comp in enumerate(nx.connected_components(forest),1):
            f.write(f"Cluster {i}: {sorted(comp)}\n")

    print("Done. Outputs prefixed with", "results")

if __name__ == '__main__':
    main()