#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

Entry point for the Nearest-Neighbor Earthquake Clustering pipeline.

Steps:
  1. Load and filter earthquake catalog
  2. Compute nearest-neighbor distances (NND) and assign parents
  3. Build spanning tree and extract cluster forest
  4. Save outputs and generate summary plots

For full usage instructions, see README.md.

Author: Ellie Johnson
"""

import argparse
import pandas as pd
import networkx as nx
import sys
import os

from utils import estimate_b_value, estimate_fractal_dimension
import config
from data_io import load_catalog
from clustering import compute_nnd, build_spanning_tree, extract_forest
from analysis import (
    find_nthresh,
    plot_log_eta_hist,
    plot_logTR_contours
)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def parse_args():
    """Parse command-line arguments for user flexibility and guidance."""
    parser = argparse.ArgumentParser(
        description="Nearest-neighbor clustering of earthquake (or AE) catalogs."
    )
    parser.add_argument('-i', '--input', required=True, 
                        help="Path to input event catalog (.csv or .mat)")
    parser.add_argument('-o', '--output_prefix', default='results',
                        help="Prefix for output files (default: results)")
    parser.add_argument('--time_col', default='time', 
                        help="Name of time column in catalog (default: time)")
    parser.add_argument('--x_col', default='x', 
                        help="Name of x (or longitude/latitude) column (default: x)")
    parser.add_argument('--y_col', default='y', 
                        help="Name of y (or latitude/longitude) column (default: y)")
    parser.add_argument('--z_col', default=None, 
                        help="Name of z/depth column if available (default: None)")
    parser.add_argument('--mag_col', default='mag', 
                        help="Name of magnitude column (default: mag)")
    parser.add_argument('--time_format', default=None,
                        help="Optional datetime format for parsing (e.g., '%%Y-%%m-%%d %%H:%%M:%%S')")
    parser.add_argument('--mag_cutoff', type=float, default=None,
                        help="Minimum magnitude to include in analysis (default: None)")
    parser.add_argument('--q', type=float, default=0.5,
                        help="Normalization exponent Q (default: 0.5)")
    parser.add_argument('--b', type=float, default=None,
                        help="b-value for magnitude normalization (default: auto-estimate)")
    parser.add_argument('--df', type=float, default=None,
                        help="Fractal dimension (default: auto-estimate)")
    parser.add_argument('--eta0', type=float, default=None,
                        help="Threshold for strong links (default: auto-estimate)")
    return parser.parse_args()


def resolve_config_param(name, cli_value, config_value, fallback_fn=None, default=None):
    """
    Resolves a config parameter priority:
        1. CLI value (explicit user input)
        2. Config value (from config.py)
        3. Fallback function (auto-calc)
        4. Hardcoded default

    Sets and returns the resolved value.
    """
    if cli_value is not None:
        final_value = cli_value
        print(f"[CONFIG] {name} set by user input: {final_value}")
    elif config_value is not None:
        final_value = config_value
        print(f"[CONFIG] {name} from config file: {final_value}")
    elif fallback_fn is not None:
        final_value = fallback_fn()
        print(f"[CONFIG] {name} estimated: {final_value}")
    elif default is not None:
        final_value = default
        print(f"[CONFIG] {name} defaulted: {final_value}")
    else:
        final_value = None
        print(f"[WARN] {name} remains unset")
    setattr(config, name, final_value)
    return final_value


def main():
    args = parse_args()

    # 1. Load and filter catalog
    print(f"\n[STEP 1] Loading catalog: {args.input}")
    try:
        df = load_catalog(
            file_path=args.input,
            time_col=args.time_col,
            x_col=args.x_col,
            y_col=args.y_col,
            mag_col=args.mag_col,
            z_col=args.z_col,
            time_format=args.time_format
        )
    except Exception as e:
        print(f"[ERROR] Failed to load catalog: {e}")
        sys.exit(1)
    print(f"    Loaded {len(df)} events.")
    
    # 2. Resolve config parameters in proper priority
    #   MAG_CUTOFF
    resolve_config_param('MAG_CUTOFF', args.mag_cutoff, getattr(config, 'MAG_CUTOFF', None))
    #   Q
    resolve_config_param('Q', args.q, getattr(config, 'Q', None), default=0.5)
    #   B (b-value)
    resolve_config_param('B', args.b, getattr(config, 'B', None), fallback_fn=lambda: estimate_b_value(df['mag']))
    #   DF (fractal dimension)
    if args.z_col and args.z_col in df.columns:
        resolve_config_param('DF', args.df, getattr(config, 'DF', None),
                            fallback_fn=lambda: estimate_fractal_dimension(df[['x', 'y', 'z']]))
    else:
        resolve_config_param('DF', args.df, getattr(config, 'DF', None),
                            fallback_fn=lambda: estimate_fractal_dimension(df[['x', 'y']]))

    print(f"    Config: B={config.B}, DF={config.DF}, Q={config.Q}, MAG_CUTOFF={config.MAG_CUTOFF}")

    # 3. Compute nearest-neighbor distances (NND)
    print("[STEP 2] Computing nearest-neighbor distances...")
    nnd_dict = compute_nnd(df)
    for k in ('nnd', 'parent', 'T', 'R'):
        df[k] = nnd_dict[k]
    nnd_csv = os.path.join(RESULTS_DIR, f"{args.output_prefix}_nnd.csv")
    df.to_csv(nnd_csv, index=False)
    print(f"    Saved nearest-neighbor results to {nnd_csv}")

    # 4. Threshold for strong links
    eta0, eta0_seed = find_nthresh(df, runs=10)
    resolve_config_param('ETA0', args.eta0, getattr(config, 'ETA0', None),
                        fallback_fn=lambda: eta0)
    print(f"    Using ETA0 (threshold): {config.ETA0}")

    # 5. Build spanning tree and cluster forest
    print("[STEP 3] Building spanning tree and extracting cluster forest...")
    tree = build_spanning_tree(nnd_dict, nthresh=config.ETA0)
    forest = extract_forest(tree)

    # Save tree edge list
    edges = pd.DataFrame([
        {'u': u, 'v': v, 'eta': a['eta'], 'strong': a['strong']}
        for u, v, a in tree.edges(data=True)
    ])
    tree_csv = os.path.join(RESULTS_DIR, f"{args.output_prefix}_tree.csv")
    edges.to_csv(tree_csv, index=False)
    print(f"    Spanning tree saved to {tree_csv}")

    # Save adjacency matrix of strong links (as .csv)
    directed_forest = nx.DiGraph()
    directed_forest.add_nodes_from(forest.nodes())
    strong_edges = edges[edges['strong']]
    for _, row in strong_edges.iterrows():
        directed_forest.add_edge(int(row['u']), int(row['v']))
    nodes = sorted(directed_forest.nodes())
    A = nx.to_numpy_array(directed_forest, nodelist=nodes, dtype=int)
    adj_df = pd.DataFrame(A, index=nodes, columns=nodes)
    adj_csv = os.path.join(RESULTS_DIR, f"{args.output_prefix}_adjacency.csv")
    adj_df.to_csv(adj_csv)
    print(f"    Strong-link adjacency matrix saved to {adj_csv}")

    # Save list of clusters (forest connected components)
    cluster_txt = os.path.join(RESULTS_DIR, f"{args.output_prefix}_clusters.txt")
    with open(cluster_txt, "w") as f:
        for i, comp in enumerate(nx.connected_components(forest), 1):
            f.write(f"Cluster {i}: {sorted(comp)}\n")
    print(f"    Cluster list saved to {cluster_txt}")

    # 6. Plot and save histogram of log10 η
    print("[STEP 4] Generating and saving plots...")
    fig1, ax = plot_log_eta_hist(df, bins=50, seed=eta0_seed)
    hist_png = os.path.join(PLOTS_DIR, f"{args.output_prefix}_hist.png")
    fig1.savefig(hist_png, dpi=300)

    fig_overlaid, ax_overlaid = plot_logTR_contours(
        df, levels=8, seed=eta0_seed
    )
    fig_overlaid.savefig(os.path.join(PLOTS_DIR, f"{args.output_prefix}_overlaid.png"), dpi=300, bbox_inches='tight')
    print(f"    Plots saved to {PLOTS_DIR}")

    print("\n[COMPLETE] Nearest-neighbor clustering pipeline finished.")
    print(f"Results saved with prefix: {args.output_prefix}_*")


if __name__ == '__main__':
    main()
