#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:24:13 2025

@author: ellie
"""

import argparse
import os
import pandas as pd
import networkx as nx
import numpy as np
from scipy.stats import truncnorm
import sys
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Patch

ROOT = os.path.dirname(__file__)
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import config
from data_io import load_catalog
from clustering import compute_nnd, build_spanning_tree, extract_forest
from analysis import find_nthresh
from utils import estimate_b_value, estimate_fractal_dimension


def set_config_param(name, value, fallback_fn=None, default=None):
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
    
def compute_metrics(forest_undirected, directed_forest, nthresh):
    # Whole‐graph metrics
    comps = list(nx.connected_components(forest_undirected))
    n_clusters = len(comps)
    sizes = [len(c) for c in comps]
    avg_nodes_per_cluster = np.mean(sizes) if sizes else 0

    # diameter = longest shortest‐path
    diameters = []
    for c in comps:
        sub = forest_undirected.subgraph(c)
        if len(sub)>1:
            diameters.append(nx.diameter(sub))
    avg_diameter = np.mean(diameters) if diameters else 0

    # depth: for directed, define roots as in_deg==0
    depths = []
    for root in [n for n,d in directed_forest.in_degree() if d==0]:
        lengths = nx.single_source_shortest_path_length(directed_forest, root)
        depths.extend(lengths.values())
    avg_depth = np.mean(depths) if depths else 0
    max_depth = np.max(depths) if depths else 0

    out_degs = [d for _,d in directed_forest.out_degree()]
    avg_out_degree = np.mean(out_degs) if out_degs else 0
    max_out_degree = np.max(out_degs) if out_degs else 0

    # branching coefficient: mean out_degree over non‐leaves
    non_leaves = [d for d in out_degs if d>0]
    avg_branching = np.mean(non_leaves) if non_leaves else 0

    # degree distribution moments
    degs = [d for _,d in forest_undirected.degree()]
    deg_mean, deg_std = (np.mean(degs), np.std(degs)) if degs else (0,0)
    max_degree = np.max(degs) if degs else 0

    # centrality summaries
    bet = nx.betweenness_centrality(forest_undirected)
    clo = nx.closeness_centrality(forest_undirected)
    mean_bet = np.mean(list(bet.values())) if bet else 0
    mean_clo = np.mean(list(clo.values())) if clo else 0
    max_bet = np.max(list(bet.values())) if bet else 0
    max_clo = np.max(list(clo.values())) if clo else 0

    # Largest subgraph (by node count)
    largest = max(comps, key=len) if comps else set()
    sub_u = forest_undirected.subgraph(largest)
    sub_d = directed_forest.subgraph(largest)
    ls_sizes = len(largest)
    # subgraph depth
    sub_depths = []
    for root in [n for n,d in sub_d.in_degree() if d==0]:
        sub_depths.extend(nx.single_source_shortest_path_length(sub_d, root).values())
    sub_avg_depth = np.mean(sub_depths) if sub_depths else 0
    sub_max_depth = np.max(sub_depths) if sub_depths else 0
    # subgraph out_degree
    sub_outs = [d for _,d in sub_d.out_degree()]
    sub_avg_out = np.mean(sub_outs) if sub_outs else 0
    sub_max_out = np.max(sub_outs) if sub_outs else 0
    # subgraph diameter
    sub_diams = [nx.diameter(sub_u)] if len(sub_u)>1 else [0]
    sub_diam = sub_diams[0]
    # subgraph branching
    sub_non_leaves = [d for d in sub_outs if d>0]
    sub_branch = np.mean(sub_non_leaves) if sub_non_leaves else 0
    # subgraph degree moments
    sub_degs = [d for _,d in sub_u.degree()]
    sub_deg_mean, sub_deg_std = (np.mean(sub_degs), np.std(sub_degs)) if sub_degs else (0,0)
    sub_max_deg = np.max(sub_degs) if sub_degs else 0
    # subgraph centralities
    sub_bet = nx.betweenness_centrality(sub_u)
    sub_clo = nx.closeness_centrality(sub_u)
    sub_mean_bet = np.mean(list(sub_bet.values())) if sub_bet else 0
    sub_mean_clo = np.mean(list(sub_clo.values())) if sub_clo else 0
    sub_max_bet = np.max(list(sub_bet.values())) if sub_bet else 0
    sub_max_clo = np.max(list(sub_clo.values())) if sub_clo else 0

    return {
        'nthresh': nthresh,
        'n_clusters': n_clusters,
        'avg_nodes_per_cluster': avg_nodes_per_cluster,
        'avg_diameter': avg_diameter,
        'avg_depth': avg_depth,
        'max_depth': max_depth,
        'avg_out_degree': avg_out_degree,
        'max_out_degree': max_out_degree,
        'avg_branching': avg_branching,
        'degree_mean': deg_mean,
        'max_degree': max_degree,
        'degree_std': deg_std,
        'mean_betweenness': mean_bet,
        'max_betweeness': max_bet,
        'mean_closeness': mean_clo,
        'max_closeness': max_clo,
        'largest_num_nodes': ls_sizes,
        'largest_avg_depth': sub_avg_depth,
        'largest_max_depth': sub_max_depth,
        'largest_avg_out_degree': sub_avg_out,
        'largest_max_out_degree': sub_max_out,
        'largest_diameter': sub_diam,
        'largest_branching': sub_branch,
        'largest_degree_mean': sub_deg_mean,
        'largest_degree_max': sub_max_deg,
        'largest_degree_std': sub_deg_std,
        'largest_mean_betweenness': sub_mean_bet,
        'largest_max_betweenness': sub_max_bet,
        'largest_mean_closeness': sub_mean_clo,
        'largest_max_closeness': sub_max_clo
    }
    
def run_simulations(df_orig, n_sims, loc_errs, output_prefix):
    # outdir = f"{output_prefix}/sim_{sim_index:03d}"
    # os.makedirs(outdir, exist_ok=True)
    all_metrics = []
    for loc_err in loc_errs:
        for sim in range(1, n_sims + 1):
            print(f"[Error={loc_err}] Simulation {sim}/{n_sims}")
            sigma_loc = loc_err / 3.0    # ≈ 1.667
            a = -loc_err / sigma_loc     # = –3
            b = +loc_err / sigma_loc     # = +3
            loc_dist = truncnorm(a, b, loc=0.0, scale=sigma_loc)
        
            # Inject location error
            df = df_orig.copy()
            df['x'] += loc_dist.rvs(size=len(df))
            df['y'] += loc_dist.rvs(size=len(df))
    
            # 1. Compute NND & parents
            nnd_dict = compute_nnd(df)
            for k in ('nnd','parent','T','R'):
                df[k] = nnd_dict[k]
            # df.to_csv(f"{outdir}/nnd.csv", index=False)
        
            # 2. Determine ETA0 threshold if not set
            eta0 = find_nthresh(df, n_init=50, random_state=42)
            # print(f"[sim {sim_index}] Estimated ETA0 = {eta0:.3e}")

            # 3. Build spanning tree & forest
            tree = build_spanning_tree(nnd_dict, eta0)
            forest = extract_forest(tree)

            # # save tree edges
            edges = pd.DataFrame([
                {'u': u, 'v': v, 'eta': a['eta'], 'strong': a['strong']}
                for u, v, a in tree.edges(data=True)
            ])
            # edges.to_csv(f"{outdir}/tree.csv", index=False)

            # undirected forest for connectivity
            forest_undirected = forest.copy()
            # build directed forest of strong edges
            directed = nx.DiGraph()
            directed.add_nodes_from(forest.nodes())
            for _, row in edges[edges['strong']].iterrows():
                directed.add_edge(int(row['u']), int(row['v']))
            # adj = nx.to_numpy_array(directed, nodelist=sorted(directed.nodes()), dtype=int)
            # pd.DataFrame(adj, index=sorted(directed.nodes()), columns=sorted(directed.nodes()))\
            #   .to_csv(f"{outdir}/adjacency.csv")

            # # 5. Clusters listing
            # with open(f"{outdir}/clusters.txt", 'w') as f:
            #     for i, comp in enumerate(nx.connected_components(forest), 1):
            #         f.write(f"Cluster {i}: {sorted(comp)}\n")
    
            metrics = compute_metrics(forest_undirected, directed, eta0)
            metrics['loc_err'] = loc_err
            metrics['sim'] = sim
            all_metrics.append(metrics)
        print(f"=== Completed simulations for loc_err = {loc_err} ===")
    return pd.DataFrame(all_metrics)


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo NND + clustering with truncated-normal location error"
    )
    parser.add_argument('-i','--input', required=True,
                        help='Input CSV or MAT catalog')
    parser.add_argument('-o','--output_prefix', default='mc_results',
                        help='Directory prefix for all simulations')
    parser.add_argument('--n_sims', type=int, default=2,
                        help='Number of Monte Carlo realizations')
    parser.add_argument('--loc_err', type=float, default=5.0,
                        help='Max absolute error (mm) in x,y')
    args = parser.parse_args()

    os.makedirs(args.output_prefix, exist_ok=True)

    df_orig = load_catalog(
        file_path=args.input,
        time_col="time",
        x_col="x",
        y_col="y",
        mag_col="mag",
        z_col=None,
        time_format=None
    )

    set_config_param('MAG_CUTOFF', config.MAG_CUTOFF)
    set_config_param('Q', config.Q, default=0.5)
    set_config_param('B', config.B, fallback_fn=lambda: estimate_b_value(df_orig['mag']))
    set_config_param('DF', config.DF, fallback_fn=lambda: estimate_fractal_dimension(
        df_orig[['x', 'y', 'z']] if 'z' in df_orig.columns else df_orig[['x', 'y']]
    ))
    config.ETA0 = None

    combined_df = run_simulations(df_orig, 100, [1,5,10,20], 'mc_results')
    combined_df.to_csv('mc_results/combined_metrics.csv', index=False)

    # Define error levels and colors
    loc_errs = sorted(combined_df['loc_err'].unique())
    colors = ['C0', 'C1', 'C2', 'C3']  # must match len(loc_errs)
    
    # Prepare metrics list (exclude sim and loc_err)
    metrics = [col for col in combined_df.columns if col not in ('loc_err', 'sim')]
    n_metrics = len(metrics)
    
    # Grid layout: 5 rows x 6 columns
    nrows, ncols = 5, 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_flat = axes.flatten()
    
    # Plot each metric with black mean marker
    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        data = [combined_df[combined_df['loc_err'] == err][metric].dropna() for err in loc_errs]
        bp = ax.boxplot(data,
                        patch_artist=True,
                        showmeans=True,
                        meanprops={
                            'marker': 'D',
                            'markeredgecolor': 'black',
                            'markerfacecolor': 'black',
                            'markersize': 5
                        })
        # Set face color for boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_title(metric)
        ax.set_xticks(range(1, len(loc_errs) + 1))
        ax.set_xticklabels([str(err) for err in loc_errs])
        ax.set_ylabel('Value')
    
    # Turn off unused subplots
    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].axis('off')
    
    # Add legend for error levels
    handles = [Patch(facecolor=col) for col in colors]
    labels = [f'{err} mm' for err in loc_errs]
    fig.legend(handles, labels, title='Location error', loc='upper right')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Save the figure
    output_file = 'mc_results/combined_boxplots.png'
    fig.savefig(output_file, dpi=300)
    print(f"Saved combined boxplots with black mean markers to {output_file}")

if __name__ == '__main__':
    main()