#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Catalina Casanueva (cicasanueva@uc.cl), 07/2025
# ============================================================
# Step 2 — Merger detection from the AMIGA merger tree (DM-only and baryonic)
# for a given central galaxy (identified by its SubFind ID at z = 0).
#
# Input structure:
# ----------------
# - Merger tree generated with the AMIGA code (Knollmann, S. R. & Knebe, A. 2009, ApJS, 182, 608).
# - HDF5 file with all simulation snapshots (CIELO format).
#
# Procedure:
# ----------
# 1. Extract the full main progenitor branch of the central galaxy (`uid_target`).
# 2. Identify all branches that physically merge into that main branch:
#    - The reversed graph (`G_inv`) is traversed from `uid_target` backward in time.
#    - Excluded:
#      · Branches that already belong to the main branch.
#      · Branches that previously merged with another branch that does merge into the main.
#
#    For example:
#       Suppose A and B are progenitor galaxies and at snapshot 60 A + B → C,
#       then at snapshot 50 C merges into the central galaxy.
#       In this case only the branch of A (the more massive progenitor) is kept.
#
#    This prevents counting the same physical merger event more than once.
#
# 3. For each detected physical merger, reconstruct its main progenitor branch:
#    - From the last snapshot before merging into the central (`uid_before_merge`,
#      i.e. the last UID that differs from the central)
#    - Traversed both backward (progenitors) and forward (descendants) combining `G` and `G_fwd`.
#
# Output file:
# ------------
# (1) Summary CSV:
#     - `uid_before_merge`: last UID before touching the central's main branch.
#     - `uid_merge`: UID where the satellite merges with the central (coincides with the main branch).
#     - `snap_merge`: snapshot of the merger.
#     - `z_merge`: corresponding redshift.
#     - `origin_node`: root node of the secondary branch in the tree.
# ============================================================

import os
import h5py
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import split_unique_id, get_main_branch_unique_ids

# ------------------------------------------------------------
# General configuration
# ------------------------------------------------------------

def detect_mergers_from_tree(sim_name, target_subfind_id, snap_target):
    #sim_name = "LG1"
    #target_subfind_id = 4337
    #snap_target = 128
    uid_target = f"{snap_target}{target_subfind_id:06d}"

    # Input and output paths
    path_sim = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}.hdf5"
    path_tree = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}_tree.dat"
    
    output_dir = "Datos/All_Mergers/"
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)   
    
    csv_out = f"{output_dir}/mergers_{sim_name}_gx_{target_subfind_id}_summary.csv"
    
    if os.path.exists(csv_out):
        print(f'Skipping {sim_name}-{target_subfind_id}: already processed')
        # Already processed: skip to the next galaxy
        return


    # ------------------------------------------------------------
    # Load redshift per snapshot
    # ------------------------------------------------------------
    snap_to_z = {}
    with h5py.File(path_sim, 'r') as f:
        for k in f.keys():
            if k.startswith("SnapNumber_"):
                snap = int(k.split("_")[1])
                try:
                    z = f[k]["Header"]["Redshift"][()]
                    snap_to_z[snap] = z
                except:
                    continue
    all_snaps = sorted(snap_to_z)

    # ------------------------------------------------------------
    # Load merger trees and main branch
    # ------------------------------------------------------------
    print(f"[INFO] Loading merger trees and main branch from UID {uid_target}")
    G = nx.read_multiline_adjlist(path_tree, create_using=nx.DiGraph)  # backward tree
    G_fwd = nx.read_multiline_adjlist(path_tree)                       # forward tree
    G_inv = G.reverse(copy=True)

    # Extract full main branch of the central galaxy
    subtree_main = nx.dfs_tree(G, uid_target)
    main_branch_ids = set(get_main_branch_unique_ids(subtree_main, uid_target))

    # ------------------------------------------------------------
    # Detect mergers from the tree
    # ------------------------------------------------------------
    print(f"[INFO] Detecting mergers from tree")
    merging_branches = {}

    # Use single-target shortest paths from all nodes to the central
    shortest_paths = nx.single_target_shortest_path(G_inv, uid_target)

    for node, path in tqdm(shortest_paths.items()):
        if node == uid_target or node in main_branch_ids:
            continue
        if len(path) < 2:
            continue

        for i, uid in enumerate(path):
            if uid in main_branch_ids:
                idx_merger = i
                break
        else:
            continue  # no real merger found

        uid_before = path[idx_merger - 1]

        # Verify that this branch actually merges and is not a prior part of the main branch
        if any(uid in main_branch_ids for uid in get_main_branch_unique_ids(nx.dfs_tree(G, node), node)):
            continue  # skip if this node's own main branch overlaps with central's

        # Backward and forward branches from the merger point
        branch_back = get_main_branch_unique_ids(nx.dfs_tree(G, uid_before), uid_before)
        try:
            branch_forward = get_main_branch_unique_ids(nx.dfs_tree(G_fwd, uid_before), uid_before)
        except:
            branch_forward = []

        # Concatenate without duplicating UIDs
        full_branch = list(dict.fromkeys(branch_back + branch_forward))
        merging_branches[uid_before] = full_branch

    # ------------------------------------------------------------
    # Build output tables: summary CSV
    # ------------------------------------------------------------
    #rows_parquet = []
    rows_csv = []

    for uid_merge, branch in merging_branches.items():
        uid_by_snap = {split_unique_id(int(uid))[0]: uid for uid in branch}
        #row = {"origin_node": uid_merge}

        #for snap in all_snaps:
        #    row[f"snap_{snap:03d}"] = uid_by_snap.get(snap, "")
        #rows_parquet.append(row)

        # Identify the merger snapshot
        snaps = sorted(uid_by_snap)
        try:
            snap_merge = min(s for s in snaps if uid_by_snap[s] in main_branch_ids)
            snap_prev = max(s for s in snaps if s < snap_merge)
            rows_csv.append({
                "origin_node": uid_merge,
                "snap_merge": snap_merge,
                "z_merge": snap_to_z.get(snap_merge, None),
                "uid_merge": uid_by_snap[snap_merge],
                "uid_before_merge": uid_by_snap[snap_prev],
            })
        except:
            rows_csv.append({
                "origin_node": uid_merge,
                "snap_merge": None,
                "z_merge": None,
                "uid_merge": None,
                "uid_before_merge": None,
            })

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------
    #df_parquet = pd.DataFrame(rows_parquet)
    #df_parquet.to_parquet(parquet_out, index=False)
    #print(f"[✅] Guardado Parquet en {parquet_out} con {len(df_parquet)} ramas fusionadas")

    if rows_csv:
        df_csv = pd.DataFrame(rows_csv)
        df_csv.to_csv(csv_out, index=False)
        print(f"[OK] Summary CSV saved to {csv_out}")
    else:
        df_csv = pd.DataFrame(columns=[
            "origin_node",
            "snap_merge",
            "z_merge",
            "uid_merge",
            "uid_before_merge"
        ])
        df_csv.to_csv(csv_out, index=False)
        print(f"[WARN] Empty CSV saved to {csv_out} (no mergers detected for {sim_name} - {target_subfind_id})")


    # ------------------------------------------------------------
    # Optional: statistics summary
    # ------------------------------------------------------------
    print("\n" + "="*40)
    print(f"\nMerger summary statistics for {sim_name} - {target_subfind_id} (Step 2)")
    print("\n" + "="*40)
    
    print(f"   - Physical mergers detected: {len(df_csv)}")

    num_ok = df_csv["snap_merge"].notna().sum()
    num_bad = df_csv["snap_merge"].isna().sum()

    print(f"   - With well-defined snap_merge: {num_ok}")
    print(f"   - With undefined snap_merge (tracking issues): {num_bad}")

    # Merger distribution per snapshot
    if num_ok > 0:
        print("\n   - Merger distribution per snapshot:")
        distrib = df_csv["snap_merge"].value_counts().sort_index().reset_index()
        distrib.columns = ["Snapshot", "N_mergers"]
        print(distrib.to_string(index=False))

