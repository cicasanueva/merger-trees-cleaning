#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Catalina Casanueva (cicasanueva@uc.cl), 2025
# ============================================================
# Step 4 — Merger tree cleaning: overlap-based deduplication and classification
#
# Reads the baryonic merger catalogue produced by Step 3, classifies each merger
# into one of six groups (A1/A2/B1/B2/C1/C2), detects particle-level overlaps
# between mergers that share the same infall snapshot, and applies four sequential
# filters to remove spurious duplicate entries arising from subhalo fragmentation.
#
# Outputs:
#   - baryon_mergers_dep_{sim}_gx{gxid}.csv      : classified catalogue with all flags
#   - FINAL_resumen_clean_mergers_{sim}_gx{gxid}.csv : final cleaned catalogue
# ============================================================

import os
import json
import argparse

import h5py
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from config import (
    FRAC_TRANSFER_THRESHOLD,
    FRAC_TRANSFER_COLS,
    OVERLAP_DOMINANT_FRAC,
    OVERLAP_SECONDARY_FRAC,
    OVERLAP_SECONDARY_MIN_N,
    KEEP_FRAC_IN_CONSERVED,
)
from utils import split_unique_id, get_main_branch_unique_ids


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_offset_safe(offset_array, subf):
    """Return [start, end] particle offsets for subhalo `subf`, or [0, 0] on error."""
    try:
        raw = offset_array[subf]
        if isinstance(raw, (np.ndarray, list)) and len(raw) == 2 and np.all(np.isfinite(raw)):
            return np.array([int(raw[0]), int(raw[1])], dtype=int)
        return np.array([0, 0], dtype=int)
    except Exception:
        return np.array([0, 0], dtype=int)


def _classify_group(row, frac_cols, threshold):
    """Return the two-letter group label (A1/A2/B1/B2/C1/C2)."""
    if row.get("flag_always_inside_r200", False):
        letter1 = "A"
    elif row.get("flag_never_inside_r200", False):
        letter1 = "B"
    else:
        letter1 = "C"

    fracs = [row.get(c, 0) for c in frac_cols]
    fracs_valid = [f for f in fracs if not (isinstance(f, float) and np.isnan(f))]
    letter2 = "2" if any(f >= threshold for f in fracs_valid) else "1"

    return letter1 + letter2


def _safe_json_list(val):
    """Serialize val (list, array, or scalar) to a JSON list string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return json.dumps([])
    if isinstance(val, str):
        try:
            return json.dumps(json.loads(val))
        except Exception:
            return json.dumps([])
    if isinstance(val, (list, np.ndarray)):
        return json.dumps([int(x) for x in val])
    return json.dumps([int(val)])


def _dominant_component(row):
    """Return ('stars'|'gas'|'dm') for the most massive component at infall."""
    m_stars = row.get("mass_stars_infall", 0) or 0
    m_gas   = row.get("mass_gas_infall",   0) or 0
    m_dm    = row.get("mass_dm_infall",    0) or 0
    if m_stars >= m_gas and m_stars >= m_dm:
        return "stars"
    elif m_gas >= m_dm:
        return "gas"
    else:
        return "dm"


def _overlap_fraction(pids_a, pids_b):
    """Fraction of pids_a that appear in pids_b (0.0 if pids_a is empty)."""
    if len(pids_a) == 0:
        return 0.0
    return float(np.isin(pids_a, pids_b).sum()) / len(pids_a)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def clean_merger_catalog(sim_name, target_subfind_id, snap_target):
    """
    Classify, de-overlap, and export the final baryonic merger catalogue.

    Parameters
    ----------
    sim_name           : simulation identifier (e.g. "LG1")
    target_subfind_id  : SubFind index of the central galaxy at z=0
    snap_target        : z=0 snapshot number
    """
    path_sim  = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}.hdf5"
    path_tree = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}_tree.dat"
    input_csv = (
        f"Datos/Baryon_Mergers/baryon_mergers_{sim_name}_gx_{target_subfind_id}_summary.csv"
    )
    output_dir = "Datos/Baryon_Mergers_Cleaned"
    os.makedirs(output_dir, exist_ok=True)

    out_classified = f"{output_dir}/baryon_mergers_dep_{sim_name}_gx{target_subfind_id}.csv"
    out_final      = (
        f"{output_dir}/FINAL_resumen_clean_mergers_{sim_name}_gx{target_subfind_id}.csv"
    )

    # ------------------------------------------------------------------
    # Load Step 3 output; keep only mergers with baryonic content at infall
    # ------------------------------------------------------------------
    df = pd.read_csv(input_csv)
    df = df[(df["mass_stars_infall"] > 0) | (df["mass_gas_infall"] > 0)].copy()
    df = df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Classify each merger into one of six groups
    # ------------------------------------------------------------------
    df["group"] = df.apply(
        lambda row: _classify_group(row, FRAC_TRANSFER_COLS, FRAC_TRANSFER_THRESHOLD),
        axis=1,
    )

    # ------------------------------------------------------------------
    # Load merger tree and central main branch (needed for Filter 4)
    # ------------------------------------------------------------------
    uid_target = f"{snap_target}{target_subfind_id:06d}"
    tree_bwd = nx.read_multiline_adjlist(path_tree, create_using=nx.DiGraph)
    subtree_main = nx.dfs_tree(tree_bwd, uid_target)
    central_main_branch_set = set(get_main_branch_unique_ids(subtree_main, uid_target))

    # ------------------------------------------------------------------
    # Load particle IDs at t_infall for every merger
    # ------------------------------------------------------------------
    print("[Step 4] Loading particle IDs at t_infall …")
    pids_by_uid = {}  # uid_infall -> {"stars": array, "gas": array, "dm": array}

    with h5py.File(path_sim, "r") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading particle IDs"):
            uid_inf = str(row["uid_infall"])
            if uid_inf in pids_by_uid:
                continue
            snap, subf = split_unique_id(int(uid_inf))
            snap_grp = f[f"SnapNumber_{snap}"]

            off_s  = _get_offset_safe(snap_grp["SubGroups/PartType4/Offsets"], subf)
            off_g  = _get_offset_safe(snap_grp["SubGroups/PartType0/Offsets"], subf)
            off_dm = _get_offset_safe(snap_grp["SubGroups/PartType1/Offsets"], subf)

            pids_stars = (
                snap_grp["PartType4/ParticleIDs"][off_s[0]:off_s[1]]
                if off_s[1] > off_s[0] else np.array([], dtype=np.int64)
            )
            pids_gas = (
                snap_grp["PartType0/ParticleIDs"][off_g[0]:off_g[1]]
                if off_g[1] > off_g[0] else np.array([], dtype=np.int64)
            )
            pids_dm = (
                snap_grp["PartType1/ParticleIDs"][off_dm[0]:off_dm[1]]
                if off_dm[1] > off_dm[0] else np.array([], dtype=np.int64)
            )

            pids_by_uid[uid_inf] = {
                "stars": pids_stars,
                "gas":   pids_gas,
                "dm":    pids_dm,
            }

    # ------------------------------------------------------------------
    # Initialise overlap flag/metadata columns
    # ------------------------------------------------------------------
    for suffix in ["start_c2", "contains_c2", "no_c"]:
        df[f"flag_overlap_remove_{suffix}"]       = False
        df[f"uids_overlap_{suffix}"]               = ""
        df[f"uid_disruption_overlap_{suffix}"]     = ""
        df[f"uid_overlap_kept_{suffix}"]           = ""
        df[f"frac_stars_A_in_B_overlap_{suffix}"]  = np.nan
        df[f"frac_stars_B_in_A_overlap_{suffix}"]  = np.nan
        df[f"frac_gas_A_in_B_overlap_{suffix}"]    = np.nan
        df[f"frac_gas_B_in_A_overlap_{suffix}"]    = np.nan
        df[f"frac_dm_A_in_B_overlap_{suffix}"]     = np.nan
        df[f"frac_dm_B_in_A_overlap_{suffix}"]     = np.nan

    # ------------------------------------------------------------------
    # Build overlap graph
    # ------------------------------------------------------------------
    print("[Step 4] Building overlap graph …")

    dom_frac  = OVERLAP_DOMINANT_FRAC  / 100.0
    sec_frac  = OVERLAP_SECONDARY_FRAC / 100.0
    sec_min_n = OVERLAP_SECONDARY_MIN_N

    G_overlap = nx.Graph()
    uids_infall = df["uid_infall"].astype(str).tolist()
    groups_arr  = df["group"].tolist()

    # Only compare pairs that share the same infall snapshot
    snap_groups = df.groupby("snap_infall", dropna=False)

    for _snap_val, idx_group in tqdm(snap_groups.groups.items(), desc="Overlap pairs"):
        idxs = list(idx_group)
        if len(idxs) < 2:
            continue

        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                ia, ib = idxs[i], idxs[j]
                uid_a  = uids_infall[ia]
                uid_b  = uids_infall[ib]

                if uid_a == uid_b:
                    continue
                if uid_a not in pids_by_uid or uid_b not in pids_by_uid:
                    continue

                pids_a = pids_by_uid[uid_a]
                pids_b = pids_by_uid[uid_b]

                dom_comp = _dominant_component(df.iloc[ia])
                pids_dom_a = pids_a[dom_comp]
                if len(pids_dom_a) == 0:
                    continue

                all_pids_b = np.concatenate(list(pids_b.values()))
                frac_dom = _overlap_fraction(pids_dom_a, all_pids_b)
                if frac_dom < dom_frac:
                    continue

                # Secondary components must also overlap if they contain enough particles
                secondary_ok = True
                for comp in ["stars", "gas", "dm"]:
                    if comp == dom_comp:
                        continue
                    pids_sec_a = pids_a[comp]
                    if len(pids_sec_a) < sec_min_n:
                        continue  # too few particles — skip this component
                    if _overlap_fraction(pids_sec_a, all_pids_b) < sec_frac:
                        secondary_ok = False
                        break

                if secondary_ok:
                    G_overlap.add_edge(ia, ib)

    # ------------------------------------------------------------------
    # Apply filters on each connected component of the overlap graph
    # ------------------------------------------------------------------
    print("[Step 4] Applying overlap filters …")

    keep_frac = KEEP_FRAC_IN_CONSERVED / 100.0

    def _latest_disruption(comp_indices):
        """Return the uid_disruption with the highest integer value in the group."""
        all_dis = [df.iloc[i]["uid_disruption"] for i in comp_indices]
        return max(all_dis, key=lambda u: int(u) if str(u).isdigit() else 0)

    def _apply_removal(conserved, removed_indices, suffix, comp_indices, pids_cons, pids_by_uid):
        """Mark `removed_indices` for removal and update metadata columns."""
        uid_cons    = uids_infall[conserved]
        uids_json   = _safe_json_list([int(uids_infall[i]) for i in comp_indices])
        latest_dis  = _latest_disruption(comp_indices)
        dom_cons    = _dominant_component(df.iloc[conserved])
        all_pids_removed = {}

        for other in removed_indices:
            uid_other  = uids_infall[other]
            pids_other = pids_by_uid.get(uid_other, {})
            dom_other  = _dominant_component(df.iloc[other])

            pids_dom_other   = pids_other.get(dom_other, np.array([]))
            all_pids_cons    = np.concatenate(list(pids_cons.values())) if pids_cons else np.array([])
            all_pids_other_c = np.concatenate(list(pids_other.values())) if pids_other else np.array([])

            frac_b_in_a = _overlap_fraction(pids_dom_other, all_pids_cons)
            frac_a_in_b = _overlap_fraction(pids_cons.get(dom_cons, np.array([])), all_pids_other_c)

            df.at[other,    f"flag_overlap_remove_{suffix}"]      = True
            df.at[other,    f"uids_overlap_{suffix}"]              = uids_json
            df.at[conserved, f"uids_overlap_{suffix}"]             = uids_json
            df.at[other,    f"uid_overlap_kept_{suffix}"]          = uid_cons
            df.at[conserved, f"uid_overlap_kept_{suffix}"]         = uid_cons
            df.at[other,    f"uid_disruption_overlap_{suffix}"]    = latest_dis
            df.at[conserved, f"uid_disruption_overlap_{suffix}"]   = latest_dis
            df.at[other,    f"frac_stars_A_in_B_overlap_{suffix}"] = frac_a_in_b
            df.at[other,    f"frac_stars_B_in_A_overlap_{suffix}"] = frac_b_in_a

    for component in nx.connected_components(G_overlap):
        comp = list(component)
        if len(comp) < 2:
            continue

        snap_by_idx  = {
            idx: (df.iloc[idx]["snap_infall"] if not pd.isna(df.iloc[idx]["snap_infall"]) else 9999)
            for idx in comp
        }
        comp_sorted  = sorted(comp, key=lambda i: (snap_by_idx[i], uids_infall[i]))
        oldest_idx   = comp_sorted[0]
        oldest_group = groups_arr[oldest_idx]
        c2_indices   = [i for i in comp if groups_arr[i] == "C2"]
        has_c        = any(groups_arr[i].startswith("C") for i in comp)

        # ---- Filter 1: C2-start conservancy ----
        if oldest_group == "C2":
            conserved     = oldest_idx
            uid_cons      = uids_infall[conserved]
            pids_cons     = pids_by_uid.get(uid_cons, {})
            all_pids_cons = np.concatenate(list(pids_cons.values())) if pids_cons else np.array([])
            dom_cons      = _dominant_component(df.iloc[conserved])

            to_remove = []
            for other in comp:
                if other == conserved:
                    continue
                uid_other  = uids_infall[other]
                pids_other = pids_by_uid.get(uid_other, {})
                dom_other  = _dominant_component(df.iloc[other])
                pids_dom_other = pids_other.get(dom_other, np.array([]))

                frac_b_in_a = _overlap_fraction(pids_dom_other, all_pids_cons)
                if frac_b_in_a < keep_frac:
                    continue

                # Secondary overlap check
                sec_ok = True
                for comp_name in ["stars", "gas", "dm"]:
                    if comp_name == dom_other:
                        continue
                    p = pids_other.get(comp_name, np.array([]))
                    if len(p) < sec_min_n:
                        continue
                    if _overlap_fraction(p, all_pids_cons) < sec_frac:
                        sec_ok = False
                        break
                if sec_ok:
                    to_remove.append(other)

            if to_remove:
                _apply_removal(conserved, to_remove, "start_c2", comp, pids_cons, pids_by_uid)
            continue

        # ---- Filter 2: C2-containing conservancy ----
        if c2_indices:
            conserved = min(
                c2_indices,
                key=lambda i: (snap_by_idx[i], uids_infall[i]),
            )
            uid_cons      = uids_infall[conserved]
            pids_cons     = pids_by_uid.get(uid_cons, {})
            all_pids_cons = np.concatenate(list(pids_cons.values())) if pids_cons else np.array([])
            dom_cons      = _dominant_component(df.iloc[conserved])

            to_remove = []
            for other in comp:
                if other == conserved:
                    continue
                uid_other  = uids_infall[other]
                pids_other = pids_by_uid.get(uid_other, {})
                dom_other  = _dominant_component(df.iloc[other])
                pids_dom_other = pids_other.get(dom_other, np.array([]))

                frac_b_in_a = _overlap_fraction(pids_dom_other, all_pids_cons)
                if frac_b_in_a < keep_frac:
                    continue

                sec_ok = True
                for comp_name in ["stars", "gas", "dm"]:
                    if comp_name == dom_other:
                        continue
                    p = pids_other.get(comp_name, np.array([]))
                    if len(p) < sec_min_n:
                        continue
                    if _overlap_fraction(p, all_pids_cons) < sec_frac:
                        sec_ok = False
                        break
                if sec_ok:
                    to_remove.append(other)

            if to_remove:
                _apply_removal(conserved, to_remove, "contains_c2", comp, pids_cons, pids_by_uid)
            continue

        # ---- Filter 3: B2/A2 overlap (no C-type mergers in the group) ----
        if not has_c:
            b2_indices = [i for i in comp if groups_arr[i] == "B2"]
            if b2_indices:
                conserved = min(
                    b2_indices,
                    key=lambda i: (snap_by_idx[i], uids_infall[i]),
                )
                uid_cons      = uids_infall[conserved]
                pids_cons     = pids_by_uid.get(uid_cons, {})
                all_pids_cons = np.concatenate(list(pids_cons.values())) if pids_cons else np.array([])

                to_remove = []
                for other in comp:
                    if other == conserved or groups_arr[other] != "A2":
                        continue
                    uid_other  = uids_infall[other]
                    pids_other = pids_by_uid.get(uid_other, {})
                    dom_other  = _dominant_component(df.iloc[other])
                    pids_dom_other = pids_other.get(dom_other, np.array([]))
                    frac_b_in_a = _overlap_fraction(pids_dom_other, all_pids_cons)
                    if frac_b_in_a >= keep_frac:
                        to_remove.append(other)

                if to_remove:
                    _apply_removal(conserved, to_remove, "no_c", comp, pids_cons, pids_by_uid)

    # ------------------------------------------------------------------
    # Filter 4 — Contiguous-snapshot test
    # ------------------------------------------------------------------
    print("[Step 4] Applying contiguous-snapshot test (Filter 4) …")

    for idx, row in df.iterrows():
        grp = row["group"]

        if grp == "A2":
            # If the dominant subhalo at the snapshot just before the satellite's
            # first appearance belongs to the central's main branch, this A2 entry
            # is a spurious fragmentation artifact.
            uid_inf = str(row["uid_infall"])
            snap_inf, subf_inf = split_unique_id(int(uid_inf))
            snap_before = snap_inf - 1
            if snap_before >= 0:
                uid_candidate = f"{snap_before:03d}{subf_inf:06d}"
                if uid_candidate in central_main_branch_set:
                    df.at[idx, "flag_overlap_remove_no_c"] = True

        # B2 mergers: the contiguous test is informational only; B2 is retained.

    # ------------------------------------------------------------------
    # Save classified catalogue (all mergers, with flags)
    # ------------------------------------------------------------------
    df.to_csv(out_classified, index=False)
    print(f"[OK] Classified catalogue saved to {out_classified} ({len(df)} rows)")

    # ------------------------------------------------------------------
    # Build and save the final cleaned catalogue
    # ------------------------------------------------------------------
    remove_mask = (
        df["flag_overlap_remove_start_c2"]   |
        df["flag_overlap_remove_contains_c2"] |
        df["flag_overlap_remove_no_c"]
    )
    df_final = df[~remove_mask].copy()

    # Update uid_disruption with the latest-disruption value set by the filters
    for idx, row_f in df_final.iterrows():
        for suffix in ["start_c2", "contains_c2", "no_c"]:
            new_uid = row_f.get(f"uid_disruption_overlap_{suffix}", "")
            if new_uid and str(new_uid).strip():
                df_final.at[idx, "uid_disruption"] = new_uid
                break

    df_final.to_csv(out_final, index=False)
    print(
        f"[OK] Final cleaned catalogue saved to {out_final} "
        f"({len(df_final)} rows, {remove_mask.sum()} removed)"
    )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"\nStep 4 summary — {sim_name} galaxy {target_subfind_id}")
    print("=" * 50)
    print(f"  Total baryonic mergers (input from Step 3): {len(df)}")
    print(f"  Removed by overlap filters:                 {remove_mask.sum()}")
    print(f"  Final clean catalogue:                      {len(df_final)}")
    print("\n  Group distribution (before cleaning):")
    for grp, cnt in df["group"].value_counts().sort_index().items():
        print(f"    {grp}: {cnt}")
    print("\n  Group distribution (after cleaning):")
    for grp, cnt in df_final["group"].value_counts().sort_index().items():
        print(f"    {grp}: {cnt}")


def final_depuracion(sim_name, target_subfind_id, snap_target):
    """
    Post-cleaning export step: read back the final catalogue and print a brief summary.
    Provided for pipeline compatibility; calls clean_merger_catalog() if the output is missing.
    """
    final_csv = (
        f"Datos/Baryon_Mergers_Cleaned/"
        f"FINAL_resumen_clean_mergers_{sim_name}_gx{target_subfind_id}.csv"
    )
    if not os.path.exists(final_csv):
        print(f"[WARN] Final CSV not found: {final_csv}. Run clean_merger_catalog() first.")
        return
    df_final = pd.read_csv(final_csv)
    print(f"\n[Step 4 / Final export] {sim_name} galaxy {target_subfind_id}")
    print(f"  Clean mergers in final catalogue: {len(df_final)}")


def run_merger_pipeline(sim_name, target_subfind_id, snap_target):
    """Run Step 4 (cleaning) followed by the final export step."""
    clean_merger_catalog(sim_name, target_subfind_id, snap_target)
    final_depuracion(sim_name, target_subfind_id, snap_target)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4: clean the baryonic merger catalogue via particle-overlap filters."
    )
    parser.add_argument("--sim",   required=True,       help="Simulation name (e.g. LG1)")
    parser.add_argument("--gx",    required=True, type=int, help="SubFind ID of the central galaxy at z=0")
    parser.add_argument("--snap0", required=True, type=int, help="z=0 snapshot number")
    args = parser.parse_args()
    run_merger_pipeline(args.sim, args.gx, args.snap0)
