#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Catalina Casanueva (cicasanueva@uc.cl), 07/2025
# ============================================================
# Step 3 — Baryonic merger characterisation for a given central galaxy
# (identified by its SubFind ID at z = 0).
#
# This step reads the `mergers_gx{ID}_summary.csv` file produced by step 2,
# which contains the full branches of physical mergers ending at a central galaxy.
#
# For each merger, the infall snapshot (`snap_infall`) is the last snapshot at which the
# satellite is outside the central's halo (R200 recomputed in step 1). From the next
# snapshot onward the satellite is considered to have crossed into the halo.
#
# The satellite UID at `t_infall` (`uid_infall`) and its last UID before disruption
# (`uid_disruption`, corresponding to the snapshot just before merger) are also computed.
#
# The following quantities are computed for each merger:
# ---------------------------------------------------------------
# * Redshift and UID at `t_infall` (`uid_infall`, `z_infall`)
# * Satellite UID at disruption (`uid_disruption`)
#
# * Satellite masses at `t_infall`:
#     - Stellar (`mass_stars_infall`)
#     - Gas (`mass_gas_infall`)
#     - Dark matter (`mass_dm_infall`)
#     - Total baryonic (`mass_baryon_infall`)
#
# * Mass fractions (at `t_infall`) that end up in the central galaxy at z = 0:
#     - Stars           → `Mstell_frac_t_infall_in_central_z0`
#     - Gas             → `Mgas_frac_t_infall_in_central_z0`
#     - Dark matter     → `Mdm_frac_t_infall_in_central_z0`
#
# * Star formation from the satellite's gas at `t_infall`:
#     - Mass fraction of 1st-generation stars → `frac_star1_from_gas_infall_in_central_z0`
#     - Mass fraction of 2nd-generation stars → `frac_star2_from_gas_infall_in_central_z0`
#     - Corresponding masses that end up in the central galaxy
#
# Additional flags:
# ---------------------------------------------------------------
# * `flag_had_stars`, `flag_had_gas`, `flag_had_dm`:
#     indicate whether the satellite had each component type at any point
#
# * `flag_always_inside_r200`: satellite was always inside the halo;
#     in this case the first UID (earliest snapshot) is used as `uid_infall`.
#
# * `flag_never_inside_r200`: satellite never crossed into the halo;
#     in this case the last UID (latest snapshot) is used as `uid_infall`.
#
# * `flag_CM_hdf5_tinfall_sat` / `flag_CM_hdf5_tinfall_cen`:
#     indicate that the shrinking-sphere CM could not be computed (too few stars);
#     the HDF5-tabulated position was used instead.
# ============================================================

# ==== IMPORTS ====
import os
import h5py
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

from utils import (
    split_unique_id, get_main_branch_unique_ids,
    recalculate_cm, pos_co_to_phys, vel_co_to_phys,
    pos_co_to_phys_cm, vel_co_to_phys_cm
)

def get_offset_safe(offset_array, subf):
    try:
        raw = offset_array[subf]
        if isinstance(raw, (np.ndarray, list)) and len(raw) == 2 and np.all(np.isfinite(raw)):
            return np.array([int(raw[0]), int(raw[1])], dtype=int)
        else:
            return np.array([0, 0], dtype=int)
    except Exception:
        return np.array([0, 0], dtype=int)


    
# ==== GENERAL CONFIGURATION ====
def characterize_baryonic_mergers(sim_name, target_subfind_id, snap_target):
    #sim_name = "LG1"
    #target_subfind_id = 4337
    #snap_target = 128
    uid_target = f"{snap_target}{target_subfind_id:06d}"
    output_dir = "Datos/Baryon_Mergers"

    # ==== INPUT/OUTPUT PATHS ====
    path_sim = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}.hdf5"
    path_tree = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}_tree.dat"
    path_csv = f"Datos/All_Mergers/mergers_{sim_name}_gx_{target_subfind_id}_summary.csv"
    path_rvir = f"Datos/CM_R200/r200_mainbranches_all_central_z0_{sim_name}.csv"
    path_out = f"{output_dir}/baryon_mergers_{sim_name}_gx_{target_subfind_id}_summary.csv"

    os.makedirs(output_dir, exist_ok=True)

    # ==== LOAD MERGER TREE AND CENTRAL MAIN BRANCH ====
    tree_1 = nx.read_multiline_adjlist(path_tree, create_using=nx.DiGraph)
    tree_full = nx.read_multiline_adjlist(path_tree)  # forward tree
    subtree_main = nx.dfs_tree(tree_1, uid_target)
    central_main_branch = get_main_branch_unique_ids(subtree_main, uid_target)
    central_branch_by_snap = {split_unique_id(int(uid))[0]: uid for uid in central_main_branch}

    # ==== LOAD INPUT TABLES ====
    df = pd.read_csv(path_csv)
    df_rvir = pd.read_csv(path_rvir)
    dict_rvir = dict(zip(df_rvir["uid"].astype(str), df_rvir["r200_kpc"]))

    # ==== REDSHIFT LOOKUP TABLE ====
    dict_snap_to_z = {}
    with h5py.File(path_sim, "r") as f:
        for key in f.keys():
            if key.startswith("SnapNumber_"):
                snap = int(key.split("_")[1])
                z = f[key]["Header/Redshift"][()]
                dict_snap_to_z[snap] = z

    # ==== INITIALISE OUTPUT LISTS ====
    had_stars_list = []
    had_gas_list = []
    had_dm_list = []

    snap_infall_list = []
    uid_infall_list = []
    z_infall_list = []
    uid_disruption_list = []

    flag_always_inside_r200_list = []
    flag_never_inside_r200_list = []
    flag_nostars_tinfall_sat_list = []
    flag_nostars_tinfall_cen_list = []

    mass_star_tinfall_list = []
    mass_gas_tinfall_list = []
    mass_dm_tinfall_list = []
    mass_baryon_tinfall_list = []

    stellar_mass_frac_infall_in_central_z0_list = []
    gas_mass_frac_infall_in_central_z0_list = []
    dm_mass_frac_infall_in_central_z0_list = []

    frac_star1_from_gas_in_central_z0_list = []
    frac_star2_from_gas_in_central_z0_list = []
    mass_star1_from_gas_in_central_z0_list = []
    mass_star2_from_gas_in_central_z0_list = []

    # ==== COUNTER: DM-ONLY MERGERS ====
    count_dm_only_mergers = 0

    with h5py.File(path_sim, "r") as f:
        snap_z0 = f[f"SnapNumber_{snap_target}"]
        subf_central_z0 = split_unique_id(int(central_branch_by_snap[int(snap_target)]))[1]  
        offset_cen_z0 = get_offset_safe(snap_z0["SubGroups/PartType4/Offsets"], subf_central_z0)
        pids_cen_z0 = snap_z0["PartType4/ParticleIDs"][offset_cen_z0[0]:offset_cen_z0[1]]

        offset_cen_z0_gas = get_offset_safe(snap_z0["SubGroups/PartType0/Offsets"], subf_central_z0)
        pids_cen_z0_gas = snap_z0["PartType0/ParticleIDs"][offset_cen_z0_gas[0]:offset_cen_z0_gas[1]]   

        offset_cen_z0_dm = get_offset_safe(snap_z0["SubGroups/PartType1/Offsets"], subf_central_z0)
        pids_cen_z0_dm = snap_z0["PartType1/ParticleIDs"][offset_cen_z0_dm[0]:offset_cen_z0_dm[1]]   

        # ==== Load star data at z=0 (loaded once for all mergers) ====
        header = snap_z0['Header']
        h = header['HubbleParam'][()]

        pids_star_z0 = snap_z0["PartType4/ParticleIDs"][:]
        masses_star_z0 = snap_z0["PartType4/Masses"][:] * 1e10 / h
        subfind_star_z0 = snap_z0["PartType4/SubFindNumber"][:]

        pid_to_mass_star_z0 = dict(zip(pids_star_z0, masses_star_z0))
        pid_to_subfind_star_z0 = dict(zip(pids_star_z0, subfind_star_z0))
        subfind_central_z0 = subf_central_z0  

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing mergers"):
                if np.isnan(row["uid_before_merge"]) == True:
                    print(f'[WARN] uid {row["origin_node"]} in {sim_name}-{target_subfind_id} has no uid_before_merge')
                    continue
                uid_sat_before_merger = str(int(row["uid_before_merge"]))
                branch = get_main_branch_unique_ids(nx.dfs_tree(tree_1, uid_sat_before_merger), uid_sat_before_merger) # da sólo branch desde snap before merger para atrás
                snap_uid_dict = {
                    int(split_unique_id(int(uid))[0]): uid
                    for uid in branch
                }            
                # ==== Check baryonic content of this satellite ====
                had_stars = False
                had_gas = False
                had_dm = False

                for uid in branch:
                    snap, subf = split_unique_id(int(uid))
                    for comp_type, parttype in [("stars", "PartType4"), ("gas", "PartType0"), ("dm", "PartType1")]:

                        offset = get_offset_safe(f[f"SnapNumber_{snap}/SubGroups/{parttype}/Offsets"], subf)

                        if comp_type == "stars" and offset[1] > offset[0]:
                            had_stars = True
                        elif comp_type == "gas" and offset[1] > offset[0]:
                            had_gas = True
                        elif comp_type == "dm" and offset[1] > offset[0]:
                            had_dm = True

                    # Stop early if all three components have been confirmed
                    if had_stars and had_gas and had_dm:
                        break

                # Skip DM-only mergers (no stars or gas throughout history).
                # Remove this check if you want to keep all mergers including DM-only.
                if had_stars == False and had_gas == False:
                    count_dm_only_mergers += 1
                    continue
                else:
                    uid_disruption_list.append(uid_sat_before_merger)
                    had_stars_list.append(had_stars)
                    had_gas_list.append(had_gas)
                    had_dm_list.append(had_dm)

                    # ==== Determine infall time ====
                    snap_first_inside = None
                    flag_nostars_tinfall_sat = False
                    flag_nostars_tinfall_cen = False
                    
                    snap_missing = False
                    for uid_sat in branch[::-1]:

                        snap_sat, subf_sat = split_unique_id(int(uid_sat))
                        if snap_sat not in central_branch_by_snap.keys():
                            snap_missing = True
                            print(f'[WARN] {snap_sat} not in main branch of {sim_name}-{target_subfind_id}; '
                                  f'minimum snap in branch is {min(central_branch_by_snap.keys())}')
                            continue
                        uid_central = central_branch_by_snap[snap_sat]

                        rvir = dict_rvir[uid_central]

                        snap_grp = f[f"SnapNumber_{snap_sat}"]
                        header = snap_grp['Header']
                        a = header['Time'][()]
                        z = header['Redshift'][()]
                        h = header['HubbleParam'][()]
                        omega0 = header['Omega0'][()]
                        omegaL = header['OmegaLambda'][()]
                        omega = omega0 + omegaL
                        omegaR = 0

                        H0 = 0.1 * h
                        adot = a * H0 * np.sqrt(omegaL + omega0 * a**-3 + omegaR * a**-4 - (omega - 1) * a**-2)

                        # ==== Central CM: recompute with shrinking sphere if >= 100 stars ====
                        subf_c = split_unique_id(int(uid_central))[1]
                        offset_c = get_offset_safe(snap_grp["SubGroups/PartType4/Offsets"], subf_c)

                        if offset_c[1] > offset_c[0] and offset_c[1]-offset_c[0]>=100:
                            pos_c = snap_grp["PartType4/Coordinates"][offset_c[0]:offset_c[1]]
                            vel_c = snap_grp["PartType4/Velocities"][offset_c[0]:offset_c[1]]
                            mass_c = snap_grp["PartType4/Masses"][offset_c[0]:offset_c[1]] * 1e10 / h
                            cm0_cen = pos_co_to_phys_cm(snap_grp["SubGroups/SubGroupPos"][subf_c], a, h)
                            vcm0_cen = vel_co_to_phys_cm(
                                snap_grp["SubGroups/SubGroupPos"][subf_c],
                                snap_grp["SubGroups/SubGroupVel"][subf_c],
                                a, h, adot
                            )
                            x_c, y_c, z_c = pos_co_to_phys(pos_c, a, h)
                            vx_c, vy_c, vz_c = vel_co_to_phys(pos_c, vel_c, a, h, adot)
                            cmx_cen, cmy_cen, cmz_cen, *_ = recalculate_cm(
                                x_c, y_c, z_c, vx_c, vy_c, vz_c, mass_c,
                                cm0_cen[0], cm0_cen[1], cm0_cen[2],
                                vcm0_cen[0], vcm0_cen[1], vcm0_cen[2]
                            )
                        else:
                            flag_nostars_tinfall_cen = True
                            cmx_cen, cmy_cen, cmz_cen = pos_co_to_phys_cm(
                                snap_grp["SubGroups/SubGroupPos"][subf_c], a, h
                            )

                        # ==== Satellite CM: recompute with shrinking sphere if >= 100 stars ====
                        offset_s = get_offset_safe(snap_grp["SubGroups/PartType4/Offsets"], subf_sat)
                        if offset_s[1] > offset_s[0] and offset_s[1]-offset_s[0]>=100:
                            pos_s = snap_grp["PartType4/Coordinates"][offset_s[0]:offset_s[1]]
                            vel_s = snap_grp["PartType4/Velocities"][offset_s[0]:offset_s[1]]
                            mass_s = snap_grp["PartType4/Masses"][offset_s[0]:offset_s[1]] * 1e10 / h
                            cm0_sat = pos_co_to_phys_cm(snap_grp["SubGroups/SubGroupPos"][subf_sat], a, h)
                            vcm0_sat = vel_co_to_phys_cm(
                                snap_grp["SubGroups/SubGroupPos"][subf_sat],
                                snap_grp["SubGroups/SubGroupVel"][subf_sat],
                                a, h, adot
                            )
                            x_s, y_s, z_s = pos_co_to_phys(pos_s, a, h)
                            vx_s, vy_s, vz_s = vel_co_to_phys(pos_s, vel_s, a, h, adot)
                            cmx_sat, cmy_sat, cmz_sat, *_ = recalculate_cm(
                                x_s, y_s, z_s, vx_s, vy_s, vz_s, mass_s,
                                cm0_sat[0], cm0_sat[1], cm0_sat[2],
                                vcm0_sat[0], vcm0_sat[1], vcm0_sat[2]
                            )
                        else:
                            flag_nostars_tinfall_sat = True
                            cmx_sat, cmy_sat, cmz_sat  = pos_co_to_phys_cm(
                                snap_grp["SubGroups/SubGroupPos"][subf_sat], a, h
                            )

                        dist = np.sqrt((cmx_sat-cmx_cen)**2+(cmy_sat-cmy_cen)**2+(cmz_sat-cmz_cen)**2)

                        if dist < rvir:
                            snap_first_inside = snap_sat
                            break

                    if snap_first_inside is not None:
                        if snap_first_inside - 1 in snap_uid_dict:
                            snap_infall = snap_first_inside - 1
                            uid_infall = snap_uid_dict[snap_infall]
                            z_infall = dict_snap_to_z[snap_infall]

                            snap_infall_list.append(snap_infall)
                            uid_infall_list.append(uid_infall)
                            z_infall_list.append(z_infall)

                            flag_always_inside_r200_list.append(False)
                            flag_never_inside_r200_list.append(False)
                            
                            infall_status_note = 'normal infall'

                        else:
                            uid_infall = branch[-1]  # always inside halo: use first (earliest) UID 
                            uid_infall_list.append(uid_infall) 
                            snap_infall_list.append(None)
                            z_infall_list.append(None)
                            flag_never_inside_r200_list.append(False)
                            flag_always_inside_r200_list.append(True) 
                            
                            infall_status_note = 'always inside r200'

                    else:
                        uid_infall = branch[0]  # never inside halo: use last (latest) UID
                        uid_infall_list.append(uid_infall) 
                        snap_infall_list.append(None)
                        z_infall_list.append(None)
                        flag_never_inside_r200_list.append(True)
                        flag_always_inside_r200_list.append(False)
                        
                        infall_status_note = 'never inside r200'
                    
                    if snap_missing:
                        print(f'[WARN] uid_infall resolved to {uid_infall} ({infall_status_note})') 


                    flag_nostars_tinfall_sat_list.append(flag_nostars_tinfall_sat)
                    flag_nostars_tinfall_cen_list.append(flag_nostars_tinfall_cen)

                    snap, subf = split_unique_id(int(uid_infall))
                    snap_grp = f[f"SnapNumber_{snap}"]
                    h = snap_grp["Header/HubbleParam"][()]

                    offset_g = get_offset_safe(snap_grp["SubGroups/PartType0/Offsets"], subf) 
                    offset_dm = get_offset_safe(snap_grp["SubGroups/PartType1/Offsets"], subf) 
                    offset_s = get_offset_safe(snap_grp["SubGroups/PartType4/Offsets"], subf)

                    pids_g = snap_grp["PartType0/ParticleIDs"][offset_g[0]:offset_g[1]]  # satellite gas PIDs at t_infall
                    pids_dm = snap_grp["PartType1/ParticleIDs"][offset_dm[0]:offset_dm[1]]  # satellite DM PIDs at t_infall
                    pids_s = snap_grp["PartType4/ParticleIDs"][offset_s[0]:offset_s[1]]  # satellite star PIDs at t_infall

                    m_star = (
                        snap_grp["PartType4/Masses"][offset_s[0]:offset_s[1]].sum() * 1e10 / h
                        if offset_s[1] > offset_s[0] else 0.0
                    )

                    m_gas = (
                        snap_grp["PartType0/Masses"][offset_g[0]:offset_g[1]].sum() * 1e10 / h
                        if offset_g[1] > offset_g[0] else 0.0
                    )

                    m_dm = (
                        (offset_dm[1] - offset_dm[0]) * snap_grp["Header/MassTable"][()][1] * 1e10 / h
                        if offset_dm[1] > offset_dm[0] else 0.0
                    )

                    mass_star_tinfall_list.append(m_star)
                    mass_gas_tinfall_list.append(m_gas)
                    mass_dm_tinfall_list.append(m_dm)
                    mass_baryon_tinfall_list.append(m_star + m_gas)

                    # ==== Compute fraction of satellite mass (stars/gas/DM at t_infall) that ends up in the central ====

                    if offset_s[1] > offset_s[0]:
                        mask_comunes = np.in1d(pids_s, pids_cen_z0, assume_unique=True)

                        if np.any(mask_comunes):
                            masses_s = snap_grp["PartType4/Masses"][offset_s[0]:offset_s[1]] * 1e10 / h
                            mass_comunes_total = masses_s[mask_comunes].sum()
                            stellar_mass_frac_infall_in_central_z0_list.append(mass_comunes_total / m_star)
                        else:
                            stellar_mass_frac_infall_in_central_z0_list.append(0)
                    else:
                        stellar_mass_frac_infall_in_central_z0_list.append(np.nan)


                    if offset_g[1] > offset_g[0]:
                        mask_comunes = np.in1d(pids_g, pids_cen_z0_gas, assume_unique=True)

                        if np.any(mask_comunes):
                            masses_g = snap_grp["PartType0/Masses"][offset_g[0]:offset_g[1]] * 1e10 / h
                            mass_comunes_total = masses_g[mask_comunes].sum()
                            gas_mass_frac_infall_in_central_z0_list.append(mass_comunes_total / m_gas)
                        else:
                            gas_mass_frac_infall_in_central_z0_list.append(0)
                    else:
                        gas_mass_frac_infall_in_central_z0_list.append(np.nan)

                    if offset_dm[1] > offset_dm[0]:
                        mask_comunes = np.in1d(pids_dm, pids_cen_z0_dm, assume_unique=True)

                        if np.any(mask_comunes):
                            mass_comunes_total = mask_comunes.sum() * snap_grp["Header/MassTable"][()][1] * 1e10 / h
                            dm_mass_frac_infall_in_central_z0_list.append(mass_comunes_total / m_dm)
                        else:
                            dm_mass_frac_infall_in_central_z0_list.append(0)
                    else:
                        dm_mass_frac_infall_in_central_z0_list.append(np.nan)




                    # ==== Mass and fraction of stars formed from the satellite's gas ====
                    if offset_g[1] > offset_g[0]:
                        pids_g = snap_grp["PartType0/ParticleIDs"][offset_g[0]:offset_g[1]]

                        mass_star1_total = 0.0
                        mass_star2_total = 0.0
                        mass_star1_cen = 0.0
                        mass_star2_cen = 0.0

                        for pid in pids_g:
                            pid_star1 = pid
                            pid_star2 = pid + 2**31

                            # First generation
                            if pid_star1 in pid_to_mass_star_z0:
                                m1 = pid_to_mass_star_z0[pid_star1]
                                mass_star1_total += m1
                                if pid_to_subfind_star_z0[pid_star1] == subfind_central_z0:
                                    mass_star1_cen += m1

                            # Second generation
                            if pid_star2 in pid_to_mass_star_z0:
                                m2 = pid_to_mass_star_z0[pid_star2]
                                mass_star2_total += m2
                                if pid_to_subfind_star_z0[pid_star2] == subfind_central_z0:
                                    mass_star2_cen += m2

                        frac1 = mass_star1_cen / mass_star1_total if mass_star1_total > 0 else np.nan
                        frac2 = mass_star2_cen / mass_star2_total if mass_star2_total > 0 else np.nan

                        frac_star1_from_gas_in_central_z0_list.append(frac1)
                        frac_star2_from_gas_in_central_z0_list.append(frac2)
                        mass_star1_from_gas_in_central_z0_list.append(mass_star1_cen)
                        mass_star2_from_gas_in_central_z0_list.append(mass_star2_cen)
                    else:
                        frac_star1_from_gas_in_central_z0_list.append(np.nan)
                        frac_star2_from_gas_in_central_z0_list.append(np.nan)
                        mass_star1_from_gas_in_central_z0_list.append(np.nan)
                        mass_star2_from_gas_in_central_z0_list.append(np.nan)



    # ==== Save output ====
    df_new = pd.DataFrame()
    df_new["uid_disruption"] = uid_disruption_list
    df_new["uid_infall"] = uid_infall_list
    df_new["snap_infall"] = snap_infall_list
    df_new["z_infall"] = z_infall_list
    df_new["mass_stars_infall"] = mass_star_tinfall_list
    df_new["mass_gas_infall"] = mass_gas_tinfall_list
    df_new["mass_dm_infall"] = mass_dm_tinfall_list
    df_new["mass_baryon_infall"] = mass_baryon_tinfall_list
    df_new["Mstell_frac_t_infall_in_central_z0"] = stellar_mass_frac_infall_in_central_z0_list
    df_new["Mgas_frac_t_infall_in_central_z0"] = gas_mass_frac_infall_in_central_z0_list
    df_new["Mdm_frac_t_infall_in_central_z0"] = dm_mass_frac_infall_in_central_z0_list
    df_new["frac_star1_from_gas_infall_in_central_z0"] = frac_star1_from_gas_in_central_z0_list
    df_new["frac_star2_from_gas_infall_in_central_z0"] = frac_star2_from_gas_in_central_z0_list
    df_new["mass_star1_from_gas_infall_in_central_z0"] = mass_star1_from_gas_in_central_z0_list
    df_new["mass_star2_from_gas_infall_in_central_z0"] = mass_star2_from_gas_in_central_z0_list
    df_new["flag_had_stars"] = had_stars_list
    df_new["flag_had_gas"] = had_gas_list
    df_new["flag_had_dm"] = had_dm_list
    df_new["flag_always_inside_r200"] = flag_always_inside_r200_list
    df_new["flag_never_inside_r200"] = flag_never_inside_r200_list
    df_new["flag_CM_hdf5_tinfall_sat"] = flag_nostars_tinfall_sat_list
    df_new["flag_CM_hdf5_tinfall_cen"] = flag_nostars_tinfall_cen_list

    # === Columnas del DataFrame df_new ===

    # uid_disruption : satellite UID at the last snapshot before the merger
    # uid_infall     : satellite UID at the infall snapshot (last snapshot outside R200; inside from the next snap onward)
    # snap_infall    : snapshot number corresponding to uid_infall
    # z_infall       : redshift of the infall snapshot

    # mass_stars_infall   : satellite stellar mass at t_infall [Msun]
    # mass_gas_infall     : satellite gas mass at t_infall [Msun]
    # mass_dm_infall      : satellite DM mass at t_infall [Msun]
    # mass_baryon_infall  : total baryonic mass at t_infall (stars + gas) [Msun]

    # Mstell_frac_t_infall_in_central_z0 : fraction of satellite stellar mass (at t_infall)
    #                                      that ends up in the central galaxy at z = 0
    # Mgas_frac_t_infall_in_central_z0 : fraction of satellite gas mass (at t_infall)
    #                                      that ends up in the central galaxy at z = 0
    # Mdm_frac_t_infall_in_central_z0 : fraction of satellite DM mass (at t_infall)
    #                                      that ends up in the central galaxy at z = 0

    # frac_star1_from_gas_infall_in_central_z0 : mass fraction of 1st-gen stars formed from satellite gas at t_infall
    #                                            that ended up in the central galaxy at z=0

    # frac_star2_from_gas_infall_in_central_z0 : mass fraction of 2nd-gen stars formed from satellite gas at t_infall
    #                                            that ended up in the central galaxy at z=0

    # mass_star1_from_gas_infall_in_central_z0 : total mass of 1st-gen stars from satellite gas at t_infall
    #                                            that ended in the central galaxy [Msun]

    # mass_star2_from_gas_infall_in_central_z0 : total mass of 2nd-gen stars from satellite gas at t_infall
    #                                            that ended in the central galaxy [Msun]


    # flag_had_stars : True if the satellite had stars at any snapshot
    # flag_had_gas   : True if the satellite had gas at any snapshot
    # flag_had_dm    : True if the satellite had DM at any snapshot

    # flag_always_inside_r200 : True if the satellite was always inside R200 -> uid_infall is set to the earliest UID
    # flag_never_inside_r200  : True if the satellite never entered R200 -> uid_infall is set to the latest UID

    # flag_CM_hdf5_tinfall_sat : True if the satellite CM at t_infall could not be recomputed
    #                             (HDF5 tabulated value used instead)
    # flag_CM_hdf5_tinfall_cen : True if the central galaxy CM at t_infall could not be recomputed
    #                             (HDF5 tabulated value used instead)


    df_new.to_csv(path_out, index=False)
    print(f"[OK] Saved CSV with baryonic mergers at infall: {path_out}")
    
    
    # ------------------------------------------------------------
    # Final statistics for Step 3
    # ------------------------------------------------------------

    print("\n" + "="*40)
    print(f"\nBaryonic merger statistics for {sim_name} - {target_subfind_id} (Step 3)")
    print("\n" + "="*40)
    
    # 1. Mergers discarded because they were DM-only throughout their history
    print(f"\n- Mergers discarded (DM-only throughout): {count_dm_only_mergers}")

    # 2. Remaining mergers
    print(f"- Branches with baryonic content retained: {len(df_new)}")

    # 3. Break down by baryonic component type at infall
    n_stars = df_new[(df_new["mass_stars_infall"] > 0) & (df_new["mass_gas_infall"] == 0)].shape[0]
    n_gas   = df_new[(df_new["mass_stars_infall"] == 0) & (df_new["mass_gas_infall"] > 0)].shape[0]
    n_both  = df_new[(df_new["mass_stars_infall"] > 0) & (df_new["mass_gas_infall"] > 0)].shape[0]
    n_dm    = df_new[(df_new["mass_stars_infall"] == 0) & (df_new["mass_gas_infall"] == 0)].shape[0]

    total_baryonic_at_infall = n_stars + n_gas + n_both

    print(f"   |- Stars only at infall: {n_stars}")
    print(f"   |- Gas only at infall:   {n_gas}")
    print(f"   |- Both stars and gas:   {n_both}")
    print(f"   |- DM only at infall:    {n_dm}")
    print(f"   `- Total baryonic at infall: {total_baryonic_at_infall}")

