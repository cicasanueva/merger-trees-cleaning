#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Catalina Casanueva (cicasanueva@uc.cl), 07/2025
# ============================================================
# Paso [3] – Caracterización de mergers bariónicos hacia una galaxia central
# dada (identificada por su subfind ID en z = 0).
#
# Este script toma como input el archivo `mergers_gx{ID}_summary.csv` 
# generado en el paso [2], el cual contiene las ramas completas de mergers 
# físicos que terminan fusionándose con una galaxia central.
#
# Para cada merger, se determina el snapshot de infall (`snap_infall`) como el último 
# snapshot en que el satélite está fuera del halo de la central, definido por su R₂₀₀ 
# recalculado manualmente en el paso [1]. A partir del siguiente snapshot, se considera
# que el satélite ya ha cruzado el halo.
#
# Se calcula además el UID del satélite en `t_infall` (`uid_infall`) y el último UID 
# disponible antes de su disrupción (`uid_disruption`), correspondiente al snapshot 
# previo a su fusión con la central.
#
# Para cada merger se calculan:
# ---------------------------------------------------------------
# • Redshift y UID en el snapshot `t_infall` (`uid_infall`, `z_infall`)
# • UID del merger al momento de disrupción (`uid_disruption`)
#
# • Masas del satélite en `t_infall`:
#     - Estelar (`mass_stars_infall`)
#     - Gas (`mass_gas_infall`)
#     - Materia oscura (`mass_dm_infall`)
#     - Bariónica total (`mass_baryon_infall`)
#
# • Fracciones de masa (en `t_infall`) que terminan en la central a z = 0:
#     - Estrellas → `Mstell_frac_t_infall_in_central_z0`
#     - Gas      → `Mgas_frac_t_infall_in_central_z0`
#     - Materia oscura → `Mdm_frac_t_infall_in_central_z0`
#
# • Formación estelar desde gas del satélite en `t_infall`:
#     - Fracción de masa de estrellas de 1ra generación → `frac_star1_from_gas_infall_in_central_z0`
#     - Fracción de masa de estrellas de 2da generación → `frac_star2_from_gas_infall_in_central_z0`
#     - Masas correspondientes que terminan en la galaxia central
#
# Flags adicionales:
# ---------------------------------------------------------------
# • `flag_tuvo_estrellas`, `flag_tuvo_gas`, `flag_tuvo_dm`:
#     indican si el satélite tuvo cada tipo de componente en su historia
#
# • `flag_always_inside_r200`: el satélite estuvo siempre dentro del halo.
#     En este caso, se usa el primer UID (snap más chico) como `uid_infall`.
#
# • `flag_never_inside_r200`: el satélite nunca cruzó el halo.
#     En este caso, se usa el último UID (snap más grande) como `uid_infall`.
#
# • `flag_CM_hdf5_tinfall_sat` / `flag_CM_hdf5_tinfall_cen`:
#     indican que no fue posible recalcular el centro de masa con shrinking sphere
#     (por falta de estrellas), por lo que se usó el valor tabulado en HDF5.
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


    
# ==== CONFIGURACIÓN GENERAL ====
def baryon_merger_tree_caract(sim_name, target_subfind_id, snap_target):
    #sim_name = "LG1"
    #target_subfind_id = 4337
    #snap_target = 128
    uid_target = f"{snap_target}{target_subfind_id:06d}"
    carpeta_out = "Datos/Baryon_Mergers/"

    # ==== PATHS DE ENTRADA/SALIDA ====
    path_sim = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}.hdf5"
    path_tree = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}_tree.dat"
    path_csv = f"Datos/All_Mergers/mergers_{sim_name}_gx_{target_subfind_id}_summary.csv"
    path_rvir = f"Datos/CM_R200/r200_mainbranches_all_central_z0_{sim_name}.csv"
    path_out = f"{carpeta_out}/baryon_mergers_{sim_name}_gx_{target_subfind_id}_summary.csv"

    os.makedirs(carpeta_out, exist_ok=True)

    # ==== CARGA DE ÁRBOLES Y MAIN BRANCH DE LA CENTRAL ====
    tree_1 = nx.read_multiline_adjlist(path_tree, create_using=nx.DiGraph)
    tree_full = nx.read_multiline_adjlist(path_tree)  # árbol hacia adelante
    subtree_main = nx.dfs_tree(tree_1, uid_target)
    main_branch_4337 = get_main_branch_unique_ids(subtree_main, uid_target)
    rama_4337_por_snap = {split_unique_id(int(uid))[0]: uid for uid in main_branch_4337}

    # ==== CARGA DE TABLAS DE ENTRADA ====
    df = pd.read_csv(path_csv)
    df_rvir = pd.read_csv(path_rvir)
    dict_rvir = dict(zip(df_rvir["uid"].astype(str), df_rvir["r200_kpc"]))

    # ==== REDSHIFT POR SNAPSHOT ====
    dict_snap_to_z = {}
    with h5py.File(path_sim, "r") as f:
        for key in f.keys():
            if key.startswith("SnapNumber_"):
                snap = int(key.split("_")[1])
                z = f[key]["Header/Redshift"][()]
                dict_snap_to_z[snap] = z

    # ==== INICIALIZACIÓN DE LISTAS DE SALIDA ====
    tuvo_estrellas_list = []
    tuvo_gas_list = []
    tuvo_dm_list = []

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

    # ==== CONTADOR DE MERGERS SÓLO-DM ====
    count_onlyDM_mergers = 0

    with h5py.File(path_sim, "r") as f:
        snap_z0 = f[f"SnapNumber_{snap_target}"]
        subf_central_z0 = split_unique_id(int(rama_4337_por_snap[int(snap_target)]))[1]  
        offset_cen_z0 = get_offset_safe(snap_z0["SubGroups/PartType4/Offsets"], subf_central_z0)
        pids_cen_z0 = snap_z0["PartType4/ParticleIDs"][offset_cen_z0[0]:offset_cen_z0[1]]

        offset_cen_z0_gas = get_offset_safe(snap_z0["SubGroups/PartType0/Offsets"], subf_central_z0)
        pids_cen_z0_gas = snap_z0["PartType0/ParticleIDs"][offset_cen_z0_gas[0]:offset_cen_z0_gas[1]]   

        offset_cen_z0_dm = get_offset_safe(snap_z0["SubGroups/PartType1/Offsets"], subf_central_z0)
        pids_cen_z0_dm = snap_z0["PartType1/ParticleIDs"][offset_cen_z0_dm[0]:offset_cen_z0_dm[1]]   

        # ==== Cargar estrellas a z=0 (solo una vez) ====
        header = snap_z0['Header']
        h = header['HubbleParam'][()]

        pids_star_z0 = snap_z0["PartType4/ParticleIDs"][:]
        masses_star_z0 = snap_z0["PartType4/Masses"][:] * 1e10 / h
        subfind_star_z0 = snap_z0["PartType4/SubFindNumber"][:]

        pid_to_mass_star_z0 = dict(zip(pids_star_z0, masses_star_z0))
        pid_to_subfind_star_z0 = dict(zip(pids_star_z0, subfind_star_z0))
        subfind_central_z0 = subf_central_z0  

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Procesando mergers"):
                if np.isnan(row["uid_before_merge"]) == True:
                    print(f'⚠️⚠️ para uid {row["origin_node"]} de {sim_name}-{target_subfind_id} no hay uid_before_merger')
                    continue
                uid_sat_before_merger = str(int(row["uid_before_merge"]))
                branch = get_main_branch_unique_ids(nx.dfs_tree(tree_1, uid_sat_before_merger), uid_sat_before_merger) # da sólo branch desde snap before merger para atrás
                snap_uid_dict = {
                    int(split_unique_id(int(uid))[0]): uid
                    for uid in branch
                }            
                #==== Verificar tipo de merger ====
                tuvo_estrellas = False
                tuvo_gas = False
                tuvo_dm = False

                for uid in branch:
                    snap, subf = split_unique_id(int(uid))
                    for tipo, parttype in [("stars", "PartType4"), ("gas", "PartType0"), ("dm", "PartType1")]:

                        offset = get_offset_safe(f[f"SnapNumber_{snap}/SubGroups/{parttype}/Offsets"], subf)

                        if tipo == "stars" and offset[1] > offset[0]:
                            tuvo_estrellas = True
                        elif tipo == "gas" and offset[1] > offset[0]:
                            tuvo_gas = True
                        elif tipo == "dm" and offset[1] > offset[0]:
                            tuvo_dm = True

                    # Salir si ya sabemos que tiene las tres componentes
                    if tuvo_estrellas and tuvo_gas and tuvo_dm:
                        break

                if tuvo_estrellas == False and tuvo_gas == False: # Si el satélite nunca tuvo estrellas ni gas lo ignoramos. Si queremos todos los mergers (incluidos los de sólo DM) borrar esto.
                    count_onlyDM_mergers+=1
                    continue
                else:
                    uid_disruption_list.append(uid_sat_before_merger)
                    tuvo_estrellas_list.append(tuvo_estrellas)
                    tuvo_gas_list.append(tuvo_gas)
                    tuvo_dm_list.append(tuvo_dm)

                    #==== Determinar t_infall ====
                    snap_post_infall = None
                    flag_nostars_tinfall_sat = False
                    flag_nostars_tinfall_cen = False
                    
                    detecta_error = False
                    for uid_sat in branch[::-1]:

                        snap_sat, subf_sat = split_unique_id(int(uid_sat))
                        if snap_sat not in rama_4337_por_snap.keys():
                            detecta_error = True
                            print(f'⚠️ {snap_sat} no estaba en rama de {sim_name}-{target_subfind_id}, el minimo en rama es {min(rama_4337_por_snap.keys())}')
                            continue
                        uid_central = rama_4337_por_snap[snap_sat]

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

                        #==== CM central ==== Se re-calcula con shrinking sólo si hay más de 100 estrellas
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

                        #==== CM satélite ==== Se re-calcula con shrinking sólo si hay más de 100 estrellas
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
                            snap_post_infall = snap_sat 
                            break

                    if snap_post_infall is not None:
                        if snap_post_infall-1 in snap_uid_dict:
                            snap_infall = snap_post_infall-1
                            uid_infall = snap_uid_dict[snap_infall]
                            z_infall = dict_snap_to_z[snap_infall]

                            snap_infall_list.append(snap_infall)
                            uid_infall_list.append(uid_infall)
                            z_infall_list.append(z_infall)

                            flag_always_inside_r200_list.append(False)
                            flag_never_inside_r200_list.append(False)
                            
                            extra_descrip = 'bien portada'

                        else:
                            uid_infall = branch[-1] #en caso que siempre estuvo dentro de rvir considero su uid al 'nacer' 
                            uid_infall_list.append(uid_infall) 
                            snap_infall_list.append(None)
                            z_infall_list.append(None)
                            flag_never_inside_r200_list.append(False)
                            flag_always_inside_r200_list.append(True) 
                            
                            extra_descrip = 'siempre dentro de rvir'

                    else:
                        uid_infall = branch[0] #en caso que nunca estuvo dentro de rvir considero su último uid disponible
                        uid_infall_list.append(uid_infall) 
                        snap_infall_list.append(None)
                        z_infall_list.append(None)
                        flag_never_inside_r200_list.append(True)
                        flag_always_inside_r200_list.append(False)
                        
                        extra_descrip = 'nunca dentro de rvir'
                    
                    if detecta_error == True:
                        print(f'el uid infall quedó como {uid_infall}, {extra_descrip}') 


                    flag_nostars_tinfall_sat_list.append(flag_nostars_tinfall_sat)
                    flag_nostars_tinfall_cen_list.append(flag_nostars_tinfall_cen)

                    snap, subf = split_unique_id(int(uid_infall))
                    snap_grp = f[f"SnapNumber_{snap}"]
                    h = snap_grp["Header/HubbleParam"][()]

                    offset_g = get_offset_safe(snap_grp["SubGroups/PartType0/Offsets"], subf) 
                    offset_dm = get_offset_safe(snap_grp["SubGroups/PartType1/Offsets"], subf) 
                    offset_s = get_offset_safe(snap_grp["SubGroups/PartType4/Offsets"], subf)

                    pids_g = snap_grp["PartType0/ParticleIDs"][offset_g[0]:offset_g[1]] # IDs del gas de la satélite en t_infall
                    pids_dm = snap_grp["PartType1/ParticleIDs"][offset_dm[0]:offset_dm[1]] # IDs de la dm de la satélite en t_infall
                    pids_s = snap_grp["PartType4/ParticleIDs"][offset_s[0]:offset_s[1]] # IDs de las estrellas de la satélite en t_infall

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

                    #==== Revisar qué fracción de la masa de estrellas/gas/DM de la satélite queda en la central ====

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




                    # ==== Masa y fracción de estrellas formadas desde el gas del satélite ====
                    if offset_g[1] > offset_g[0]:
                        pids_g = snap_grp["PartType0/ParticleIDs"][offset_g[0]:offset_g[1]]

                        mass_star1_total = 0.0
                        mass_star2_total = 0.0
                        mass_star1_cen = 0.0
                        mass_star2_cen = 0.0

                        for pid in pids_g:
                            pid_star1 = pid
                            pid_star2 = pid + 2**31

                            # Primera generación
                            if pid_star1 in pid_to_mass_star_z0:
                                m1 = pid_to_mass_star_z0[pid_star1]
                                mass_star1_total += m1
                                if pid_to_subfind_star_z0[pid_star1] == subfind_central_z0:
                                    mass_star1_cen += m1

                            # Segunda generación
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



    #==== Guardar ====
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
    df_new["flag_tuvo_estrellas"] = tuvo_estrellas_list
    df_new["flag_tuvo_gas"] = tuvo_gas_list
    df_new["flag_tuvo_dm"] = tuvo_dm_list
    df_new["flag_always_inside_r200"] = flag_always_inside_r200_list
    df_new["flag_never_inside_r200"] = flag_never_inside_r200_list
    df_new["flag_CM_hdf5_tinfall_sat"] = flag_nostars_tinfall_sat_list
    df_new["flag_CM_hdf5_tinfall_cen"] = flag_nostars_tinfall_cen_list

    # === Columnas del DataFrame df_new ===

    # uid_disruption : UID del satélite en el último snapshot antes de la fusión
    # uid_infall     : UID del satélite en el snapshot de infall (último snapshot donde está fuera de r_vir; en el siguiente ya está dentro)
    # snap_infall    : Snapshot de infall correspondiente al uid_infall
    # z_infall       : Redshift del snapshot de infall

    # mass_stars_infall   : Masa estelar del satélite en t_infall [Msun]
    # mass_gas_infall     : Masa de gas del satélite en t_infall [Msun]
    # mass_dm_infall      : Masa de materia oscura del satélite en t_infall [Msun]
    # mass_baryon_infall  : Masa bariónica total en t_infall (estrellas + gas) [Msun]

    # Mstell_frac_t_infall_in_central_z0 : Fracción de la masa estelar del satélite (en t_infall)
    #                                      que termina en la galaxia central a z = 0
    # Mgas_frac_t_infall_in_central_z0 : Fracción de la masa de gas del satélite (en t_infall)
    #                                      que termina en la galaxia central a z = 0
    # Mdm_frac_t_infall_in_central_z0 : Fracción de la masa DM del satélite (en t_infall)
    #                                      que termina en la galaxia central a z = 0

    # frac_star1_from_gas_infall_in_central_z0 : Fracción (en masa) de las estrellas de 1ra generación formadas a partir del gas del satélite en t_infall
    #                                            que terminaron en la galaxia central a z=0

    # frac_star2_from_gas_infall_in_central_z0 : Fracción (en masa) de las estrellas de 2da generación formadas a partir del gas del satélite en t_infall
    #                                            que terminaron en la galaxia central a z=0

    # mass_star1_from_gas_infall_in_central_z0 : Masa total de estrellas de 1ra generación formadas desde el gas del satélite en t_infall
    #                                            que terminaron en la galaxia central a z=0 [Msun]

    # mass_star2_from_gas_infall_in_central_z0 : Masa total de estrellas de 2da generación formadas desde el gas del satélite en t_infall
    #                                            que terminaron en la galaxia central a z=0 [Msun]


    # flag_tuvo_estrellas : True si el satélite tuvo estrellas en algún snapshot
    # flag_tuvo_gas       : True si el satélite tuvo gas en algún snapshot
    # flag_tuvo_dm        : True si el satélite tuvo DM en algún snapshot

    # flag_always_inside_r200 : True si el satélite estuvo siempre dentro de r_vir -> en ese caso me quedo con su primer uid (snap más chico)
    # flag_never_inside_r200  : True si el satélite nunca entró dentro de r_vir -> en ese caso me quedo con su último uid (snap más grande)

    # flag_CM_hdf5_tinfall_sat : True si NO se pudo recalcular el centro de masa del satélite en t_infall
    #                             (se usó el valor de HDF5 en su lugar)
    # flag_CM_hdf5_tinfall_cen : True si NO se pudo recalcular el centro de masa de la galaxia central en t_infall
    #                             (se usó el valor de HDF5 en su lugar)


    df_new.to_csv(path_out, index=False)
    print(f"[✅] Guardado CSV sólo mergers con componente bariónica al infall: {path_out}")
    
    
    # ------------------------------------------------------------
    # Estadísticas finales del Paso [3]
    # ------------------------------------------------------------

    print("\n" + "="*40)
    print(f"\nEstadísticas del resumen de mergers bariónicos para {sim_name} - {target_subfind_id} (Paso [3])")
    print("\n" + "="*40)
    
    # 1. Mergers descartados por ser sólo materia oscura durante toda su historia
    print(f"\n🔸 Mergers descartados por ser exclusivamente de DM en todos los snapshots: {count_onlyDM_mergers}")

    # 2. Mergers que quedan
    print(f"🔸 Ramas con contenido bariónico conservadas: {len(df_new)}")

    # 3. Desglose por tipo de componente bariónica presente en el snapshot de infall
    n_stars = df_new[(df_new["mass_stars_infall"] > 0) & (df_new["mass_gas_infall"] == 0)].shape[0]
    n_gas   = df_new[(df_new["mass_stars_infall"] == 0) & (df_new["mass_gas_infall"] > 0)].shape[0]
    n_both  = df_new[(df_new["mass_stars_infall"] > 0) & (df_new["mass_gas_infall"] > 0)].shape[0]
    n_dm    = df_new[(df_new["mass_stars_infall"] == 0) & (df_new["mass_gas_infall"] == 0)].shape[0]

    total_barionicos_infall = n_stars + n_gas + n_both

    print(f"   ├─ Con solo estrellas al infall: {n_stars}")
    print(f"   ├─ Con solo gas al infall:       {n_gas}")
    print(f"   ├─ Con ambos componentes:        {n_both}")
    print(f"   ├─ Con solo DM al infall:        {n_dm}")
    print(f"   └─ Total bariónicos al infall:   {total_barionicos_infall}")

