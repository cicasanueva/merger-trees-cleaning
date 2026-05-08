#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Catalina Casanueva (cicasanueva@uc.cl), 07/2025
# ============================================================
# Paso [1]  – Cálculo del centro de masa (CM) y de R200
# para galaxias centrales con M⋆ > mass_cut M☉ en z = 0
# Simulaciones de CIELO (archivos HDF5 + árbol de mergers)
#
# - Se identifican todas las galaxias centrales en z = 0
#   (SubGroupNumber = 0) con masa estelar M⋆ > mass_cut M☉.
#
# - Para cada galaxia, se recorre su main branch completa.
#
# - En cada snapshot:
#     • El CM se recalcula mediante shrinking sphere usando
#       estrellas del subhalo (PartType4) si hay al menos 100.
#     • Si no hay suficientes estrellas, se usa el CM tabulado
#       en el HDF5 (SubGroupPos).
#     • R200 se define como el radio esférico dentro del
#       cual la densidad promedio encerrada es igual a 200 veces
#       la densidad crítica del universo a ese redshift.
#     • R200 se calcula construyendo el perfil de densidad con todas
#       las partículas del grupo (gas, materia oscura, estrellas),
#       centradas en el CM correspondiente.
#
# - El resultado es una tabla con el CM y R200 de cada galaxia
#   a lo largo de su main branch, que se guarda como archivo CSV.
# ============================================================



import h5py
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from scipy.interpolate import interp1d
from astropy import constants as const
import astropy.units as u
import os


from utils import (
    recalculate_cm,
    split_unique_id,
    pos_co_to_phys,
    pos_co_to_phys_cm,
    vel_co_to_phys_cm,
    vel_co_to_phys,
    get_main_branch_unique_ids,
)

# ============================================================
# Funciones auxiliares
# ============================================================

def rho_crit_phys(z, h, Omega0, OmegaL):
    """Densidad crítica física del universo a redshift z [Msun/kpc^3]."""
    G = const.G.to(u.m**3 / (u.kg * u.s**2)).value
    M_sun = const.M_sun.to(u.kg).value
    kpc_to_m = (1 * u.kpc).to(u.m).value
    mpc_to_m = (1 * u.Mpc).to(u.m).value

    H0_SI = (100 * h) * 1e3 / mpc_to_m  # Hubble en s^-1
    Hz = H0_SI * np.sqrt(Omega0 * (1 + z)**3 + OmegaL)
    rho_crit_SI = 3 * Hz**2 / (8 * np.pi * G)
    return rho_crit_SI / M_sun * kpc_to_m**3

def get_particle_data(snap, ptype, center_phys, a, h, group_id=None):
    """Distancias y masas físicas de partículas de un tipo, opcionalmente filtradas por GroupNumber."""
    key = f"PartType{ptype}"
    if key not in snap:
        return np.array([]), np.array([])
    part = snap[key]
    pos = part["Coordinates"][:] * a / h
    if ptype == 1:
        mval = snap["Header/MassTable"][1] * 1e10 / h
        masses = np.full(pos.shape[0], mval)
    else:
        masses = part["Masses"][:] * 1e10 / h
    dists = np.linalg.norm(pos - center_phys, axis=1)
    if group_id is not None:
        mask = part["GroupNumber"][:] == group_id
        return dists[mask], masses[mask]
    return dists, masses

def compute_density_profile(dists, masses):
    """Perfil de densidad promedio encerrada ρ(<r)."""
    idx = np.argsort(dists)
    r = dists[idx]
    m_enc = np.cumsum(masses[idx])
    volumes = (4/3) * np.pi * r**3
    rho = m_enc / volumes
    return r, rho

def find_r200_interp(r, rho, rho_crit_200):
    """Interpolación log-log para encontrar R donde ρ(<r) = 200 × ρ_crit."""
    mask = (rho > 0) & (r > 0)
    r, rho = r[mask], rho[mask]
    if not np.any(rho <= rho_crit_200):
        return np.nan
    interp = interp1d(np.log10(rho), np.log10(r), kind="linear", fill_value="extrapolate", assume_sorted=False)
    return 10**interp(np.log10(rho_crit_200))


def calcula_cm_r200(sim_name, snap_z0, mass_cut, tipo='central'):
    # ============================================================
    # Configuración de simulación
    # ============================================================

    #sim = "LG1"
    path_sim = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}.hdf5"
    path_tree = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}_tree.dat"
    #snap_z0 = 128
    #mass_cut = 1e9  # [M☉] umbral en masa estelar para considerar la central

    carpeta_out = "Datos/CM_R200/"

    # Crea la carpeta si no existe
    os.makedirs(carpeta_out, exist_ok=True)

    # ============================================================
    # Loop principal
    # ============================================================

    rows = []

    with h5py.File(path_sim, "r") as f:
        snap_grp = f[f"SnapNumber_{snap_z0}"]
        header = snap_grp["Header"]
        a = header["Time"][()]
        h = header["HubbleParam"][()]

        # Identificar subhalos centrales/satelites en z=0 con M* > mass_cut
        subf_ids = []
        if tipo == 'central':
            for subf in range(len(snap_grp["SubGroups/GroupNumber"][:])):
                if snap_grp["SubGroups/SubGroupNumber"][subf] != 0:
                    continue
                offset_raw = snap_grp["SubGroups/PartType4/Offsets"][subf]

                # Si el subhalo no tiene estrellas → no calcular CM, seguir
                if np.any(np.isnan(offset_raw)) or offset_raw[1] <= offset_raw[0]:
                    continue
                start, end = snap_grp["SubGroups/PartType4/Offsets"][subf].astype(int)
                if end <= start:
                    continue
                masses = snap_grp["PartType4/Masses"][start:end] * 1e10 / h
                mstar = np.sum(masses)
                if mstar > mass_cut:
                    subf_ids.append(subf)

            print(f"✅ {len(subf_ids)} centrales en {sim_name} con M* > {mass_cut:.2E}")
        elif tipo == 'satelite':
            for subf in range(len(snap_grp["SubGroups/GroupNumber"][:])):
                if snap_grp["SubGroups/SubGroupNumber"][subf] == 0:
                    continue
                offset_raw = snap_grp["SubGroups/PartType4/Offsets"][subf]

                # Si el subhalo no tiene estrellas → no calcular CM, seguir
                if np.any(np.isnan(offset_raw)) or offset_raw[1] <= offset_raw[0]:
                    continue
                start, end = snap_grp["SubGroups/PartType4/Offsets"][subf].astype(int)
                if end <= start:
                    continue
                masses = snap_grp["PartType4/Masses"][start:end] * 1e10 / h
                mstar = np.sum(masses)
                if mstar > mass_cut:
                    subf_ids.append(subf)    
            print(f"✅ {len(subf_ids)} satélites en {sim_name} con M* > {mass_cut:.2E}")


        tree = nx.read_multiline_adjlist(path_tree, create_using=nx.DiGraph)
        
        if len(subf_ids) == 0: 
            return
        for subf in tqdm(subf_ids, desc=f"{tipo} z=0"):
            # Obtener UID del snapshot z=0 y su main branch
            uid_z0 = f"{snap_z0:03}{subf:06}"
            main_branch = get_main_branch_unique_ids(nx.dfs_tree(tree, uid_z0), uid_z0)

            for uid in main_branch:
                snap, subf = split_unique_id(int(uid))

                # Extraer propiedades del snapshot
                snap_grp = f[f"SnapNumber_{snap}"]
                header = snap_grp["Header"]
                a = header["Time"][()]
                z = header["Redshift"][()]
                h = header["HubbleParam"][()]
                Omega0 = header["Omega0"][()]
                OmegaL = header["OmegaLambda"][()]
                omega = Omega0 + OmegaL
                H0 = 0.1 * h
                adot = a * H0 * np.sqrt(OmegaL + Omega0 * a**-3 - (omega - 1) * a**-2)

                # Centro tabulado del subhalo
                group_id = snap_grp["SubGroups/GroupNumber"][subf]
                cm_sub = snap_grp["SubGroups/SubGroupPos"][subf] 
                vel_sub = snap_grp["SubGroups/SubGroupVel"][subf]
                x0, y0, z0 = pos_co_to_phys_cm(cm_sub, a, h)
                vx0, vy0, vz0 = vel_co_to_phys_cm(cm_sub, vel_sub, a, h, adot)

                # === CM con shrinking (solo estrellas del subhalo) ===
                offset_raw = snap_grp["SubGroups/PartType4/Offsets"][subf]

                # Si no hay estrellas → usar centro HDF5 para calcular R200 igual
                if np.any(np.isnan(offset_raw)) or offset_raw[1] <= offset_raw[0]:
                    # Usar centro tabulado en HDF5
                    xcm_s, ycm_s, zcm_s = x0, y0, z0
                    flag_recalc = False
                else:
                    start, end = snap_grp["SubGroups/PartType4/Offsets"][subf].astype(int)
                    Nstars = end - start
                    if Nstars < 100:
                        # Menos de 100 estrellas: usar centro tabulado
                        xcm_s, ycm_s, zcm_s = x0, y0, z0
                        flag_recalc = False
                    else:
                        # Calcular CM con shrinking sphere
                        coords = snap_grp["PartType4/Coordinates"][start:end]
                        vels = snap_grp["PartType4/Velocities"][start:end]
                        masses = snap_grp["PartType4/Masses"][start:end] * 1e10 / h
                        x_pos, y_pos, z_pos = pos_co_to_phys(coords, a, h)
                        vx, vy, vz = vel_co_to_phys(coords, vels, a, h, adot)

                        xcm_s, ycm_s, zcm_s, *_ = recalculate_cm(
                            x_pos, y_pos, z_pos, vx, vy, vz, masses, x0, y0, z0, vx0, vy0, vz0
                        )

                        flag_recalc = True

                center_phys = np.array([xcm_s, ycm_s, zcm_s])

                # Usar el centro elegido (recalculado o tabulado) para construir el perfil de densidad
                # con TODAS las partículas del grupo (DM, gas, estrellas)

                d_grp_all, m_grp_all = [], []
                for ptype in [0, 1, 4]:
                    d, m = get_particle_data(snap_grp, ptype, center_phys, a, h, group_id)
                    d_grp_all.append(d)
                    m_grp_all.append(m)

                r_grp_all, rho_grp_all = compute_density_profile(
                    np.concatenate(d_grp_all), np.concatenate(m_grp_all)
                )

                rho_c = rho_crit_phys(z, h, Omega0, OmegaL)
                rho_200 = 200 * rho_c
                r200_kpc = find_r200_interp(r_grp_all, rho_grp_all, rho_200)

                rows.append({
                    "uid": uid,  # Identificador único de la galaxia (snap * 1e6 + subfind)
                    "snap": snap,  # Número de snapshot correspondiente al registro
                    "z": z,  # Redshift del snapshot

                    # Centro de masa calculado con shrinking sphere (si es posible)
                    "xcm_star": xcm_s,  # Coordenada X del CM (en kpc físicos)
                    "ycm_star": ycm_s,  # Coordenada Y del CM (en kpc físicos)
                    "zcm_star": zcm_s,  # Coordenada Z del CM (en kpc físicos)

                    "r200_kpc": r200_kpc,  # r200 recalculado (en kpc físicos), usando todas las partículas del grupo

                    # Centro de masa tabulado en el HDF5 para el subhalo (coordenadas físicas)
                    "xcm_star_old": x0,  # Coordenada X del CM HDF5
                    "ycm_star_old": y0,  # Coordenada Y del CM HDF5
                    "zcm_star_old": z0,  # Coordenada Z del CM HDF5

                    "r200_kpc_group_old": snap_grp['Groups/Group_R_Crit200'][group_id] * a * h**(-1),
                    # Radio virial del grupo directamente desde el HDF5 (convertido a unidades físicas)

                    "flag_cm_recalculado": flag_recalc  # True si el CM fue recalculado con estrellas, False si se usó el tabulado
                })




    # ============================================================
    # Guardar tabla de resultados
    # ============================================================

    df = pd.DataFrame(rows)
    df.to_csv(f"{carpeta_out}r200_mainbranches_all_{tipo}_z0_{sim_name}.csv", index=False)
    print(f"✅ Guardado {carpeta_out}/r200_mainbranches_all_{tipo}_z0_{sim_name}.csv con {len(df)} filas")
    
    # PLOTS OPCIONALES
    # ============================================================
    # COMPARACIÓN DE R200 Y CENTRO DE MASA (CM) CALCULADOS VS TABULADOS
    # ============================================================

    print("\n" + "="*40)
    print(f" PLOTS PARA {tipo} DE {sim_name} ")
    print("="*40 + "\n")
    
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Plot configuration:
    plt.style.context("seaborn-paper")
    plt.rcParams["font.family"]         = "DejaVu Serif"
    plt.rcParams['font.serif']          = 'Computer Modern'
    plt.rcParams["xtick.direction"]     = "in"
    plt.rcParams["ytick.direction"]     = "in"
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.top"]           = True
    plt.rcParams["ytick.right"]         = True

    SMALL_SIZE  = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    plt.rc('font'  , size      = SMALL_SIZE)   # controls default text sizes
    plt.rc('axes'  , titlesize = MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes'  , labelsize = MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick' , labelsize = SMALL_SIZE)   # fontsize of the tick labels
    plt.rc('ytick' , labelsize = SMALL_SIZE)   # fontsize of the tick labels
    plt.rc('legend', fontsize  = SMALL_SIZE)   # legend fontsize
    plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title

    # ================================
    # PARÁMETROS Y CARGA DE DATOS
    # ================================

    #sim_name = "LG1"
    #snap_z0 = 128
    df = pd.read_csv(f"{carpeta_out}/r200_mainbranches_all_{tipo}_z0_{sim_name}.csv")

    path_tree = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}_tree.dat"
    tree = nx.read_multiline_adjlist(path_tree)

    uid_to_gxid = {}
    for subf in subf_ids:
        uid_z0 = f"{snap_z0:03}{subf:06}"
        rama = get_main_branch_unique_ids(nx.dfs_tree(tree, uid_z0), uid_z0)
        for uid in rama:
            uid_to_gxid[int(uid)] = int(uid_z0)

    df["gxid"] = df["uid"].map(uid_to_gxid)

    # Lista de gxid únicos
    gxids = sorted(df["gxid"].unique())

    # Crear una paleta de colores con tantos colores como galaxias haya
    palette = sns.color_palette("hsv", n_colors=len(gxids)) 

    # Mapear cada gxid a un color
    color_dict = {gxid: palette[i] for i, gxid in enumerate(gxids)}

    # ============================================================
    # PLOT 1 — VALORES ABSOLUTOS: R200 y CM vs z
    # ============================================================

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for gxid in df["gxid"].unique():
        subdf = df[df["gxid"] == gxid]
        color = color_dict.get(gxid, "gray")  # fallback por si falta color

        # -------- Panel 1: R200 --------
        axs[0].plot(subdf["z"], subdf["r200_kpc"], color=color,
                    linestyle="-", marker="s", mec='black', mew=0.5)
        axs[0].plot(subdf["z"], subdf["r200_kpc_group_old"], color=color,
                    linestyle="--", marker="*", mec='black', mew=0.5, alpha=0.7)

        # -------- Panel 2: Centros de masa --------
        sub_cm = subdf[subdf["flag_cm_recalculado"] == True]
        if not sub_cm.empty:
            axs[1].plot(sub_cm["z"], sub_cm["xcm_star"], color=color, linestyle="-", marker="s", mec='black', mew=0.5)
            axs[1].plot(sub_cm["z"], sub_cm["xcm_star_old"], color=color, linestyle="--", marker="*", mec='black', mew=0.5, alpha=0.7)

            axs[1].plot(sub_cm["z"], sub_cm["ycm_star"], color=color, linestyle="-", marker="s", mec='black', mew=0.5)
            axs[1].plot(sub_cm["z"], sub_cm["ycm_star_old"], color=color, linestyle="--", marker="*", mec='black', mew=0.5, alpha=0.7)

            axs[1].plot(sub_cm["z"], sub_cm["zcm_star"], color=color, linestyle="-", marker="s", mec='black', mew=0.5)
            axs[1].plot(sub_cm["z"], sub_cm["zcm_star_old"], color=color, linestyle="--", marker="*", mec='black', mew=0.5, alpha=0.7)

    # Ejes
    axs[0].set_xlabel("Redshift")
    axs[0].set_ylabel("R₂₀₀ [kpc]")
    axs[1].set_xlabel("Redshift")
    axs[1].set_ylabel("Coordenadas del CM [kpc]")

    for ax in axs:
        ax.grid(False)
        ax.set_title("")

    # Leyenda de galaxias (colores)
    handles_gx = [
        plt.Line2D([0], [0], color=color_dict[gxid], marker="s", mec='black', mew=1.2,
                   linestyle="None", label=f"Gx {int(gxid % 1e6)}")
        for gxid in df["gxid"].unique()
    ]

    # Leyenda de símbolos (negro)
    handles_symbols = [
        plt.Line2D([0], [0], color="black", marker="s", linestyle="None", label="Nuevo"),
        plt.Line2D([0], [0], color="black", marker="*", linestyle="None", label="Tabulado"),
    ]

    fig.legend(handles=handles_gx, loc="upper center", ncol=5, frameon=False)
    axs[0].legend(handles=handles_symbols, loc="upper right", frameon=False)
    axs[1].legend(handles=handles_symbols, loc="upper right", frameon=False)

    fig.subplots_adjust(top=3.7) #más espacio arriba
    plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.show()

    # ============================================================
    # PLOT 2 — DIFERENCIAS ENTRE VERSIONES (old - new)
    # ============================================================

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

    for gxid in df["gxid"].unique():
        subdf = df[df["gxid"] == gxid]
        color = color_dict.get(gxid, "gray")

        # -------- Panel 1: ΔR200 --------
        delta_r = subdf["r200_kpc_group_old"] - subdf["r200_kpc"]
        axs[0].plot(subdf["z"], delta_r, color=color, linestyle="-", marker="o", mec='black', mew=1.2)

        # -------- Panel 2: Δ CM --------
        sub_cm = subdf[subdf["flag_cm_recalculado"] == True]
        if not sub_cm.empty:
            dx = sub_cm["xcm_star_old"] - sub_cm["xcm_star"]
            dy = sub_cm["ycm_star_old"] - sub_cm["ycm_star"]
            dz = sub_cm["zcm_star_old"] - sub_cm["zcm_star"]
            
            dr = np.sqrt(dx**2+dy**2+dz**2)

            axs[2].plot(sub_cm["z"], dx, color=color, linestyle="-", marker="o", mec='black', mew=1.2)
            axs[2].plot(sub_cm["z"], dy, color=color, linestyle="--", marker="^", mec='black', mew=1.2, alpha=0.7)
            axs[2].plot(sub_cm["z"], dz, color=color, linestyle=":", marker="v", mec='black', mew=1.2, alpha=0.7)
            axs[1].plot(sub_cm["z"], dr, color=color, linestyle=":", marker="v", mec='black', mew=1.2, alpha=0.7)

    # Ejes
    axs[0].set_xlabel("Redshift")
    axs[0].set_ylabel("Δ R₂₀₀ [kpc] (old - new)")
    axs[1].set_xlabel("Redshift")
    axs[1].set_ylabel("Δ CM [kpc] (old - new)")
    axs[2].set_xlabel("Redshift")
    axs[2].set_ylabel("Δ CM [kpc] (old - new)")
    
    for ax in axs:
        ax.grid(False)
        ax.set_title("")

    # Leyenda general
    fig.legend(handles=handles_gx, loc="upper center", ncol=5, frameon=False)

    # Leyenda de deltas
    handles_delta = [
        plt.Line2D([0], [0], color="black", marker="o", linestyle="None", label="Δx"),
        plt.Line2D([0], [0], color="black", marker="^", linestyle="None", label="Δy"),
        plt.Line2D([0], [0], color="black", marker="v", linestyle="None", label="Δz"),
    ]
    axs[2].legend(handles=handles_delta, loc="upper right", frameon=False)

    fig.subplots_adjust(top=3.7)
    plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.show()


