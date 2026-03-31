#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Catalina Casanueva (cicasanueva@uc.cl), 2025
# ============================================================
# Step 1 — Centre of mass (CM) and R200 along the main progenitor branch
#
# For each central galaxy at z=0 with M* > mass_cut, this step traverses
# the full main progenitor branch snapshot by snapshot and computes:
#
#   - Centre of mass: recomputed with an iterative shrinking-sphere algorithm
#     applied to stellar particles (PartType4). Only attempted when the subhalo
#     has >= 100 stellar particles; otherwise the SubFind-tabulated position
#     (SubGroupPos) is used and flag_cm_recomputed is set to False.
#
#   - R200: computed from an enclosed-density profile rho(<r) built from all
#     particle types (gas, dark matter, stars; PartTypes 0, 1, 4), centred on
#     the chosen CM. The profile is interpolated in log-log space to find the
#     radius where rho(<r) = 200 * rho_crit(z).
#
# Output: Datos/CM_R200/r200_mainbranches_all_{type}_z0_{sim}.csv
#         One row per snapshot per central galaxy.
# ============================================================

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from astropy import constants as const
import astropy.units as u
from scipy.interpolate import interp1d
from tqdm import tqdm

from utils import (
    recalculate_cm,
    split_unique_id,
    pos_co_to_phys,
    pos_co_to_phys_cm,
    vel_co_to_phys_cm,
    vel_co_to_phys,
    get_main_branch_unique_ids,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def rho_crit_phys(z, h, Omega0, OmegaL):
    """Return the physical critical density of the universe at redshift z [Msun/kpc^3]."""
    G          = const.G.to(u.m**3 / (u.kg * u.s**2)).value
    M_sun      = const.M_sun.to(u.kg).value
    kpc_to_m   = (1 * u.kpc).to(u.m).value
    mpc_to_m   = (1 * u.Mpc).to(u.m).value

    H0_SI = (100 * h) * 1e3 / mpc_to_m          # Hubble constant in s^-1
    Hz    = H0_SI * np.sqrt(Omega0 * (1 + z)**3 + OmegaL)
    rho_SI = 3 * Hz**2 / (8 * np.pi * G)
    return rho_SI / M_sun * kpc_to_m**3


def get_particle_data(snap, ptype, center_phys, a, h, group_id=None):
    """
    Return physical distances and masses for particles of a given type.

    Optionally restrict to particles belonging to a specific GroupNumber.
    """
    key = f"PartType{ptype}"
    if key not in snap:
        return np.array([]), np.array([])

    part   = snap[key]
    pos    = part["Coordinates"][:] * a / h
    if ptype == 1:
        mval   = snap["Header/MassTable"][1] * 1e10 / h
        masses = np.full(pos.shape[0], mval)
    else:
        masses = part["Masses"][:] * 1e10 / h

    dists = np.linalg.norm(pos - center_phys, axis=1)
    if group_id is not None:
        mask = part["GroupNumber"][:] == group_id
        return dists[mask], masses[mask]
    return dists, masses


def compute_density_profile(dists, masses):
    """Return the mean enclosed density profile rho(<r) given particle distances and masses."""
    idx    = np.argsort(dists)
    r      = dists[idx]
    m_enc  = np.cumsum(masses[idx])
    volumes = (4 / 3) * np.pi * r**3
    rho    = m_enc / volumes
    return r, rho


def find_r200_interp(r, rho, rho_crit_200):
    """
    Find R200 via log-log interpolation of the enclosed-density profile.

    Returns the radius where rho(<r) = 200 * rho_crit, or NaN if the profile
    never reaches that threshold.
    """
    mask = (rho > 0) & (r > 0)
    r, rho = r[mask], rho[mask]
    if not np.any(rho <= rho_crit_200):
        return np.nan
    interp = interp1d(
        np.log10(rho), np.log10(r),
        kind="linear", fill_value="extrapolate", assume_sorted=False,
    )
    return 10 ** interp(np.log10(rho_crit_200))


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_cm_r200(sim_name, snap_z0, mass_cut, galaxy_type='central'):
    """
    Compute the shrinking-sphere CM and R200 along the main progenitor branch
    of all central galaxies at z=0 with M* > mass_cut.

    Parameters
    ----------
    sim_name     : simulation identifier (e.g. "LG1", "P7")
    snap_z0      : z=0 snapshot number (usually 128; 127 for LG2)
    mass_cut     : minimum stellar mass threshold [M_sun]
    galaxy_type  : 'central' or 'satellite' — which subhalo type to select
    """
    path_sim  = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}.hdf5"
    path_tree = f"/data1/CIELO/Simulations/{sim_name}/{sim_name}_tree.dat"
    output_dir = "Datos/CM_R200/"
    os.makedirs(output_dir, exist_ok=True)

    rows = []

    with h5py.File(path_sim, "r") as f:
        snap_grp = f[f"SnapNumber_{snap_z0}"]
        header   = snap_grp["Header"]
        a = header["Time"][()]
        h = header["HubbleParam"][()]

        # ------------------------------------------------------------------
        # Select target subhalos at z=0
        # ------------------------------------------------------------------
        central_subf_ids = []
        subgroup_numbers = snap_grp["SubGroups/SubGroupNumber"][:]
        offsets_pt4      = snap_grp["SubGroups/PartType4/Offsets"]

        target_subgroup = 0 if galaxy_type == 'central' else None  # None means satellite (!=0)

        for subf in range(len(snap_grp["SubGroups/GroupNumber"][:])):
            sgn = subgroup_numbers[subf]
            if galaxy_type == 'central' and sgn != 0:
                continue
            if galaxy_type == 'satellite' and sgn == 0:
                continue

            offset_raw = offsets_pt4[subf]
            if np.any(np.isnan(offset_raw)) or offset_raw[1] <= offset_raw[0]:
                continue

            start, end = int(offset_raw[0]), int(offset_raw[1])
            if end <= start:
                continue

            mstar = np.sum(snap_grp["PartType4/Masses"][start:end] * 1e10 / h)
            if mstar > mass_cut:
                central_subf_ids.append(subf)

        print(f"[Step 1] {len(central_subf_ids)} {galaxy_type}s in {sim_name} with M* > {mass_cut:.2E}")

        tree = nx.read_multiline_adjlist(path_tree, create_using=nx.DiGraph)

        if len(central_subf_ids) == 0:
            return

        for subf in tqdm(central_subf_ids, desc=f"{galaxy_type} z=0"):
            uid_z0      = f"{snap_z0:03}{subf:06}"
            main_branch = get_main_branch_unique_ids(nx.dfs_tree(tree, uid_z0), uid_z0)

            for uid in main_branch:
                snap, subf = split_unique_id(int(uid))

                snap_grp = f[f"SnapNumber_{snap}"]
                header   = snap_grp["Header"]
                a      = header["Time"][()]
                z      = header["Redshift"][()]
                h      = header["HubbleParam"][()]
                Omega0 = header["Omega0"][()]
                OmegaL = header["OmegaLambda"][()]
                omega  = Omega0 + OmegaL
                H0     = 0.1 * h
                adot   = a * H0 * np.sqrt(OmegaL + Omega0 * a**-3 - (omega - 1) * a**-2)

                group_id = snap_grp["SubGroups/GroupNumber"][subf]

                # SubFind tabulated position (fallback)
                cm_sub  = snap_grp["SubGroups/SubGroupPos"][subf]
                vel_sub = snap_grp["SubGroups/SubGroupVel"][subf]
                x0, y0, z0     = pos_co_to_phys_cm(cm_sub,  a, h)
                vx0, vy0, vz0  = vel_co_to_phys_cm(cm_sub, vel_sub, a, h, adot)

                # ----------------------------------------------------------
                # Shrinking-sphere CM (stars only, if >= 100 particles)
                # ----------------------------------------------------------
                offset_raw = snap_grp["SubGroups/PartType4/Offsets"][subf]
                if np.any(np.isnan(offset_raw)) or offset_raw[1] <= offset_raw[0]:
                    xcm_s, ycm_s, zcm_s = x0, y0, z0
                    flag_recalc = False
                else:
                    start, end = int(offset_raw[0]), int(offset_raw[1])
                    n_stars = end - start
                    if n_stars < 100:
                        xcm_s, ycm_s, zcm_s = x0, y0, z0
                        flag_recalc = False
                    else:
                        coords = snap_grp["PartType4/Coordinates"][start:end]
                        vels   = snap_grp["PartType4/Velocities"][start:end]
                        masses = snap_grp["PartType4/Masses"][start:end] * 1e10 / h
                        x_pos, y_pos, z_pos = pos_co_to_phys(coords, a, h)
                        vx, vy, vz          = vel_co_to_phys(coords, vels, a, h, adot)

                        xcm_s, ycm_s, zcm_s, *_ = recalculate_cm(
                            x_pos, y_pos, z_pos, vx, vy, vz, masses,
                            x0, y0, z0, vx0, vy0, vz0,
                        )
                        flag_recalc = True

                center_phys = np.array([xcm_s, ycm_s, zcm_s])

                # ----------------------------------------------------------
                # R200 from enclosed-density profile (all particle types)
                # ----------------------------------------------------------
                d_all, m_all = [], []
                for ptype in [0, 1, 4]:
                    d, m = get_particle_data(snap_grp, ptype, center_phys, a, h, group_id)
                    d_all.append(d)
                    m_all.append(m)

                r_prof, rho_prof = compute_density_profile(
                    np.concatenate(d_all), np.concatenate(m_all)
                )

                rho_c     = rho_crit_phys(z, h, Omega0, OmegaL)
                r200_kpc  = find_r200_interp(r_prof, rho_prof, 200 * rho_c)

                rows.append({
                    "uid":               uid,
                    "snap":              snap,
                    "z":                 z,
                    "xcm_star":          xcm_s,
                    "ycm_star":          ycm_s,
                    "zcm_star":          zcm_s,
                    "r200_kpc":          r200_kpc,
                    "xcm_star_old":      x0,
                    "ycm_star_old":      y0,
                    "zcm_star_old":      z0,
                    "r200_kpc_group_old": snap_grp['Groups/Group_R_Crit200'][group_id] * a * h**(-1),
                    "flag_cm_recomputed": flag_recalc,
                })

    # -----------------------------------------------------------------------
    # Save output
    # -----------------------------------------------------------------------
    df = pd.DataFrame(rows)
    out_path = f"{output_dir}r200_mainbranches_all_{galaxy_type}_z0_{sim_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"[Step 1] Saved {out_path} ({len(df)} rows)")

    # -----------------------------------------------------------------------
    # Optional diagnostic plots: CM and R200 comparison (recalculated vs HDF5)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 50)
    print(f" Diagnostic plots — {galaxy_type.upper()} in {sim_name}")
    print("=" * 50 + "\n")

    plt.style.use("seaborn-v0_8-paper")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(df["xcm_star_old"], df["xcm_star"], s=2, alpha=0.5)
    axes[0].set_xlabel("x_CM (HDF5) [kpc]")
    axes[0].set_ylabel("x_CM (shrinking sphere) [kpc]")
    axes[0].set_title("CM x-coordinate comparison")

    axes[1].scatter(df["r200_kpc_group_old"], df["r200_kpc"], s=2, alpha=0.5)
    axes[1].set_xlabel("R200 (HDF5) [kpc]")
    axes[1].set_ylabel("R200 (recomputed) [kpc]")
    axes[1].set_title("R200 comparison")

    flag_counts = df["flag_cm_recomputed"].value_counts()
    axes[2].bar(flag_counts.index.astype(str), flag_counts.values)
    axes[2].set_xlabel("flag_cm_recomputed")
    axes[2].set_ylabel("count")
    axes[2].set_title("Fraction with shrinking-sphere CM")

    plt.tight_layout()
    plt.show()
