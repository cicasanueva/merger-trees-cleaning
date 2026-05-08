#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Catalina Casanueva (cicasanueva@uc.cl)
#
# Utility functions shared across all pipeline steps.
# Covers: merger tree traversal, unit conversions (comoving -> physical),
# and the shrinking-sphere centre-of-mass algorithm.

import numpy as np
import networkx as nx


# ──────────────────────────────────────────────
# Merger tree navigation
# ──────────────────────────────────────────────

def get_main_branch_unique_ids(subtree, node):
    """
    Walk the main progenitor branch of a given node in the merger tree.

    The AMIGA tree stores each merger event as a directed graph where each
    node can have multiple successors (progenitors). The main branch always
    follows the *first* successor at each step, which by convention in AMIGA
    corresponds to the most-massive progenitor.

    Returns a list of unique subhalo IDs from the given node back to the
    earliest available snapshot.
    """
    mpb = [node]
    while True:
        successors = list(subtree.successors(node))
        if not successors:
            break
        node = successors[0]
        mpb.append(node)
    return mpb


def split_unique_id(unique_id):
    """
    Decode an AMIGA unique subhalo ID into its snapshot number and SubFind index.

    The convention used in CIELO is:
        unique_id = snap_number * 1_000_000 + subfind_number

    Returns (snap_number, subfind_number) as integers.
    """
    subfind_number = int(unique_id % 1_000_000)
    snap_number    = int((unique_id - subfind_number) / 1_000_000)
    return snap_number, subfind_number


# ──────────────────────────────────────────────
# Coordinate and velocity conversions
# (comoving internal units → physical kpc / km s⁻¹)
# ──────────────────────────────────────────────

def pos_co_to_phys_cm(cm, a, h):
    """
    Convert a centre-of-mass position from GADGET comoving to physical kpc.

    cm : array-like of length 3, in GADGET internal comoving units
    a  : scale factor
    h  : dimensionless Hubble parameter
    """
    return [cm[i] * a / h for i in range(len(cm))]


def vel_co_to_phys_cm(cm, vcm, a, h, adot):
    """
    Convert a centre-of-mass velocity from GADGET comoving to physical km/s.

    For SubFind-tabulated positions/velocities the sqrt(a) factor is already
    folded in by the code, so only the Hubble flow correction is applied here.

    adot : da/dt in units of km/s/kpc
    """
    return [vcm[i] + adot * cm[i] / h for i in range(len(vcm))]


def pos_co_to_phys(pos, a, h):
    """
    Convert a (N, 3) particle position array from GADGET comoving to physical kpc.
    """
    factor = a / h
    return pos[:, 0] * factor, pos[:, 1] * factor, pos[:, 2] * factor


def vel_co_to_phys(pos, vel, a, h, adot):
    """
    Convert a (N, 3) particle velocity array from GADGET comoving to physical km/s.

    The peculiar velocity in GADGET is stored as sqrt(a)*v_pec; the Hubble flow
    term adot * x / h accounts for the comoving-to-physical correction.
    """
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]
    vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]
    return (
        vx * np.sqrt(a) + adot * xs / h,
        vy * np.sqrt(a) + adot * ys / h,
        vz * np.sqrt(a) + adot * zs / h,
    )


# ──────────────────────────────────────────────
# Shrinking-sphere centre of mass
# ──────────────────────────────────────────────

def recalculate_cm(x, y, z, vx, vy, vz, m,
                   cmx_subf, cmy_subf, cmz_subf,
                   vcmx_subf, vcmy_subf, vcmz_subf):
    """
    Compute the centre of mass and centre-of-mass velocity using the
    iterative shrinking-sphere method (Power et al. 2003).

    At each iteration, particles beyond 97.5% of the current maximum radius
    (measured from the running CM estimate) are discarded. Iteration stops
    when fewer than 10% of the original particles remain, or when the count
    drops below 100.

    If the input already has fewer than 100 particles, the SubFind-tabulated
    CM is returned unchanged.

    Parameters
    ----------
    x, y, z         : physical coordinates of stellar particles [kpc]
    vx, vy, vz      : physical velocities [km/s]
    m               : particle masses [M_sun]
    cmx/y/z_subf    : fallback CM from SubFind [kpc]
    vcmx/y/z_subf   : fallback CM velocity from SubFind [km/s]

    Returns
    -------
    x_cm, y_cm, z_cm, vx_cm, vy_cm, vz_cm  (physical, mass-weighted)
    """
    xsh  = np.array(x,  dtype=float)
    ysh  = np.array(y,  dtype=float)
    zsh  = np.array(z,  dtype=float)
    vxsh = np.array(vx, dtype=float)
    vysh = np.array(vy, dtype=float)
    vzsh = np.array(vz, dtype=float)
    msh  = np.array(m,  dtype=float)

    na   = len(xsh)
    nlow = 0.1 * na  # stop when 10% of particles remain

    if na < 100:
        # too few particles to iterate reliably — use SubFind value
        return cmx_subf, cmy_subf, cmz_subf, vcmx_subf, vcmy_subf, vcmz_subf

    # initial distances measured from SubFind CM
    dx    = xsh - cmx_subf
    dy    = ysh - cmy_subf
    dz    = zsh - cmz_subf
    rpris = np.sqrt(dx**2 + dy**2 + dz**2)
    rmaxs = rpris.max()

    # running CM initialised to SubFind value (overwritten on first iteration)
    xcm0,  ycm0,  zcm0  = cmx_subf,  cmy_subf,  cmz_subf
    vxcm0, vycm0, vzcm0 = vcmx_subf, vcmy_subf, vcmz_subf

    while na >= nlow and na >= 100:
        mask = rpris <= 0.975 * rmaxs
        na   = mask.sum()
        if na == 0:
            break

        xsh  = xsh[mask];  ysh  = ysh[mask];  zsh  = zsh[mask]
        vxsh = vxsh[mask]; vysh = vysh[mask];  vzsh = vzsh[mask]
        msh  = msh[mask]

        Mtot  = msh.sum()
        xcm0  = (xsh  * msh).sum() / Mtot
        ycm0  = (ysh  * msh).sum() / Mtot
        zcm0  = (zsh  * msh).sum() / Mtot
        vxcm0 = (vxsh * msh).sum() / Mtot
        vycm0 = (vysh * msh).sum() / Mtot
        vzcm0 = (vzsh * msh).sum() / Mtot

        rpris = np.sqrt((xsh - xcm0)**2 + (ysh - ycm0)**2 + (zsh - zcm0)**2)
        rmaxs = rpris.max()

    return xcm0, ycm0, zcm0, vxcm0, vycm0, vzcm0
