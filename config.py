#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Catalina Casanueva (cicasanueva@uc.cl), 2025
#
# Pipeline configuration: paths, output directory names, and algorithm thresholds.
# Edit this file when running on a different machine — no step script needs to change.

from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

# Root directory containing the CIELO simulation files (HDF5 + AMIGA trees).
SIM_ROOT = Path("/data1/CIELO/Simulations")

# Root directory for all pipeline outputs (CSV files).
DATA_ROOT = Path("Datos")

# ---------------------------------------------------------------------------
# Output subdirectory names (all relative to DATA_ROOT)
# ---------------------------------------------------------------------------
DIR_CM_R200            = "CM_R200"
DIR_ALL_MERGERS        = "All_Mergers"
DIR_BARYON_MERGERS     = "Baryon_Mergers"
DIR_BARYON_MERGERS_DEP = "Baryon_Mergers_Cleaned"

# ---------------------------------------------------------------------------
# Snapshot parameters
# ---------------------------------------------------------------------------

# Most CIELO runs use snapshot 128 as z=0; LG2 is the exception.
SNAP_Z0_DEFAULT   = 128
SNAP_Z0_OVERRIDES = {"LG2": 127}


def get_snap_z0(sim_name: str) -> int:
    """Return the z=0 snapshot number for the given simulation name."""
    return SNAP_Z0_OVERRIDES.get(sim_name, SNAP_Z0_DEFAULT)


# ---------------------------------------------------------------------------
# Shrinking-sphere centre-of-mass algorithm (Steps 1 and 3)
# ---------------------------------------------------------------------------
SHRINK_KEEP_FRACTION = 0.975  # fraction of particles kept per iteration (outer 2.5% removed)
SHRINK_MIN_FRAC      = 0.10   # stop iterating when particle count < 10% of initial
SHRINK_MIN_N         = 100    # minimum particle count required to attempt the algorithm

# ---------------------------------------------------------------------------
# R200 definition (Step 1)
# ---------------------------------------------------------------------------
R200_OVERDENSITY = 200   # overdensity threshold in units of rho_crit(z)
MASS_CUT_STELLAR = 1e9   # minimum central stellar mass [M_sun] to include a galaxy

# ---------------------------------------------------------------------------
# Merger classification (Step 4)
# ---------------------------------------------------------------------------

# A merger is "high-transfer" (groups A2, B2, C2) if at least one of the
# mass-fraction columns below exceeds this threshold.
FRAC_TRANSFER_THRESHOLD = 0.10

FRAC_TRANSFER_COLS = [
    "Mstell_frac_t_infall_in_central_z0",
    "Mgas_frac_t_infall_in_central_z0",
    "Mdm_frac_t_infall_in_central_z0",
    "frac_star1_from_gas_infall_in_central_z0",
    "frac_star2_from_gas_infall_in_central_z0",
]

# ---------------------------------------------------------------------------
# Overlap-detection thresholds (Step 4)
# ---------------------------------------------------------------------------
OVERLAP_DOMINANT_FRAC   = 90   # [%] dominant component must overlap by at least this fraction
OVERLAP_SECONDARY_FRAC  = 50   # [%] secondary components (if > 100 particles) must overlap
OVERLAP_SECONDARY_MIN_N = 100  # minimum particle count for a secondary component to be tested
KEEP_FRAC_IN_CONSERVED  = 75   # [%] dominant component of B must be contained in A to eliminate B
