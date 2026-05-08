#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Catalina Casanueva (cicasanueva@uc.cl), 2025
#
# Pipeline configuration: paths, output directory names, and algorithm thresholds.
# Edit this file when running on a different machine — no step script needs to change.

from pathlib import Path

# ----- paths -----

# Root directory containing the CIELO simulation files (HDF5 + AMIGA trees).
# On the cluster this points to the shared data volume.
SIM_ROOT = Path("/data1/CIELO/Simulations")

# Root directory for all pipeline outputs (CSVs).
DATA_ROOT = Path("Datos")

# Output subdirectory names (all relative to DATA_ROOT).
DIR_CM_R200            = "CM_R200"
DIR_ALL_MERGERS        = "All_Mergers"
DIR_BARYON_MERGERS     = "Baryon_Mergers"
DIR_BARYON_MERGERS_DEP = "Baryon_Mergers_Depurados"


def get_sim_paths(sim_name, sim_root=SIM_ROOT):
    """Return HDF5 and merger-tree file paths for a given simulation."""
    base = sim_root / sim_name
    return {
        "hdf5": base / f"{sim_name}.hdf5",
        "tree": base / f"{sim_name}_tree.dat",
    }


def get_output_paths(sim_name, gx_id, data_root=DATA_ROOT):
    """Return the expected output file paths for each pipeline step."""
    return {
        "r200":         data_root / DIR_CM_R200 / f"r200_mainbranches_all_central_z0_{sim_name}.csv",
        "all_mergers":  data_root / DIR_ALL_MERGERS / f"mergers_{sim_name}_gx_{gx_id}_summary.csv",
        "baryon_mergers": data_root / DIR_BARYON_MERGERS / f"baryon_mergers_{sim_name}_gx_{gx_id}_summary.csv",
        "dep_mergers":  data_root / DIR_BARYON_MERGERS_DEP / f"baryon_mergers_dep_{sim_name}_gx_{gx_id}.csv",
        "final_mergers": data_root / DIR_BARYON_MERGERS_DEP / f"FINAL_clean_mergers_{sim_name}_gx{gx_id}.csv",
    }


# ----- snapshot parameters -----

# Most CIELO runs use snapshot 128 as z=0; LG2 is the exception.
SNAP_Z0_DEFAULT  = 128
SNAP_Z0_OVERRIDES = {"LG2": 127}

def get_snap_z0(sim_name):
    """Return the z=0 snapshot number for the given simulation."""
    return SNAP_Z0_OVERRIDES.get(sim_name, SNAP_Z0_DEFAULT)


# ----- shrinking-sphere CM (Steps 1 and 3) -----

SHRINK_KEEP_FRACTION = 0.975   # fraction of particles kept per iteration (outer 2.5% removed)
SHRINK_MIN_FRAC      = 0.10    # stop when particle count < 10% of initial
SHRINK_MIN_N         = 100     # minimum count to attempt the algorithm

# ----- R200 definition (Step 1) -----

R200_OVERDENSITY  = 200        # mean enclosed density threshold in units of rho_crit(z)
MASS_CUT_STELLAR  = 1e9        # minimum stellar mass [M_sun] to include a galaxy

# ----- merger classification (Step 4) -----

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

# Six-group classification scheme:
#   A = satellite always inside R200 of the central
#   B = satellite never crossed R200
#   C = satellite crossed R200 at some point
#   1 = low mass transfer; 2 = high mass transfer
MERGER_GROUPS = ["A1", "A2", "B1", "B2", "C1", "C2"]

# ----- overlap detection (Step 4) -----

OVERLAP_DOMINANT_THRESH  = 0.90   # fraction of dominant component shared to flag an overlap
OVERLAP_SECONDARY_THRESH = 0.50   # fraction for secondary components (if > 100 particles)
OVERLAP_MIN_PARTICLES    = 100    # minimum count before secondary overlap is enforced

# ----- GADGET-3 particle types -----

PARTTYPE_GAS   = "PartType0"
PARTTYPE_DM    = "PartType1"
PARTTYPE_STARS = "PartType4"
