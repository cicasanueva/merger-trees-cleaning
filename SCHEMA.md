# Pipeline output schema

This file documents every CSV and parquet file produced by the pipeline, column by column.
All masses are in M☉ unless stated otherwise. All distances are in physical kpc.
Redshifts and scale factors follow standard cosmological conventions.

---

## Step 1 output

### `Datos/CM_R200/r200_mainbranches_all_central_z0_{sim}.csv`

One row per snapshot along the main progenitor branch of each central galaxy.
Produced by `step1_CM_r200.py → calcula_cm_r200()`.

| Column | Type | Description |
|---|---|---|
| `uid` | int | Unique subhalo ID: `snap * 1_000_000 + subfind` |
| `snap` | int | Snapshot number |
| `subfind` | int | SubFind subhalo number within the snapshot |
| `z` | float | Redshift at this snapshot |
| `a` | float | Scale factor at this snapshot |
| `cmx` | float | Shrinking-sphere CM x-coordinate [physical kpc] |
| `cmy` | float | Shrinking-sphere CM y-coordinate [physical kpc] |
| `cmz` | float | Shrinking-sphere CM z-coordinate [physical kpc] |
| `vcmx` | float | CM velocity x-component [km/s] |
| `vcmy` | float | CM velocity y-component [km/s] |
| `vcmz` | float | CM velocity z-component [km/s] |
| `r200_kpc` | float | Virial radius R₂₀₀ [physical kpc] |
| `m200` | float | Virial mass M₂₀₀ [M☉] |
| `flag_cm_hdf5` | int | 1 if shrinking sphere failed (< 100 stars) and SubFind CM was used |

---

## Step 2 output

### `Datos/All_Mergers/mergers_{sim}_gx_{gxid}_summary.csv`

One row per merger event detected from the AMIGA tree for the target central galaxy.
Produced by `step2_all_mergers_tree.py → all_mergers_tree()`.

| Column | Type | Description |
|---|---|---|
| `origin_node` | str | UID of the satellite's root node in the merger tree |
| `uid_before_merge` | float | UID of the satellite at the last snapshot before it merges into the central |
| `snap_merge` | int | Snapshot at which the merger is detected |
| `z_merge` | float | Redshift at the merger snapshot |
| `mu` | float | Baryonic mass ratio (satellite / central) at the merger snapshot |

---

## Step 3 output

### `Datos/Baryon_Mergers/baryon_mergers_{sim}_gx_{gxid}_summary.csv`

One row per baryonic merger (satellites that had stars or gas at some point).
Produced by `step3_baryonic_mergers_caract.py → baryon_merger_tree_caract()`.

| Column | Type | Description |
|---|---|---|
| `origin_node` | str | UID of the satellite root in the merger tree |
| `uid_before_merge` | float | UID at last snapshot before disruption |
| `uid_infall` | str | UID at t_infall (last snapshot outside R₂₀₀) |
| `uid_disruption` | str | UID at last available snapshot before disruption |
| `snap_infall` | int | Snapshot number at t_infall |
| `z_infall` | float | Redshift at t_infall |
| `mass_stars_infall` | float | Stellar mass of satellite at t_infall [M☉] |
| `mass_gas_infall` | float | Gas mass of satellite at t_infall [M☉] |
| `mass_dm_infall` | float | Dark matter mass of satellite at t_infall [M☉] |
| `mass_baryon_infall` | float | Baryonic mass (stars + gas) at t_infall [M☉] |
| `Mstell_frac_t_infall_in_central_z0` | float | Fraction of satellite stellar mass at t_infall that ends in the central at z=0 |
| `Mgas_frac_t_infall_in_central_z0` | float | Fraction of satellite gas mass at t_infall that ends in the central at z=0 |
| `Mdm_frac_t_infall_in_central_z0` | float | Fraction of satellite DM at t_infall that ends in the central at z=0 |
| `frac_star1_from_gas_infall_in_central_z0` | float | Fraction (by central z=0 stellar mass) of 1st-gen stars formed from satellite's infall gas |
| `frac_star2_from_gas_infall_in_central_z0` | float | Fraction (by central z=0 stellar mass) of 2nd-gen stars formed from satellite's infall gas |
| `mass_star1_from_gas_infall_in_central_z0` | float | Mass of 1st-gen stars in central at z=0 that formed from satellite's infall gas [M☉] |
| `mass_star2_from_gas_infall_in_central_z0` | float | Mass of 2nd-gen stars in central at z=0 that formed from satellite's infall gas [M☉] |
| `flag_tuvo_estrellas` | bool | Satellite had stellar particles at some point in its history |
| `flag_tuvo_gas` | bool | Satellite had gas particles at some point in its history |
| `flag_tuvo_dm` | bool | Satellite had DM particles at some point in its history |
| `flag_always_inside_r200` | bool | Satellite was always inside the central's R₂₀₀ (→ groups A1/A2) |
| `flag_never_inside_r200` | bool | Satellite never crossed the central's R₂₀₀ (→ groups B1/B2) |
| `flag_CM_hdf5_tinfall_sat` | bool | SubFind CM used for satellite at t_infall (shrinking sphere not possible) |
| `flag_CM_hdf5_tinfall_cen` | bool | SubFind CM used for central at t_infall (shrinking sphere not possible) |

---

## Step 4 output

### `Datos/Baryon_Mergers_Depurados/baryon_mergers_dep_{sim}_gx_{gxid}.csv`

Classified and overlap-filtered merger catalogue.
Produced by `step4_depuracion_mergers_trees.py → depura_mergers_trees()`.

Contains all columns from Step 3, plus:

| Column | Type | Description |
|---|---|---|
| `grupo` | str | Six-group classification: A1, A2, B1, B2, C1, C2 |
| `flag_overlap_eliminar_inicioC2` | bool | This merger is eliminated because it overlaps with an earlier C2 merger |
| `flag_overlap_eliminar_contieneC2` | bool | This merger is eliminated within an overlap group that contains a C2 |
| `flag_overlap_eliminar_sinC` | bool | This merger is eliminated in an overlap group with no C merger |
| `uids_asociadas_overlap_inicioC2` | str | JSON list of UIDs in the same C2-conservancy overlap group |
| `uids_asociadas_overlap_contieneC2` | str | JSON list of UIDs in the same C2-containing overlap group |
| `uids_asociadas_overlap_sinC` | str | JSON list of UIDs in the same no-C overlap group |
| `uid_disruption_overlap_inicioC2` | str | Updated disruption UID after C2 conservancy |
| `uid_disruption_overlap_contieneC2` | str | Updated disruption UID for C2-containing group |
| `uid_disruption_overlap_sinC` | str | Updated disruption UID for no-C group |
| `uid_overlap_conservado_inicioC2` | str | UID of the conserved merger in C2-start group |
| `uid_overlap_conservado_contieneC2` | str | UID of the conserved merger in C2-containing group |
| `uid_overlap_conservado_sinC` | str | UID of the conserved merger in no-C group |

### `Datos/Baryon_Mergers_Depurados/FINAL_clean_mergers_{sim}_gx{gxid}.csv`

Final, human-facing cleaned catalogue. Same schema as above but with eliminated
mergers removed and disruption UIDs already updated. This is the recommended file
for all downstream analyses.
