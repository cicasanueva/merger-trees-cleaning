# Pipeline output schema

This file documents every CSV file produced by the pipeline, column by column.
All masses are in M☉ unless stated otherwise. All distances are in physical kpc.
Redshifts and scale factors follow standard cosmological conventions.

---

## Spanish → English column name mapping

The table below lists all column names that were renamed from Spanish to English
during the v2.0 translation of this repository.

| Old name (Spanish) | New name (English) |
|---|---|
| `flag_cm_recalculado` | `flag_cm_recomputed` |
| `flag_tuvo_estrellas` | `flag_had_stars` |
| `flag_tuvo_gas` | `flag_had_gas` |
| `flag_tuvo_dm` | `flag_had_dm` |
| `grupo` | `group` |
| `flag_overlap_eliminar_inicioC2` | `flag_overlap_remove_start_c2` |
| `flag_overlap_eliminar_contieneC2` | `flag_overlap_remove_contains_c2` |
| `flag_overlap_eliminar_sinC` | `flag_overlap_remove_no_c` |
| `uids_asociadas_overlap_inicioC2` | `uids_overlap_start_c2` |
| `uids_asociadas_overlap_contieneC2` | `uids_overlap_contains_c2` |
| `uids_asociadas_overlap_sinC` | `uids_overlap_no_c` |
| `uid_disruption_overlap_inicioC2` | `uid_disruption_overlap_start_c2` |
| `uid_disruption_overlap_contieneC2` | `uid_disruption_overlap_contains_c2` |
| `uid_disruption_overlap_sinC` | `uid_disruption_overlap_no_c` |
| `uid_overlap_conservado_inicioC2` | `uid_overlap_kept_start_c2` |
| `uid_overlap_conservado_contieneC2` | `uid_overlap_kept_contains_c2` |
| `uid_overlap_conservado_sinC` | `uid_overlap_kept_no_c` |
| `frac_star_A_en_Bs_overlap_inicioC2` | `frac_stars_A_in_B_overlap_start_c2` |
| `frac_star_Bs_en_A_overlap_inicioC2` | `frac_stars_B_in_A_overlap_start_c2` |
| `frac_gas_A_en_Bs_overlap_inicioC2` | `frac_gas_A_in_B_overlap_start_c2` |
| `frac_gas_Bs_en_A_overlap_inicioC2` | `frac_gas_B_in_A_overlap_start_c2` |
| `frac_dm_A_en_Bs_overlap_inicioC2` | `frac_dm_A_in_B_overlap_start_c2` |
| `frac_dm_Bs_en_A_overlap_inicioC2` | `frac_dm_B_in_A_overlap_start_c2` |
| *(same pattern for `_contieneC2` → `_contains_c2` and `_sinC` → `_no_c`)* | |

---

## Step 1 output

### `Datos/CM_R200/r200_mainbranches_all_{type}_z0_{sim}.csv`

One row per snapshot along the main progenitor branch of each central galaxy.
Produced by `step1_CM_r200.py → compute_cm_r200()`.

| Column | Type | Description |
|---|---|---|
| `uid` | int | Unique subhalo ID: `snap * 1_000_000 + subfind` |
| `snap` | int | Snapshot number |
| `z` | float | Redshift at this snapshot |
| `xcm_star` | float | Shrinking-sphere CM x-coordinate [physical kpc] |
| `ycm_star` | float | Shrinking-sphere CM y-coordinate [physical kpc] |
| `zcm_star` | float | Shrinking-sphere CM z-coordinate [physical kpc] |
| `r200_kpc` | float | Recomputed virial radius R₂₀₀ [physical kpc]; interpolated from enclosed-density profile |
| `xcm_star_old` | float | SubFind-tabulated CM x-coordinate [physical kpc] (for comparison) |
| `ycm_star_old` | float | SubFind-tabulated CM y-coordinate [physical kpc] (for comparison) |
| `zcm_star_old` | float | SubFind-tabulated CM z-coordinate [physical kpc] (for comparison) |
| `r200_kpc_group_old` | float | SubFind-tabulated R₂₀₀ [physical kpc] (for comparison) |
| `flag_cm_recomputed` | bool | `True` if shrinking-sphere was used; `False` if the subhalo had < 100 stars and SubFind CM was used instead |

---

## Step 2 output

### `Datos/All_Mergers/mergers_{sim}_gx_{gxid}_summary.csv`

One row per merger event detected from the AMIGA tree for the target central galaxy.
Produced by `step2_all_mergers_tree.py → detect_mergers_from_tree()`.

| Column | Type | Description |
|---|---|---|
| `origin_node` | str | UID of the satellite's root node in the merger tree |
| `snap_merge` | int | Snapshot at which the merger is detected |
| `z_merge` | float | Redshift at the merger snapshot |
| `uid_merge` | str | UID of the node in the main branch where the satellite merges |
| `uid_before_merge` | float | UID of the satellite at the last snapshot before it merges into the central |

---

## Step 3 output

### `Datos/Baryon_Mergers/baryon_mergers_{sim}_gx_{gxid}_summary.csv`

One row per baryonic merger (satellites that had stars or gas at some point in their history).
Produced by `step3_baryonic_mergers_caract.py → characterize_baryonic_mergers()`.

| Column | Type | Description |
|---|---|---|
| `uid_disruption` | str | UID of the satellite at its last snapshot before disruption (`uid_before_merge` from Step 2) |
| `uid_infall` | str | UID at t_infall (last snapshot the satellite is outside R₂₀₀; from the next snapshot it is inside) |
| `snap_infall` | int | Snapshot number at t_infall (NaN if `flag_always_inside_r200` or `flag_never_inside_r200` is set) |
| `z_infall` | float | Redshift at t_infall |
| `mass_stars_infall` | float | Stellar mass of satellite at t_infall [M☉] |
| `mass_gas_infall` | float | Gas mass of satellite at t_infall [M☉] |
| `mass_dm_infall` | float | Dark matter mass of satellite at t_infall [M☉] |
| `mass_baryon_infall` | float | Baryonic mass (stars + gas) at t_infall [M☉] |
| `Mstell_frac_t_infall_in_central_z0` | float | Fraction of satellite stellar mass at t_infall that ends up in the central at z=0 |
| `Mgas_frac_t_infall_in_central_z0` | float | Fraction of satellite gas mass at t_infall that ends up in the central at z=0 |
| `Mdm_frac_t_infall_in_central_z0` | float | Fraction of satellite DM at t_infall that ends up in the central at z=0 |
| `frac_star1_from_gas_infall_in_central_z0` | float | Fraction (by mass) of 1st-gen stars from the satellite's infall gas that end up in the central at z=0 |
| `frac_star2_from_gas_infall_in_central_z0` | float | Fraction (by mass) of 2nd-gen stars from the satellite's infall gas that end up in the central at z=0 |
| `mass_star1_from_gas_infall_in_central_z0` | float | Mass of 1st-gen stars (from satellite infall gas) in the central at z=0 [M☉] |
| `mass_star2_from_gas_infall_in_central_z0` | float | Mass of 2nd-gen stars (from satellite infall gas) in the central at z=0 [M☉] |
| `flag_had_stars` | bool | Satellite had stellar particles at some point in its history |
| `flag_had_gas` | bool | Satellite had gas particles at some point in its history |
| `flag_had_dm` | bool | Satellite had DM particles at some point in its history |
| `flag_always_inside_r200` | bool | Satellite was always inside the central's R₂₀₀; `uid_infall` is set to the earliest tracked UID |
| `flag_never_inside_r200` | bool | Satellite never crossed the central's R₂₀₀; `uid_infall` is set to the latest tracked UID |
| `flag_CM_hdf5_tinfall_sat` | bool | SubFind CM used for satellite at t_infall (shrinking sphere not possible) |
| `flag_CM_hdf5_tinfall_cen` | bool | SubFind CM used for central at t_infall (shrinking sphere not possible) |

---

## Step 4 output

### `Datos/Baryon_Mergers_Cleaned/baryon_mergers_dep_{sim}_gx_{gxid}.csv`

Classified and overlap-filtered merger catalogue.
Produced by `step4_depuracion_mergers_trees.py → clean_merger_catalog()`.

Contains all columns from Step 3, plus:

| Column | Type | Description |
|---|---|---|
| `group` | str | Six-group classification: A1, A2, B1, B2, C1, C2 |
| `flag_overlap_remove_start_c2` | bool | This merger is flagged for removal: it overlaps with an earlier C2 merger that starts the group |
| `flag_overlap_remove_contains_c2` | bool | This merger is flagged for removal: it is in an overlap group that contains a C2 (but the C2 is not the oldest) |
| `flag_overlap_remove_no_c` | bool | This merger is flagged for removal: it is in an overlap group with no C-type merger, or it is a spurious A2 fragmentation (Filter 4) |
| `uids_overlap_start_c2` | str | JSON list of UIDs in the same C2-start overlap group |
| `uids_overlap_contains_c2` | str | JSON list of UIDs in the same C2-containing overlap group |
| `uids_overlap_no_c` | str | JSON list of UIDs in the same no-C overlap group |
| `uid_disruption_overlap_start_c2` | str | Updated disruption UID after C2-start filter (latest disruption in the group) |
| `uid_disruption_overlap_contains_c2` | str | Updated disruption UID after C2-containing filter |
| `uid_disruption_overlap_no_c` | str | Updated disruption UID after no-C filter |
| `uid_overlap_kept_start_c2` | str | UID of the merger conserved within the C2-start group |
| `uid_overlap_kept_contains_c2` | str | UID of the merger conserved within the C2-containing group |
| `uid_overlap_kept_no_c` | str | UID of the merger conserved within the no-C group |
| `frac_stars_A_in_B_overlap_start_c2` | float | Fraction of A's stellar particles contained in B (C2-start filter) |
| `frac_stars_B_in_A_overlap_start_c2` | float | Fraction of B's stellar particles contained in A (C2-start filter) |
| `frac_gas_A_in_B_overlap_start_c2` | float | Fraction of A's gas particles contained in B (C2-start filter) |
| `frac_gas_B_in_A_overlap_start_c2` | float | Fraction of B's gas particles contained in A (C2-start filter) |
| `frac_dm_A_in_B_overlap_start_c2` | float | Fraction of A's DM particles contained in B (C2-start filter) |
| `frac_dm_B_in_A_overlap_start_c2` | float | Fraction of B's DM particles contained in A (C2-start filter) |
| *(same `frac_*` columns for `_contains_c2` and `_no_c`)* | float | Equivalent fraction columns for filters 2 and 3 |

### `Datos/Baryon_Mergers_Cleaned/FINAL_resumen_clean_mergers_{sim}_gx{gxid}.csv`

Final cleaned catalogue — same column schema as above but with flagged mergers
already removed and disruption UIDs updated to their final values. This is the
recommended file for all downstream analyses.
