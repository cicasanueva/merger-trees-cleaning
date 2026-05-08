<div align="center">

# CIELO Merger-Tree Cleaning Pipeline

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19338366-c9b8e8?style=flat-square)](https://doi.org/10.5281/zenodo.19338366)
[![License: MIT](https://img.shields.io/badge/License-MIT-b8e8d0?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-≥_3.8-b8d4e8?style=flat-square)](https://www.python.org/)
[![Pipeline](https://img.shields.io/badge/Pipeline-4_steps-f0d4b8?style=flat-square)](#pipeline-structure)

**Author:** Catalina Casanueva-Villarreal (cicasanueva@uc.cl) · **Version:** 1.0 · **License:** MIT

</div>

---

## Overview

This repository contains a Python pipeline to detect, characterize, and clean merger trees for central galaxies in cosmological zoom-in simulations. It is designed to work with simulations based on the GADGET-3 N-body/SPH code that use the AMIGA Halo Finder (AHF; Knollmann & Knebe 2009) merger tree format.

The main goal of the pipeline is to produce a physically consistent, duplicate-free catalogue of baryonic merger events for each central galaxy. The key contribution is the **overlap-based cleaning algorithm** (Step 4), which classifies each merger according to its orbital history and mass-transfer efficiency, then applies eight sequential particle-ID overlap filters to consolidate spurious tree entries arising from the fragmentation of single infalling objects into multiple subhaloes by the structure finder.

---

## Pipeline Structure

The pipeline runs in four sequential steps:

| Step | Script | Description |
|---|---|---|
| 1 | `step1_CM_r200.py` | Computes the centre of mass (CM) and virial radius R₂₀₀ along the main progenitor branch of each central galaxy |
| 2 | `step2_all_mergers_tree.py` | Detects all merger branches that physically merge into the central's main branch using the AMIGA tree |
| 3 | `step3_baryonic_mergers_caract.py` | Characterizes baryonic and dark matter content at infall, and measures mass transfer fractions to the z=0 central |
| 4 | `step4_depuracion_mergers_trees.py` | Core cleaning: classifies mergers into six groups, detects particle-level overlaps, and applies eight sequential filters to produce the final catalogue |

---

## Step-by-step Description

### Step 1 — Centre of Mass and R₂₀₀

**Script:** `step1_CM_r200.py`
**Function:** `calcula_cm_r200(sim_name, snap_z0, mass_cut, tipo='central')`

At z=0, all central galaxies (`SubGroupNumber = 0`) with stellar mass M★ > `mass_cut` are selected. For each central, the full main progenitor branch is traversed snapshot by snapshot.

At each snapshot:
- **Centre of mass:** Recomputed using an iterative shrinking-sphere algorithm applied to stellar particles (PartType4) belonging to the subhalo. The sphere is iteratively shrunk (by a fixed fraction per iteration) until convergence. This is applied only when the subhalo contains ≥ 100 stellar particles; otherwise the SubFind-tabulated position (`SubGroupPos`) from the HDF5 is used (`flag_cm_recalculado = False`).
- **R₂₀₀:** Computed by building an enclosed-density profile ρ(<r) using all particles of the group (gas, dark matter, and stars; PartType 0, 1, and 4). The profile is interpolated in log–log space to find the radius where ρ(<r) = 200 × ρ_crit(z), where ρ_crit(z) is the physical critical density at that redshift.

**Output:** `Datos/CM_R200/r200_mainbranches_all_central_z0_{sim}.csv`
One row per snapshot per central. Key columns: `uid`, `snap`, `z`, `xcm_star`, `ycm_star`, `zcm_star`, `r200_kpc`, `flag_cm_recalculado`.

---

### Step 2 — Merger Detection from the Tree

**Script:** `step2_all_mergers_tree.py`
**Function:** `all_mergers_tree(sim_name, target_subfind_id, snap_target)`

The AMIGA merger tree is loaded as a directed graph using NetworkX. The main progenitor branch of the target central (`uid_target`) is extracted via depth-first search (DFS).

All secondary branches that merge into the central's main branch are identified by computing `single_target_shortest_path` on the reversed graph. For each path from any node to `uid_target`:
- Nodes already belonging to the main branch are excluded.
- Branches that previously merged into another secondary branch (before reaching the central) are excluded, so that the pre-merger object is not double-counted. Only the most massive progenitor at the point of an intermediate merger is retained.

For each valid secondary merger, the following are recorded: the UID of the satellite at the last snapshot before merging (`uid_before_merge`), the snapshot at which it merges (`snap_merge`), the corresponding redshift (`z_merge`), and the root node of the secondary branch in the tree (`origin_node`).

**Output:** `Datos/All_Mergers/mergers_{sim}_gx_{gxid}_summary.csv`

---

### Step 3 — Baryonic Characterization and Infall Time

**Script:** `step3_baryonic_mergers_caract.py`
**Function:** `baryon_merger_tree_caract(sim_name, target_subfind_id, snap_target)`

For each merger detected in Step 2, this step defines the **infall time** (t_infall) and measures the baryonic content of the satellite at that moment.

**Definition of t_infall:** The last snapshot at which the physical distance between the satellite's CM and the central's CM exceeds the central's R₂₀₀ (from Step 1). The snapshot immediately after this is when the satellite is first considered to be inside the halo.
- If the satellite is always inside R₂₀₀ throughout its tracked history: `flag_always_inside_r200 = True`, and the earliest available snapshot is used as the reference.
- If the satellite never crosses R₂₀₀: `flag_never_inside_r200 = True`, and the latest available snapshot is used.

At t_infall, CMs are recomputed with the shrinking-sphere algorithm (if ≥ 100 stars are available; otherwise the HDF5 value is used), for both the satellite (`flag_CM_hdf5_tinfall_sat`) and the central (`flag_CM_hdf5_tinfall_cen`).

**Masses at t_infall:** Stellar (`mass_stars_infall`), gas (`mass_gas_infall`), and dark matter (`mass_dm_infall`) masses are measured from the HDF5 at the infall snapshot.

**Mass transfer fractions:** For each component, the fraction of the satellite's infall mass that ends up in the z=0 central galaxy is computed by direct particle-ID matching between the infall snapshot and the z=0 snapshot. This yields `Mstell_frac_t_infall_in_central_z0`, `Mgas_frac_t_infall_in_central_z0`, and `Mdm_frac_t_infall_in_central_z0`.

Additionally, stellar particles in the z=0 central that formed from the satellite's gas (both those that were already stars at infall — 1st generation — and those that formed from the satellite's gas after infall — 2nd generation) are identified and their masses recorded.

**Output:** `Datos/Baryon_Mergers/baryon_mergers_{sim}_gx_{gxid}_summary.csv`

---

### Step 4 — Merger Tree Cleaning (Core Algorithm)

**Script:** `step4_depuracion_mergers_trees.py`
**Function:** `depura_mergers_trees(sim_name, target_subfind_id, snap_target)`

This is the core of the pipeline. Only mergers with non-zero stellar or gas mass at infall are retained. Eight sequential filters are applied.

#### Classification into six groups

Each merger is first assigned to one of six groups based on its orbital history and mass-transfer efficiency:

| Group | Orbital history | Transfer |
|---|---|---|
| A1 | Always inside R₂₀₀ | Low (all fractions < 10%) |
| A2 | Always inside R₂₀₀ | High (at least one fraction ≥ 10%) |
| B1 | Never inside R₂₀₀ | Low |
| B2 | Never inside R₂₀₀ | High |
| C1 | Crossed R₂₀₀ at some point | Low |
| C2 | Crossed R₂₀₀ at some point | High |

The transfer threshold is 10% for all mass-fraction columns (`Mstell_frac`, `Mgas_frac`, `Mdm_frac`, `frac_star1`, `frac_star2`).

#### Particle-ID loading at t_infall

For every merger, the particle IDs of all gas (PartType0), stellar (PartType4), and dark matter (PartType1) particles belonging to the satellite subhalo at t_infall are loaded from the HDF5. This particle catalogue is the basis for overlap detection.

#### Overlap graph construction

For each pair of mergers (A, B) at the same infall snapshot, the overlap is evaluated as follows:
1. The **dominant component** (the one with the largest mass at infall: stars, gas, or DM) is identified for merger A.
2. A strong overlap is declared if ≥ 90% of merger A's dominant-component particles also belong to merger B.
3. If a secondary component of merger A contains > 100 particles, that component must also overlap merger B at ≥ 50%.

Pairs satisfying these criteria are connected as edges in a NetworkX overlap graph. Connected components of this graph form groups of physically redundant merger events.

#### Filter 1 — C2-conservancy (C2 starts the group)

If the earliest merger within an overlap-connected group is of type C2, that C2 is conserved. All other members of the group that are contained within the C2 (dominant overlap ≥ 75% and secondary overlap ≥ 50% if > 100 particles) are flagged for removal. The disruption UID of the conserved C2 is updated to reflect the latest disruption time in the group.

#### Filter 2 — C2-conservancy (C2 not first)

If the group contains a C2 but the earliest merger is not a C2, the oldest C2 in the group is conserved and later mergers contained within it are removed.

#### Filter 3 — B2/A2 overlap (no C merger in group)

If the group contains only B2 and A2 mergers (no C type), the oldest B2 is conserved and any A2 mergers contained within it are removed.

#### Filter 4 — Contiguous-snapshot test

For A2 mergers: the dominant subhalo in the snapshot immediately before the satellite's birth snapshot is checked. If that subhalo belongs to the central's main branch, the A2 is considered a spurious fragmentation of the central and is removed.

For B2 mergers: the dominant subhalo in the snapshot immediately after the satellite's disruption is checked. If it corresponds to the central, the B2 is retained as a real instantaneous merger.

#### Filter 5 — Spurious or already-captured branches

A2 mergers that have already been absorbed into another merger's branch (i.e. their particles were captured by a C2 or B2) are removed. A2 mergers that are always unbound from the central are also removed. The `uid_infall_conservado` field is updated to point to the absorbing merger when applicable.

#### Filter 6 — Broken-branch validation

For A2 branches that were truncated in the tree, the pipeline checks whether the branch re-emerges as a coherent structure in later snapshots. If it does, it is explicitly retained.

#### Filter 7 — Low-transfer removal

All mergers belonging to groups A1, B1, or C1 (low mass transfer in all components) are removed.

#### Filter 8 — Post-infall unregistered mergers

For each retained merger, the pipeline searches for additional progenitors that merged into the satellite between its disruption and infall snapshots but were not captured as separate entries in the tree. Their masses are computed and added to the satellite's total mass at infall (`uids_extra_infall`, `mass_extra_stars`, `mass_extra_baryon`). Mergers identified via Filter 5 contribute an additional mass correction (`uids_extra_infall_filtro5`).

#### Final flags

- `flag_eliminar_total`: logical OR of all removal flags across all eight filters.
- `flag_conservar_total`: mergers explicitly conserved by the overlap filters, plus all high-transfer crossed-R₂₀₀ mergers not removed.
- Conflicts (simultaneous eliminate and conserve flags) are detected and logged for manual inspection.

**Outputs:**
- `Datos/Baryon_Mergers_Depurados/baryon_mergers_dep_{sim}_gx{gxid}.csv` — classified catalogue with all flags
- `Datos/Baryon_Mergers_Depurados/FINAL_clean_mergers_{sim}_gx{gxid}.csv` — final cleaned catalogue (eliminated entries removed, disruption UIDs updated)
- `Datos/Baryon_Mergers_Depurados/FINAL_resumen_clean_mergers_{sim}_gx{gxid}.csv` — per-merger summary for downstream analysis

---

## Key Parameters and Thresholds

| Parameter | Value | Description |
|---|---|---|
| Minimum stars for shrinking-sphere CM | 100 | Below this, the SubFind-tabulated CM is used |
| Low-transfer threshold | 10% (0.1) | Applied to all mass-fraction columns |
| Dominant component overlap (overlap detection) | ≥ 90% | Required to declare a strong overlap between two mergers |
| Dominant component overlap (C2-conservancy removal) | ≥ 75% | Required to flag a merger for removal within a C2 group |
| Secondary component overlap | ≥ 50% | Required only when the secondary component has > 100 particles |
| R₂₀₀ definition | 200 × ρ_crit(z) | Mean enclosed density criterion; all group particles used |

---

## Dependencies

- Python ≥ 3.8
- `numpy`, `scipy`, `pandas`, `h5py`, `networkx`, `astropy`, `tqdm`

See `requirements.txt` for exact versions.

---

## Data Format and Units

This pipeline is designed to process raw HDF5 snapshots from the CIELO cosmological simulations. The merger tree must be in AMIGA multiline adjacency list format. Input paths are configured via `src/config.py`.

It is crucial to note that the code explicitly assumes the input HDF5 files are in **cosmological code units** (specifically those used by the CIELO simulation). 

For a detailed list of the exact physical variables and the internal units assumed by the Python scripts, please see [UNITS.md](UNITS.md).

For detailed column-by-column descriptions of all output files, see [SCHEMA.md](SCHEMA.md).

---

## Usage

Steps must be run in order. Each step reads the outputs of the previous one.

```python
from src.step1_CM_r200 import calcula_cm_r200
from src.step2_all_mergers_tree import all_mergers_tree
from src.step3_baryonic_mergers_caract import baryon_merger_tree_caract
from src.step4_depuracion_mergers_trees import depura_mergers_trees

sim_name = "LG1"
snap_z0  = 128
gxid     = 4337
mass_cut = 1e9  # M☉

calcula_cm_r200(sim_name, snap_z0, mass_cut)
all_mergers_tree(sim_name, gxid, snap_z0)
baryon_merger_tree_caract(sim_name, gxid, snap_z0)
depura_mergers_trees(sim_name, gxid, snap_z0)
```

---

## Example results

Results are shown for the six central galaxies of the CIELO sample used in Casanueva-Villarreal et al. (in prep.), following the simulation naming convention of Tissera et al. (2025). Counts refer to individual subhalo infall events detected at each stage.

### Pipeline throughput

| Galaxy | Tree events (Step 2) | With baryons (Step 3) | Clean catalogue (Step 4) | Removed (%) |
|--------|---------------------:|----------------------:|-------------------------:|------------:|
| LG1-gx4337 |  866 |  383 | 178 | 54% |
| P3-gx298   |   26 |   17 |  10 | 41% |
| P4-gx18    |  104 |   48 |  29 | 40% |
| P4-gx428   |  249 |  100 |  57 | 43% |
| P4-gx1258  |   50 |   30 |  20 | 33% |
| P7-gx2389  | 2513 | 1027 | 391 | 62% |
| **Total**  | **3808** | **1605** | **685** | **57%** |

The reduction from Step 3 to Step 4 reflects two sequential filters: removal of low mass-transfer events (baryonic mass fraction transferred to the central by z = 0 below 10%) and the overlap-based cleaning algorithm that consolidates spurious duplicate entries caused by structure-finder fragmentation.

### Baryonic mass ratio (μ_b) distribution — clean catalogue

Distribution of baryonic mass ratios across the 685 entries in the clean catalogue (all six galaxies combined):

| μ_b range | N | Fraction |
|-----------|--:|--------:|
| < 0.01    | 596 | 87.0% |
| 0.01–0.05 |  45 |  6.6% |
| 0.05–0.10 |  18 |  2.6% |
| 0.10–0.30 |  21 |  3.1% |
| 0.30–1.00 |   4 |  0.6% |
| > 1.00    |   1 |  0.1% |

Median μ_b = 0.0002; 90th percentile = 0.016; maximum = 2.14. The strongly skewed distribution reflects the dominance of minor and micro-merger events in the histories of these galaxies.

### Satellites with no detected R₂₀₀ crossing (A-group)

A subset of baryonic mergers are flagged as "always inside R₂₀₀": the satellite was found within the central's R₂₀₀ at every recorded snapshot, so no outside-to-inside crossing could be identified. These objects are typically satellites that were already inside R₂₀₀ before the earliest available snapshot, or whose identification by the structure finder began only after capture. Rather than discarding them, the pipeline assigns the earliest available snapshot as the reference time and computes masses and particle inventories at that epoch. The particle-ID overlap analysis in Step 4 can in some cases trace their baryonic contribution back to an earlier interaction, allowing them to be associated with a physically distinct merger event.

| Galaxy | Baryonic mergers (Step 3) | Always inside R₂₀₀ | Fraction |
|--------|-------------------------:|-------------------:|---------:|
| LG1-gx4337 |  383 | 41 | 11% |
| P3-gx298   |   17 |  4 | 24% |
| P4-gx18    |   48 |  4 |  8% |
| P4-gx428   |  100 |  9 |  9% |
| P4-gx1258  |   30 |  4 | 13% |
| P7-gx2389  | 1027 | 83 |  8% |

### Satellite–satellite mergers inside R₂₀₀ and effective μ_b

In some cases the structure finder records what appears to be two separate infall events — two satellites crossing R₂₀₀ at different snapshots — when in fact one satellite merged with the other inside R₂₀₀ before either reached the central. Because the merged object is subsequently fragmented back into two subhaloes by the structure finder, both fragments appear independently in the merger catalogue with their own (underestimated) baryonic mass ratios.

The overlap-based cleaning algorithm in Step 4 identifies these pairs through particle-ID comparison: if two catalogue entries share ≥ 90% of their dominant baryonic component at their respective infall snapshots, they are grouped together and one is flagged for removal. The surviving entry has its mass updated to include the contribution from the absorbed entry, so that the effective μ_b used in the analysis correctly reflects the total baryonic mass of the combined system at the time it first crossed R₂₀₀, rather than the mass of only one of its fragments.

---

## Papers using this pipeline

- Casanueva-Villarreal, C., et al., in prep., *Evolution of angular momentum in the CIELO simulations. I. Temporal evolution of gas–stellar misalignments and their merger context*

If you use this pipeline in your work, please cite the software record:

> Casanueva-Villarreal, C. (2026). *CIELO Merger-Tree Cleaning Pipeline* [Software]. Zenodo. https://doi.org/10.5281/zenodo.19338366

---

## References

- Knollmann, S. R. & Knebe, A. 2009, ApJS, 182, 608
- Springel, V. 2005, MNRAS, 364, 1105
- Tissera, P. B., Bignone, L., Gonzalez-Jara, J., et al. 2025, A&A, 697, A134
