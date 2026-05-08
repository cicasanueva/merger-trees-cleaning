# Internal Units Assumption

This pipeline explicitly assumes that the input HDF5 snapshot files are structured following typical cosmological code units (such as those used by the CIELO simulation). 

If you are using this code with other simulations, your raw data must match the following internal code units, as the Python scripts hardcode these conversion factors (e.g., `* 1e10 / h` and `* a / h`).

| Physical Quantity | Expected Raw Code Unit | Script Internal Conversion | Final Physical Unit Output |
| --- | --- | --- | --- |
| **Mass** (`PartType*/Masses`) | $10^{10} M_{\odot} / h$ | `mass * 1e10 / h` | $M_{\odot}$ (Physical) |
| **Coordinates** (`PartType*/Coordinates`) | Comoving kpc / $h$ (ckpc/$h$) | `pos * a / h` | Physical kpc |
| **Subhalo CM** (`SubGroupPos`) | Comoving kpc / $h$ (ckpc/$h$) | `pos * a / h` | Physical kpc |
| **Critical Density** ($\rho_{\rm crit}$) | — | Computed internally using $H_0$ and $\Omega_m$ | $M_{\odot} / \mathrm{kpc}^3$ |

> [!IMPORTANT]
> Do not input data that has already been converted to physical $M_{\odot}$ or physical kpc into the raw HDF5 files. The pipeline relies on fetching $a$ (scale factor) and $h$ (Hubble parameter) from the `Header` to perform the comoving-to-physical translations dynamically at each snapshot.
