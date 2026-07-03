# 2νββ Decay Spectrum Calculator (¹³⁶Xe)

A Python reproduction of the **RADIAL** FORTRAN subroutine (Francesc Salvat) for numerically solving the radial Dirac equation, applied to computing electron energy spectra and angular correlations for **two-neutrino double beta decay (2νββ)**. Developed as part of a Master's thesis.

The code numerically integrates the Dirac equation for an electron in a nuclear electrostatic potential, builds relativistic electron wavefunctions, and uses them to compute the Fermi function, phase-space factors (G/H), the single/double differential decay spectrum, the angular correlation coefficient, and the total half-life — benchmarked against Šimkovič et al. (2018) reference values.

## Features

- Numerical Dirac equation solver on a RADIAL-style non-uniform radial mesh (fine near the origin, coarser outward)
- Four nuclear electrostatic potential models:
  - `0` — pure Coulomb (point charge)
  - `1` — Coulomb + exponential correction
  - `2` — finite nuclear size (uniformly charged sphere)
  - `3` — Thomas-Fermi screened potential
- Numeric Fermi function F(Z, E), compared against analytic (pure Coulomb) and FORTRAN reference values
- Phase-space integrals (G/H factors), single- and double-differential decay spectra, and the angular correlation coefficient κ²ν
- Half-life calculation and comparison against nuclear-matrix-element-based references
- Built-in isotope database (Ca48, Ge76, Se82, Zr96, Mo100, Pd110, Cd116, Sn124, Te130, Xe136, Nd150), exposed via CLI for Xe136, Mo100, Nd150
- Extensive plotting: wavefunction comparisons, Fermi function curves, energy spectra, and angular/double-differential contour maps

## Requirements

- Python 3
- `numpy`
- `scipy`
- `mpmath`
- `matplotlib`
- `tqdm`

```bash
pip install numpy scipy mpmath matplotlib tqdm
```

## Usage

The program runs in one of four modes, selected via `--mode`. Defaults are read from `configurations/config.json` and can be overridden on the command line.

```bash
# Run with defaults from config.json (default mode is "all": generate then data)
python main.py

python main.py --mode generate     # solve the Dirac equation, save wavefunctions
python main.py --mode plot         # visualize saved wavefunction / Fermi data
python main.py --mode data         # compute G/H factors, spectrum, half-life
python main.py --mode all          # explicit: generate + data in sequence

# Common overrides
python main.py --mode data --isotope Xe136 --potential 2
python main.py --mode generate --potential 3 --num_samples 50
```

Potential options: `0`=pure Coulomb, `1`=Coulomb+exp, `2`=finite-size Coulomb, `3`=Thomas-Fermi.
Isotope options (CLI): `Xe136`, `Mo100`, `Nd150`.

### Verbose output and progress info

By default the program runs quietly, writing only data files and a `results.txt` summary. Pass `-v`/`--verbose` for progress prints and richer output:

```bash
python main.py --mode generate -v          # also writes Z=0 free-particle reference files
python main.py --mode generate -v --no-z0  # progress prints only, skips Z=0 (faster)
python main.py --mode data -v              # prints comparison totals; -v alone -> all plots
```

### Plots (data mode)

Available plot keys: `gf`, `fermi`, `epsilon`, `double_diff`, `electron`, `energy_diff`, `grid`.

```bash
python main.py --mode data --plots fermi double_diff   # only these plots
python main.py --mode data -v --plots fermi double_diff
python main.py --mode data --plots                     # no plots, still computes G/H + half-life
python main.py --mode data                             # quiet mode, no plots
```

Run `python main.py --help` for the full list of CLI options (energy range, mesh resolution, Z, A, kappa, angular momentum, output directory, etc.).

## Architecture

`main.py` is the sole entry point. It reads `configurations/config.json` for defaults, applies CLI overrides, builds a per-isotope config dict from `configurations/isotopes.py` (Z, A, Q, G/H reference values, nuclear matrix elements), and dispatches to the selected mode.

| Mode | Function | Description |
|---|---|---|
| `generate` | `Generate_Fermi_Data` (`dirac_numerical_radial.py`) | Solves the Dirac equation numerically over a range of electron energies; saves P/Q wavefunctions to `.npz` files under `output/` |
| `plot` | `Visualize` (`dirac_numerical_radial.py`) | Loads saved `.npz` data and produces wavefunction / Fermi function comparison plots |
| `data` | `calc_double_beta_decay_spectrum` (`radial_data_handler.py`) | Computes phase-space factors (G/H), the decay spectrum, angular correlation, and half-life |
| `all` | — | Runs `generate` then `data` in sequence (default) |

### Modules

| Module | Role |
|---|---|
| `algorithms/dirac_numerical_radial.py` | Dirac ODE solver, potential definitions, wavefunction normalization, Fermi data generation |
| `algorithms/radial_data_handler.py` | Phase-space integrals, spectrum computation, half-life, result output |
| `algorithms/integrands.py` | Pure integrand functions (two-electron kernel, epsilon spectrum, triple-differential angular spectrum) |
| `algorithms/fermi_function_utlities.py` | Fermi function F(Z,E), E-function, wavefunction plot helpers |
| `algorithms/grid_creation.py` | RADIAL-style non-uniform radial mesh construction |
| `algorithms/thomas_fermi.py` | Thomas-Fermi screening potential via power-series solution |
| `configurations/physics_constants.py` | Physical constants (mₑ, α, G_F, V_ud, ℏc, g_A, ...) |
| `configurations/isotopes.py` | Per-isotope data: Z, A, Q, G/H reference values, nuclear matrix elements |
| `configurations/config.json` | Default runtime parameters |

### Data flow

1. `grid_creation.py` builds the radial mesh (logarithmic near the origin, linear far out).
2. `dirac_numerical_radial.py` integrates the Dirac equation on this mesh for each electron energy, saving the upper (P) and lower (Q) wavefunction components.
3. `fermi_function_utlities.py` constructs the numeric Fermi function and E-function from the saved wavefunctions at the nuclear radius.
4. `integrands.py` defines the phase-space kernels; `radial_data_handler.py` integrates them (via `scipy.integrate.nquad`/`quad`) over the two-electron energy domain to obtain G/H factors and the spectrum.
5. For the angular spectrum, `integrands.py` computes the triple-differential kernel dΓ/(dEe1 dEe2 d cos θ) using the angular correlation coefficient κ²ν; `radial_data_handler.py` integrates over cos θ to recover dΓ/(dEe1 dEe2).

### Reference data

Each potential has a matching FORTRAN reference directory (`out_fortran`, `fortran_output_finite_size`, `fortran_output_thomas_fermi`) containing RADIAL output (`FermiN.dat`, `cxem_*.out`) used for comparison against the numeric results, where available. Potential 1 has no FORTRAN or Šimkovič reference.

## Key physical conventions

- Energies are total energies E = T + mₑ unless noted otherwise (mₑ = 0.511 MeV).
- Kinematic boundary: E1 + E2 ≤ Q + 2mₑ, with E ∈ [mₑ, Q + mₑ].
- Momenta: p = √(E² − mₑ²).
- Default isotope ¹³⁶Xe → ¹³⁶Ba: daughter Z = 56, Q = 2.457 MeV, T½ ≈ 2.2 × 10²¹ yr.
- The radial grid and Dirac solver use atomic units; `mpmath` precision is set to 30 decimal places for near-origin wavefunction accuracy.

## Status

This is research code developed for an ongoing Master's thesis; there is no automated test suite or linter configured. Some numerical issues are under active investigation — see `CLAUDE.md` for developer-facing implementation notes and known caveats.
