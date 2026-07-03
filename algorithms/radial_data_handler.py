#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import nquad
from scipy.integrate import quad
from pathlib import Path
import os
from configurations.physics_constants import ME, GF, VUD, HBAR_C, GA, E_HARTREE
from algorithms.integrands import (two_electron_kernel, two_electron_spectrum_integrand,
                                    spectrum_epsilon_integrand, single_electron_spectrum_integrand,
                                    energy_diff_spectrum_integrand,
                                    double_differential_spectrum_kernel,
                                    angular_correlation_distribution,
                                    angular_correlation_weighted_kernel)
from algorithms.fermi_function_utlities import Fermi, Fermi_numerical, E_function, plot_f_and_g, find_mesh_point_R_au, Fermi_numerical_division

plt.rcParams.update({
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

Gbeta = GF * VUD


def _gamma_combination(G, xi31, xi51):
    return G[0] + xi31*G[1] + 1/3*xi31**2*G[3] + (1/3*xi31**2 + xi51)*G[2]
c2n = ME * (Gbeta * ME ** 2)**4 / (8 * np.pi**7)
MeVtoyr = (365.25 * 24 * 3600) * 2.998e8 / (1e-15 * HBAR_C)




def load_data(directory_name, file_name, verbose=True):
    """
    Extracts Energy T, radial grid r, upper component wave function P and lower component
    wave function Q from the given .npz file.

    Parameters
    ----------
    directory_name : string
        Name of the directory in which .npz file is located.
    file_name : string
        Name of the .npz file
    verbose : bool
        If True, print the path of the loaded file.

    Returns
    -------
    T : list
        List of kinetic energy values for which P and Q where evaluated
    r : list
        Radial grid points on which P and Q are calculated
    P : list
        List of upper component of Dirac wave function for each T on radial grid r
    Q : list
        List of lower component of Dirac wave function for each T on radial grid r
    """
    full_path = Path(directory_name)/ file_name
    data = np.load(full_path)

    if verbose:
        print(f"loading file from {full_path}")

    T = data["T"]
    r = data["r"]
    P = data["P"]
    Q = data["Q"]
    return T, r, P, Q





# ========== PHASE-SPACE FACTORS CALCULATION ==========

def calculate_phase_space_factors(Q, Fermi_analytic, Fermi_numeric, E_numeric, nuclear_matrix_elements):
    """
    Calculate phase-space factors G and H for double beta decay.

    Parameters
    ----------
    Q : float
        Q-value of the decay process.
    Fermi_analytic : callable
        Analytic Fermi function.
    Fermi_numeric : callable
        Numerical Fermi function from Dirac solutions.
    E_numeric : callable
        Numerical E function.

    Returns
    -------
    dict
        Dictionary containing G_results, G_errors, G_results_num, G_errors_num, H_results_num, H_errors_num
    """
    def bounds_Ee1():
        return [ME, Q + ME]
    
    def bounds_Ee2(Ee1):
        return [ME, Q + 2.0 * ME - Ee1]

    opts_Ee1 = {"epsabs": 1e-18, "epsrel": 1e-16, "limit": 50000, "points": [ME, Q/2 + ME, Q + ME]}
    opts_Ee2 = {"epsabs": 1e-18, "epsrel": 1e-16, "limit": 50000, "points": [ME]}

    tags = [0, 2, 4, 22]
    G_results_list, G_errors_list = [], []
    G_results_num_list, G_errors_num_list = [], []
    H_results_num_list, H_errors_num_list = [], []
    
    for t in tags:
        # Analytic G values
        G_output = nquad(two_electron_spectrum_integrand(t, Q, Fermi_analytic, nuclear_matrix_elements), [bounds_Ee2, bounds_Ee1], opts=[opts_Ee2, opts_Ee1])
        G_result, G_error = G_output[:2]
        G_results_list.append(G_result)
        G_errors_list.append(G_error)

        # Numeric G values
        G_output_num = nquad(two_electron_spectrum_integrand(t, Q, Fermi_numeric, nuclear_matrix_elements), [bounds_Ee2, bounds_Ee1], opts=[opts_Ee2, opts_Ee1])
        G_result_num, G_error_num = G_output_num[:2]
        G_results_num_list.append(G_result_num)
        G_errors_num_list.append(G_error_num)

        # H values (numeric only)
        H_output_num = nquad(two_electron_spectrum_integrand(t, Q, E_numeric, nuclear_matrix_elements), [bounds_Ee2, bounds_Ee1], opts=[opts_Ee2, opts_Ee1])
        H_result, H_error = H_output_num[:2]
        H_results_num_list.append(H_result)
        H_errors_num_list.append(H_error)

    return {
        "G_results": np.asarray(G_results_list),
        "G_errors": np.asarray(G_errors_list),
        "G_results_num": np.asarray(G_results_num_list),
        "G_errors_num": np.asarray(G_errors_num_list),
        "H_results_num": np.asarray(H_results_num_list),
        "H_errors_num": np.asarray(H_errors_num_list),
    }


# ========== TOTAL GAMMA COMPARISON ==========

def compute_and_compare_total_gamma(Q, Fermi_numeric, E_numeric, nuclear_matrix_elements, phase_space_data, verbose=True):
    """
    Integrate dΓ/(dEe1 dEe2 d(cos θ)) over all three variables and compare
    the result with the phase-space combination
    G_0 + xi31*G_2 + 1/3*xi31^2*G_22 + (1/3*xi31^2 + xi51)*G_4
    constructed from the numerical G factors.

    Both quantities carry the same prefactor (MeVtoyr*c2n / (ME^11 * log2))
    and do not include the GA^4 * MGT1^2 nuclear factors.

    Parameters
    ----------
    Q : float
        Q-value of the decay (MeV).
    Fermi_numeric : callable
        Numerical Fermi function F(Ee).
    E_numeric : callable
        Numerical E function E(Ee) = 2*g*f from Dirac wavefunctions.
    nuclear_matrix_elements : dict
        Must contain 'xi31' and 'xi51'.
    phase_space_data : dict
        Output of calculate_phase_space_factors; must contain 'G_results_num'.
    verbose : bool
        If True, print the comparison table.

    Returns
    -------
    dict
        gamma_triple       : Γ from triple-differential integration
        gamma_triple_err   : absolute integration error estimate
        gamma_ps           : Γ from numerical phase-space G combination
        percent_diff       : 100*(gamma_triple - gamma_ps)/gamma_ps
    """
    xi31 = nuclear_matrix_elements["xi31"]
    xi51 = nuclear_matrix_elements["xi51"]
    G_num = phase_space_data["G_results_num"]  # indices: [G0, G2, G4, G22]

    gamma_ps = _gamma_combination(G_num, xi31, xi51)

    def integrand(cos_theta, Ee2, Ee1):
        return double_differential_spectrum_kernel(
            Ee1, Ee2, cos_theta, Q, Fermi_numeric, E_numeric, nuclear_matrix_elements
        )

    def bounds_cos_theta(Ee2, Ee1):
        return [-1.0, 1.0]

    def bounds_Ee2(Ee1):
        return [ME, Q + 2.0 * ME - Ee1]

    bounds_Ee1 = [ME, Q + ME]

    # cos θ integrand is linear → very few evaluations needed; looser tol is fine
    opts_cos = {"epsabs": 1e-14, "epsrel": 1e-12, "limit": 500,   "points": [0.0]}
    opts_Ee2 = {"epsabs": 1e-18, "epsrel": 1e-16, "limit": 50000, "points": [ME]}
    opts_Ee1 = {"epsabs": 1e-18, "epsrel": 1e-16, "limit": 50000, "points": [ME, Q/2 + ME, Q + ME]}

    if verbose:
        print("\nComputing total Γ via triple-differential integration (this may take several minutes)...")
    nquad_output = nquad(
        integrand,
        [bounds_cos_theta, bounds_Ee2, bounds_Ee1],
        opts=[opts_cos, opts_Ee2, opts_Ee1]
    )
    gamma_triple, gamma_triple_err = nquad_output[:2]

    percent_diff = 100.0 * (gamma_triple - gamma_ps) / gamma_ps if gamma_ps != 0.0 else float("nan")

    if verbose:
        print(f"\n{'='*65}")
        print(f"Total Γ Comparison  (units: G factors, no G_A^4 * M_GT1^2)")
        print(f"{'='*65}")
        print(f"  Triple-diff integral : {gamma_triple:.8e}  ±  {gamma_triple_err:.2e}")
        print(f"  Phase-space formula  : {gamma_ps:.8e}")
        print(f"  Difference           : {percent_diff:+.6f}%")
        print(f"{'='*65}\n")

    return {
        "gamma_triple":     gamma_triple,
        "gamma_triple_err": gamma_triple_err,
        "gamma_ps":         gamma_ps,
        "percent_diff":     percent_diff,
    }


# ========== DECAY-RATE-WEIGHTED ANGULAR CORRELATION ==========

def compute_kappa_weighted_average(Q, Fermi_numeric, E_numeric, nuclear_matrix_elements, phase_space_data):
    """
    Compute the decay-rate-weighted average angular correlation coefficient
    at three truncation orders of the phase-space polynomial expansion, plus
    a 'total' value obtained two independent ways for cross-validation.

    ⟨κ²ν⟩ = ∫∫ [dΓ/(dEe1 dEe2)] · κ²ν dEe1 dEe2 / Γ²ν

    The truncation order governs which polynomial terms are included in the
    dΓ integration weight (and, for 'total', whether the full κ²ν ratio
    or its LO approximation is used):

      LO    : κ²ν ratio = 1,    weight = G0/H0 only   → −H0 / G0
      1st   : κ²ν ratio = 1,    weight includes ξ31    → −(H0+ξ31·H2)/(G0+ξ31·G2)
      total (formula)  : full κ²ν ratio, Γ_full from the G0..G22 combination
                          (4 pre-computed tagged integrals, reused from phase_space_data)
      total (direct)   : full κ²ν ratio, Γ_full from one direct integral of the
                          untagged two-electron kernel over the same (Ee1, Ee2) domain

    The two 'total' variants integrate the same numerator
    (angular_correlation_weighted_kernel) over identical bounds; they only
    differ in how the normalizing Γ²ν is obtained, so agreement between them
    is a consistency check on the G0..G22 combination formula.

    Parameters
    ----------
    Q : float
        Q-value of the decay (MeV).
    Fermi_numeric : callable
        Scalar Fermi function F(Ee).
    E_numeric : callable
        Scalar E-function E(Ee) = 2·g·f from Dirac wavefunctions.
    nuclear_matrix_elements : dict
        Must contain 'xi31' and 'xi51'.
    phase_space_data : dict
        Output of calculate_phase_space_factors; must contain
        'G_results_num' and 'H_results_num' (indices: [G0,G2,G4,G22]).

    Returns
    -------
    dict
        kappa_LO, kappa_1st,
        kappa_total, kappa_total_err (formula-based Γ²ν),
        kappa_direct, kappa_direct_err (direct-integral Γ²ν),
        gamma_formula, gamma_direct, gamma_percent_diff
    """
    xi31 = nuclear_matrix_elements["xi31"]
    xi51 = nuclear_matrix_elements["xi51"]
    G = phase_space_data["G_results_num"]  # [G0, G2, G4, G22]
    H = phase_space_data["H_results_num"]  # [H0, H2, H4, H22]

    kappa_LO  = -H[0] / G[0]
    kappa_1st = -(H[0] + xi31 * H[1]) / (G[0] + xi31 * G[1])

    gamma_formula = _gamma_combination(G, xi31, xi51)

    def bounds_Ee2(Ee1):
        return [ME, Q + 2.0 * ME - Ee1]

    bounds_Ee1 = [ME, Q + ME]
    opts_Ee1 = {"epsabs": 1e-18, "epsrel": 1e-16, "limit": 50000,
                "points": [ME, Q / 2 + ME, Q + ME]}
    opts_Ee2 = {"epsabs": 1e-18, "epsrel": 1e-16, "limit": 50000, "points": [ME]}

    numerator, numerator_err = nquad(
        lambda Ee2, Ee1: angular_correlation_weighted_kernel(
            Ee1, Ee2, Q, Fermi_numeric, E_numeric, nuclear_matrix_elements
        ),
        [bounds_Ee2, bounds_Ee1],
        opts=[opts_Ee2, opts_Ee1]
    )[:2]

    # Direct method: Γ²ν from a single integral of the untagged kernel over
    # the same domain, instead of combining the 4 separately-tagged G integrals.
    gamma_direct, gamma_direct_err = nquad(
        two_electron_spectrum_integrand(None, Q, Fermi_numeric, nuclear_matrix_elements),
        [bounds_Ee2, bounds_Ee1],
        opts=[opts_Ee2, opts_Ee1]
    )[:2]

    kappa_total     = numerator / gamma_formula
    kappa_total_err = abs(numerator_err / gamma_formula)

    kappa_direct     = numerator / gamma_direct
    kappa_direct_err = abs(numerator_err / gamma_direct)

    gamma_percent_diff = 100.0 * (gamma_direct - gamma_formula) / gamma_formula
    kappa_percent_diff = 100.0 * (kappa_direct - kappa_total) / kappa_total

    return {
        "kappa_LO":           kappa_LO,
        "kappa_1st":          kappa_1st,
        "kappa_total":        kappa_total,
        "kappa_total_err":    kappa_total_err,
        "kappa_direct":       kappa_direct,
        "kappa_direct_err":   kappa_direct_err,
        "kappa_percent_diff": kappa_percent_diff,
        "gamma_formula":      gamma_formula,
        "gamma_direct":       gamma_direct,
        "gamma_direct_err":   gamma_direct_err,
        "gamma_percent_diff": gamma_percent_diff,
    }


# ========== DATA FILE OUTPUT (always written, independent of verbose) ==========

def save_fermi_function(Q, Z, A, Fermi_numeric, E_numeric, output_dir, potential_index):
    """
    Evaluate and save the numeric Fermi function F(Ee) and E-function E(Ee)
    on a fine energy grid to an NPZ file.

    Parameters
    ----------
    Q : float
        Q-value (MeV).
    Z : int
        Atomic number.
    A : int
        Mass number.
    Fermi_numeric : callable
        Scalar function F(Ee).
    E_numeric : callable
        Scalar function E(Ee) = 2*g*f.
    output_dir : Path or str
        Directory in which to write the file.
    potential_index : int
        Used in the output filename.
    """
    E_grid = np.linspace(ME + 1e-3, Q + ME, 2000)
    F_vals = np.array([Fermi_numeric(e) for e in E_grid])
    E_vals = np.array([E_numeric(e)     for e in E_grid])
    path = Path(output_dir) / f"fermi_function_potential_{potential_index}_Z{Z}_A{A}.npz"
    np.savez_compressed(path, E=E_grid, F_numeric=F_vals, E_function=E_vals)
    print(f"Fermi function saved  → {path}")


def save_triple_differential_grid(Q, Fermi_numeric, E_numeric, nuclear_matrix_elements,
                                   output_dir, potential_index, Z, A, n_Ee=40, n_cos=20):
    """
    Evaluate dΓ/(dEe1 dEe2 d(cos θ)) on a regular (Ee1, Ee2, cos θ) grid
    and save the result to an NPZ file.  Points outside the kinematic boundary
    (Ee1 + Ee2 > Q + 2*ME) are stored as 0.

    Parameters
    ----------
    Q : float
        Q-value (MeV).
    Fermi_numeric : callable
        Scalar Fermi function F(Ee).
    E_numeric : callable
        Scalar E-function E(Ee) = 2*g*f.
    nuclear_matrix_elements : dict
        Must contain 'xi31' and 'xi51'.
    output_dir : Path or str
        Directory in which to write the file.
    potential_index : int
        Used in the output filename.
    Z, A : int
        Atomic and mass numbers (for filename).
    n_Ee : int
        Number of grid points along each energy axis.
    n_cos : int
        Number of grid points along the cos(θ) axis.
    """
    Ee1_grid = np.linspace(ME + 1e-3, Q + ME, n_Ee)
    Ee2_grid = np.linspace(ME + 1e-3, Q + ME, n_Ee)
    cos_grid = np.linspace(-1.0, 1.0, n_cos)
    integrand = np.zeros((n_Ee, n_Ee, n_cos))

    for i, Ee1 in enumerate(Ee1_grid):
        for j, Ee2 in enumerate(Ee2_grid):
            if Ee1 + Ee2 > Q + 2.0 * ME:
                continue
            for k, ct in enumerate(cos_grid):
                integrand[i, j, k] = double_differential_spectrum_kernel(
                    Ee1, Ee2, ct, Q, Fermi_numeric, E_numeric, nuclear_matrix_elements
                )

    path = Path(output_dir) / f"triple_differential_potential_{potential_index}_Z{Z}_A{A}.npz"
    np.savez_compressed(path, Ee1=Ee1_grid, Ee2=Ee2_grid, cos_theta=cos_grid, integrand=integrand)
    print(f"Triple-differential grid saved  → {path}")


# ========== FERMI FUNCTION PLOTTING ==========

def plot_fermi_function(Q, Z, A, rN, data, plots_output_directory, potential_index, isotope):
    """
    Plot analytical vs numerical Fermi functions.

    Parameters
    ----------
    Q : float
        Q-value of the decay process.
    Z : int
        Atomic number.
    A : int
        Mass number.
    rN : float
        Nuclear radius (dimensionless).
    data : dict
        Wavefunction data dictionary.
    plots_output_directory : str or Path
        Directory to save plots.
    potential_index : int
        Index specifying which potential model is used.
    isotope : str
        Isotope name, used in plot titles.
    """
    show_analytic = (potential_index == 0)
    potential_names = {0: "Pure Coulomb", 2: "finite sized nucleus", 3: "Thomas-Fermi"}
    potential_label = potential_names.get(potential_index)

    x_plot = np.linspace(ME + 1e-3, Q + ME, 2000)
    Numeric_fermi_plot = Fermi_numerical(x_plot, Z, data)
    Numeric_fermi_plot_division = None
    if data.get("P_n_Z0") is not None:
        Numeric_fermi_plot_division = Fermi_numerical_division(x_plot, Z, data)
    Analytic_fermi_plot = np.zeros_like(x_plot)
    if show_analytic:
        Analytic_fermi_plot = Fermi(x_plot, Z, A, rN)

    fortran_dirs = {0: 'out_fortran', 2: 'fortran_output_finite_size', 3: 'fortran_output_thomas_fermi'}
    fortran_dir = fortran_dirs.get(potential_index)
    fortran_dat_path = os.path.join(os.path.dirname(__file__), '..', fortran_dir, 'FermiN.dat') if fortran_dir else None
    fortran_energies, fortran_fermi = None, None
    if fortran_dat_path is not None and os.path.isfile(fortran_dat_path):
        raw = np.loadtxt(fortran_dat_path, comments='#')
        # FermiN.dat stores kinetic energies; convert to total energy
        fortran_energies = raw[:, 0] + ME
        fortran_fermi = raw[:, 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # Top plot: comparison
    if show_analytic:
        ax1.plot(x_plot, Analytic_fermi_plot, lw=1.5, label='Pure Coulomb Analytic')
    ax1.plot(x_plot, Numeric_fermi_plot, lw=1.5, label='Numerical')
    if Numeric_fermi_plot_division is not None:
        ax1.plot(x_plot, Numeric_fermi_plot_division, lw=1.5, ls='--', label='Numerical (Division)')
    if fortran_energies is not None:
        ax1.plot(fortran_energies, fortran_fermi, lw=1.5, ls='--', label='FORTRAN (FermiN.dat)')
    ax1.set_ylabel('Fermi Function Value')
    title = f'{isotope}: Fermi Function vs Electron Energy'
    if potential_label:
        title += f' ({potential_label})'
    ax1.set_title(title)
    ax1.grid(True)
    ax1.legend()

    # Bottom plot: percentage differences
    if show_analytic:
        percent_diff_fermi = 100.0 * (Numeric_fermi_plot - Analytic_fermi_plot) / Analytic_fermi_plot
        ax2.plot(x_plot, percent_diff_fermi, lw=1.5, label='Numerical vs Analytic')
    if fortran_energies is not None:
        fortran_numeric_interp = np.interp(fortran_energies, x_plot, Numeric_fermi_plot)
        percent_diff_fortran_numeric = 100.0 * (fortran_fermi - fortran_numeric_interp) / fortran_numeric_interp
        if show_analytic:
            fortran_analytic_interp = np.interp(fortran_energies, x_plot, Analytic_fermi_plot)
            percent_diff_fortran_analytic = 100.0 * (fortran_fermi - fortran_analytic_interp) / fortran_analytic_interp
            ax2.plot(fortran_energies, percent_diff_fortran_analytic, lw=1.5, ls='--', label='FORTRAN vs Analytic')
        ax2.plot(fortran_energies, percent_diff_fortran_numeric, lw=1.5, ls=':', label='FORTRAN vs Numeric')
        if Numeric_fermi_plot_division is not None:
            fortan_numeric_interp_division = np.interp(fortran_energies, x_plot, Numeric_fermi_plot_division)
            percent_diff_fortran_numeric_division = 100.0 * (fortran_fermi - fortan_numeric_interp_division) / fortan_numeric_interp_division
            ax2.plot(fortran_energies, percent_diff_fortran_numeric_division, lw=1.5, ls='-.', label='FORTRAN vs Numeric (Division)')
        ax2.legend()
    ax2.set_xlabel('Electron Energy (MeV)')
    ax2.set_ylabel('% Difference')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory, f"Fermi_Function_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300, bbox_inches='tight')
    plt.show()




# ========== SPECTRUM CALCULATION AND PLOTTING ==========

def calculate_and_plot_spectra(Q, Z, A, Fermi_analytic, Fermi_numeric, plots_output_directory, potential_index, nuclear_matrix_elements, isotope):
    """
    Calculate and plot 2νββ epsilon spectrum.

    Parameters
    ----------
    Q : float
        Q-value of the decay process.
    Z : int
        Atomic number.
    A : int
        Mass number.
    Fermi_analytic : callable
        Analytic Fermi function.
    Fermi_numeric : callable
        Numerical Fermi function.
    plots_output_directory : str or Path
        Directory to save plots.
    potential_index : int
        Index specifying which potential model is used.
    isotope : str
        Isotope name, used in plot titles.

    Returns
    -------
    dict
        Dictionary containing spectrum data and rates
    """
    show_analytic = (potential_index == 0)
    potential_names = {0: "Pure Coulomb", 2: "Finite-Sized Nucleus", 3: "Thomas-Fermi"}
    potential_label = potential_names.get(potential_index, f"Potential {potential_index}")

    eps_grid = np.linspace(0.0, Q, 400)
    spectrum_vals_num = np.array([spectrum_epsilon_integrand(eps, Q, Fermi_numeric, nuclear_matrix_elements) for eps in eps_grid])
    spectrum_vals = np.zeros_like(spectrum_vals_num)
    if show_analytic:
        spectrum_vals = np.array([spectrum_epsilon_integrand(eps, Q, Fermi_analytic, nuclear_matrix_elements) for eps in eps_grid])

    # ========== Plot differential spectrum ==========
    if show_analytic:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(eps_grid, spectrum_vals, lw=1.5, label="Pure Coulomb Analytic")
        ax1.plot(eps_grid, spectrum_vals_num, lw=1.5, label="Numerical")
        ax1.set_ylabel('dΓ/dε (1/yr per MeV)')
        ax1.set_title(f'{isotope}: 2νββ Spectrum vs epsilon ({potential_label})')
        ax1.legend()
        ax1.grid(True)
        percent_diff_eps = 100.0 * (spectrum_vals_num - spectrum_vals) / spectrum_vals
        ax2.plot(eps_grid, percent_diff_eps, lw=1.5)
        ax2.set_xlabel('epsilon = Ee1 + Ee2 − 2 me (MeV)')
        ax2.set_ylabel('% Difference')
        ax2.grid(True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
        ax1.plot(eps_grid, spectrum_vals_num, lw=1.5, label="Numerical")
        ax1.set_xlabel('epsilon = Ee1 + Ee2 − 2 me (MeV)')
        ax1.set_ylabel('dΓ/dε (1/yr per MeV)')
        ax1.set_title(f'{isotope}: 2νββ Spectrum vs epsilon ({potential_label})')
        ax1.legend()
        ax1.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory, f"Spectrum_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # ========== Calculate total rates ==========
    total_rate_num, total_err_num = quad(lambda eps: spectrum_epsilon_integrand(eps, Q, Fermi_numeric, nuclear_matrix_elements), 0.0, Q,
                                         epsabs=1e-18, epsrel=1e-16, limit=20000, points=[0.0, Q/2.0, Q])
    if show_analytic:
        total_rate, total_err = quad(lambda eps: spectrum_epsilon_integrand(eps, Q, Fermi_analytic, nuclear_matrix_elements), 0.0, Q,
                                     epsabs=1e-18, epsrel=1e-16, limit=20000, points=[0.0, Q/2.0, Q])
    else:
        total_rate, total_err = None, None

    # ========== Plot normalized spectrum ==========
    spec_vals_norm_num = spectrum_vals_num / total_rate_num

    plt.figure(figsize=(12, 8))
    if show_analytic:
        spec_vals_norm = spectrum_vals / total_rate
        plt.plot(eps_grid, spec_vals_norm, lw=1.5, label="Analytic")
    plt.plot(eps_grid, spec_vals_norm_num, lw=1.5, label="Numeric")
    plt.xlabel('epsilon = Ee1 + Ee2 − 2 me (MeV)')
    plt.ylabel('1/Γ dΓ/dε')
    plt.title(f"{isotope}: Normalized 2νββ Spectrum ({potential_label})")
    plt.legend()
    plt.grid(True)
    plt.show()

    norm_check_num = np.trapezoid(spec_vals_norm_num, eps_grid)
    if show_analytic:
        norm_check = np.trapezoid(spec_vals_norm, eps_grid)
        print(f"Normalization check - Analytic: {norm_check:.10g}, Numeric: {norm_check_num:.10g}")
    else:
        print(f"Normalization check - Numeric: {norm_check_num:.10g}")

    return {
        "eps_grid": eps_grid,
        "spectrum_vals": spectrum_vals if show_analytic else None,
        "spectrum_vals_num": spectrum_vals_num,
        "total_rate": total_rate,
        "total_rate_num": total_rate_num,
        "total_err": total_err,
        "total_err_num": total_err_num,
    }

# ========== dGamma/dEe SINGLE-ELECTRON SPECTRUM CALCULATION AND PLOTTING ==========

def calculate_and_plot_single_electron_spectra(Q, Z, A, Fermi_analytic, Fermi_numeric,
                                               plots_output_directory, potential_index,
                                               nuclear_matrix_elements, isotope):
    """
    Calculate and plot single-electron energy spectra dΓ/dEe1 and dΓ/dEe2.

    Each marginal is obtained by integrating two_electron_kernel over the
    opposite electron energy (cos(θ) drops out upon integration over [-1,1]).
    Both spectra are identical by kernel symmetry — plotting them overlaid
    serves as a sanity check.

    Parameters
    ----------
    Q : float
        Q-value of the decay process (MeV).
    Z : int
        Atomic number.
    A : int
        Mass number.
    Fermi_analytic : callable
        Analytic Fermi function (pure Coulomb).
    Fermi_numeric : callable
        Numerical Fermi function from Dirac solutions.
    plots_output_directory : str or Path
        Directory to save plots.
    potential_index : int
        Index specifying which potential model is used.
    nuclear_matrix_elements : dict
        Nuclear matrix elements for the selected isotope.
    isotope : str
        Isotope name, used in plot titles.
    """
    from configurations.physics_constants import ME as _ME
    show_analytic = (potential_index == 0)
    potential_names = {0: "Pure Coulomb", 2: "Finite-Sized Nucleus", 3: "Thomas-Fermi"}
    potential_label = potential_names.get(potential_index, f"Potential {potential_index}")

    T_grid = np.linspace(1e-3, Q, 200)
    Ee_grid = T_grid + _ME

    spec_Ee1_num = np.array([single_electron_spectrum_integrand(Ee, Q, Fermi_numeric, nuclear_matrix_elements)
                              for Ee in Ee_grid])
    spec_Ee2_num = np.array([single_electron_spectrum_integrand(Ee, Q, Fermi_numeric, nuclear_matrix_elements)
                              for Ee in Ee_grid])

    spec_Ee1_ana = np.zeros_like(spec_Ee1_num)
    if show_analytic:
        spec_Ee1_ana = np.array([single_electron_spectrum_integrand(Ee, Q, Fermi_analytic, nuclear_matrix_elements)
                                  for Ee in Ee_grid])

    # ========== Figure 1: dΓ/dEe ==========
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    if show_analytic:
        ax1.plot(T_grid, spec_Ee1_ana, lw=1.5, label="Pure Coulomb Analytic")
    ax1.plot(T_grid, spec_Ee1_num, lw=1.5, label=r"Numeric $d\Gamma/dE_{e_1}$", ls='-')
    ax1.plot(T_grid, spec_Ee2_num, lw=1.5, label=r"Numeric $d\Gamma/dE_{e_2}$", ls='--')
    ax1.set_xlabel(r"Kinetic energy $T = E_e - m_e$ (MeV)")
    ax1.set_ylabel(r"$d\Gamma/dE_e$ (1/yr per MeV)")
    ax1.set_title(rf"{isotope}: Single-Electron Spectrum $d\Gamma/dE_e$ ({potential_label})")
    ax1.legend()
    ax1.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory,
                             f"Single_Electron_Spectrum_potential_{potential_index}_Z{Z}_A{A}.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    # ========== Total rate and normalisation ==========
    total_rate_num, _ = quad(
        lambda Ee: single_electron_spectrum_integrand(Ee, Q, Fermi_numeric, nuclear_matrix_elements),
        _ME, Q + _ME,
        epsabs=1e-18, epsrel=1e-16, limit=20000, points=[_ME, _ME + Q / 2.0, _ME + Q],
    )
    if show_analytic:
        total_rate_ana, _ = quad(
            lambda Ee: single_electron_spectrum_integrand(Ee, Q, Fermi_analytic, nuclear_matrix_elements),
            _ME, Q + _ME,
            epsabs=1e-18, epsrel=1e-16, limit=20000, points=[_ME, _ME + Q / 2.0, _ME + Q],
        )
    else:
        total_rate_ana = None

    # ========== Figure 2: normalized spectrum ==========
    norm_num = spec_Ee1_num / total_rate_num
    norm_ana = spec_Ee1_ana / total_rate_ana if show_analytic and total_rate_ana else None

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    # if norm_ana is not None:
    #     ax1.plot(T_grid, norm_ana, lw=1.5, label="Analytic")
    ax1.plot(T_grid, norm_num, lw=1.5, label="Numeric")
    ax1.set_xlabel(r"Kinetic energy $T = E_e - m_e$ (MeV)")
    ax1.set_ylabel(r"$1/\Gamma\;d\Gamma/dE_e$")
    ax1.set_title(f"{isotope}: Normalized Single-Electron Spectrum ({potential_label})")
    ax1.legend()
    ax1.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory,
                             f"Single_Electron_Spectrum_normalized_potential_{potential_index}_Z{Z}_A{A}.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    norm_check_num = np.trapezoid(norm_num, T_grid)
    if norm_ana is not None:
        norm_check_ana = np.trapezoid(norm_ana, T_grid)
        print(f"Normalization check - Analytic: {norm_check_ana:.10g}, Numeric: {norm_check_num:.10g}")
    else:
        print(f"Normalization check - Numeric: {norm_check_num:.10g}")


# ========== dGamma/dEe1dEe2dcos(θ) SPECTRUM CALCULATION AND PLOTTING ==========

def calculate_and_plot_double_differential_spectrum(Q, Z, A, Fermi_analytic, Fermi_numeric, E_function_numeric, plots_output_directory, potential_index, nuclear_matrix_elements, isotope, phase_space_data=None):
    """
    Calculate and plot the double-differential spectrum dΓ/dEe1dEe2 for 2νββ decay.
    
    This is the workflow:
    
    1. **Compute triple-differential spectrum**: dΓ/(dEe1 dEe2 d(cos θ))
       - 3D function: (Ee1, Ee2, cos θ) → spectrum value
       - Cannot be easily plotted as 3D, but we can evaluate it at points
       - Stored for angular dependence analysis
    
    2. **Integrate over cos(θ)**: ∫_{-1}^{+1} dΓ/(dEe1 dEe2 d(cos θ)) d(cos θ)
       - Results in: dΓ/(dEe1 dEe2)
       - 2D function: (Ee1, Ee2) → spectrum value
       - **This is what we plot as 2D contours**
    
    3. **Show angular dependence**:
       - Plot triple-differential spectrum at fixed energy pairs
       - Shows how spectrum varies with angle for specific electron energies

    Parameters
    ----------
    Q : float
        Q-value of the decay process (MeV).
    Z : int
        Atomic number.
    A : int
        Mass number.
    Fermi_analytic : callable
        Analytic Fermi function (pure Coulomb) - only depends on Ee.
    Fermi_numeric : callable
        Numerical Fermi function from Dirac solutions - only depends on Ee.
    E_function_numeric : callable
        E function (E(Ee) = 2*g*f) from Dirac solutions - only depends on Ee.
    plots_output_directory : str or Path
        Directory to save plots.
    potential_index : int
        Index specifying which potential model is used.
    nuclear_matrix_elements : dict
        Dictionary containing nuclear matrix elements needed for spectrum calculation.
    isotope : str
        Isotope name, used in plot titles.

    Returns
    -------
    dict
        Dictionary containing:
        - 'Ee1_grid': Energy grid for first electron
        - 'Ee2_grid': Energy grid for second electron
        - 'cos_theta_grid': Angular grid
        - 'spectrum_double_diff_analytic': dΓ/(dEe1 dEe2) integrated over cos(θ) - PRIMARY
        - 'spectrum_double_diff_numeric': dΓ/(dEe1 dEe2) integrated over cos(θ) - PRIMARY
        - 'spectrum_triple_diff_analytic': dΓ/(dEe1 dEe2 d(cos θ)) at grid points - Reference
        - 'spectrum_triple_diff_numeric': dΓ/(dEe1 dEe2 d(cos θ)) at grid points - Reference
        - 'spectrum_dEe2_dcos_theta_analytic': dΓ/(dEe2 d(cos θ)) integrated over Ee1
        - 'spectrum_dEe2_dcos_theta_numeric': dΓ/(dEe2 d(cos θ)) integrated over Ee1
        - 'spectrum_dEe1_dcos_theta_analytic': dΓ/(dEe1 d(cos θ)) integrated over Ee2
        - 'spectrum_dEe1_dcos_theta_numeric': dΓ/(dEe1 d(cos θ)) integrated over Ee2
    """
    
    show_analytic = (potential_index == 0)
    potential_names = {0: "Pure Coulomb", 2: "Finite-Sized Nucleus", 3: "Thomas-Fermi"}
    potential_label = potential_names.get(potential_index, f"Potential {potential_index}")

    # ========== DEFINE EVALUATION GRIDS ==========
    Ee1_grid = np.linspace(ME + 1e-3, Q + ME, 40)
    Ee2_grid = np.linspace(ME + 1e-3, Q + ME, 40)
    cos_theta_grid = np.linspace(-1.0, 1.0, 20)

    print(f"Calculating triple-differential spectrum using integration:")
    print(f"  Ee1 range: [{Ee1_grid[0]:.3f}, {Ee1_grid[-1]:.3f}] MeV with {len(Ee1_grid)} evaluation points")
    print(f"  Ee2 range: [{Ee2_grid[0]:.3f}, {Ee2_grid[-1]:.3f}] MeV with {len(Ee2_grid)} evaluation points")
    print(f"  cos(θ) range: [{cos_theta_grid[0]:.2f}, {cos_theta_grid[-1]:.2f}] with {len(cos_theta_grid)} evaluation points")
    print(f"  Note: INTEGRATING over cos(θ) to obtain dΓ/(dEe1 dEe2) at each energy pair")

    # ========== COMPUTE TRIPLE-DIFFERENTIAL SPECTRUM FOR REFERENCE ==========
    spectrum_triple_diff_numeric = np.zeros((len(Ee1_grid), len(Ee2_grid), len(cos_theta_grid)))
    spectrum_triple_diff_analytic = np.zeros_like(spectrum_triple_diff_numeric)

    for i, Ee1 in enumerate(Ee1_grid):
        for j, Ee2 in enumerate(Ee2_grid):
            if Ee1 + Ee2 > Q + 2*ME:
                continue
            for k, cos_theta in enumerate(cos_theta_grid):
                if show_analytic:
                    spec_val_analytic = double_differential_spectrum_kernel(
                        Ee1, Ee2, cos_theta, Q,
                        fermi_func=Fermi_analytic,
                        E_func=lambda Ee: 0.0,
                        nuclear_matrix_elements=nuclear_matrix_elements
                    )
                    spectrum_triple_diff_analytic[i, j, k] = max(spec_val_analytic, 0.0)

                spec_val_numeric = double_differential_spectrum_kernel(
                    Ee1, Ee2, cos_theta, Q,
                    fermi_func=Fermi_numeric,
                    E_func=E_function_numeric,
                    nuclear_matrix_elements=nuclear_matrix_elements
                )
                spectrum_triple_diff_numeric[i, j, k] = max(spec_val_numeric, 0.0)

    # ========== COMPUTE DOUBLE-DIFFERENTIAL SPECTRUM VIA INTEGRATION ==========
    spectrum_double_diff_numeric = np.zeros((len(Ee1_grid), len(Ee2_grid)))
    spectrum_double_diff_analytic = np.zeros_like(spectrum_double_diff_numeric)

    epsabs_angle = 1e-14
    epsrel_angle = 1e-10
    limit_angle = 5000

    for i, Ee1 in enumerate(Ee1_grid):
        if i % 5 == 0:
            print(f"  Processing Ee1 point {i+1}/{len(Ee1_grid)}")

        for j, Ee2 in enumerate(Ee2_grid):
            if Ee1 + Ee2 > Q + 2*ME:
                continue

            if show_analytic:
                integrand_analytic = lambda cos_theta: double_differential_spectrum_kernel(
                    Ee1, Ee2, cos_theta, Q,
                    fermi_func=Fermi_analytic,
                    E_func=lambda Ee: 0.0,
                    nuclear_matrix_elements=nuclear_matrix_elements
                )
                integral_analytic, _ = quad(integrand_analytic, -1.0, 1.0,
                                            epsabs=epsabs_angle, epsrel=epsrel_angle,
                                            limit=limit_angle, points=[0.0])
                spectrum_double_diff_analytic[i, j] = max(integral_analytic, 0.0)

            integrand_numeric = lambda cos_theta: double_differential_spectrum_kernel(
                Ee1, Ee2, cos_theta, Q,
                fermi_func=Fermi_numeric,
                E_func=E_function_numeric,
                nuclear_matrix_elements=nuclear_matrix_elements
            )
            integral_numeric, _ = quad(integrand_numeric, -1.0, 1.0,
                                       epsabs=epsabs_angle, epsrel=epsrel_angle,
                                       limit=limit_angle, points=[-1.0, 0.0, 1.0])
            spectrum_double_diff_numeric[i, j] = max(integral_numeric, 0.0)

    # ========== COMPUTE DIRECT SPECTRUM USING two_electron_kernel ==========
    print("Computing direct two-electron kernel spectrum for verification...")
    spectrum_direct_numeric = np.zeros((len(Ee1_grid), len(Ee2_grid)))

    for i, Ee1 in enumerate(Ee1_grid):
        if i % 5 == 0:
            print(f"  Direct computation - Ee1 point {i+1}/{len(Ee1_grid)}")
        for j, Ee2 in enumerate(Ee2_grid):
            if Ee1 + Ee2 > Q + 2*ME:
                continue
            spectrum_direct_numeric[i, j] = two_electron_kernel(
                Ee1, Ee2, Q, tag=None,
                fermi_func=Fermi_numeric,
                nuclear_matrix_elements=nuclear_matrix_elements
            )

    # ========== PLOT 1: 2D CONTOURS - DOUBLE DIFFERENTIAL SPECTRUM ==========
    print("\nGenerating double-differential spectrum contour plots...")

    if show_analytic:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(rf'{isotope}: $d\Gamma/(dE_{{e_1}}\,dE_{{e_2}})$ - {potential_label}', fontsize=14)

        ax_a = axes[0]
        contour_a = ax_a.contourf(Ee2_grid, Ee1_grid, spectrum_double_diff_analytic, levels=15, cmap='viridis')
        ax_a.set_xlabel(r'$E_{e_2}$ (MeV)'); ax_a.set_ylabel(r'$E_{e_1}$ (MeV)')
        ax_a.set_title('Analytic Fermi\n(Integrated over cos(θ))')
        plt.colorbar(contour_a, ax=ax_a).set_label(r'$d\Gamma/(dE_{e_1}\,dE_{e_2})$')

        ax_n = axes[1]
        contour_n = ax_n.contourf(Ee2_grid, Ee1_grid, spectrum_double_diff_numeric, levels=15, cmap='viridis')
        ax_n.set_xlabel(r'$E_{e_2}$ (MeV)'); ax_n.set_ylabel(r'$E_{e_1}$ (MeV)')
        ax_n.set_title('Numeric Fermi\n(Integrated over cos(θ))')
        plt.colorbar(contour_n, ax=ax_n).set_label(r'$d\Gamma/(dE_{e_1}\,dE_{e_2})$')

        ax_d = axes[2]
        contour_d = ax_d.contourf(Ee2_grid, Ee1_grid, spectrum_direct_numeric, levels=15, cmap='viridis')
        ax_d.set_xlabel(r'$E_{e_2}$ (MeV)'); ax_d.set_ylabel(r'$E_{e_1}$ (MeV)')
        ax_d.set_title('Numeric Fermi\n(Direct two_electron_kernel)')
        plt.colorbar(contour_d, ax=ax_d).set_label(r'$d\Gamma/(dE_{e_1}\,dE_{e_2})$')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(rf'{isotope}: $d\Gamma/(dE_{{e_1}}\,dE_{{e_2}})$ - {potential_label}', fontsize=14)

        ax_n = axes[0]
        contour_n = ax_n.contourf(Ee2_grid, Ee1_grid, spectrum_double_diff_numeric, levels=15, cmap='viridis')
        ax_n.set_xlabel(r'$E_{e_2}$ (MeV)'); ax_n.set_ylabel(r'$E_{e_1}$ (MeV)')
        ax_n.set_title('Numeric Fermi\n(Integrated over cos(θ))')
        plt.colorbar(contour_n, ax=ax_n).set_label(r'$d\Gamma/(dE_{e_1}\,dE_{e_2})$')

        ax_d = axes[1]
        contour_d = ax_d.contourf(Ee2_grid, Ee1_grid, spectrum_direct_numeric, levels=15, cmap='viridis')
        ax_d.set_xlabel(r'$E_{e_2}$ (MeV)'); ax_d.set_ylabel(r'$E_{e_1}$ (MeV)')
        ax_d.set_title('Numeric Fermi\n(Direct two_electron_kernel)')
        plt.colorbar(contour_d, ax=ax_d).set_label(r'$d\Gamma/(dE_{e_1}\,dE_{e_2})$')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory, f"DoublesDiff_Spectrum_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # ========== PLOT 1b: NORMALIZED BY Γ^{2ν} ==========
    if phase_space_data is not None:
        xi31 = nuclear_matrix_elements["xi31"]
        xi51 = nuclear_matrix_elements["xi51"]
        G_num = phase_space_data["G_results_num"]
        gamma_2nu_numeric = _gamma_combination(G_num, xi31, xi51)

        spec_norm_numeric = spectrum_double_diff_numeric / gamma_2nu_numeric
        spec_norm_direct  = spectrum_direct_numeric      / gamma_2nu_numeric

        if show_analytic:
            G = phase_space_data["G_results"]
            gamma_2nu_analytic = _gamma_combination(G, xi31, xi51)
            spec_norm_analytic = spectrum_double_diff_analytic / gamma_2nu_analytic

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(rf'{isotope}: Normalized: $d\Gamma/(dE_{{e_1}}\,dE_{{e_2}})\,/\,\Gamma^{{2\nu}}$ — {potential_label}', fontsize=14)

            ax_a = axes[0]
            contour_a = ax_a.contourf(Ee2_grid, Ee1_grid, spec_norm_analytic, levels=15, cmap='viridis')
            ax_a.set_xlabel(r'$E_{e_2}$ (MeV)'); ax_a.set_ylabel(r'$E_{e_1}$ (MeV)')
            ax_a.set_title('Analytic Fermi\n(Integrated over cos(θ))')
            plt.colorbar(contour_a, ax=ax_a).set_label(r'$d\Gamma/(dE_{e_1}\,dE_{e_2})\,/\,\Gamma^{2\nu}$')

            ax_n = axes[1]
            ax_d = axes[2]
        else:
            fig, (ax_n, ax_d) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(rf'{isotope}: Normalized: $d\Gamma/(dE_{{e_1}}\,dE_{{e_2}})\,/\,\Gamma^{{2\nu}}$ — {potential_label}', fontsize=14)

        contour_n = ax_n.contourf(Ee2_grid, Ee1_grid, spec_norm_numeric, levels=15, cmap='viridis')
        ax_n.set_xlabel(r'$E_{e_2}$ (MeV)'); ax_n.set_ylabel(r'$E_{e_1}$ (MeV)')
        ax_n.set_title('Numeric Fermi\n(Integrated over cos(θ))')
        plt.colorbar(contour_n, ax=ax_n).set_label(r'$d\Gamma/(dE_{e_1}\,dE_{e_2})\,/\,\Gamma^{2\nu}$')

        contour_d = ax_d.contourf(Ee2_grid, Ee1_grid, spec_norm_direct, levels=15, cmap='viridis')
        ax_d.set_xlabel(r'$E_{e_2}$ (MeV)'); ax_d.set_ylabel(r'$E_{e_1}$ (MeV)')
        ax_d.set_title('Numeric Fermi\n(Direct two_electron_kernel)')
        plt.colorbar(contour_d, ax=ax_d).set_label(r'$d\Gamma/(dE_{e_1}\,dE_{e_2})\,/\,\Gamma^{2\nu}$')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_output_directory, f"DoublesDiff_Spectrum_Normalized_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300, bbox_inches='tight')
        plt.show()
        # ========== PLOT 1c: NORMALIZED SPECTRUM (NUMERIC, INTEGRATED ONLY) WITH KINEMATIC LIMIT & ANGULAR CORRELATION ==========
        fig, ax = plt.subplots(figsize=(9, 7))
        contour = ax.contourf(Ee2_grid, Ee1_grid, spec_norm_numeric, levels=15, cmap='viridis')
        plt.colorbar(contour, ax=ax).set_label(r'$d\Gamma/(dE_{e_1}\,dE_{e_2})\,/\,\Gamma^{2\nu}$')

        # Kinematic limit Ee1 + Ee2 = Q + 2*ME, connecting (Ee2=ME, Ee1=Q+ME) to (Ee2=Q+ME, Ee1=ME)
        kinematic_Ee2 = np.array([ME, Q + ME])
        kinematic_Ee1 = Q + 2 * ME - kinematic_Ee2
        ax.plot(kinematic_Ee2, kinematic_Ee1, 'r--', linewidth=2, label=r'Kinematic limit ($E_{e_1}+E_{e_2}=Q+2m_e$)')

        # Angular correlation coefficient kappa^2nu overlaid as labeled contour lines
        kappa_grid = np.full((len(Ee1_grid), len(Ee2_grid)), np.nan)
        for i, Ee1 in enumerate(Ee1_grid):
            for j, Ee2 in enumerate(Ee2_grid):
                if Ee1 + Ee2 > Q + 2 * ME:
                    continue
                kappa_grid[i, j] = angular_correlation_distribution(
                    Ee1, Ee2, Q,
                    fermi_func=Fermi_numeric,
                    E_func=E_function_numeric,
                    nuclear_matrix_elements=nuclear_matrix_elements
                )

        kappa_contour = ax.contour(Ee2_grid, Ee1_grid, kappa_grid, levels=6, colors='white', linewidths=1.0)
        ax.clabel(kappa_contour, inline=True, fontsize=8, fmt=r'$\kappa^{2\nu}=%.2f$')

        ax.set_xlabel(r'$E_{e_2}$ (MeV)'); ax.set_ylabel(r'$E_{e_1}$ (MeV)')
        ax.set_title(rf'{isotope}: Normalized $d\Gamma/(dE_{{e_1}}\,dE_{{e_2}})\,/\,\Gamma^{{2\nu}}$ (Numeric, Integrated) - {potential_label}')
        ax.set_xlim(Ee2_grid[0], Ee2_grid[-1])
        ax.set_ylim(Ee1_grid[0], Ee1_grid[-1])
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_output_directory, f"DoublesDiff_Spectrum_Normalized_Numeric_Kinematic_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("Skipping normalized plot: phase_space_data not provided.")

    # ========== VERIFICATION: Compare integrated vs direct methods ==========
    print("\nVerification: Comparing integration and direct methods...")
    with np.errstate(divide='ignore', invalid='ignore'):
        percent_diff_methods = 100.0 * (spectrum_double_diff_numeric - spectrum_direct_numeric) / (np.abs(spectrum_direct_numeric) + 1e-30)
        percent_diff_methods[~np.isfinite(percent_diff_methods)] = 0.0

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = np.percentile(np.abs(percent_diff_methods[np.isfinite(percent_diff_methods)]), 95) if np.any(np.isfinite(percent_diff_methods)) else 1.0
    contour_diff = ax.contourf(Ee2_grid, Ee1_grid, percent_diff_methods, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xlabel(r'$E_{e_2}$ (MeV)'); ax.set_ylabel(r'$E_{e_1}$ (MeV)')
    ax.set_title(f'{isotope}: Relative Difference (%) between Integrated and Direct Methods - {potential_label}')
    plt.colorbar(contour_diff, ax=ax).set_label('% Difference')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory, f"DoublesDiff_Integration_vs_Direct_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300, bbox_inches='tight')
    plt.show()

    diff_valid = percent_diff_methods[np.isfinite(percent_diff_methods)]
    if len(diff_valid) > 0:
        print(f"  Mean difference: {np.mean(np.abs(diff_valid)):.4f}%")
        print(f"  Max difference: {np.max(np.abs(diff_valid)):.4f}%")
        print(f"  Median difference: {np.median(np.abs(diff_valid)):.4f}%")

    # ========== PLOT 2: ANGULAR DEPENDENCE AT FIXED ENERGIES ==========
    print("Generating angular dependence plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(rf'{isotope}: Angular Integrand Shape: $d\Gamma/(dE_{{e_1}}\,dE_{{e_2}}\,d(\cos\theta))$ at Different Energy Pairs - {potential_label}', fontsize=14)

    i_max_symm = int(np.searchsorted(2 * Ee1_grid, Q + 2 * ME)) - 1
    i_low  = max(2, i_max_symm // 4)
    i_high = max(i_low + 2, i_max_symm - 1)
    i_mid  = (i_low + i_high) // 2
    j_asym = int(np.searchsorted(Ee2_grid, Q + 2 * ME - Ee1_grid[i_low])) - 1
    j_asym = min(j_asym, len(Ee2_grid) - 1)
    energy_indices = [(i_low, i_low), (i_high, i_high), (i_low, j_asym), (i_mid, i_mid)]
    energy_labels  = [f'$E_{{e_1}}$={Ee1_grid[i]:.2f}, $E_{{e_2}}$={Ee2_grid[j]:.2f} MeV' for i, j in energy_indices]

    for plot_idx, ((idx_ee1, idx_ee2), energy_label) in enumerate(zip(energy_indices, energy_labels)):
        ax = axes[plot_idx // 2, plot_idx % 2]
        if show_analytic:
            data_analytic_1d = spectrum_triple_diff_analytic[idx_ee1, idx_ee2, :]
            ax.plot(cos_theta_grid, data_analytic_1d, 'o-', linewidth=2, markersize=6, label='Analytic (Coulomb)')
        data_numeric_1d = spectrum_triple_diff_numeric[idx_ee1, idx_ee2, :]
        ax.plot(cos_theta_grid, data_numeric_1d, 's-', linewidth=2, markersize=6, label='Numeric (Dirac)')
        ax.set_xlabel(r'$\cos\theta$')
        ax.set_ylabel(r'$d\Gamma/(dE_{e_1}\,dE_{e_2}\,d(\cos\theta))$')
        ax.set_title(energy_label)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory, f"TripleDiff_Angular_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # ========== INTEGRATE OVER Ee1 → dΓ/(dEe2 d(cos θ)) ==========
    print("\nComputing dΓ/(dEe2 d(cos θ)) by integrating over Ee1...")

    spectrum_dEe2_dcos_theta_numeric = np.zeros((len(Ee2_grid), len(cos_theta_grid)))
    spectrum_dEe2_dcos_theta_analytic = np.zeros_like(spectrum_dEe2_dcos_theta_numeric)

    for j, Ee2 in enumerate(Ee2_grid):
        if j % 5 == 0:
            print(f"  Processing Ee2 point {j+1}/{len(Ee2_grid)}")
        for k, cos_theta in enumerate(cos_theta_grid):
            Ee1_min = ME
            Ee1_max = Q + 2*ME - Ee2
            if Ee1_max <= Ee1_min:
                continue

            if show_analytic:
                integrand_analytic = lambda Ee1: double_differential_spectrum_kernel(
                    Ee1, Ee2, cos_theta, Q,
                    fermi_func=Fermi_analytic,
                    E_func=lambda Ee: 0.0,
                    nuclear_matrix_elements=nuclear_matrix_elements
                )
                integral_analytic, _ = quad(integrand_analytic, Ee1_min, Ee1_max,
                                            epsabs=epsabs_angle, epsrel=epsrel_angle,
                                            limit=limit_angle, points=[Ee1_min, (Ee1_min + Ee1_max)/2, Ee1_max])
                spectrum_dEe2_dcos_theta_analytic[j, k] = max(integral_analytic, 0.0)

            integrand_numeric = lambda Ee1: double_differential_spectrum_kernel(
                Ee1, Ee2, cos_theta, Q,
                fermi_func=Fermi_numeric,
                E_func=E_function_numeric,
                nuclear_matrix_elements=nuclear_matrix_elements
            )
            integral_numeric, _ = quad(integrand_numeric, Ee1_min, Ee1_max,
                                       epsabs=epsabs_angle, epsrel=epsrel_angle,
                                       limit=limit_angle, points=[Ee1_min, (Ee1_min + Ee1_max)/2, Ee1_max])
            spectrum_dEe2_dcos_theta_numeric[j, k] = max(integral_numeric, 0.0)

    # ========== INTEGRATE OVER Ee2 → dΓ/(dEe1 d(cos θ)) ==========
    print("Computing dΓ/(dEe1 d(cos θ)) by integrating over Ee2...")

    spectrum_dEe1_dcos_theta_numeric = np.zeros((len(Ee1_grid), len(cos_theta_grid)))
    spectrum_dEe1_dcos_theta_analytic = np.zeros_like(spectrum_dEe1_dcos_theta_numeric)

    for i, Ee1 in enumerate(Ee1_grid):
        if i % 5 == 0:
            print(f"  Processing Ee1 point {i+1}/{len(Ee1_grid)}")
        for k, cos_theta in enumerate(cos_theta_grid):
            Ee2_min = ME
            Ee2_max = Q + 2*ME - Ee1
            if Ee2_max <= Ee2_min:
                continue

            if show_analytic:
                integrand_analytic = lambda Ee2: double_differential_spectrum_kernel(
                    Ee1, Ee2, cos_theta, Q,
                    fermi_func=Fermi_analytic,
                    E_func=lambda Ee: 0.0,
                    nuclear_matrix_elements=nuclear_matrix_elements
                )
                integral_analytic, _ = quad(integrand_analytic, Ee2_min, Ee2_max,
                                            epsabs=epsabs_angle, epsrel=epsrel_angle,
                                            limit=limit_angle, points=[Ee2_min, (Ee2_min + Ee2_max)/2, Ee2_max])
                spectrum_dEe1_dcos_theta_analytic[i, k] = max(integral_analytic, 0.0)

            integrand_numeric = lambda Ee2: double_differential_spectrum_kernel(
                Ee1, Ee2, cos_theta, Q,
                fermi_func=Fermi_numeric,
                E_func=E_function_numeric,
                nuclear_matrix_elements=nuclear_matrix_elements
            )
            integral_numeric, _ = quad(integrand_numeric, Ee2_min, Ee2_max,
                                       epsabs=epsabs_angle, epsrel=epsrel_angle,
                                       limit=limit_angle, points=[Ee2_min, (Ee2_min + Ee2_max)/2, Ee2_max])
            spectrum_dEe1_dcos_theta_numeric[i, k] = max(integral_numeric, 0.0)

    # ========== PLOT 3: dΓ/(dEe2 d(cos θ)) ==========
    print("Generating dΓ/(dEe2 d(cos θ)) contour plots...")

    if show_analytic:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(rf'{isotope}: $d\Gamma/(dE_{{e_2}}\,d(\cos\theta))$ - Integrated over $E_{{e_1}}$ - {potential_label}', fontsize=14)
        ax_a = axes[0]
        contour_a = ax_a.contourf(cos_theta_grid, Ee2_grid, spectrum_dEe2_dcos_theta_analytic, levels=15, cmap='viridis')
        ax_a.set_xlabel(r'$\cos\theta$'); ax_a.set_ylabel(r'$E_{e_2}$ (MeV)')
        ax_a.set_title('Analytic Fermi (Pure Coulomb)')
        plt.colorbar(contour_a, ax=ax_a).set_label(r'$d\Gamma/(dE_{e_2}\,d(\cos\theta))$')
        ax_n = axes[1]
    else:
        fig, ax_n = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle(rf'{isotope}: $d\Gamma/(dE_{{e_2}}\,d(\cos\theta))$ - Integrated over $E_{{e_1}}$ - {potential_label}', fontsize=14)

    contour_n = ax_n.contourf(cos_theta_grid, Ee2_grid, spectrum_dEe2_dcos_theta_numeric, levels=15, cmap='viridis')
    ax_n.set_xlabel(r'$\cos\theta$'); ax_n.set_ylabel(r'$E_{e_2}$ (MeV)')
    ax_n.set_title('Numeric Fermi (Dirac Wavefunctions)')
    plt.colorbar(contour_n, ax=ax_n).set_label(r'$d\Gamma/(dE_{e_2}\,d(\cos\theta))$')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory, f"dEe2_dcos_theta_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # ========== PLOT 4: dΓ/(dEe1 d(cos θ)) ==========
    print("Generating dΓ/(dEe1 d(cos θ)) contour plots...")

    if show_analytic:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(rf'{isotope}: $d\Gamma/(dE_{{e_1}}\,d(\cos\theta))$ - Integrated over $E_{{e_2}}$ - {potential_label}', fontsize=14)
        ax_a = axes[0]
        contour_a = ax_a.contourf(cos_theta_grid, Ee1_grid, spectrum_dEe1_dcos_theta_analytic, levels=15, cmap='viridis')
        ax_a.set_xlabel(r'$\cos\theta$'); ax_a.set_ylabel(r'$E_{e_1}$ (MeV)')
        ax_a.set_title('Analytic Fermi (Pure Coulomb)')
        plt.colorbar(contour_a, ax=ax_a).set_label(r'$d\Gamma/(dE_{e_1}\,d(\cos\theta))$')
        ax_n = axes[1]
    else:
        fig, ax_n = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle(rf'{isotope}: $d\Gamma/(dE_{{e_1}}\,d(\cos\theta))$ - Integrated over $E_{{e_2}}$ - {potential_label}', fontsize=14)

    contour_n = ax_n.contourf(cos_theta_grid, Ee1_grid, spectrum_dEe1_dcos_theta_numeric, levels=15, cmap='viridis')
    ax_n.set_xlabel(r'$\cos\theta$'); ax_n.set_ylabel(r'$E_{e_1}$ (MeV)')
    ax_n.set_title('Numeric Fermi (Dirac Wavefunctions)')
    plt.colorbar(contour_n, ax=ax_n).set_label(r'$d\Gamma/(dE_{e_1}\,d(\cos\theta))$')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory, f"dEe1_dcos_theta_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # ========== PLOT 5: NUMERIC vs ANALYTIC % DIFF (Coulomb only) ==========
    if show_analytic:
        print("Generating relative difference plots...")
        with np.errstate(divide='ignore', invalid='ignore'):
            percent_diff = 100.0 * (spectrum_double_diff_numeric - spectrum_double_diff_analytic) / (np.abs(spectrum_double_diff_analytic) + 1e-20)
            percent_diff[~np.isfinite(percent_diff)] = 0.0

        fig, ax = plt.subplots(figsize=(10, 8))
        vmax = np.percentile(np.abs(percent_diff[np.isfinite(percent_diff)]), 95)
        contour = ax.contourf(Ee2_grid, Ee1_grid, percent_diff, levels=20, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xlabel(r'$E_{e_2}$ (MeV)'); ax.set_ylabel(r'$E_{e_1}$ (MeV)')
        ax.set_title(f'{isotope}: Relative Difference (%) between Numeric and Analytic - {potential_label}')
        plt.colorbar(contour, ax=ax).set_label('% Difference')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_output_directory, f"DoubleDiff_Difference_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300, bbox_inches='tight')
        plt.show()

    print("\nTriple-differential spectrum calculation complete!")

    return {
        "Ee1_grid": Ee1_grid,
        "Ee2_grid": Ee2_grid,
        "cos_theta_grid": cos_theta_grid,
        "spectrum_double_diff_analytic": spectrum_double_diff_analytic if show_analytic else None,
        "spectrum_double_diff_numeric": spectrum_double_diff_numeric,
        "spectrum_triple_diff_analytic": spectrum_triple_diff_analytic if show_analytic else None,
        "spectrum_triple_diff_numeric": spectrum_triple_diff_numeric,
        "spectrum_dEe2_dcos_theta_analytic": spectrum_dEe2_dcos_theta_analytic if show_analytic else None,
        "spectrum_dEe2_dcos_theta_numeric": spectrum_dEe2_dcos_theta_numeric,
        "spectrum_dEe1_dcos_theta_analytic": spectrum_dEe1_dcos_theta_analytic if show_analytic else None,
        "spectrum_dEe1_dcos_theta_numeric": spectrum_dEe1_dcos_theta_numeric,
    }

# ========== dGamma/d|ΔT| ENERGY DIFFERENCE SPECTRUM ==========

def calculate_and_plot_energy_diff_spectrum(Q, Z, A, Fermi_numeric,
                                             plots_output_directory, potential_index,
                                             nuclear_matrix_elements, isotope):
    """
    Calculate and plot dΓ/d(|ΔT|) where ΔT = T_{e1} − T_{e2}.

    Parameters
    ----------
    Q : float
        Q-value (MeV).
    Z, A : int
        Atomic and mass numbers.
    Fermi_numeric : callable
        Numerical Fermi function from Dirac solutions.
    plots_output_directory : str or Path
        Directory to save plots.
    potential_index : int
        Potential model index.
    nuclear_matrix_elements : dict
        Nuclear matrix elements for the selected isotope.
    isotope : str
        Isotope name, used in plot titles.
    """
    potential_names = {0: "Pure Coulomb", 2: "Finite-Sized Nucleus", 3: "Thomas-Fermi"}
    potential_label = potential_names.get(potential_index, f"Potential {potential_index}")
    nme_LO = {**nuclear_matrix_elements, "xi31": 0.0, "xi51": 0.0}

    delta_T_grid = np.linspace(0.0, Q, 300)

    spec_num = np.array([energy_diff_spectrum_integrand(dT, Q, Fermi_numeric, nuclear_matrix_elements)
                         for dT in delta_T_grid])
    spec_LO  = np.array([energy_diff_spectrum_integrand(dT, Q, Fermi_numeric, nme_LO)
                         for dT in delta_T_grid])

    # ========== Differential spectrum plot ==========
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(delta_T_grid, spec_num, lw=1.5, label=r'Full ($\xi_{31},\xi_{51}$ included)')
    ax.plot(delta_T_grid, spec_LO,  lw=1.5, ls='--', label=r'LO ($\xi_{31}=\xi_{51}=0$)')
    ax.set_xlabel(r'$|\Delta T| = |T_{e_1} - T_{e_2}|$ (MeV)')
    ax.set_ylabel(r'$d\Gamma/d|\Delta T|$ (1/yr per MeV)')
    ax.set_title(rf'{isotope}: Energy Difference Spectrum $d\Gamma/d|\Delta T|$'
                 rf' ($\Delta T = T_{{e_1}} - T_{{e_2}}$) ({potential_label})')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory,
                             f"EnergyDiff_Spectrum_potential_{potential_index}_Z{Z}_A{A}.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    # ========== Total rate and normalized spectrum ==========
    total_rate_num, _ = quad(
        lambda dT: energy_diff_spectrum_integrand(dT, Q, Fermi_numeric, nuclear_matrix_elements),
        0.0, Q,
        epsabs=1e-18, epsrel=1e-16, limit=20000, points=[0.0, Q / 2.0, Q]
    )
    total_rate_LO, _ = quad(
        lambda dT: energy_diff_spectrum_integrand(dT, Q, Fermi_numeric, nme_LO),
        0.0, Q,
        epsabs=1e-18, epsrel=1e-16, limit=20000, points=[0.0, Q / 2.0, Q]
    )

    norm_num = spec_num / total_rate_num
    norm_LO  = spec_LO  / total_rate_LO

    plt.figure(figsize=(12, 5))
    plt.plot(delta_T_grid, norm_num, lw=1.5, label=r'Full ($\xi_{31},\xi_{51}$ included)')
    plt.plot(delta_T_grid, norm_LO,  lw=1.5, ls='--', label=r'LO ($\xi_{31}=\xi_{51}=0$)')
    plt.xlabel(r'$|\Delta T| = |T_{e_1} - T_{e_2}|$ (MeV)')
    plt.ylabel(r'$1/\Gamma\;d\Gamma/d|\Delta T|$')
    plt.title(rf'{isotope}: Normalized Energy Difference Spectrum ({potential_label})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory,
                             f"EnergyDiff_Spectrum_Normalized_potential_{potential_index}_Z{Z}_A{A}.png"),
                dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Normalization check — Full: {np.trapezoid(norm_num, delta_T_grid):.10g}, "
          f"LO: {np.trapezoid(norm_LO, delta_T_grid):.10g}")


# ========== RESULTS OUTPUT ==========

def write_results_to_file(results_output_path, config, potential_index, Z, A, G_results, G_results_num,
                          H_results_num, halflife, halflife_num, cnf,
                          total_rate=None, total_rate_num=None,
                          total_err=None, total_err_num=None,
                          kappa_results=None):
    """
    Write phase-space factors and half-lives to output file.

    Parameters
    ----------
    results_output_path : str or Path
        Path to output file.
    config : dict
        Configuration dictionary.
    potential_index : int
        Potential model index.
    Z, A : int
        Atomic and mass numbers.
    G_results, G_results_num : ndarray
        Analytic and numeric G factors.
    H_results_num : ndarray
        Numeric H factors.
    halflife, halflife_num : float
        Analytic and numeric half-lives.
    total_rate, total_rate_num : float
        Total decay rates.
    total_err, total_err_num : float
        Integration errors.
    cnf : dict
        Isotope configuration dictionary.
    """
    potential_names = {0: "Pure Coulomb", 2: "finite sized nucleus", 3: "Thomas-Fermi"}
    potential_label = potential_names.get(potential_index, "Unknown")

    Simkovic_Gs = cnf.get("Simkovic_Gs", None)
    Simkovic_Hs = cnf.get("Simkovic_Hs", None)

    with open(results_output_path, "w") as f:
        f.write(f"### Double Beta Decay Results for {config['isotope']} using {potential_label} ###\n\n")

        # ========== G factors ==========
        tags = ["G0", "G2", "G4", "G22"]
        f.write("G FACTORS (Phase-Space)\n")
        if Simkovic_Gs is not None:
            f.write(f"{'Tags':<10} {'Numeric':<15} {'Simkovic':<15} {'% Diff (vs Simkovic)':<20}\n")
            f.write("-" * 60 + "\n")
            for i, tag in enumerate(tags):
                percent_diff = 100.0 * (G_results_num[i] - Simkovic_Gs[i]) / Simkovic_Gs[i]
                f.write(f"{tag:<10} {G_results_num[i]:<15.6e} {Simkovic_Gs[i]:<15.6e} {percent_diff:<20.6f}\n")
            if potential_index == 0:
                f.write(f"\nAnalytic reference: {[float(f'{v:.6e}') for v in G_results]}\n")
        else:
            f.write(f"{'Tags':<10} {'Numeric':<15}\n")
            f.write("-" * 25 + "\n")
            for i, tag in enumerate(tags):
                f.write(f"{tag:<10} {G_results_num[i]:<15.6e}\n")
            if potential_index == 0:
                f.write(f"\nAnalytic reference: {[float(f'{v:.6e}') for v in G_results]}\n")
        f.write("\n")

        # ========== H factors ==========
        tags = ["H0", "H2", "H4", "H22"]
        f.write("H FACTORS\n")
        if Simkovic_Hs is not None:
            f.write(f"{'Tags':<10} {'Numeric':<15} {'Simkovic':<15} {'% Diff (vs Simkovic)':<20}\n")
            f.write("-" * 60 + "\n")
            for i, tag in enumerate(tags):
                percent_diff = 100.0 * (H_results_num[i] - Simkovic_Hs[i]) / Simkovic_Hs[i]
                f.write(f"{tag:<10} {H_results_num[i]:<15.6e} {Simkovic_Hs[i]:<15.6e} {percent_diff:<20.6f}\n")
        else:
            f.write(f"{'Tags':<10} {'Numeric':<15}\n")
            f.write("-" * 25 + "\n")
            for i, tag in enumerate(tags):
                f.write(f"{tag:<10} {H_results_num[i]:<15.6e}\n")
        f.write("\n")

        # ========== Half-lives ==========
        f.write("HALF-LIVES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Calculated (Analytic):  {halflife:.6e} yr\n")
        f.write(f"Calculated (Numeric):   {halflife_num:.6e} yr\n")
        f.write(f"Experimental:           {2.19e21:.6e} yr\n\n")

        # ========== Decay rates ==========
        if total_rate is not None:
            f.write("DECAY RATES FROM ε-SPECTRUM\n")
            f.write("-" * 40 + "\n")
            f.write(f"Analytic:  {1/total_rate:.6e} ± {total_err:.6e} yr\n")
            f.write(f"Numeric:   {1/total_rate_num:.6e} ± {total_err_num:.6e} yr\n")
        else:
            f.write("DECAY RATES FROM ε-SPECTRUM\n")
            f.write("-" * 40 + "\n")
            f.write("Not computed (epsilon spectrum was not selected).\n")

        # ========== Angular correlation ==========
        f.write("\n")
        f.write("ANGULAR CORRELATION ⟨κ²ν⟩  (decay-rate-weighted average, numeric F and E)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Truncation order convention:\n")
        f.write(f"    LO     : κ²ν ratio = 1, weight = G0/H0 only\n")
        f.write(f"    1st    : κ²ν ratio = 1, weight includes ξ31 term\n")
        f.write(f"    total  : full κ²ν ratio, full weight; Γ²ν obtained two ways:\n")
        f.write(f"             formula : Γ²ν = G0..G22 combination (4 pre-computed tagged integrals)\n")
        f.write(f"             direct  : Γ²ν = single direct integral of the untagged two-electron kernel\n")
        f.write("-" * 70 + "\n")
        if kappa_results is not None:
            f.write(f"{'kappa_LO':<20} {'kappa_1st_order':<20} {'kappa_total_formula':<20} {'kappa_total_direct':<20}\n")
            f.write(f"{kappa_results['kappa_LO']:<20.6e} "
                    f"{kappa_results['kappa_1st']:<20.6e} "
                    f"{kappa_results['kappa_total']:<20.6e} "
                    f"{kappa_results['kappa_direct']:<20.6e}\n")
            f.write(f"(integration error: formula ±{kappa_results['kappa_total_err']:.2e}, "
                    f"direct ±{kappa_results['kappa_direct_err']:.2e})\n")
            f.write(f"% diff (direct vs formula): kappa {kappa_results['kappa_percent_diff']:+.6f}%, "
                    f"Γ²ν {kappa_results['gamma_percent_diff']:+.6f}%\n")
        else:
            f.write("Not computed.\n")


# ========== MAIN SPECTRUM CALCULATION ==========

def calc_double_beta_decay_spectrum(config, cnf):
    """
    Compute phase-space factors, half-lives, and energy spectra for
    two-neutrino double beta decay (2νββ).

    Orchestrates the calculation pipeline:
    1. Load wavefunction data from NPZ files
    2. Plot f, g, and Fermi functions
    3. Calculate phase-space factors (G and H)
    4. Calculate and plot spectra
    5. Write results to file

    Parameters
    ----------
    config : dict
        Main configuration dictionary
    cnf : dict
        Isotope configuration dictionary with Z, A, Q values
    """
    # ========== SETUP ==========
    verbose = config.get("verbose", True)
    Z = cnf["Z"]
    A = cnf["A"]
    Q = cnf["Q"]
    nuclear_matrix_elements = cnf["nuclear_matrix_elements"]

    rN = 1.2 * A ** (1 / 3) / HBAR_C
    potential_index = config["generator"]["potential_index"]

    # ========== DIRECTORIES ==========
    main_output_directory = Path(config["paths"]["output_directory"]) / f"Dirac_{config['isotope']}"
    directory_name = main_output_directory / "NPZ_files"
    plots_output_directory = main_output_directory / "plots"

    results_output_directory = main_output_directory / "phase_space_results"
    results_output_directory.mkdir(parents=True, exist_ok=True)
    results_output_path = results_output_directory / f"results_potential_{potential_index}_Z{Z}_A{A}.txt"

    data_files_directory = main_output_directory / "data_files"
    data_files_directory.mkdir(parents=True, exist_ok=True)

    # ========== LOAD DATA ==========
    file_name_kappa_n = f"{config['isotope']}_potential_{potential_index}_kappa_-1_Z{Z}_A{A}.npz"
    file_name_kappa_p = f"{config['isotope']}_potential_{potential_index}_kappa_+1_Z{Z}_A{A}.npz"

    file_name_kappa_n_Z0 = f"{config['isotope']}_potential_{potential_index}_kappa_-1_Z0_A{A}.npz"
    file_name_kappa_p_Z0 = f"{config['isotope']}_potential_{potential_index}_kappa_+1_Z0_A{A}.npz"

    _standard_npz_dir = Path("output") / f"Dirac_{config['isotope']}" / "NPZ_files"
    if not (directory_name / file_name_kappa_n).exists() and directory_name != _standard_npz_dir:
        print(
            f"\n[WARNING] No .npz files found in '{directory_name}'.\n"
            f"          Falling back to standard output directory: '{_standard_npz_dir}'\n"
        )
        directory_name = _standard_npz_dir

    T_n, r_n, P_n, Q_n = load_data(directory_name, file_name_kappa_n, verbose=verbose)
    T_p, r_p, P_p, Q_p = load_data(directory_name, file_name_kappa_p, verbose=verbose)
    P_n_Z0, Q_p_Z0 = None, None
    z0_n_path = directory_name / file_name_kappa_n_Z0
    z0_p_path = directory_name / file_name_kappa_p_Z0
    if z0_n_path.exists() and z0_p_path.exists():
        _, _, P_n_Z0, _ = load_data(directory_name, file_name_kappa_n_Z0, verbose=verbose)
        _, _, _, Q_p_Z0 = load_data(directory_name, file_name_kappa_p_Z0, verbose=verbose)
    elif verbose:
        print(
            f"[INFO] Z=0 reference files not found — Fermi division plot will be skipped.\n"
            f"       Re-run with '--mode generate -v' to generate them."
        )

    idx_R, mesh_point_R_au = find_mesh_point_R_au(r_n, rN, verbose=verbose)
    data = {"T": T_n, "P_n": P_n, "P_n_Z0": P_n_Z0, "Q_p": Q_p, "Q_p_Z0": Q_p_Z0, "idx_R": idx_R, "mesh_point_R_au": mesh_point_R_au}

    T_MeV = T_n * E_HARTREE / 1e6
    Ee = T_MeV + ME

    # ========== SETUP FERMI FUNCTIONS ==========
    Fermi_analytic = lambda Ee: Fermi(Ee, Z, A, rN)
    Fermi_numeric = lambda Ee: Fermi_numerical(Ee, Z, data)
    E_numeric = lambda Ee: E_function(Ee, Z, A, data)

    # ========== SAVE DATA FILES (always written) ==========
    save_fermi_function(Q, Z, A, Fermi_numeric, E_numeric, data_files_directory, potential_index)
    save_triple_differential_grid(Q, Fermi_numeric, E_numeric, nuclear_matrix_elements,
                                   data_files_directory, potential_index, Z, A)

    plots = config.get("plots", ["gf", "fermi", "epsilon", "double_diff"])
    if plots:
        plots_output_directory.mkdir(parents=True, exist_ok=True)

    # ========== PLOT WAVEFUNCTION COMPARISONS ==========
    if "gf" in plots:
        plot_f_and_g(Ee, potential_index, cnf, data, plots_output_directory, config['isotope'])

    # ========== PLOT FERMI FUNCTION ==========
    if "fermi" in plots:
        plot_fermi_function(Q, Z, A, rN, data, plots_output_directory, potential_index, config['isotope'])

    # ========== CALCULATE PHASE-SPACE FACTORS ==========
    phase_space_data = calculate_phase_space_factors(Q, Fermi_analytic, Fermi_numeric, E_numeric, nuclear_matrix_elements)
    G_results = phase_space_data["G_results"]
    G_results_num = phase_space_data["G_results_num"]
    H_results_num = phase_space_data["H_results_num"]

    # ========== VERIFY: TRIPLE-DIFFERENTIAL INTEGRAL vs PHASE-SPACE FORMULA ==========
    if verbose:
        compute_and_compare_total_gamma(Q, Fermi_numeric, E_numeric, nuclear_matrix_elements,
                                         phase_space_data, verbose=verbose)

    # ========== COMPUTE WEIGHTED ANGULAR CORRELATION ==========
    print("\nComputing decay-rate-weighted ⟨κ²ν⟩ at three truncation orders...")
    kappa_results = compute_kappa_weighted_average(Q, Fermi_numeric, E_numeric,
                                                    nuclear_matrix_elements, phase_space_data)
    print(f"  ⟨κ²ν⟩ LO            = {kappa_results['kappa_LO']:.6e}")
    print(f"  ⟨κ²ν⟩ 1st           = {kappa_results['kappa_1st']:.6e}")
    print(f"  ⟨κ²ν⟩ total (formula) = {kappa_results['kappa_total']:.6e}  (±{kappa_results['kappa_total_err']:.2e})"
          f"   [Γ²ν from G0..G22 combination = {kappa_results['gamma_formula']:.6e}]")
    print(f"  ⟨κ²ν⟩ total (direct)  = {kappa_results['kappa_direct']:.6e}  (±{kappa_results['kappa_direct_err']:.2e})"
          f"   [Γ²ν from direct untagged integral = {kappa_results['gamma_direct']:.6e}]")
    print(f"  Difference: κ {kappa_results['kappa_percent_diff']:+.6f}%,  Γ²ν {kappa_results['gamma_percent_diff']:+.6f}%")

    # ========== CALCULATE HALF-LIVES ==========
    MGT1 = nuclear_matrix_elements["MGT1"]
    xi31 = nuclear_matrix_elements["xi31"]
    xi51 = nuclear_matrix_elements["xi51"]
    halflife     = 1 / (GA ** 4 * MGT1 ** 2 * _gamma_combination(G_results,     xi31, xi51))
    halflife_num = 1 / (GA ** 4 * MGT1 ** 2 * _gamma_combination(G_results_num, xi31, xi51))

    # ========== CALCULATE AND PLOT TRIPLE-DIFFERENTIAL SPECTRUM ==========
    if "double_diff" in plots:
        calculate_and_plot_double_differential_spectrum(Q, Z, A, Fermi_analytic, Fermi_numeric,
                                                        E_numeric, plots_output_directory, potential_index, nuclear_matrix_elements, config['isotope'], phase_space_data)

    # ========== CALCULATE AND PLOT SPECTRA ==========
    spectrum_data = None
    if "epsilon" in plots:
        spectrum_data = calculate_and_plot_spectra(Q, Z, A, Fermi_analytic, Fermi_numeric,
                                                   plots_output_directory, potential_index, nuclear_matrix_elements, config['isotope'])

    # ========== CALCULATE AND PLOT ENERGY DIFFERENCE SPECTRUM ==========
    if "energy_diff" in plots:
        calculate_and_plot_energy_diff_spectrum(Q, Z, A, Fermi_numeric,
                                                plots_output_directory, potential_index,
                                                nuclear_matrix_elements, config['isotope'])

    # ========== CALCULATE AND PLOT SINGLE-ELECTRON SPECTRA ==========
    if "electron" in plots:
        calculate_and_plot_single_electron_spectra(Q, Z, A, Fermi_analytic, Fermi_numeric,
                                                   plots_output_directory, potential_index,
                                                   nuclear_matrix_elements, config['isotope'])

    # ========== WRITE RESULTS ==========
    write_results_to_file(
        results_output_path, config, potential_index, Z, A, G_results, G_results_num,
        H_results_num, halflife, halflife_num, cnf,
        total_rate=spectrum_data["total_rate"] if spectrum_data else None,
        total_rate_num=spectrum_data["total_rate_num"] if spectrum_data else None,
        total_err=spectrum_data["total_err"] if spectrum_data else None,
        total_err_num=spectrum_data["total_err_num"] if spectrum_data else None,
        kappa_results=kappa_results,
    )

    print(f"\nResults written to: {results_output_path}")



