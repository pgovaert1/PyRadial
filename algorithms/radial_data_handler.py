#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import nquad
from scipy.integrate import quad
from pathlib import Path
import os
from configurations.physics_constants import ME, GF, VUD, HBAR_C, GA, E_HARTREE
from algorithms.integrands import two_electron_kernel, two_electron_spectrum_integrand, spectrum_epsilon_integrand
from algorithms.fermi_function_utlities import Fermi, Fermi_numerical, E_function, plot_f_and_g, find_mesh_point_R_au

Gbeta = GF * VUD # Effective weak coupling constant in MeV^-2
c2n = ME * (Gbeta * ME ** 2)**4 / (8 * np.pi**7)
MeVtoyr = (365.25 * 24 * 3600) * 2.998e8 / (1e-15 * HBAR_C)




def load_data(directory_name, file_name):
    """
    Extracts Energy T, radial grid r, upper component wave function P and lower component
    wave function Q from the given .npz file.

    Parameters
    ----------
    directory_name : string
        Name of the directory in which .npz file is located.
    file_name : string
        Name of the .npz file

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


# ========== FERMI FUNCTION PLOTTING ==========

def plot_fermi_function(Q, Z, A, rN, data, plots_output_directory, potential_index):
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
    """
    x_plot = np.linspace(ME + 1e-3, Q + ME, 2000)
    Analytic_fermi_plot = Fermi(x_plot, Z, A, rN)
    Numeric_fermi_plot = Fermi_numerical(x_plot, Z, A, data)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    
    # Top plot: comparison
    ax1.plot(x_plot, Analytic_fermi_plot, lw=1.5, label='Pure Coulomb Analytic')
    ax1.plot(x_plot, Numeric_fermi_plot, lw=1.5, label='Numerical')
    ax1.set_ylabel('Fermi Function Value')
    ax1.set_title('Fermi Function vs Electron Energy')
    ax1.grid(True)
    ax1.legend()

    # Bottom plot: percentage difference
    percent_diff_fermi = 100.0 * (Numeric_fermi_plot - Analytic_fermi_plot) / Analytic_fermi_plot
    ax2.plot(x_plot, percent_diff_fermi, lw=1.5)
    ax2.set_xlabel('Electron Energy (MeV)')
    ax2.set_ylabel('% Difference')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory, f"Fermi_Function_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300)
    plt.show()


# ========== SPECTRUM CALCULATION AND PLOTTING ==========

def calculate_and_plot_spectra(Q, Z, A, Fermi_analytic, Fermi_numeric, plots_output_directory, potential_index, nuclear_matrix_elements):
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

    Returns
    -------
    dict
        Dictionary containing spectrum data and rates
    """
    eps_grid = np.linspace(0.0, Q, 400)
    spectrum_vals = np.array([spectrum_epsilon_integrand(eps, Q, Fermi_analytic, nuclear_matrix_elements) for eps in eps_grid])
    spectrum_vals_num = np.array([spectrum_epsilon_integrand(eps, Q, Fermi_numeric, nuclear_matrix_elements) for eps in eps_grid])

    # ========== Plot differential spectrum ==========
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    
    ax1.plot(eps_grid, spectrum_vals, lw=1.5, label="Pure Coulomb Analytic")
    ax1.plot(eps_grid, spectrum_vals_num, lw=1.5, label="Numerical")
    ax1.set_ylabel('dΓ/dε (1/yr per MeV)')
    ax1.set_title('2νββ Spectrum vs epsilon')
    ax1.legend()
    ax1.grid(True)

    percent_diff_eps = 100.0 * (spectrum_vals_num - spectrum_vals) / spectrum_vals
    ax2.plot(eps_grid, percent_diff_eps, lw=1.5)
    ax2.set_xlabel('epsilon = Ee1 + Ee2 − 2 me (MeV)')
    ax2.set_ylabel('% Difference')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory, f"Spectrum_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300)
    plt.show()

    # ========== Calculate total rates ==========
    total_rate, total_err = quad(lambda eps: spectrum_epsilon_integrand(eps, Q, Fermi_analytic, nuclear_matrix_elements), 0.0, Q,
                                 epsabs=1e-18, epsrel=1e-16, limit=20000, points=[0.0, Q/2.0, Q])
    total_rate_num, total_err_num = quad(lambda eps: spectrum_epsilon_integrand(eps, Q, Fermi_numeric, nuclear_matrix_elements), 0.0, Q,
                                         epsabs=1e-18, epsrel=1e-16, limit=20000, points=[0.0, Q/2.0, Q])

    # ========== Plot normalized spectrum ==========
    spec_vals_norm = spectrum_vals / total_rate
    spec_vals_norm_num = spectrum_vals_num / total_rate_num

    plt.figure(figsize=(12, 8))
    plt.plot(eps_grid, spec_vals_norm, lw=1.5, label="Analytic")
    plt.plot(eps_grid, spec_vals_norm_num, lw=1.5, label="Numeric")
    plt.xlabel('epsilon = Ee1 + Ee2 − 2 me (MeV)')
    plt.ylabel('1/Γ dΓ/dε')
    plt.title("Normalized 2νββ Spectrum")
    plt.legend()
    plt.grid(True)
    plt.show()

    norm_check = np.trapezoid(spec_vals_norm, eps_grid)
    norm_check_num = np.trapezoid(spec_vals_norm_num, eps_grid)
    print(f"Normalization check - Analytic: {norm_check:.10g}, Numeric: {norm_check_num:.10g}")

    return {
        "eps_grid": eps_grid,
        "spectrum_vals": spectrum_vals,
        "spectrum_vals_num": spectrum_vals_num,
        "total_rate": total_rate,
        "total_rate_num": total_rate_num,
        "total_err": total_err,
        "total_err_num": total_err_num,
    }

# ========== dGamma/dEe1dEe2dcos(θ) SPECTRUM CALCULATION AND PLOTTING ==========

def calculate_and_plot_double_differential_spectrum(Q, Z, A, Fermi_analytic, Fermi_numeric, E_function_numeric, plots_output_directory, potential_index, nuclear_matrix_elements, phase_space_data=None):
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
    
    # Import integrand functions
    from algorithms.integrands import triple_differential_spectrum_integrand_cos_theta, triple_differential_spectrum_integrand_ee1, triple_differential_spectrum_integrand_ee2
    
    # ========== DEFINE EVALUATION GRIDS ==========
    # Grid points for evaluating/plotting results
    Ee1_grid = np.linspace(ME + 1e-3, Q + ME, 40)
    Ee2_grid = np.linspace(ME + 1e-3, Q + ME, 40)
    cos_theta_grid = np.linspace(-1.0, 1.0, 20)
    
    print(f"Calculating triple-differential spectrum using integration:")
    print(f"  Ee1 range: [{Ee1_grid[0]:.3f}, {Ee1_grid[-1]:.3f}] MeV with {len(Ee1_grid)} evaluation points")
    print(f"  Ee2 range: [{Ee2_grid[0]:.3f}, {Ee2_grid[-1]:.3f}] MeV with {len(Ee2_grid)} evaluation points")
    print(f"  cos(θ) range: [{cos_theta_grid[0]:.2f}, {cos_theta_grid[-1]:.2f}] with {len(cos_theta_grid)} evaluation points")
    print(f"  Note: INTEGRATING over cos(θ) to obtain dΓ/(dEe1 dEe2) at each energy pair")
    
    # ========== COMPUTE DOUBLE-DIFFERENTIAL SPECTRUM VIA INTEGRATION ==========
    # For each (Ee1, Ee2) pair, INTEGRATE over cos_theta from -1 to +1 using quad
    # This gives us the double-differential spectrum dΓ/(dEe1 dEe2)
    spectrum_double_diff_analytic = np.zeros((len(Ee1_grid), len(Ee2_grid)))
    spectrum_double_diff_numeric = np.zeros((len(Ee1_grid), len(Ee2_grid)))
    
    # Integration options for cos_theta
    epsabs_angle = 1e-14
    epsrel_angle = 1e-10
    limit_angle = 5000
    
    for i, Ee1 in enumerate(Ee1_grid):
        if i % 5 == 0:
            print(f"  Processing Ee1 point {i+1}/{len(Ee1_grid)}")
        
        for j, Ee2 in enumerate(Ee2_grid):
            # Kinematic check: Ee1 + Ee2 must not exceed Q + 2*ME
            if Ee1 + Ee2 > Q + 2*ME:
                spectrum_double_diff_analytic[i, j] = 0.0
                spectrum_double_diff_numeric[i, j] = 0.0
                continue
            
            # ========== INTEGRATE OVER cos(θ) ==========
            # Use quad to integrate the triple-differential spectrum over cos_theta
            # This is analogous to how spectrum_epsilon_integrand integrates over D
            
            # ANALYTIC FERMI: Integrate triple-diff over cos_theta
            integrand_analytic = lambda cos_theta: triple_differential_spectrum_integrand_cos_theta(
                cos_theta, Ee1, Ee2, Q,
                fermi_func=Fermi_analytic,
                E_func=lambda Ee: 0.0,
                nuclear_matrix_elements=nuclear_matrix_elements
            )
            integral_analytic, _ = quad(integrand_analytic, -1.0, 1.0,
                                        epsabs=epsabs_angle, epsrel=epsrel_angle,
                                        limit=limit_angle, points=[0.0])
            spectrum_double_diff_analytic[i, j] = max(integral_analytic, 0.0)

            # NUMERIC FERMI: Integrate triple-diff over cos_theta
            integrand_numeric = lambda cos_theta: triple_differential_spectrum_integrand_cos_theta(
                cos_theta, Ee1, Ee2, Q,
                fermi_func=Fermi_numeric,
                E_func=E_function_numeric,
                nuclear_matrix_elements=nuclear_matrix_elements
            )
            integral_numeric, _ = quad(integrand_numeric, -1.0, 1.0,
                                       epsabs=epsabs_angle, epsrel=epsrel_angle,
                                       limit=limit_angle, points=[-1.0, 0.0, 1.0])
            spectrum_double_diff_numeric[i, j] = max(integral_numeric, 0.0)
    
    # ========== COMPUTE TRIPLE-DIFFERENTIAL SPECTRUM FOR REFERENCE (NO INTEGRATION) ==========
    # Store the triple-differential spectrum at grid points for angular dependence analysis
    spectrum_triple_diff_analytic = np.zeros((len(Ee1_grid), len(Ee2_grid), len(cos_theta_grid)))
    spectrum_triple_diff_numeric = np.zeros((len(Ee1_grid), len(Ee2_grid), len(cos_theta_grid)))
    
    for i, Ee1 in enumerate(Ee1_grid):
        for j, Ee2 in enumerate(Ee2_grid):
            if Ee1 + Ee2 > Q + 2*ME:
                continue
            for k, cos_theta in enumerate(cos_theta_grid):
                spec_val_analytic = triple_differential_spectrum_integrand_cos_theta(
                    cos_theta, Ee1, Ee2, Q,
                    fermi_func=Fermi_analytic,
                    E_func=lambda Ee: 0.0,
                    nuclear_matrix_elements=nuclear_matrix_elements
                )
                spectrum_triple_diff_analytic[i, j, k] = max(spec_val_analytic, 0.0)

                spec_val_numeric = triple_differential_spectrum_integrand_cos_theta(
                    cos_theta, Ee1, Ee2, Q,
                    fermi_func=Fermi_numeric,
                    E_func=E_function_numeric,
                    nuclear_matrix_elements=nuclear_matrix_elements
                )
                spectrum_triple_diff_numeric[i, j, k] = max(spec_val_numeric, 0.0)
    
    # ========== COMPUTE DIRECT SPECTRUM USING two_electron_kernel ==========
    # For verification: compute dΓ/(dEe1 dEe2) directly without integration over cos(θ)
    print("Computing direct two-electron kernel spectrum for verification...")
    spectrum_direct_analytic = np.zeros((len(Ee1_grid), len(Ee2_grid)))
    spectrum_direct_numeric = np.zeros((len(Ee1_grid), len(Ee2_grid)))
    
    for i, Ee1 in enumerate(Ee1_grid):
        if i % 5 == 0:
            print(f"  Direct computation - Ee1 point {i+1}/{len(Ee1_grid)}")
        for j, Ee2 in enumerate(Ee2_grid):
            if Ee1 + Ee2 > Q + 2*ME:
                spectrum_direct_analytic[i, j] = 0.0
                spectrum_direct_numeric[i, j] = 0.0
                continue
            
            # Direct kernel with tag=None uses full polynomial expansion
            spectrum_direct_analytic[i, j] = two_electron_kernel(
                Ee1, Ee2, Q, tag=None, 
                fermi_func=Fermi_analytic,
                nuclear_matrix_elements=nuclear_matrix_elements
            )
            spectrum_direct_numeric[i, j] = two_electron_kernel(
                Ee1, Ee2, Q, tag=None,
                fermi_func=Fermi_numeric,
                nuclear_matrix_elements=nuclear_matrix_elements
            )
    
    # ========== PLOT 1: 2D CONTOURS - DOUBLE DIFFERENTIAL SPECTRUM ==========
    print("\nGenerating double-differential spectrum contour plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'dΓ/(dEe1 dEe2) - Potential {potential_index}', fontsize=14)
    
    # Plot 1: Analytic - Integrated over cos(θ)
    ax_a = axes[0]
    contour_a = ax_a.contourf(Ee2_grid, Ee1_grid, spectrum_double_diff_analytic, levels=15, cmap='viridis')
    ax_a.set_xlabel('Ee2 (MeV)')
    ax_a.set_ylabel('Ee1 (MeV)')
    ax_a.set_title('Analytic Fermi\n(Integrated over cos(θ))')
    cbar_a = plt.colorbar(contour_a, ax=ax_a)
    cbar_a.set_label('dΓ/(dEe1 dEe2)')
    
    # Plot 2: Numeric - Integrated over cos(θ)
    ax_n = axes[1]
    contour_n = ax_n.contourf(Ee2_grid, Ee1_grid, spectrum_double_diff_numeric, levels=15, cmap='viridis')
    ax_n.set_xlabel('Ee2 (MeV)')
    ax_n.set_ylabel('Ee1 (MeV)')
    ax_n.set_title('Numeric Fermi\n(Integrated over cos(θ))')
    cbar_n = plt.colorbar(contour_n, ax=ax_n)
    cbar_n.set_label('dΓ/(dEe1 dEe2)')
    
    # Plot 3: Direct two_electron_kernel computation (should match the integration)
    ax_d = axes[2]
    # Use numeric for direct as the key comparison
    contour_d = ax_d.contourf(Ee2_grid, Ee1_grid, spectrum_direct_numeric, levels=15, cmap='viridis')
    ax_d.set_xlabel('Ee2 (MeV)')
    ax_d.set_ylabel('Ee1 (MeV)')
    ax_d.set_title('Numeric Fermi\n(Direct two_electron_kernel)')
    cbar_d = plt.colorbar(contour_d, ax=ax_d)
    cbar_d.set_label('dΓ/(dEe1 dEe2)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory, f"DoublesDiff_Spectrum_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300)
    plt.show()

    # ========== PLOT 1b: NORMALIZED BY Γ^{2ν} ==========
    if phase_space_data is not None:
        xi31 = nuclear_matrix_elements["xi31"]
        xi51 = nuclear_matrix_elements["xi51"]

        G = phase_space_data["G_results"]        # analytic; indices: 0=G0, 1=G2, 2=G4, 3=G22
        G_num = phase_space_data["G_results_num"]

        gamma_2nu_analytic = G[0] + xi31*G[1] + 1/3*xi31**2*G[3] + (1/3*xi31**2 + xi51)*G[2]
        gamma_2nu_numeric  = G_num[0] + xi31*G_num[1] + 1/3*xi31**2*G_num[3] + (1/3*xi31**2 + xi51)*G_num[2]

        spec_norm_analytic = spectrum_double_diff_analytic / gamma_2nu_analytic
        spec_norm_numeric  = spectrum_double_diff_numeric  / gamma_2nu_numeric
        spec_norm_direct   = spectrum_direct_numeric       / gamma_2nu_analytic

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Normalized: dΓ/(dEe1 dEe2) / Γ^{{2ν}} — Potential {potential_index}', fontsize=14)

        ax_a = axes[0]
        contour_a = ax_a.contourf(Ee2_grid, Ee1_grid, spec_norm_analytic, levels=15, cmap='viridis')
        ax_a.set_xlabel('Ee2 (MeV)')
        ax_a.set_ylabel('Ee1 (MeV)')
        ax_a.set_title('Analytic Fermi\n(Integrated over cos(θ))')
        cbar_a = plt.colorbar(contour_a, ax=ax_a)
        cbar_a.set_label('dΓ/(dEe1 dEe2) / Γ^{2ν}')

        ax_n = axes[1]
        contour_n = ax_n.contourf(Ee2_grid, Ee1_grid, spec_norm_numeric, levels=15, cmap='viridis')
        ax_n.set_xlabel('Ee2 (MeV)')
        ax_n.set_ylabel('Ee1 (MeV)')
        ax_n.set_title('Numeric Fermi\n(Integrated over cos(θ))')
        cbar_n = plt.colorbar(contour_n, ax=ax_n)
        cbar_n.set_label('dΓ/(dEe1 dEe2) / Γ^{2ν}')

        ax_d = axes[2]
        contour_d = ax_d.contourf(Ee2_grid, Ee1_grid, spec_norm_direct, levels=15, cmap='viridis')
        ax_d.set_xlabel('Ee2 (MeV)')
        ax_d.set_ylabel('Ee1 (MeV)')
        ax_d.set_title('Numeric Fermi\n(Direct two_electron_kernel)')
        cbar_d = plt.colorbar(contour_d, ax=ax_d)
        cbar_d.set_label('dΓ/(dEe1 dEe2) / Γ^{2ν}')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_output_directory, f"DoublesDiff_Spectrum_Normalized_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300)
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
    
    contour_diff = ax.contourf(Ee2_grid, Ee1_grid, percent_diff_methods, levels=20, 
                               cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Ee2 (MeV)')
    ax.set_ylabel('Ee1 (MeV)')
    ax.set_title(f'Relative Difference (%) between Integrated and Direct Methods - Potential {potential_index}')
    cbar = plt.colorbar(contour_diff, ax=ax)
    cbar.set_label('% Difference')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory, f"DoublesDiff_Integration_vs_Direct_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300)
    plt.show()
    
    # Print statistics
    diff_valid = percent_diff_methods[np.isfinite(percent_diff_methods)]
    if len(diff_valid) > 0:
        print(f"  Mean difference: {np.mean(np.abs(diff_valid)):.4f}%")
        print(f"  Max difference: {np.max(np.abs(diff_valid)):.4f}%")
        print(f"  Median difference: {np.median(np.abs(diff_valid)):.4f}%")
    
    # ========== PLOT 2: ANGULAR DEPENDENCE AT FIXED ENERGIES ==========
    # Show the INTEGRAND FUNCTION dΓ/(dEe1 dEe2 d(cosθ)) at specific energy pairs.
    # This visualizes the angular shape of the spectrum - how (1 + κ²ν*cos(θ)) modulates the distribution.
    # Demonstrates back-to-back vs collinear correlation effects at different energy pairs.
    print("Generating angular dependence plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Angular Integrand Shape: dΓ/(dEe1 dEe2 d(cosθ)) at Different Energy Pairs - Potential {potential_index}', fontsize=14)
    
    # Build 4 kinematically allowed energy pairs spread across the phase space:
    #   symmetric low, symmetric near-boundary, asymmetric, symmetric mid
    i_max_symm = int(np.searchsorted(2 * Ee1_grid, Q + 2 * ME)) - 1
    i_low  = max(2, i_max_symm // 4)
    i_high = max(i_low + 2, i_max_symm - 1)
    i_mid  = (i_low + i_high) // 2
    j_asym = int(np.searchsorted(Ee2_grid, Q + 2 * ME - Ee1_grid[i_low])) - 1
    j_asym = min(j_asym, len(Ee2_grid) - 1)
    energy_indices = [(i_low, i_low), (i_high, i_high), (i_low, j_asym), (i_mid, i_mid)]
    energy_labels = [
        f'Ee1={Ee1_grid[i]:.2f}, Ee2={Ee2_grid[j]:.2f} MeV'
        for i, j in energy_indices
    ]
    
    for plot_idx, ((idx_ee1, idx_ee2), energy_label) in enumerate(zip(energy_indices, energy_labels)):
        ax = axes[plot_idx // 2, plot_idx % 2]
        
        data_analytic_1d = spectrum_triple_diff_analytic[idx_ee1, idx_ee2, :]
        data_numeric_1d = spectrum_triple_diff_numeric[idx_ee1, idx_ee2, :]
        
        ax.plot(cos_theta_grid, data_analytic_1d, 'o-', linewidth=2, 
                markersize=6, label='Analytic (Coulomb)')
        ax.plot(cos_theta_grid, data_numeric_1d, 's-', linewidth=2, 
                markersize=6, label='Numeric (Dirac)')
        
        ax.set_xlabel('cos(θ)')
        ax.set_ylabel('dΓ/(dEe1 dEe2 d(cosθ))')
        ax.set_title(energy_label)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory, f"TripleDiff_Angular_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300)
    plt.show()
    
    # ========== INTEGRATE OVER Ee1 ==========
    # For each (Ee2, cos θ) pair, integrate over Ee1 to get dΓ/(dEe2 d(cos θ))
    print("\nComputing dΓ/(dEe2 d(cos θ)) by integrating over Ee1...")
    
    spectrum_dEe2_dcos_theta_analytic = np.zeros((len(Ee2_grid), len(cos_theta_grid)))
    spectrum_dEe2_dcos_theta_numeric = np.zeros((len(Ee2_grid), len(cos_theta_grid)))
    
    for j, Ee2 in enumerate(Ee2_grid):
        if j % 5 == 0:
            print(f"  Processing Ee2 point {j+1}/{len(Ee2_grid)}")
        
        for k, cos_theta in enumerate(cos_theta_grid):
            # Integration bounds for Ee1: from ME to Q+2*ME-Ee2
            Ee1_min = ME
            Ee1_max = Q + 2*ME - Ee2
            
            if Ee1_max <= Ee1_min:
                spectrum_dEe2_dcos_theta_analytic[j, k] = 0.0
                spectrum_dEe2_dcos_theta_numeric[j, k] = 0.0
                continue
            
            # ANALYTIC FERMI: Integrate triple-diff over Ee1
            integrand_analytic = lambda Ee1: triple_differential_spectrum_integrand_ee1(
                Ee1, Ee2, cos_theta, Q,
                fermi_func=Fermi_analytic,
                E_func=lambda Ee: 0.0,
                nuclear_matrix_elements=nuclear_matrix_elements
            )
            integral_analytic, _ = quad(integrand_analytic, Ee1_min, Ee1_max,
                                        epsabs=epsabs_angle, epsrel=epsrel_angle,
                                        limit=limit_angle, points=[Ee1_min, (Ee1_min + Ee1_max)/2, Ee1_max])
            spectrum_dEe2_dcos_theta_analytic[j, k] = max(integral_analytic, 0.0)

            # NUMERIC FERMI: Integrate triple-diff over Ee1
            integrand_numeric = lambda Ee1: triple_differential_spectrum_integrand_ee1(
                Ee1, Ee2, cos_theta, Q,
                fermi_func=Fermi_numeric,
                E_func=E_function_numeric,
                nuclear_matrix_elements=nuclear_matrix_elements
            )
            integral_numeric, _ = quad(integrand_numeric, Ee1_min, Ee1_max,
                                       epsabs=epsabs_angle, epsrel=epsrel_angle,
                                       limit=limit_angle, points=[Ee1_min, (Ee1_min + Ee1_max)/2, Ee1_max])
            spectrum_dEe2_dcos_theta_numeric[j, k] = max(integral_numeric, 0.0)
    
    # ========== INTEGRATE OVER Ee2 ==========
    # For each (Ee1, cos θ) pair, integrate over Ee2 to get dΓ/(dEe1 d(cos θ))
    print("Computing dΓ/(dEe1 d(cos θ)) by integrating over Ee2...")
    
    spectrum_dEe1_dcos_theta_analytic = np.zeros((len(Ee1_grid), len(cos_theta_grid)))
    spectrum_dEe1_dcos_theta_numeric = np.zeros((len(Ee1_grid), len(cos_theta_grid)))
    
    for i, Ee1 in enumerate(Ee1_grid):
        if i % 5 == 0:
            print(f"  Processing Ee1 point {i+1}/{len(Ee1_grid)}")
        
        for k, cos_theta in enumerate(cos_theta_grid):
            # Integration bounds for Ee2: from ME to Q+2*ME-Ee1
            Ee2_min = ME
            Ee2_max = Q + 2*ME - Ee1
            
            if Ee2_max <= Ee2_min:
                spectrum_dEe1_dcos_theta_analytic[i, k] = 0.0
                spectrum_dEe1_dcos_theta_numeric[i, k] = 0.0
                continue
            
            # ANALYTIC FERMI: Integrate triple-diff over Ee2
            integrand_analytic = lambda Ee2: triple_differential_spectrum_integrand_ee2(
                Ee2, Ee1, cos_theta, Q,
                fermi_func=Fermi_analytic,
                E_func=lambda Ee: 0.0,
                nuclear_matrix_elements=nuclear_matrix_elements
            )
            integral_analytic, _ = quad(integrand_analytic, Ee2_min, Ee2_max,
                                        epsabs=epsabs_angle, epsrel=epsrel_angle,
                                        limit=limit_angle, points=[Ee2_min, (Ee2_min + Ee2_max)/2, Ee2_max])
            spectrum_dEe1_dcos_theta_analytic[i, k] = max(integral_analytic, 0.0)

            # NUMERIC FERMI: Integrate triple-diff over Ee2
            integrand_numeric = lambda Ee2: triple_differential_spectrum_integrand_ee2(
                Ee2, Ee1, cos_theta, Q,
                fermi_func=Fermi_numeric,
                E_func=E_function_numeric,
                nuclear_matrix_elements=nuclear_matrix_elements
            )
            integral_numeric, _ = quad(integrand_numeric, Ee2_min, Ee2_max,
                                       epsabs=epsabs_angle, epsrel=epsrel_angle,
                                       limit=limit_angle, points=[Ee2_min, (Ee2_min + Ee2_max)/2, Ee2_max])
            spectrum_dEe1_dcos_theta_numeric[i, k] = max(integral_numeric, 0.0)
    
    # ========== PLOT 3: 2D CONTOURS - dΓ/(dEe2 d(cos θ)) ==========
    print("Generating dΓ/(dEe2 d(cos θ)) contour plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'dΓ/(dEe2 d(cos θ)) - Integrated over Ee1 - Potential {potential_index}', fontsize=14)
    
    ax_a = axes[0]
    contour_a = ax_a.contourf(cos_theta_grid, Ee2_grid, spectrum_dEe2_dcos_theta_analytic, levels=15, cmap='viridis')
    ax_a.set_xlabel('cos(θ)')
    ax_a.set_ylabel('Ee2 (MeV)')
    ax_a.set_title('Analytic Fermi (Pure Coulomb)')
    cbar_a = plt.colorbar(contour_a, ax=ax_a)
    cbar_a.set_label('dΓ/(dEe2 d(cos θ))')
    
    ax_n = axes[1]
    contour_n = ax_n.contourf(cos_theta_grid, Ee2_grid, spectrum_dEe2_dcos_theta_numeric, levels=15, cmap='viridis')
    ax_n.set_xlabel('cos(θ)')
    ax_n.set_ylabel('Ee2 (MeV)')
    ax_n.set_title('Numeric Fermi (Dirac Wavefunctions)')
    cbar_n = plt.colorbar(contour_n, ax=ax_n)
    cbar_n.set_label('dΓ/(dEe2 d(cos θ))')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory, f"dEe2_dcos_theta_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300)
    plt.show()
    
    # ========== PLOT 4: 2D CONTOURS - dΓ/(dEe1 d(cos θ)) ==========
    print("Generating dΓ/(dEe1 d(cos θ)) contour plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'dΓ/(dEe1 d(cos θ)) - Integrated over Ee2 - Potential {potential_index}', fontsize=14)
    
    ax_a = axes[0]
    contour_a = ax_a.contourf(cos_theta_grid, Ee1_grid, spectrum_dEe1_dcos_theta_analytic, levels=15, cmap='viridis')
    ax_a.set_xlabel('cos(θ)')
    ax_a.set_ylabel('Ee1 (MeV)')
    ax_a.set_title('Analytic Fermi (Pure Coulomb)')
    cbar_a = plt.colorbar(contour_a, ax=ax_a)
    cbar_a.set_label('dΓ/(dEe1 d(cos θ))')
    
    ax_n = axes[1]
    contour_n = ax_n.contourf(cos_theta_grid, Ee1_grid, spectrum_dEe1_dcos_theta_numeric, levels=15, cmap='viridis')
    ax_n.set_xlabel('cos(θ)')
    ax_n.set_ylabel('Ee1 (MeV)')
    ax_n.set_title('Numeric Fermi (Dirac Wavefunctions)')
    cbar_n = plt.colorbar(contour_n, ax=ax_n)
    cbar_n.set_label('dΓ/(dEe1 d(cos θ))')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory, f"dEe1_dcos_theta_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300)
    plt.show()
    
    # ========== PLOT 5: COMPARISON - RELATIVE DIFFERENCE (DOUBLE DIFF) ==========
    print("Generating relative difference plots...")
    
    with np.errstate(divide='ignore', invalid='ignore'):
        percent_diff = 100.0 * (spectrum_double_diff_numeric - spectrum_double_diff_analytic) / (np.abs(spectrum_double_diff_analytic) + 1e-20)
        percent_diff[~np.isfinite(percent_diff)] = 0.0
    
    fig, ax = plt.subplots(figsize=(10, 8))
    data_diff = percent_diff
    
    vmax = np.percentile(np.abs(data_diff[np.isfinite(data_diff)]), 95)
    
    contour = ax.contourf(Ee2_grid, Ee1_grid, data_diff, levels=20, 
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Ee2 (MeV)')
    ax.set_ylabel('Ee1 (MeV)')
    ax.set_title(f'Relative Difference (%) between Numeric and Analytic - Potential {potential_index}')
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('% Difference')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_output_directory, f"DoubleDiff_Difference_potential_{potential_index}_Z{Z}_A{A}.png"), dpi=300)
    plt.show()
    
    print("\nTriple-differential spectrum calculation complete!")
    
    return {
        "Ee1_grid": Ee1_grid,
        "Ee2_grid": Ee2_grid,
        "cos_theta_grid": cos_theta_grid,
        "spectrum_double_diff_analytic": spectrum_double_diff_analytic,
        "spectrum_double_diff_numeric": spectrum_double_diff_numeric,
        "spectrum_triple_diff_analytic": spectrum_triple_diff_analytic,
        "spectrum_triple_diff_numeric": spectrum_triple_diff_numeric,
        "spectrum_dEe2_dcos_theta_analytic": spectrum_dEe2_dcos_theta_analytic,
        "spectrum_dEe2_dcos_theta_numeric": spectrum_dEe2_dcos_theta_numeric,
        "spectrum_dEe1_dcos_theta_analytic": spectrum_dEe1_dcos_theta_analytic,
        "spectrum_dEe1_dcos_theta_numeric": spectrum_dEe1_dcos_theta_numeric,
    }

# ========== RESULTS OUTPUT ==========

def write_results_to_file(results_output_path, config, potential_index, Z, A, G_results, G_results_num,
                          H_results_num, halflife, halflife_num, cnf,
                          total_rate=None, total_rate_num=None,
                          total_err=None, total_err_num=None):
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
    # Determine scheme
    scheme_map = {0: "B", 2: "A", 3: "C"}
    scheme = scheme_map.get(potential_index, "Unknown")

    Simkovic_Gs = cnf.get("Simkovic_Gs", None)
    Simkovic_Hs = cnf.get("Simkovic_Hs", None)

    with open(results_output_path, "w") as f:
        f.write(f"### Double Beta Decay Results for {config['isotope']} using Scheme {scheme} ###\n\n")

        # ========== G factors ==========
        tags = ["G0", "G2", "G4", "G22"]
        f.write("G FACTORS (Phase-Space)\n")
        if Simkovic_Gs is not None:
            f.write(f"{'Tags':<10} {'Numeric':<15} {'Simkovic':<15} {'% Diff (vs Simkovic)':<20}\n")
            f.write("-" * 60 + "\n")
            for i, tag in enumerate(tags):
                percent_diff = 100.0 * (G_results_num[i] - Simkovic_Gs[i]) / Simkovic_Gs[i]
                f.write(f"{tag:<10} {G_results_num[i]:<15.6e} {Simkovic_Gs[i]:<15.6e} {percent_diff:<20.6f}\n")
            if scheme == "B":
                f.write(f"\nAnalytic reference: {[float(f'{v:.6e}') for v in G_results]}\n")
        else:
            f.write(f"{'Tags':<10} {'Numeric':<15}\n")
            f.write("-" * 25 + "\n")
            for i, tag in enumerate(tags):
                f.write(f"{tag:<10} {G_results_num[i]:<15.6e}\n")
            if scheme == "B":
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
    plots_output_directory.mkdir(parents=True, exist_ok=True)

    results_output_directory = main_output_directory / "phase_space_results"
    results_output_directory.mkdir(parents=True, exist_ok=True)
    results_output_path = results_output_directory / f"results_potential_{potential_index}_Z{Z}_A{A}.txt"

    # ========== LOAD DATA ==========
    file_name_kappa_n = f"{config['isotope']}_potential_{potential_index}_kappa_-1_Z{Z}_A{A}.npz"
    file_name_kappa_p = f"{config['isotope']}_potential_{potential_index}_kappa_+1_Z{Z}_A{A}.npz"

    T_n, r_n, P_n, Q_n = load_data(directory_name, file_name_kappa_n)
    T_p, r_p, P_p, Q_p = load_data(directory_name, file_name_kappa_p)

    idx_R, mesh_point_R_au = find_mesh_point_R_au(r_n, rN)
    data = {"T": T_n, "P_n": P_n, "Q_p": Q_p, "idx_R": idx_R, "mesh_point_R_au": mesh_point_R_au}

    T_MeV = T_n * E_HARTREE / 1e6
    Ee = T_MeV + ME

    plots = config.get("plots", ["gf", "fermi", "epsilon", "double_diff"])

    # ========== PLOT WAVEFUNCTION COMPARISONS ==========
    if "gf" in plots:
        plot_f_and_g(Ee, potential_index, cnf, data, plots_output_directory)

    # ========== PLOT FERMI FUNCTION ==========
    if "fermi" in plots:
        plot_fermi_function(Q, Z, A, rN, data, plots_output_directory, potential_index)

    # ========== SETUP FERMI FUNCTIONS ==========
    Fermi_analytic = lambda Ee: Fermi(Ee, Z, A, rN)
    Fermi_numeric = lambda Ee: Fermi_numerical(Ee, Z, A, data)
    E_numeric = lambda Ee: E_function(Ee, Z, A, data)

    # ========== CALCULATE PHASE-SPACE FACTORS ==========
    phase_space_data = calculate_phase_space_factors(Q, Fermi_analytic, Fermi_numeric, E_numeric, nuclear_matrix_elements)
    G_results = phase_space_data["G_results"]
    G_results_num = phase_space_data["G_results_num"]
    H_results_num = phase_space_data["H_results_num"]

    # ========== CALCULATE HALF-LIVES ==========
    MGT1 = nuclear_matrix_elements["MGT1"]
    xi31 = nuclear_matrix_elements["xi31"]
    xi51 = nuclear_matrix_elements["xi51"]
    halflife = 1 / (GA ** 4 * MGT1 ** 2 * (G_results[0] + xi31 * G_results[1] +
                    1/3 * xi31 ** 2 * G_results[3] + (1/3 * xi31 ** 2 + xi51) * G_results[2]))

    halflife_num = 1 / (GA ** 4 * MGT1 ** 2 * (G_results_num[0] + xi31 * G_results_num[1] +
                        1/3 * xi31 ** 2 * G_results_num[3] + (1/3 * xi31 ** 2 + xi51) * G_results_num[2]))
    
    # ========== CALCULATE AND PLOT TRIPLE-DIFFERENTIAL SPECTRUM ==========
    if "double_diff" in plots:
        calculate_and_plot_double_differential_spectrum(Q, Z, A, Fermi_analytic, Fermi_numeric,
                                                        E_numeric, plots_output_directory, potential_index, nuclear_matrix_elements, phase_space_data)

    # ========== CALCULATE AND PLOT SPECTRA ==========
    spectrum_data = None
    if "epsilon" in plots:
        spectrum_data = calculate_and_plot_spectra(Q, Z, A, Fermi_analytic, Fermi_numeric,
                                                   plots_output_directory, potential_index, nuclear_matrix_elements)

    # ========== WRITE RESULTS ==========
    write_results_to_file(
        results_output_path, config, potential_index, Z, A, G_results, G_results_num,
        H_results_num, halflife, halflife_num, cnf,
        total_rate=spectrum_data["total_rate"] if spectrum_data else None,
        total_rate_num=spectrum_data["total_rate_num"] if spectrum_data else None,
        total_err=spectrum_data["total_err"] if spectrum_data else None,
        total_err_num=spectrum_data["total_err_num"] if spectrum_data else None,
    )
    
    print(f"\nResults written to: {results_output_path}")



