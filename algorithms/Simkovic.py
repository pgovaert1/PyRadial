#!/usr/bin/env python

# Generated from: Simkovic.ipynb
# Converted at: 2026-02-28T13:04:04.740Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from sys import prefix
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.integrate._ivp.radau import predict_factor
import scipy.special as sp
import matplotlib.pyplot as plt
from scipy.integrate import nquad
from scipy.integrate import quad
from scipy.interpolate import interp1d
from pathlib import Path
import mpmath as mp
import os
import glob, re
from physics_constants import ME, ALPHA, GF, VUD, HBAR_C, GA, AU, C, E_HARTREE


Gbeta = GF * VUD # Effective weak coupling constant in MeV^-2
c2n = ME * (Gbeta * ME ** 2)**4 / (8 * np.pi**7)
MeVtoyr = (365.25 * 24 * 3600) * 2.998e8 / (1e-15 * HBAR_C)
MGT1, MGT3, MGT5, xi31, xi51 = 0.0104, 0.00403, 0.00126, 0.3867, 0.1207 # Simkovic NMEs for 136Xe

#### FORTRAN DATA EXTRACTION

def Fortran_data():
    directory = "out_fortran"
    P_values = []
    E_values = []

    pattern = os.path.join(directory, "cxem_*.out")
    R_value_fortran = None

    for filename in sorted(glob.glob(pattern)):
        E = None
        with open(filename, "r") as f:
            for line in f:
                if "Free state:" in line:
                    match = re.search(r"E=\s*([+-]?\d+\.\d+E[+-]\d+)", line)

                    if match:
                        E = float(match.group(1))

                parts = line.split()

                # Skip header/non-data lines
                if len(parts) != 3:
                    continue

                try:
                    R = float(parts[0])
                    P = float(parts[1])

                    # First nonzero R
                    if R != 0.0:
                        if R_value_fortran is None:
                            R_value_fortran = R
                        P_values.append(P)
                        E_values.append(E)
                        break

                except ValueError:
                    continue
    if R_value_fortran is None:
        raise RuntimeError("No matching radius extracted")

    P_values = np.array(P_values)
    E_values = np.array(E_values) * E_HARTREE/1e6

    print(f"R = {R_value_fortran}")
    print(f"R in a.u = {R_value_fortran/ AU}")
    print(f"E values = {E_values}")
    print(f"P(R) values = {P_values}")



    p = np.sqrt(E_values * (E_values + 2*C**2))/C
    total_E = E_values + ME
    g_func = np.sqrt((total_E + ME)/(2*total_E)) * P_values /(p * R_value_fortran/AU)

    plt.figure(figsize=(12,8))
    plt.plot(E_values, g_func)
    plt.xscale("log")
    plt.show()

    return g_func, E_values



def load_data(directory_name, file_name):

    full_path = Path(directory_name)/ file_name
    data = np.load(full_path)

    print(f"loading file from {full_path}")

    T = data["T"]
    r = data["r"]
    P = data["P"]
    Q = data["Q"]
    return T, r, P, Q



def find_mesh_point_R_au(mesh_points, rN):
    R_au = rN * HBAR_C * 1e-15 /AU
    idx_R = 0
    mesh_point_R_au = 0
    for i in range(0,len(mesh_points)-1):
        if mesh_points[i] <= R_au  and mesh_points[i+1] > R_au:
            mesh_point_R_au = mesh_points[i]
            idx_R = i
            print(f"closest mesh point to R_au = {R_au: .5g} is at i ={idx_R} w/ mesh point value ={mesh_point_R_au: .5g}")
            break
    return idx_R, mesh_point_R_au


def calc_g_and_f(Ee, Z, A, data):
    Ee = np.asarray(Ee, dtype = float)

    T_mesh = data["T"]
    P_n = data["P_n"]
    Q_p = data["Q_p"]
    idx_R = data["idx_R"]
    mesh_point_R_au = data["mesh_point_R_au"]


    T_MeV = T_mesh * E_HARTREE/1e6
    Ee = T_MeV + ME
    p_au = np.sqrt(T_mesh * (T_mesh + 2*C**2))/C


    g_n1 = np.sqrt((Ee + ME)/(2*Ee)) * P_n[:,idx_R] /(p_au * mesh_point_R_au)
    f_p1 = np.sqrt((Ee + ME)/(2*Ee)) * Q_p[:,idx_R] / (p_au * mesh_point_R_au)

    return g_n1, f_p1



def Fermi_numerical(Ee, Z, A, data):
    s = np.sqrt(1-(Z/A)**2)

    T = Ee - ME
    T_mesh = data["T"]
    T_MeV = T_mesh * E_HARTREE/1e6

    g, f = calc_g_and_f(Ee, Z, A, data)


    Fermi_vals = (g**2 + f**2)

    return 2/(s+1) * np.interp(T, T_MeV, Fermi_vals)

def E_function(Ee, Z, A, data):
    s = np.sqrt(1-(Z/A)**2)

    T = Ee - ME
    T_mesh = data["T"]
    T_MeV = T_mesh * E_HARTREE/1e6

    g, f = calc_g_and_f(Ee, Z, A, data)
    E_vals = 2*g * f


    return np.interp(T, T_MeV, E_vals)


def Fermi(Ee, Z, A, rN):
    # Guard against Ee slightly below me due to numerical issues
    Ee = np.asarray(Ee)
    p2 = Ee**2 - ME**2
    # Set negative p2 to a tiny positive to avoid NaNs exactly at the threshold
    p2 = np.where(p2 > 0, p2, np.finfo(float).tiny)
    p = np.sqrt(p2)
    y = ALPHA * Z * Ee / p
    gamma0 = np.sqrt(1 - (ALPHA * Z)**2)
    F = (sp.gamma(3) / (sp.gamma(1) * sp.gamma(1 + 2 * gamma0)))**2
    F *= (2 * p * rN)**(2 * (gamma0 - 1)) * np.exp(np.pi * y)
    F *= np.abs(sp.gamma(gamma0 + 1j * y))**2

    firstorder_corr = 6 * p * rN * np.sqrt((1-gamma0)/(1+gamma0)) * (1+gamma0 - 2*gamma0* (p/Ee)**2/3)/((1+ 2*gamma0)*p/Ee)

    secondorder_corr = -(p * rN)**2 * (1-gamma0)/(1+gamma0) * (-2*(1+gamma0)* (5 + 4*gamma0) + (1 + 6*gamma0 + 4*gamma0**2)*(p/Ee)**2)/((1 + 2*gamma0) * p/Ee)**2

    return F * (1 -firstorder_corr - secondorder_corr)



#TODO write a func to compare f and g individually

def plot_f_and_g(Ee, potential_index, cnf, data, output_directory_name):
    Z = cnf["Z"] # Atomic number
    A = cnf["A"] # Atomic Mass number
    Q = cnf["Q"] # MeV Q-value
    rN = 1.2 * A ** (1 / 3) / HBAR_C # Nuclear radius

    scheme = None
    f_analytic = g_analytic = None

    g_numeric, f_numeric = calc_g_and_f(Ee, Z, A, data)

    g_fortran, E_fortran = Fortran_data()

    if potential_index == 0:
        scheme = "B"

        out_g = np.empty_like(Ee, dtype = float)
        out_f = np.empty_like(Ee, dtype = float)

        kappa_g = -1
        kappa_f = 1


        for i , E in enumerate(Ee):
            p = np.sqrt(E**2 - ME**2)
            eta = ALPHA * Z* E/ p
            gammak = np.sqrt(1 - (ALPHA * Z)**2)

            numerator = np.abs(sp.gamma(1 + gammak + 1j * eta))
            denominator = sp.gamma(1 + 2 * gammak)
            gamma_ratio = numerator/denominator

            sqrt_term_plus = np.sqrt((E + ME) / (2 * E))
            sqrt_term_min = np.sqrt((E - ME) / (2 * E))

            exp_zeta_g = np.sqrt((kappa_g - 1j * eta * ME/E) / (gammak - 1j * eta))
            exp_zeta_f = np.sqrt((kappa_f - 1j * eta * ME/E) / (gammak - 1j * eta))


            hyp = mp.hyp1f1(gammak - 1j*eta, 1 + 2*gammak, -2*1j*p*rN)
            Im_term = np.imag(np.exp(1j*p*rN) * exp_zeta_g * hyp )
            Re_term = np.real(np.exp(1j*p*rN) * exp_zeta_f * hyp)

            out_g[i] = np.sign(kappa_g) * 1.0/(p*rN) * sqrt_term_plus * gamma_ratio * (2*p*rN)**gammak * np.exp(np.pi * eta/2) * Im_term
            out_f[i] = np.sign(kappa_f) * 1.0/(p*rN) * sqrt_term_min * gamma_ratio * (2*p*rN)**gammak * np.exp(np.pi * eta/2) * Re_term

        g_analytic = out_g
        f_analytic = out_f
    elif potential_index == 2:
        scheme = "A"
        g_analytic = np.sqrt(Fermi(Ee, Z, A, rN)) * np.sqrt((Ee + ME)/(2*Ee))
        f_analytic = np.sqrt(Fermi(Ee, Z, A, rN)) * np.sqrt((Ee - ME)/ (2*Ee))

    elif potential_index == 3:
        scheme = "C"



    else:
        raise RuntimeError("No proper potential index has been passed")





    plt.figure(figsize=(12,8))
    plt.plot(Ee-ME, abs(f_numeric), label = "f numeric")
    if f_analytic is not None:
        plt.plot(Ee-ME, f_analytic, label = "f analytic")
    plt.xlabel(r"$T$ (Mev)")
    plt.ylabel(r"$f_{\kappa=1}(R)$")
    plt.xlim((Ee-ME)[0], (Ee-ME)[-1])
    plt.xscale("log")
    plt.ylim(0, 4)
    plt.title(f"f wave func comparison, Scheme {scheme} ")
    plt.legend()
    plt.savefig(os.path.join(output_directory_name, f"f_comparison_scheme_{scheme}.png"), dpi = 300)
    plt.show()


    plt.figure(figsize=(12,8))
    plt.plot(E_fortran, abs(g_fortran), label = "g Fortrant")
    plt.plot(Ee-ME, abs(g_numeric), label = "g numeric")
    if g_analytic is not None:
        plt.plot(Ee-ME, g_analytic, label = "g analytic")
    plt.xlabel(r"$T$ (Mev)")
    plt.ylabel(r"$g_{\kappa=-1}(R)$")
    plt.xlim((Ee-ME)[0], (Ee-ME)[-1])
    plt.xscale("log")
    plt.ylim(0, 20)
    plt.title(f"g wave func comparison, Scheme {scheme} ")
    plt.legend()
    plt.savefig(os.path.join(output_directory_name, f"g_comparison_scheme_{scheme}.png"), dpi = 300)
    plt.show()

    return None




def electron_momentum(E):
    p_squared = E * E - ME * ME
    if p_squared <= 0.0:
        return 0.0
    return np.sqrt(p_squared)


def phase_space_polynomial(tag, x, y):
    if tag == 0:
        return x**5 / 30.0
    elif tag == 2:
        return x**5 * (x*x + 7*y*y) / (420.0 * (2*ME)**2)
    elif tag == 4:
        return x**5 * (x**4 + 18*x*x*y*y + 21*y**4) / (5040.0 * (2*ME)**4)
    elif tag == 22:
        return x**5 * (x**4 - 6*x*x*y*y + 21*y**4) / (10080.0 * (2*ME)**4)
    else:
        raise ValueError(f"Unknowown tag: {tag}")


def two_electron_integrand(Ee1, Ee2, Q_value, tag, fermi_func):
    if not (ME <= Ee1 <= Q_value + ME):
        return 0.0

    Ee2_max = Q_value + 2.0 * ME - Ee1
    if not (ME <= Ee2 <= Ee2_max):
        return 0.0

    p1 = electron_momentum(Ee1)
    p2 = electron_momentum(Ee2)
    if p1 == 0 or p2 == 0:
        return 0.0

    F1 = fermi_func(Ee1)
    F2 = fermi_func(Ee2)

    neutrino_energy_sum = Q_value + 2*ME - Ee1 - Ee2
    energy_assymetry = Ee1 - Ee2

    polynomail = phase_space_polynomial(tag, neutrino_energy_sum, energy_assymetry)

    pre_factor = MeVtoyr * c2n / (ME**11 * np.log(2))
    return float(pre_factor * F1 * F2 * p1 * Ee1 * p2 * Ee2 * polynomail)


def phase_space_integrand(tag, Q_value, fermi_func):
    return lambda Ee2, Ee1: two_electron_integrand(Ee1, Ee2, Q_value, tag, fermi_func)


def energies_eps_D(eps, D):
    Ee1 = ME + D + 0.5*eps
    Ee2 = ME - D + 0.5*eps
    return Ee1, Ee2

def phase_combo(Q_minus_eps, D):
    x = Q_minus_eps
    y = 2*D  # y = Ee1 - Ee2 = 2Δ

    A0  = x**5 / 30.0
    A2  = x**5 * (x*x + 7*y*y) / (420.0 * (2.0*ME)**2)
    A4  = x**5 * (x**4 + 18.0*x*x*y*y + 21.0*y**4) / (5040.0 * (2.0*ME)**4)
    A22 = x**5 * (x**4 - 6.0*x*x*y*y + 21.0*y**4) / (10080.0 * (2.0*ME)**4)

    return A0 + A2*xi31 + A4*(xi31**2/3.0 + xi51) + A22*(xi31**2/3.0)

def epsilon_spectrum_integrand(eps, D, Q_value, fermi_func):
    if not (0.0 <= eps <= Q_value):
        return 0.0
    if not (-eps/2.0 <= D <= eps/2.0):
        return 0.0

    Ee1, Ee2 = energies_eps_D(eps, D)

    p1 = electron_momentum(Ee1)
    p2 = electron_momentum(Ee2)
    if p1 == 0 or p2 == 0:
        return 0.0

    F1 = fermi_func(Ee1)
    F2 = fermi_func(Ee2)

    poly = phase_combo(Q_value - eps, D)
    pre_factor = (GA ** 4 * MGT1**2 *MeVtoyr * c2n) / (ME**11)
    return float(pre_factor * F1 * F2 * p1 * Ee1 * p2 * Ee2 * poly)

def spectrum_epsilon(eps, Q_value, fermi_func):
    if not (0.0 <= eps <= Q_value):
                return 0.0
    value, _ = quad(lambda D: epsilon_spectrum_integrand(eps, D, Q_value, fermi_func),
                    -eps/2.0,
                    eps/2.0,
                    epsabs=1e-16,
                    epsrel=1e-14,
                    limit=20000,
                    points=[-eps/2.0, 0.0, eps/2.0])
    return value


def Calc_double_beta_decay_spectrum(config,cnf):
    Z = cnf["Z"] # Atomic number
    A = cnf["A"] # Atomic Mass number
    Q = cnf["Q"] # MeV Q-value
    rN = 1.2 * A ** (1 / 3) / HBAR_C # Nuclear radius

    potential_index = config["generator"]["potential_index"]


    directory_name = Path("Dirac_" + config["isotope"])/ "NPZ_files"
    file_name_kappa_p = f"{config["isotope"]}_potential_{potential_index}_kappa_+1_Z{Z:}_A{A}.npz"
    file_name_kappa_n = f"{config["isotope"]}_potential_{potential_index}_kappa_-1_Z{Z:}_A{A}.npz"

    main_output_directory = Path("Dirac_" + config["isotope"])
    results_output_directory = main_output_directory/ "phase_space_results"

    results_output_directory.mkdir(parents = True, exist_ok = True)

    output_file = f"results_potential_{potential_index}_Z{Z}_A{A}.txt"
    results_output_path = results_output_directory /output_file



    plots_output_directory =  main_output_directory/ "plots"
    plots_output_directory.mkdir(parents = True, exist_ok = True)



    T_n, r_n, P_n, Q_n = load_data(directory_name, file_name_kappa_n)
    T_p, r_p, P_p, Q_p = load_data(directory_name, file_name_kappa_p)



    idx_R, mesh_point_R_au = find_mesh_point_R_au(r_n, rN)

    data = {"T":T_n, "P_n":P_n, "Q_p":Q_p, "idx_R":idx_R, "mesh_point_R_au":mesh_point_R_au}

    T_MeV = T_n * E_HARTREE/1e6
    Ee = T_MeV + ME

    plot_f_and_g(Ee, potential_index,cnf, data, plots_output_directory)

    # Optional quick visualization of Fermi function
    x_plot = np.linspace(ME+1e-3, Q + ME, 2000)
    Analytic_fermi_plot = Fermi(x_plot, Z, A, rN)
    Numeric_fermi_plot = Fermi_numerical(x_plot, Z, A, data)


    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8), sharex = True, gridspec_kw={"height_ratios": [3,1]})
    ax1.plot(x_plot, Analytic_fermi_plot, label='Pure Coulomb Fermi Analytical')
    ax1.plot(x_plot, Numeric_fermi_plot , label='Fermi Numerical')

    ax1.set_xlabel('Electron Energy (MeV)')
    ax1.set_ylabel('Fermi Function Value')
    ax1.set_title('Fermi Function vs Electron Energy')
    ax1.grid(True)
    ax1.legend()

    percent_diff_fermi_func = 100.0 * (Numeric_fermi_plot - Analytic_fermi_plot)/ Analytic_fermi_plot

    ax2.plot(x_plot, percent_diff_fermi_func)
    ax2.set_ylabel(f"% difference")
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(plots_output_directory, f"Fermi_Function_potential_{potential_index}_Z{Z}_A{A}_test.png"), dpi = 300)
    plt.show()



    Fermi_analytic = lambda Ee: Fermi(Ee, Z , A, rN)
    Fermi_numeric = lambda Ee: Fermi_numerical(Ee, Z , A, data)
    E_numeric = lambda Ee: E_function(Ee, Z, A, data)

    def bounds_Ee1():
        return [ME, Q + ME]
    def bounds_Ee2(Ee1):
        return [ME, Q + 2.0 * ME - Ee1]

    opts_Ee1 = {"epsabs": 1e-18, "epsrel": 1e-16, "limit": 50000, "points": [ME, Q/2 + ME, Q + ME]}
    opts_Ee2 = {"epsabs": 1e-18, "epsrel": 1e-16, "limit": 50000, "points": [ME]}

    tags = [0, 2, 4, 22]
    G_results_list, G_errors_list = [], []
    G_results_num_list, G_errors_num_list = [] , []
    H_results_num_list, H_errors_num_list = [] , []
    for t in tags:
        G_output = nquad(phase_space_integrand(t, Q, Fermi_analytic), [bounds_Ee2, bounds_Ee1], opts=[opts_Ee2, opts_Ee1])
        G_result, G_error = G_output[:2] #### Needed this extra line to avoid diagnostic warning about wrong output??
        G_results_list.append(G_result); G_errors_list.append(G_error)

        G_output_num = nquad(phase_space_integrand(t, Q, Fermi_numeric), [bounds_Ee2, bounds_Ee1], opts=[opts_Ee2, opts_Ee1])
        G_result_num, G_error_num = G_output_num[:2]
        G_results_num_list.append(G_result_num); G_errors_num_list.append(G_error_num)

        H_output_num = nquad(phase_space_integrand(t, Q, E_numeric),  [bounds_Ee2, bounds_Ee1], opts=[opts_Ee2, opts_Ee1])
        H_result, H_error = H_output_num[:2]
        H_results_num_list.append(H_result); H_errors_num_list.append(H_error)

    G_results = np.asarray(G_results_list)
    G_errors = np.asarray(G_errors_list)

    G_results_num = np.asarray(G_results_num_list)
    G_errors_num = np.asarray(G_errors_num_list)

    H_result_num = np.asarray(H_results_num_list)
    H_errors_num = np.asarray(H_errors_num_list)


    Glit = [1.793e-18, 5.516e-19, 2.110e-19, 4.994e-20]  # literature: G0, G2, G4, G22
    Glit = np.asarray(Glit)


    halflife = 1/(GA ** 4 * MGT1**2 * (G_results[0] + xi31 * G_results[1] + 1/3 * xi31 ** 2 * G_results[3] + (1/3 * xi31**2 + xi51)* G_results[2]))
    halflife_num = 1/(GA ** 4 * MGT1**2 * (G_results_num[0] + xi31 * G_results_num[1] + 1/3 * xi31 ** 2 * G_results_num[3] + (1/3 * xi31**2 + xi51)* G_results_num[2]))




    eps_grid = np.linspace(0.0, Q, 400)
    spectrum_vals = np.array([spectrum_epsilon(eps, Q, Fermi_analytic) for eps in eps_grid])
    spectrum_vals_num = np.array([spectrum_epsilon(eps, Q, Fermi_numeric) for eps in eps_grid])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), sharex = True, gridspec_kw = {"height_ratios": [3,1]})
    ax1.plot(eps_grid, spectrum_vals, lw=1.5, label = "pure Coulomb analytical")
    ax1.plot(eps_grid, spectrum_vals_num, lw = 1.5 ,label = "numerical")
    ax1.set_xlabel('epsilon = Ee1 + Ee2 − 2 me (MeV)')
    ax1.set_ylabel('dΓ/dε (1/yr per MeV)')
    ax1.legend()
    ax1.set_title('2νββ Spectrum vs epsilon')
    ax1.grid(True)

    percent_diff_epsilon_spectrum = 100.0 * (spectrum_vals_num - spectrum_vals) / spectrum_vals

    ax2.plot(eps_grid, percent_diff_epsilon_spectrum)
    ax2.set_ylabel(f"% diff")
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(plots_output_directory, f"Spectrum_potential_{potential_index}_Z{Z}_A{A}_test.png"), dpi = 300)
    plt.show()

    total_rate, total_err = quad(lambda eps: spectrum_epsilon(eps, Q, Fermi_analytic), 0.0, Q,
                                epsabs=1e-18, epsrel=1e-16, limit=20000, points=[0.0, Q/2.0, Q])
    total_rate_num, total_err_num = quad(lambda eps: spectrum_epsilon(eps, Q, Fermi_numeric), 0.0, Q,
                                epsabs=1e-18, epsrel=1e-16, limit=20000, points=[0.0, Q/2.0, Q])

    spec_vals_norm = spectrum_vals/total_rate
    spec_val_norm_num = spectrum_vals_num/total_rate_num

    plt.figure(figsize=(12,8))
    plt.plot(eps_grid, spec_vals_norm, lw = 1.5, label = "analytic")
    plt.plot(eps_grid, spec_val_norm_num, lw = 1.5, label = "numeric")
    plt.xlabel('epsilon = Ee1 + Ee2 − 2 me (MeV)')
    plt.ylabel('1/Γ dΓ/dε')
    plt.legend()
    plt.title("Normalized 2νββ Spectrum")
    plt.grid(True)
    plt.show()

    norm_check = np.trapezoid(spec_vals_norm, eps_grid)
    norm_check_num = np.trapezoid(spec_val_norm_num, eps_grid)
    print(f"analytic norm check: {norm_check:.10g}, numeric norm check: {norm_check_num:.10g}")

    with open(results_output_path, "w") as f:
        scheme = None
        if potential_index == 0:
            scheme = "B"
        elif potential_index == 2:
            scheme = "A"
        elif potential_index == 3:
            scheme = "C"

        f.write(f"### Double Beta Decay Results for {config["isotope"]} using Scheme {scheme} ###\n\n")

        f.write(f"G analytic converted to 1/yr units [G0, G2, G4, G22]:  {G_results} \n") #### Get rid of this, as it only shows Pure coulomb fermi G values
        f.write(f"G numerical converted to 1/yr units [G0, G2, G4, G22]: {G_results_num} \n")
        f.write(f"H numerical ----------------------->[H0, H2, H4, H22]: {H_result_num}\n\n")

        f.write(f"Calculated analytic half life: {halflife: .6e} \n")
        f.write(f"Calculated numeric half life:  {halflife_num: .6e} \n")
        f.write(f"Experimental half life:        {2.19e21: .6e} \n\n")

        f.write(f"Total rate from ε-spectrum [yr]: {1/total_rate} ± {total_err} \n")
        f.write(f"Total rate from ε-spectrum [yr]: {1/total_rate_num} ± {total_err_num} \n")



