#!/usr/bin/env python

# Generated from: Simkovic.ipynb
# Converted at: 2026-02-28T13:04:04.740Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from scipy.integrate import nquad
from scipy.integrate import quad
from scipy.interpolate import interp1d
from pathlib import Path
import os
from physics_constants import ME, ALPHA, GF, VUD, HBAR_C, GA, AU, C, E_HARTREE


Gbeta = GF * VUD # Effective weak coupling constant in MeV^-2
c2n = ME * (Gbeta * ME ** 2)**4 / (8 * np.pi**7 * np.log(2))
MeVtoyr = (365.25 * 24 * 3600) * 2.998e8 / (1e-15 * HBAR_C)
MGT1, MGT3, MGT5, xi31, xi51 = 0.0104, 0.00403, 0.00126, 0.3867, 0.1207 # Simkovic NMEs for 136Xe



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
    for i in range(0,len(mesh_points)):
        if mesh_points[i] <= R_au  and mesh_points[i+1] > R_au:
            mesh_point_R_au = mesh_points[i]
            idx_R = i
            print(f"closest mesh point to R_au = {R_au: .5g} is at i ={idx_R} w/ mesh point value ={mesh_point_R_au: .5g}")
            break
    return idx_R, mesh_point_R_au



def Fermi_numerical(Ee, Z, A, data):
    s = np.sqrt(1-(Z/A)**2)
    Ee = np.asarray(Ee, dtype = float)
    T = Ee - ME


    #### FERMI FUNCTION METHOD 1
    # Fermi_vals = abs((P_n[:, idx_R]**2 + Q_p[:, idx_R]**2) / (P_n_V0[:, idx_R]**2 + Q_p_V0[:, idx_R]**2))

    T_mesh = data["T"]
    P_n = data["P_n"]
    Q_p = data["Q_p"]
    idx_R = data["idx_R"]
    mesh_point_R_au = data["mesh_point_R_au"]


    T_MeV = T_mesh * E_HARTREE/1e6
    Ee = T_MeV + ME
    p_au = np.sqrt(T_mesh * (T_mesh + 2*C**2))/C


    g_B = np.sqrt((Ee + ME)/(2*Ee)) * P_n[:,idx_R] /(p_au * mesh_point_R_au)
    f_B = np.sqrt((Ee + ME)/(2*Ee)) * Q_p[:,idx_R] / (p_au * mesh_point_R_au)

    Fermi_vals = (g_B**2 + f_B**2)


    return 2/(s+1) * np.interp(T, T_MeV, Fermi_vals)


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
    return F



def integrand_factory(tag, Q, fermi_func):
    def f(Ee2, Ee1):
        if not (ME <= Ee1 <= Q + ME): return 0.0
        ub = Q + 2.0 * ME - Ee1
        if not (ME <= Ee2 <= ub): return 0.0

        p1_sq, p2_sq = Ee1*Ee1 - ME*ME, Ee2*Ee2 - ME*ME
        if p1_sq <= 0.0 or p2_sq <= 0.0: return 0.0
        p1, p2 = np.sqrt(p1_sq), np.sqrt(p2_sq)
        F1, F2 = fermi_func(Ee1), fermi_func(Ee2)

        x, y = 2.0*ME + Q - Ee1 - Ee2, Ee1 - Ee2
        if tag == 0:
            poly = x**5 / 30.0
        elif tag == 2:
            poly = x**5 * (x*x + 7*y*y) / (420.0 * (2*ME)**2)
        elif tag == 4:
            poly = x**5 * (x**4 + 18*x*x*y*y + 21*y**4) / (5040.0 * (2*ME)**4)
        else:  # tag == 22
            poly = x**5 * (x**4 - 6*x*x*y*y + 21*y**4) / (10080.0 * (2*ME)**4)

        return float(MeVtoyr * c2n * F1 * F2 * p1 * Ee1 * p2 * Ee2 * poly / (ME**11))
    return f




def energies_eps_D(eps, D):
    Ee1 = ME + D + 0.5*eps; Ee2 = ME - D + 0.5*eps
    return Ee1, Ee2

def phase_combo(Qminus_eps, D):
    x, y = Qminus_eps, 2*D  # y = Ee1 - Ee2 = 2Δ
    A0  = x**5 / 30.0
    A2  = x**5 * (x*x + 7*y*y) / (420.0 * (2.0*ME)**2)
    A4  = x**5 * (x**4 + 18.0*x*x*y*y + 21.0*y**4) / (5040.0 * (2.0*ME)**4)
    A22 = x**5 * (x**4 - 6.0*x*x*y*y + 21.0*y**4) / (10080.0 * (2.0*ME)**4)
    return A0 + A2*xi31 + A4*(xi31**2/3.0 + xi51) + A22*(xi31**2/3.0)

def integrand_eps_D(eps, D, Q, fermi_func):
    if eps < 0.0 or eps > Q or D < -eps/2.0 or D > eps/2.0: return 0.0
    Ee1, Ee2 = energies_eps_D(eps, D)
    if Ee1 < ME or Ee2 < ME: return 0.0
    p1_sq, p2_sq = Ee1*Ee1 - ME*ME, Ee2*Ee2 - ME*ME
    if p1_sq <= 0.0 or p2_sq <= 0.0: return 0.0
    p1, p2 = np.sqrt(p1_sq), np.sqrt(p2_sq)
    F1, F2 = fermi_func(Ee1), fermi_func(Ee2)
    poly = phase_combo(Q - eps, D)
    return float(GA ** 4 * MGT1**2 *MeVtoyr * c2n * F1 * F2 * p1 * Ee1 * p2 * Ee2 * poly / (ME**11))

def spectrum_epsilon(eps, Q, fermi_func):
    if eps < 0.0 or eps > Q: return 0.0
    res, _ = quad(lambda D: integrand_eps_D(eps, D, Q, fermi_func), -eps/2.0, eps/2.0,
                  epsabs=1e-16, epsrel=1e-14, limit=20000, points=[-eps/2.0, 0.0, eps/2.0])
    return res


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

    # Optional quick visualization of Fermi function
    plt.plot(np.linspace(ME+1e-3, Q + ME, 2000), Fermi(np.linspace(ME+1e-3, Q + ME, 2000), Z, A, rN), label='Fermi Analytical')
    plt.plot(np.linspace(ME+1e-3, Q + ME, 2000), Fermi_numerical(np.linspace(ME+1e-3, Q + ME, 2000), Z, A, data), label='Fermi Numerical')

    plt.xlabel('Electron Energy (MeV)')
    plt.ylabel('Fermi Function Value')
    plt.title('Fermi Function vs Electron Energy')
    plt.legend()
    plt.savefig(os.path.join(plots_output_directory, f"Fermi_Function_potential_{potential_index}_Z{Z}_A{A}.png"), dpi = 300)
    plt.show()

    Fermi_analytic = lambda Ee: Fermi(Ee, Z , A, rN)
    Fermi_numeric = lambda Ee: Fermi_numerical(Ee, Z , A, data)

    def bounds_Ee1(): return [ME, Q + ME]
    def bounds_Ee2(Ee1): return [ME, Q + 2.0 * ME - Ee1]

    opts_Ee1 = {"epsabs": 1e-18, "epsrel": 1e-16, "limit": 50000, "points": [ME, Q/2 + ME, Q + ME]}
    opts_Ee2 = {"epsabs": 1e-18, "epsrel": 1e-16, "limit": 50000, "points": [ME]}

    tags = [0, 2, 4, 22]
    results, errors = [], []
    results_num, errors_num = [] , [] #### NEW LINE
    for t in tags:
        r, e = nquad(integrand_factory(t, Q, Fermi_analytic), [bounds_Ee2, bounds_Ee1], opts=[opts_Ee2, opts_Ee1])
        results.append(r); errors.append(e)

        rn, en = nquad(integrand_factory(t, Q, Fermi_numeric), [bounds_Ee2, bounds_Ee1], opts=[opts_Ee2, opts_Ee1])
        results_num.append(rn); errors_num.append(en)

    results = np.asarray(results)
    errors = np.asarray(errors)

    #### NEW LINES
    results_num = np.asarray(results_num)
    errors_num = np.asarray(errors_num)

    Glit = [1.793e-18, 5.516e-19, 2.110e-19, 4.994e-20]  # literature: G0, G2, G4, G22
    Glit = np.asarray(Glit)


    halflife = 1/(GA ** 4 * MGT1**2 * (results[0] + xi31 * results[1] + 1/3 * xi31 ** 2 * results[3] + (1/3 * xi31**2 + xi51)* results[2]))
    #### NEW LINE
    halflife_num = 1/(GA ** 4 * MGT1**2 * (results_num[0] + xi31 * results_num[1] + 1/3 * xi31 ** 2 * results_num[3] + (1/3 * xi31**2 + xi51)* results_num[2]))




    eps_grid = np.linspace(0.0, Q, 400)
    spec_vals = np.array([spectrum_epsilon(eps, Q, Fermi_analytic) for eps in eps_grid])
    spec_vals_num = np.array([spectrum_epsilon(eps, Q, Fermi_numeric) for eps in eps_grid])

    plt.figure(figsize=(8,5))
    plt.plot(eps_grid, spec_vals, lw=1.5, label = "analytical")
    plt.plot(eps_grid, spec_vals_num, lw = 1.5 ,label = "numerical")
    plt.xlabel('epsilon = Ee1 + Ee2 − 2 me (MeV)')
    plt.ylabel('dΓ/dε (1/yr per MeV)')
    plt.legend()
    plt.title('2νββ Spectrum vs epsilon')
    plt.grid(True)
    plt.savefig(os.path.join(plots_output_directory, f"Spectrum_potential_{potential_index}_Z{Z}_A{A}.png"), dpi = 300)
    plt.show()

    total_rate, total_err = quad(lambda eps: spectrum_epsilon(eps, Q, Fermi_analytic), 0.0, Q,
                                epsabs=1e-18, epsrel=1e-16, limit=20000, points=[0.0, Q/2.0, Q])
    total_rate_num, total_err_num = quad(lambda eps: spectrum_epsilon(eps, Q, Fermi_numeric), 0.0, Q,
                                epsabs=1e-18, epsrel=1e-16, limit=20000, points=[0.0, Q/2.0, Q])



    with open(results_output_path, "w") as f:
        scheme = None
        if potential_index == 0:
            scheme = "B"
        elif potential_index == 2:
            scheme = "A"
        elif potential_index == 3:
            scheme = "C"

        f.write(f"### Double Beta Decay Results for Z = {Z}, A = {A} using Scheme {scheme} ###\n\n")

        f.write(f"G analytic converted to 1/yr units [G0, G2, G4, G22]:  {results} \n")
        f.write(f"G numerical converted to 1/yr units [G0, G2, G4, G22]: {results_num} \n")


        f.write(f"Calculated analytic half life: {halflife: .6e} \n")
        f.write(f"Calculated numeric half life:  {halflife_num: .6e} \n")
        f.write(f"Experimental half life:        {2.19e21: .6e} \n\n")

        f.write(f"Total rate from ε-spectrum [yr]: {1/total_rate} ± {total_err} \n")
        f.write(f"Total rate from ε-spectrum [yr]: {1/total_rate_num} ± {total_err_num} \n")



