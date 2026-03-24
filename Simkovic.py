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


# Got the values from CODATA, probably not used by Simkovic but they are accurate
me = 0.51099895069 # Electron rest mass in MeV/c^2
alpha = 7.2973525643e-3 # Fine-structure constant
GF = 1.1663787e-11 # Fermi coupling constant in MeV^-2
Vud =  0.973735 # CKM matrix element
Gbeta = GF * Vud # Effective weak coupling constant in MeV^-2
c2n = me * (Gbeta * me ** 2)**4 / (8 * np.pi**7 * np.log(2))
hbar_c = 197.3 # MeV*fm
MeVtoyr = (365.25 * 24 * 3600) * 2.998e8 / (1e-15 * hbar_c)
gA = 1.269 # Axial vector coupling constant
MGT1, MGT3, MGT5, xi31, xi51 = 0.0104, 0.00403, 0.00126, 0.3867, 0.1207 # Simkovic NMEs for 136Xe
Z, A = 54, 136 # Atomic number and mass number for Xenon
Z += 2 # Daughter nucleus
rN = 1.2 * A ** (1 / 3) / hbar_c # Nuclear radius
Q = 2.45791 # MeV Q-value for 136Xe
Tlit = 2.19e21 # yr half-life
au = 5.29177210903e-11 #Bohr radius




#### BEGIN MY CODE

def load_data(name):
    npz_file = name

    data = np.load(npz_file)


    T = data["T"]
    r = data["r"]
    P = data["P"]
    Q = data["Q"]
    return T, r, P, Q


T_n, r_n, P_n, Q_n = load_data("Dirac_run_kappa_n_schemeB_V_Z56.npz")
T_n_V0, r_n_V0, P_n_V0, Q_n_V0 = load_data("Dirac_run_kappa_n_schemeB_V0.npz")
T_p, r_p, P_p, Q_p = load_data("Dirac_run_kappa_p_schemeB_V_Z56.npz")
T_p_V0, r_p_V0, P_p_V0, Q_p_V0 = load_data("Dirac_run_kappa_p_schemeB_V0.npz")

T_nA, r_nA, P_nA, Q_nA = load_data("Dirac_run_kappa_n_schemeA_V_Z56.npz")
T_n_V0A, r_n_V0A, P_n_V0A, Q_n_V0A = load_data("Dirac_run_kappa_n_schemeA_V0.npz")
T_pA, r_pA, P_pA, Q_pA = load_data("Dirac_run_kappa_p_schemeA_V_Z56.npz")
T_p_V0A, r_p_V0A, P_p_V0A, Q_p_V0A = load_data("Dirac_run_kappa_p_schemeA_V0.npz")


R_au = rN * hbar_c * 1e-15 /au
idx_R = 0
mesh_point_R_au = 0
for i in range(0,len(r_n)):
    if r_n[i] <= R_au  and r_n[i+1] > R_au:
        mesh_point_R_au = r_n[i]
        idx_R = i
        print(f"closest mesh point to R_au = {R_au: .5g} is at i ={idx_R} w/ mesh point value ={mesh_point_R_au: .5g}")
        break


Fermi_vals = abs((P_n[:, idx_R]**2 + Q_p[:, idx_R]**2) / (P_n_V0[:, idx_R]**2 + Q_p_V0[:, idx_R]**2))
Fermi_valsA = abs((P_nA[:, idx_R]**2 + Q_pA[:, idx_R]**2) / (P_n_V0A[:, idx_R]**2 + Q_p_V0A[:, idx_R]**2))



E_hatree = 27.211386
T_MeV = T_n * E_hatree/1e6


print(f"T range numerical: ", T_MeV[0], T_MeV[-1])
print(f"1st 5 Fermi vals:", Fermi_vals[:5])
Ee_grid = T_MeV + me

def Fermi_numerical(Ee):
    s = np.sqrt(1-(56/137)**2)
    Ee = np.asarray(Ee, dtype = float)
    T = Ee - me

    out = np.empty_like(T, dtype = float)


    return 2/(s+1) * np.interp(T, T_MeV, Fermi_vals)

#### END MY CODE

def Fermi(Ee):
    # Guard against Ee slightly below me due to numerical issues
    Ee = np.asarray(Ee)
    p2 = Ee**2 - me**2
    # Set negative p2 to a tiny positive to avoid NaNs exactly at the threshold
    p2 = np.where(p2 > 0, p2, np.finfo(float).tiny)
    p = np.sqrt(p2)
    y = alpha * Z * Ee / p
    gamma0 = np.sqrt(1 - (alpha * Z)**2)
    F = (sp.gamma(3) / (sp.gamma(1) * sp.gamma(1 + 2 * gamma0)))**2
    F *= (2 * p * rN)**(2 * (gamma0 - 1)) * np.exp(np.pi * y)
    F *= np.abs(sp.gamma(gamma0 + 1j * y))**2
    return F



# Optional quick visualization of Fermi function
plt.plot(np.linspace(me+1e-3, Q + me, 2000), Fermi(np.linspace(me+1e-3, Q + me, 2000)), label='Fermi Analytical')
plt.plot(np.linspace(me+1e-3, Q + me, 2000), Fermi_numerical(np.linspace(me+1e-3, Q + me, 2000)), label='Fermi Numerical')

plt.xlabel('Electron Energy (MeV)')
plt.ylabel('Fermi Function Value')
plt.title('Fermi Function vs Electron Energy')
plt.legend()
plt.show()

Glit = [1.793e-18, 5.516e-19, 2.110e-19, 4.994e-20]  # literature: G0, G2, G4, G22
def integrand_factory(tag):
    def f(Ee2, Ee1):
        if not (me <= Ee1 <= Q + me): return 0.0
        ub = Q + 2.0 * me - Ee1
        if not (me <= Ee2 <= ub): return 0.0

        p1_sq, p2_sq = Ee1*Ee1 - me*me, Ee2*Ee2 - me*me
        if p1_sq <= 0.0 or p2_sq <= 0.0: return 0.0
        p1, p2 = np.sqrt(p1_sq), np.sqrt(p2_sq)
        F1, F2 = Fermi(Ee1), Fermi(Ee2)

        x, y = 2.0*me + Q - Ee1 - Ee2, Ee1 - Ee2
        if tag == 0:
            poly = x**5 / 30.0
        elif tag == 2:
            poly = x**5 * (x*x + 7*y*y) / (420.0 * (2*me)**2)
        elif tag == 4:
            poly = x**5 * (x**4 + 18*x*x*y*y + 21*y**4) / (5040.0 * (2*me)**4)
        else:  # tag == 22
            poly = x**5 * (x**4 - 6*x*x*y*y + 21*y**4) / (10080.0 * (2*me)**4)

        return float(MeVtoyr * c2n * F1 * F2 * p1 * Ee1 * p2 * Ee2 * poly / (me**11))
    return f

#### BEGIN MY CODE
def integrand_factory_numerical(tag):
    def f(Ee2, Ee1):
        if not (me <= Ee1 <= Q + me): return 0.0
        ub = Q + 2.0 * me - Ee1
        if not (me <= Ee2 <= ub): return 0.0

        p1_sq, p2_sq = Ee1*Ee1 - me*me, Ee2*Ee2 - me*me
        if p1_sq <= 0.0 or p2_sq <= 0.0: return 0.0
        p1, p2 = np.sqrt(p1_sq), np.sqrt(p2_sq)
        F1, F2 = Fermi_numerical(Ee1), Fermi_numerical(Ee2)

        x, y = 2.0*me + Q - Ee1 - Ee2, Ee1 - Ee2
        if tag == 0:
            poly = x**5 / 30.0
        elif tag == 2:
            poly = x**5 * (x*x + 7*y*y) / (420.0 * (2*me)**2)
        elif tag == 4:
            poly = x**5 * (x**4 + 18*x*x*y*y + 21*y**4) / (5040.0 * (2*me)**4)
        else:  # tag == 22
            poly = x**5 * (x**4 - 6*x*x*y*y + 21*y**4) / (10080.0 * (2*me)**4)

        return float(MeVtoyr * c2n * F1 * F2 * p1 * Ee1 * p2 * Ee2 * poly / (me**11))
    return f

#### END MY CODE

def bounds_Ee1(): return [me, Q + me]
def bounds_Ee2(Ee1): return [me, Q + 2.0 * me - Ee1]

opts_Ee1 = {"epsabs": 1e-18, "epsrel": 1e-16, "limit": 50000, "points": [me, Q/2 + me, Q + me]}
opts_Ee2 = {"epsabs": 1e-18, "epsrel": 1e-16, "limit": 50000, "points": [me]}

tags = [0, 2, 4, 22]
results, errors = [], []
results_num, errors_num = [] , [] #### NEW LINE
for t in tags:
    r, e = nquad(integrand_factory(t), [bounds_Ee2, bounds_Ee1], opts=[opts_Ee2, opts_Ee1])
    results.append(r); errors.append(e)
    #### NEW LINE
    rn, en = nquad(integrand_factory_numerical(t), [bounds_Ee2, bounds_Ee1], opts=[opts_Ee2, opts_Ee1])
    results_num.append(rn); errors_num.append(en)

results = np.asarray(results)
errors = np.asarray(errors)

#### NEW LINES
results_num = np.asarray(results_num)
errors_num = np.asarray(errors_num)

Glit = np.asarray(Glit)

print("Converted to 1/yr units [G0, G2, G4, G22]:", results)
print("Numerical [G0, G2, G4, G22]:" , results_num)
print("Literature values:", Glit)
print("Relative differences:", 100*(results - Glit) / Glit)
#### NEW LINES
print("Relative difference numerical vs analytical", 100*(results_num - results)/ results)
print("Relative difference numerical vs literature", 100*(results_num - Glit)/ Glit)
# You can use the values from Simkovic's paper directly

halflife = 1/(gA ** 4 * MGT1**2 * (results[0] + xi31 * results[1] + 1/3 * xi31 ** 2 * results[3] + (1/3 * xi31**2 + xi51)* results[2]))
#### NEW LINE
halflife_num = 1/(gA ** 4 * MGT1**2 * (results_num[0] + xi31 * results_num[1] + 1/3 * xi31 ** 2 * results_num[3] + (1/3 * xi31**2 + xi51)* results_num[2]))

print("Calculation Results", halflife)
#### NEW LINE
print("Numerical Results", halflife_num)
print("Experimental Measurement", 2.19e21)

def energies_eps_D(eps, D):
    Ee1 = me + D + 0.5*eps; Ee2 = me - D + 0.5*eps
    return Ee1, Ee2

def phase_combo(Qminus_eps, D):
    x, y = Qminus_eps, 2*D  # y = Ee1 - Ee2 = 2Δ
    A0  = x**5 / 30.0
    A2  = x**5 * (x*x + 7*y*y) / (420.0 * (2.0*me)**2)
    A4  = x**5 * (x**4 + 18.0*x*x*y*y + 21.0*y**4) / (5040.0 * (2.0*me)**4)
    A22 = x**5 * (x**4 - 6.0*x*x*y*y + 21.0*y**4) / (10080.0 * (2.0*me)**4)
    return A0 + A2*xi31 + A4*(xi31**2/3.0 + xi51) + A22*(xi31**2/3.0)

def integrand_eps_D(eps, D):
    if eps < 0.0 or eps > Q or D < -eps/2.0 or D > eps/2.0: return 0.0
    Ee1, Ee2 = energies_eps_D(eps, D)
    if Ee1 < me or Ee2 < me: return 0.0
    p1_sq, p2_sq = Ee1*Ee1 - me*me, Ee2*Ee2 - me*me
    if p1_sq <= 0.0 or p2_sq <= 0.0: return 0.0
    p1, p2 = np.sqrt(p1_sq), np.sqrt(p2_sq)
    F1, F2 = Fermi(Ee1), Fermi(Ee2)
    poly = phase_combo(Q - eps, D)
    return float(gA ** 4 * MGT1**2 *MeVtoyr * c2n * F1 * F2 * p1 * Ee1 * p2 * Ee2 * poly / (me**11))

#### NEW LINES
def integrand_eps_D_num(eps, D):
    if eps < 0.0 or eps > Q or D < -eps/2.0 or D > eps/2.0: return 0.0
    Ee1, Ee2 = energies_eps_D(eps, D)
    if Ee1 < me or Ee2 < me: return 0.0
    p1_sq, p2_sq = Ee1*Ee1 - me*me, Ee2*Ee2 - me*me
    if p1_sq <= 0.0 or p2_sq <= 0.0: return 0.0
    p1, p2 = np.sqrt(p1_sq), np.sqrt(p2_sq)
    F1, F2 = Fermi_numerical(Ee1), Fermi_numerical(Ee2)
    poly = phase_combo(Q - eps, D)
    return float(gA ** 4 * MGT1**2 *MeVtoyr * c2n * F1 * F2 * p1 * Ee1 * p2 * Ee2 * poly / (me**11))

def spectrum_epsilon(eps):
    if eps < 0.0 or eps > Q: return 0.0
    res, _ = quad(lambda D: integrand_eps_D(eps, D), -eps/2.0, eps/2.0,
                  epsabs=1e-16, epsrel=1e-14, limit=20000, points=[-eps/2.0, 0.0, eps/2.0])
    return res

#### NEW LINES
def spectrum_epsilon_num(eps):
    if eps < 0.0 or eps > Q: return 0.0
    res, _ = quad(lambda D: integrand_eps_D_num(eps, D), -eps/2.0, eps/2.0,
                  epsabs=1e-16, epsrel=1e-14, limit=20000, points=[-eps/2.0, 0.0, eps/2.0])
    return res

eps_grid = np.linspace(0.0, Q, 400)
spec_vals = np.array([spectrum_epsilon(eps) for eps in eps_grid])
#### NEW LINE
spec_vals_num = np.array([spectrum_epsilon_num(eps) for eps in eps_grid])

# percent_diff = 100.0 * (spec_vals_num - spec_vals)/ spec_vals
#
# fig, (ax1, ax2) = plt.subplots(2,1, figsize = (12,8), sharex = True, gridspec_kw ={"height_ratios": [3, 1]})
#
# ax1.plot(eps_grid, spec_vals, lw=1.5, label = "analytical")
# ax1.plot(eps_grid, spec_vals_num, lw = 1.5 ,label = "numerical")
# ax1.set_title('2νββ Spectrum vs epsilon')
# ax1.set_ylabel('dΓ/dε (1/yr per MeV)')
# ax1.grid(True)
# ax1.legend()
#
# ax2. plot(eps_grid, percent_diff)
# ax2.axhline(0.0, color= "black", lw= 1)
# ax2.set_ylabel(f'% difference')
# ax2.set_xlabel('epsilon = Ee1 + Ee2 − 2 me (MeV)')
# ax2.grid(True)
#
# plt.tight_layout()
# plt.show()
#
plt.figure(figsize=(8,5))
plt.plot(eps_grid, spec_vals, lw=1.5, label = "analytical")
plt.plot(eps_grid, spec_vals_num, lw = 1.5 ,label = "numerical")
plt.xlabel('epsilon = Ee1 + Ee2 − 2 me (MeV)')
plt.ylabel('dΓ/dε (1/yr per MeV)')
plt.legend()
plt.title('2νββ Spectrum vs epsilon')
plt.grid(True); plt.show()

total_rate, total_err = quad(lambda eps: spectrum_epsilon(eps), 0.0, Q,
                             epsabs=1e-18, epsrel=1e-16, limit=20000, points=[0.0, Q/2.0, Q])
print("Total rate from ε-spectrum [yr]:", 1/total_rate, "±", total_err)
