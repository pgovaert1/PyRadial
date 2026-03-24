#!/usr/bin/env python


####
#   I ADDED IN THIS LINE TO MAKE SURE EVERYTHING IS WORKING PROPERLY
####



from matplotlib.typing import LineStyleType
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize._lsq.lsq_linear import TERMINATION_MESSAGES
import scipy.special as sp
import mpmath as mp


E_hatree = 27.211386 #eV
c = 137.03599
alpha = 1.0/c


au = 5.29177210903e-11 #Bohr radius
Rn = 6.17e-15 #Nuclear radius Simkovic

R_au = Rn/au
Fermi = 1e-15/au
# R_au = 1e-6print(f"R (a.u) = {R_au}")
print(f"Fermi (a.u) = {Fermi}")

Z_sing = -56

def potential(r):
    return alpha * Z_sing/r


hbar_c = 197.3 # MeV*fm
A = 136
rN = 1.2e-15 * A ** (1 / 3)
R_simkovic = 1.2 * A ** (1 / 3) / hbar_c
R_au = rN/au

print(f"R (a.u) = {R_au}")
print(f"V(R_au) = {potential(R_au)}")


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

T_n1, r_n1, P_n1, Q_n1 = load_data("Dirac_run_kappa_n_schemeA_V_Z56.npz")
T_n_V01, r_n_V01, P_n_V01, Q_n_V01 = load_data("Dirac_run_kappa_n_schemeA_V0.npz")
T_p1, r_p1, P_p1, Q_p1 = load_data("Dirac_run_kappa_p_schemeA_V_Z56.npz")
T_p_V01, r_p_V01, P_p_V01, Q_p_V01 = load_data("Dirac_run_kappa_p_schemeA_V0.npz")


T_MeV = T_n * E_hatree/1e6


aT_n, ar_n, aP_n, aQ_n = load_data("Analytic_Coulomb_kappa_n_V_Z56.npz")
aT_n_V0, ar_n_V0, aP_n_V0, aQ_n_V0 = load_data("Analytic_Coulomb_kappa_n_V0.npz")
aT_p, ar_p, aP_p, aQ_p = load_data("Analytic_Coulomb_kappa_p_V_Z56.npz")
aT_p_V0, ar_p_V0, aP_p_V0, aQ_p_V0 = load_data("Analytic_Coulomb_kappa_p_V0.npz")



idx_R = a_idx_R = 0
mesh_point_R_au = a_mesh_point_R_au = 0
for i in range(0,len(r_n)):
    if r_n[i] <= R_au  and r_n[i+1] > R_au:
        mesh_point_R_au = r_n[i]
        idx_R = i
        print(f"Numeric: closest mesh point to R_au = {R_au: .5g} is at i ={idx_R} w/ mesh point value ={mesh_point_R_au: .5g}")
        break


for i in range(0,len(ar_n)):
    if ar_n[i] <= R_au  and ar_n[i+1] > R_au:
        a_mesh_point_R_au = ar_n[i]
        a_idx_R = i
        print(f"Analytic: closest mesh point to R_au = {R_au: .5g} is at i ={a_idx_R} w/ mesh point value ={a_mesh_point_R_au: .5g}")
        break


# Fermi_num = (P_n[:, idx_R]**2 * cos^2(delta)+ Q_p[:, idx_R]**2) * sin^2(delta)
Fermi_num = (P_n[:, idx_R]**2 + Q_p[:, idx_R]**2)
Fermi_den = (P_n_V0[:, idx_R]**2 + Q_p_V0[:, idx_R]**2)

Fermi_numA = (P_n1[:, idx_R]**2 + Q_p1[:, idx_R]**2)
Fermi_denA = (P_n_V01[:, idx_R]**2 + Q_p_V01[:, idx_R]**2)


s = np.sqrt(1-(56/137)**2)
Fermi = 2/(s+1)*abs(Fermi_num / Fermi_den)
FermiA = abs(Fermi_numA/ Fermi_denA)
Fermi_analytic = 2/(s+1) * abs((aP_n[:, a_idx_R]**2 + aQ_p[:, a_idx_R]**2) / (aP_n_V0[:, a_idx_R]**2 + aQ_p_V0[:, a_idx_R]**2))





### interpolate P and Q for point R_au
aP_n_R = np.array([np.interp(R_au, r_n, P_n[i]) for i in range(len(T_n))])
aP_n_V0_R = np.array([np.interp(R_au, r_n, P_n_V0[i]) for i in range(len(T_n))])
aQ_n_R = np.array([np.interp(R_au, r_n ,Q_n[i]) for i in range(len(T_n))])
aQ_n_V0_R = np.array([np.interp(R_au, r_n ,Q_n_V0[i]) for i in range(len(T_n))])


aP_p_R = np.array([np.interp(R_au, r_p, P_p[i]) for i in range(len(T_p))])
aP_p_V0_R = np.array([np.interp(R_au, r_p, P_p_V0[i]) for i in range(len(T_p))])
aQ_p_R = np.array([np.interp(R_au, r_p ,Q_p[i]) for i in range(len(T_p))])
aQ_p_V0_R = np.array([np.interp(R_au, r_p ,Q_p_V0[i]) for i in range(len(T_p))])


Fermi2 = 2/(s+1) * abs((aP_n_R**2 + aQ_p_R**2) / (aP_n_V0_R**2 + aQ_p_V0_R**2))

#### Interpolate density rho for R_au
# Fermi = np.zeros(len(T_n))
#
# for i in range(len(T_n)):
#
#     rho_num = P_n[i]**2 + Q_p[i]**2
#     rho_den = P_n_V0[i]**2 + Q_p_V0[i]**2
#
#     rho_num_R = np.interp(R_au, r_n, rho_num)
#     rho_den_R = np.interp(R_au, r_n_V0, rho_den)
#
#     Fermi[i] = abs(rho_num_R/rho_den_R)







# Fermi_interp = interp1d(T_MeV, Fermi, kind = "cubic", bounds_error= False, fill_value = "extrapolate")


# E_range = np.linspace(T_MeV[0],T_MeV[-1], 1000)
# Fermi_range = np.zeros(len(E_range))
# for i in range(len(E_range)):
#     Fermi_range[i] = Fermi_interp(E_range[i])




####
me = 0.51099895069
hbar_c = 197.3 # MeV*fm
A = 136
rN = 1.2 * A ** (1 / 3) / hbar_c
print(f"rN = {rN}")
Z = 56
kappa = -1

def Fermi_Analytical(Ee):
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
    F *= np.abs(sp.gamma(gamma0 + 1j * y))**2*(gamma0+1)/2
    return F




#### g_kappa comparison


def gk_analytic(Ee,r):
    out = np.empty_like(Ee, dtype = float)

    for i , E in enumerate(Ee):
        p = np.sqrt(E**2 - me**2)
        # p = np.sqrt((E-me) * (E-me + 2*c**2))/c
        eta = alpha* Z* E/ p
        gammak = np.sqrt(kappa** 2 - (alpha * Z)**2)

        numerator = np.abs(sp.gamma(1 + gammak + 1j * eta))
        denominator = sp.gamma(1 + 2 * gammak)
        gamma_ratio = numerator/denominator

        sqrt_term = np.sqrt((E + me)/ (2 * E))

        exp_zeta = np.sqrt((kappa - 1j * eta * me/E) / (gammak - 1j * eta))


        hyp = mp.hyp1f1(gammak - 1j*eta, 1 + 2*gammak, -2*1j*p*r)
        Im_term = np.imag(np.exp(1j*p*r) * exp_zeta * hyp )

        out[i] = np.sign(kappa) * 1.0/(p*r) * sqrt_term * gamma_ratio * (2*p*r)**gammak * np.exp(np.pi * eta/2) * Im_term

    return out

Ee = T_MeV + me
p_au = np.sqrt(T_n * (T_n + 2*c**2))/c
g_n = abs((P_n[:, idx_R]))/mesh_point_R_au



g_B = np.sqrt((Ee + me)/(2*Ee)) * P_n[:,idx_R] /(p_au * mesh_point_R_au)
f_B = np.sqrt((Ee + me)/(2*Ee)) * Q_p[:,idx_R] / (p_au * mesh_point_R_au)

Fermi_B = 2/(s+1)*(g_B**2 + f_B **2)


g_A = np.sqrt((Ee + me)/(2*Ee)) * P_n1[:,idx_R] /(p_au * mesh_point_R_au)
f_A = np.sqrt((Ee + me)/(2*Ee)) * Q_p1[:,idx_R] / (p_au * mesh_point_R_au)

Fermi_A = 2/(s+1) * (g_A**2 + f_A**2)



#### g{-1} Scheme A

Simkovic_gk_A = np.sqrt(Fermi_Analytical(Ee)) * np.sqrt((Ee + me)/(2*Ee))


plt.figure(figsize = (12,8))
# plt.plot(T_MeV, g_n, marker = "o", linestyle = "-", label = r"$g_{\kappa=-1}(R)$")
plt.plot(T_MeV, gk_analytic(T_MeV + me, R_simkovic), lw = 2.0 ,label = "Simkovic g_{-1}B")
plt.plot(T_MeV, abs(g_B), linestyle = "--", lw= 2.0 ,label = "g_B")

plt.plot(T_MeV, abs(g_A), linestyle = "--", lw = 2.0, label = "g_A")
plt.plot(T_MeV, Simkovic_gk_A, lw = 2.0, label = "Simkovic g_{-1}A")


plt.xlim(T_MeV[0], T_MeV[-1])
plt.ylim(0,20)
plt.xscale("log")
plt.title("eta = -alpha*Z*W/(k*c)")
plt.xlabel(r"$T$ (Mev)")
plt.ylabel(r"$g_{\kappa=-1}(R)$")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()


# plt.figure(figsize = (12,8))
# plt.plot(T_MeV[1:], Fermi[1:], marker = "o", linestyle = "-", label = f"Numerical @ R = {mesh_point_R_au: .5g}")
# plt.plot(T_MeV[1:], Fermi_Analytical(T_MeV + me)[1:], label = f"Analytical @ R = {rN*hbar_c*1e-15/au: .5g} ")
# plt.title("Fermi function ")
# plt.xlabel(r"$T$ (Mev)")
# plt.ylabel(r"Fermi")
# plt.grid(True)
# plt.legend()
# plt.show()


#### Numerator vs denominator
# plt.figure(figsize=(12,8))
# plt.plot(T_MeV, Fermi_num, label = "numerator")
# plt.plot(T_MeV, Fermi_den, label = "denominator")
# plt.title("Fermi function num vs denominator comparison")
# plt.xlabel(r"$T$ (Mev)")
# plt.ylabel(r"Fermi")
# plt.grid(True)
# plt.legend()
# plt.show()



F_analytic = Fermi_Analytical(T_MeV + me)

# Fermi_std *= F_analytic/Fermi_std

percent_diff_numeric = 100.0 * (Fermi - F_analytic)/F_analytic
percent_diff_analytic = 100.0 * (Fermi2 - F_analytic) / F_analytic
percent_diff_analytic_A = 100.0 * (FermiA - F_analytic) / F_analytic

percent_diff_B = 100.0 * (Fermi_B - F_analytic)/ F_analytic
percent_diff_A = 100.0 * (Fermi_A - F_analytic) / F_analytic




fig, (ax1, ax2) = plt.subplots(2 , 1, figsize =(12,8) , sharex = True, gridspec_kw ={"height_ratios": [3, 1]})

ax1.plot(T_MeV[:], Fermi[:], marker = "o", linestyle = "-", label = f"Numerical B @ R = {mesh_point_R_au: .5g}")
ax1.plot(T_MeV[:], F_analytic[:], label = "Saad ")
ax1.plot(T_MeV, Fermi_B, label = f"Fermi B @ R = {mesh_point_R_au: .5g}")


# ax1.plot(T_MeV, Fermi_A, label = f"Fermi A @ R = {mesh_point_R_au: .5g}")

# ax1.plot(T_MeV[:], Fermi2[:], label = f"analytic via code @ R = {a_mesh_point_R_au: .5g}")
# ax1.plot(T_MeV, FermiA, label = f"Numerical A @ R = {mesh_point_R_au: .5g}")
ax1.set_title("Fermi function ")
ax1.set_ylabel(r"Fermi")
ax1.grid(True)
ax1.legend()


ax2.plot(T_MeV[:], percent_diff_numeric[:], label = "Numeric B vs literature Fermi")
ax2.plot(T_MeV, percent_diff_B, label = "Numeric scaled B vs literature Fermi")
# ax2.plot(T_MeV, percent_diff_A, label =  "Numeric scaled A vs literature Fermi")


# ax2.plot(T_MeV[:], percent_diff_analytic[:], color = "green", label = "Analytic vs literature Fermi")
# ax2.plot(T_MeV, percent_diff_analytic_A, label = "Numeric A vs literature Fermi")
# ax2.axhline(0.0, color = "black", linewidth =1)
ax2.set_xlabel(r"$T$ (Mev)")
ax2.set_ylabel(f"% diff")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()











