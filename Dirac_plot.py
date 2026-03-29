#!/usr/bin/env python



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



Fortran_x = [
    7.4045963689e-04, 9.7931438035e-04, 1.2952184354e-03, 1.7130255935e-03,
    2.2656074811e-03, 2.9964389133e-03, 3.9630199977e-03, 5.2413973063e-03,
    6.9321495640e-03, 9.1682996313e-03, 1.2125780003e-02, 1.6037272520e-02,
    2.1210521632e-02, 2.8052544364e-02, 3.7101636356e-02, 4.9069748146e-02,
    6.4898502537e-02, 8.5833255819e-02, 1.1352103155e-01, 1.5014026446e-01,
    1.9857201445e-01, 2.6262669191e-01, 3.4734381830e-01, 4.5938887707e-01,
    6.0757718718e-01, 8.0356719484e-01, 1.0627790483e+00, 1.4056068650e+00,
    1.8590228428e+00, 2.4587000204e+00
]

Fortran_y = [
    1.3088801778e+02, 1.1383543144e+02, 9.9010968359e+01, 8.6124454479e+01,
    7.4923641685e+01, 6.5189273322e+01, 5.6730780664e+01, 4.9382548386e+01,
    4.3000662550e+01, 3.7460083005e+01, 3.2652203890e+01, 2.8482808111e+01,
    2.4870420679e+01, 2.1745002527e+01, 1.9046790421e+01, 1.6724985953e+01,
    1.4736105928e+01, 1.3042079344e+01, 1.1608431030e+01, 1.0402921997e+01,
    9.3948201226e+00, 8.5547281759e+00, 7.8547494556e+00, 7.2687901180e+00,
    6.7728786533e+00, 6.3454618457e+00, 5.9676545027e+00, 5.6233945998e+00,
    5.2994327425e+00, 4.9851067449e+00
]



Simkovic_paper_x= [0.00010181995439284004, 0.00011482916831344526, 0.00012642346121318789, 0.00014430086335830004, 0.000165699481263457, 0.00019607769749551714, 0.000247888211840957, 0.0003096432511816367, 0.00039146171555687044, 0.0005069449901306541, 0.0006971753711517342, 0.0008761102204861905, 0.001087811197113462, 0.0014087212482585282, 0.0018913086206379759, 0.002554528221431459, 0.0033280758176176426, 0.004414775423147398, 0.005927150475157031, 0.007675691184403614, 0.00988048055700296, 0.013506670426795632, 0.01802497510742809, 0.023202524933686915, 0.029157612383677534, 0.03753293551347307, 0.04495122322394991, 0.05962891664876913, 0.0886708106383785, 0.07403749172824403, 0.11976474157693831, 0.17075600248213096, 0.2348319476955167, 0.3004739239131386, 0.3753293551347309, 0.5161711577478199, 0.7013793942000445, 0.8867081063837855, 1.121007078948298, 1.4960107898396027, 1.9490242036826286, 2.419981302163716]
Simkovic_paper_y = [0.18170019467878, 0.18948734587929913, 0.1946787800129786, 0.21025308241401686, 0.21544451654769628, 0.2206359506813757, 0.24140168721609342, 0.2569759896171317, 0.26476314081765084, 0.290720311486048, 0.31667748215444513, 0.33484750162232313, 0.3530175210902011, 0.37897469175859827, 0.40752757949383517, 0.4412719013627514, 0.4516547696301103, 0.4879948085658663, 0.5191434133679429, 0.563270603504218, 0.6048020765736535, 0.6385463984425698, 0.6826735885788449, 0.7319922128487995, 0.7735236859182348, 0.8254380272550291, 0.8695652173913043, 0.9162881245944191, 1.0045425048669694, 0.9682024659312134, 1.0694354315379624, 1.136924075275795, 1.2147955872809864, 1.2615184944841011, 1.318624269954575, 1.3575600259571705, 1.427644386761843, 1.4561972744970797, 1.4951330304996755, 1.5236859182349123, 1.5314730694354315, 1.5314730694354315]

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


def B_wavefuncs_analytic(Ee,r):
    out_g = np.empty_like(Ee, dtype = float)
    out_f = np.empty_like(Ee, dtype = float)

    kappa_g = -1
    kappa_f = 1


    for i , E in enumerate(Ee):
        p = np.sqrt(E**2 - me**2)
        eta = alpha* Z* E/ p
        gammak = np.sqrt(1 - (alpha * Z)**2)

        numerator = np.abs(sp.gamma(1 + gammak + 1j * eta))
        denominator = sp.gamma(1 + 2 * gammak)
        gamma_ratio = numerator/denominator

        sqrt_term_plus = np.sqrt((E + me) / (2 * E))
        sqrt_term_min = np.sqrt((E - me) / (2 * E))

        exp_zeta_g = np.sqrt((kappa_g - 1j * eta * me/E) / (gammak - 1j * eta))
        exp_zeta_f = np.sqrt((kappa_f - 1j * eta * me/E) / (gammak - 1j * eta))


        hyp = mp.hyp1f1(gammak - 1j*eta, 1 + 2*gammak, -2*1j*p*r)
        Im_term = np.imag(np.exp(1j*p*r) * exp_zeta_g * hyp )
        Re_term = np.real(np.exp(1j*p*r) * exp_zeta_f * hyp)

        out_g[i] = np.sign(kappa_g) * 1.0/(p*r) * sqrt_term_plus * gamma_ratio * (2*p*r)**gammak * np.exp(np.pi * eta/2) * Im_term
        out_f[i] = np.sign(kappa_f) * 1.0/(p*r) * sqrt_term_min * gamma_ratio * (2*p*r)**gammak * np.exp(np.pi * eta/2) * Re_term

    return out_g, out_f


#### SCHEME B
Ee = T_MeV + me
p_au = np.sqrt(T_n * (T_n + 2*c**2))/c


g_B = np.sqrt((Ee + me)/(2*Ee)) * P_n[:,idx_R] /(p_au * mesh_point_R_au)
f_B = np.sqrt((Ee + me)/(2*Ee)) * Q_p[:,idx_R] / (p_au * mesh_point_R_au)

Fermi_B = 2/(s+1)*(g_B**2 + f_B **2)

gB_analytic , fB_analytic = B_wavefuncs_analytic(Ee, R_simkovic)
####



#### SCHEMA A
g_A = np.sqrt((Ee + me)/(2*Ee)) * P_n1[:,idx_R] /(p_au * mesh_point_R_au)
f_A = np.sqrt((Ee - me)/(2*Ee)) * Q_p1[:,idx_R] / (p_au * mesh_point_R_au)
Fermi_A = 2/(s+1) * (g_A**2 + f_A**2)



#### g{-1} Scheme A

Simkovic_gk_A = np.sqrt(Fermi_Analytical(Ee)) * np.sqrt((Ee + me)/(2*Ee))
Simkovic_fk_A = np.sqrt(Fermi_Analytical(Ee)) * np.sqrt((Ee - me)/ (2*Ee))


plt.figure(figsize = (12,8))
# plt.plot(T_MeV, g_n, marker = "o", linestyle = "-", label = r"$g_{\kappa=-1}(R)$")
# plt.plot(T_MeV, gB_analytic, lw = 2.0 ,label = "Simkovic g_{-1}B")
# plt.plot(T_MeV, abs(g_B), linestyle = "--", lw= 2.0 ,label = "g_B")
#
# plt.plot(T_MeV, fB_analytic, lw = 2.0 , label = "Simkovic f_{-1}B")
# plt.plot(T_MeV, abs(f_B), linestyle = "--" , lw = 2.0 , label = "f_B")

# #
# plt.plot(T_MeV, abs(g_A), linestyle = "--", color = "magenta" , lw = 2.0, label = "g_A")
# plt.plot(T_MeV, Simkovic_gk_A, lw = 2.0, color = "dodgerblue",label = "Simkovic g_{-1}A")

plt.plot(T_MeV, abs(f_A), ls= "--", lw = 2.0, label = "f_A" )
plt.plot(T_MeV, Simkovic_fk_A, lw = 2.0, label = "Simkovic f_{+1} A")
# plt.plot(Simkovic_paper_x , Simkovic_paper_y, lw = 2.0, label = "Simkovic paper")


plt.xlim(T_MeV[0], T_MeV[-1])
plt.ylim(0,4)
plt.xscale("log")
plt.title(r"Analytic vs Numeric reduced wavefunction $g_{-1}$ & $f_{+1}$ comparison")
plt.xlabel(r"$T$ (Mev)")
plt.ylabel(r"$g_{\kappa=-1}(R)$")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()



percent_diff_f_func = 100.0 * (abs(g_A) - Simkovic_gk_A)/ Simkovic_gk_A


fig, (ax1, ax2) = plt.subplots(2 , 1, figsize =(12,8) , sharex = True, gridspec_kw ={"height_ratios": [3, 1]})

ax1.plot(T_MeV, abs(g_A), linestyle = "--", color = "magenta" , lw = 2.0, label = "g_A")
ax1.plot(T_MeV, Simkovic_gk_A, lw = 2.0, color = "dodgerblue",label = "Simkovic g_{-1}A")
ax1.set_xscale("log")
ax1.set_title("Scheme A g_{-1} wave function comparison")
ax1.set_xlabel(r"$T$ (Mev)")
# ax1.set_ylabel(r"$g_{\kappa=-1}(R)$")
ax1.grid(True)
ax1.legend()


ax2.plot(T_MeV, percent_diff_f_func, label = " g_{-1} Numeric vs Simkovic")
# ax2.plot(T_MeV, abs(g_A)/Simkovic_gk_A, label = "ratio")
# ax2.plot(T_MeV, abs(abs(g_A) -Simkovic_gk_A) , label = "absolut difference")
# ax2.set_xlabel(r"$T$ (Mev)")
ax2.set_ylabel(f"% diff")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
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

# ax1.plot(T_MeV[:], Fermi[:], marker = "o", linestyle = "-", label = f"Numerical B @ R = {mesh_point_R_au: .5g}")
ax1.plot(T_MeV[:], F_analytic[:], label = "Saad ")
ax1.plot(T_MeV, Fermi_B, label = f"Fermi_B f&g  @ R = {mesh_point_R_au: .5g}")
ax1.plot(Fortran_x,Fortran_y, label = "FORTRAN")


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











