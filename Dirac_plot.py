#!/usr/bin/env python



from matplotlib.typing import LineStyleType
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize._lsq.lsq_linear import TERMINATION_MESSAGES
import scipy.special as sp
import mpmath as mp
import os


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


def load_data(directory_name, file_name):

    full_path = os.path.join(directory_name, file_name)
    data = np.load(full_path)

    print(f"loading file from {full_path}")

    T = data["T"]
    r = data["r"]
    P = data["P"]
    Q = data["Q"]
    return T, r, P, Q


T_n, r_n, P_n, Q_n = load_data("out", "potential_0_kappa_-1.npz")
T_n_V0, r_n_V0, P_n_V0, Q_n_V0 = load_data("out", "Dirac_run_kappa_n_schemeB_V0.npz")
T_p, r_p, P_p, Q_p = load_data("out", "potential_0_kappa_+1.npz")
T_p_V0, r_p_V0, P_p_V0, Q_p_V0 = load_data("out", "Dirac_run_kappa_p_schemeB_V0.npz")

T_n1, r_n1, P_n1, Q_n1 = load_data("out", "Dirac_run_kappa_n_schemeA_V_Z56.npz")
T_n_V01, r_n_V01, P_n_V01, Q_n_V01 = load_data("out", "Dirac_run_kappa_n_schemeA_V0.npz")
T_p1, r_p1, P_p1, Q_p1 = load_data("out", "Dirac_run_kappa_p_schemeA_V_Z56.npz")
T_p_V01, r_p_V01, P_p_V01, Q_p_V01 = load_data("out", "Dirac_run_kappa_p_schemeA_V0.npz")


T_MeV = T_n * E_hatree/1e6


aT_n, ar_n, aP_n, aQ_n = load_data("out", "Analytic_Coulomb_kappa_n_V_Z56.npz")
aT_n_V0, ar_n_V0, aP_n_V0, aQ_n_V0 = load_data("out", "Analytic_Coulomb_kappa_n_V0.npz")
aT_p, ar_p, aP_p, aQ_p = load_data("out", "Analytic_Coulomb_kappa_p_V_Z56.npz")
aT_p_V0, ar_p_V0, aP_p_V0, aQ_p_V0 = load_data("out", "Analytic_Coulomb_kappa_p_V0.npz")




T_n3, r_n3, P_n3, Q_n3 = load_data("out", "potential_3_kappa_-1.npz")
T_p3, r_p3, P_p3, Q_p3 = load_data("out", "potential_3_kappa_+1.npz")


T_n_diff, r_n_diff, P_n_diff, Q_n_diff = load_data("resolution_test", "potential_0_kappa_-1_Z-56_A136.npz")
T_p_diff, r_p_diff, P_p_diff, Q_p_diff = load_data("resolution_test", "potential_0_kappa_+1_Z-56_A136.npz")





Scheme_Cgx = [0.00010092528860766844, 0.00011334438229710347, 0.00013085791612450995, 0.0001519147698447522, 0.0001763599633897995, 0.00020473872763416396, 0.0002470586136038433, 0.000303109815358405, 0.00040179081084894003, 0.0005355499708401138, 0.000682967440807964, 0.0009306791831626531, 0.0013551894123510365, 0.001995262314968879, 0.0029702978440178628, 0.004138090648167254, 0.0062287370894402875, 0.009069847815057921, 0.015332041990349142, 0.024524490860720193, 0.07741052270741218, 0.07741052270741218, 0.1158777356155125, 0.19806147089717377, 0.3908408957924017, 0.7099046158552872, 1.3477210138513502, 2.2532013356294827, 2.434445183692199]
Scheme_Cgy = [17.841561423650976, 17.451205510907005, 16.831228473019518, 16.234213547646384, 15.591274397244547, 14.982778415614238, 14.270952927669347, 13.559127439724456, 12.640642939150403, 11.733639494833525, 10.964408725602755, 10.218140068886338, 9.265212399540758, 8.33524684270953, 7.531572904707233, 6.911595866819748, 6.199770378874857, 5.6371986222732495, 4.913892078071183, 4.316877152698049, 3.180252583237658, 3.180252583237658, 2.8702640642939152, 2.479908151549943, 2.101033295063146, 1.8484500574052813, 1.5958668197474168, 1.4351320321469576, 1.4351320321469576]


Scheme_Cfx = [0.00009888846397011232, 0.0001563776769679538, 0.0002557213510616451, 0.00041352793657061903, 0.000668717545931981, 0.001307690155672795, 0.002325438478543525, 0.004348600531112699, 0.008456379525161753, 0.013372530349989442, 0.02859635498543255, 0.046502438616317705, 0.07819948330170486, 0.1462340619871048, 0.28756621691011724, 0.5142390449924272, 0.9195857504964805, 1.6261666439658746, 2.378009538150679]
Scheme_Cfy = [3.4885404101326896, 3.1290711700844387, 2.769601930036188, 2.4632086851628467, 2.195416164053076, 1.8769601930036186, 1.6381182147165259, 1.4354644149577804, 1.2738238841978287, 1.189384800965018, 1.1194209891435463, 1.1121833534378769, 1.1314837153196622, 1.1917973462002411, 1.278648974668275, 1.3462002412545235, 1.4016887816646562, 1.430639324487334, 1.4209891435464415]


def find_Rau(r_arr):
    for i in range(0,len(r_arr)-1):
        if r_arr[i] <= R_au  and r_arr[i+1] > R_au:
            return i, r_arr[i]
    raise RuntimeError("Could not find R_au")





idx_R , mesh_point_R_au = find_Rau(r_n)
print(f"numeric potential 0: closest mesh point to R_au = {R_au: .5g} is at i ={idx_R} w/ mesh point value ={mesh_point_R_au: .5g}")

idx_R_3, mesh_point_R_au_3 = find_Rau(r_n3)
print(f"numeric potential 3: closest mesh point to R_au = {R_au: .5g} is at i ={idx_R_3} w/ mesh point value ={mesh_point_R_au_3: .5g}")

a_idx_R , a_mesh_point_R_au = find_Rau(ar_n)
print(f"Analytic: closest mesh point to R_au = {R_au: .5g} is at i ={a_idx_R} w/ mesh point value ={a_mesh_point_R_au: .5g}")





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

idx_R_diff , mesh_point_R_au_diff = find_Rau(r_n_diff)
print(f"res 200000 closest mesh point to R_au = {R_au: .5g} is at i ={idx_R_diff} w/ mesh point value ={mesh_point_R_au_diff: .5g}")

idx_fB2, mesh_point_fB2 = find_Rau(r_p_diff)
print(f"res 110000: closest mesh point to R_au = {R_au: .5g} is at i ={idx_fB2} w/ mesh point value ={mesh_point_fB2: .5g}")


g_B2 = np.sqrt((Ee + me)/(2*Ee)) * P_n_diff[:,idx_R_diff] /(p_au * mesh_point_R_au_diff)
f_B2 = np.sqrt((Ee + me)/(2*Ee)) * Q_p_diff[:,idx_fB2] / (p_au * mesh_point_fB2)

Fermi_B = 2/(s+1)*(g_B2**2 + f_B2 **2)


gB_analytic , fB_analytic = B_wavefuncs_analytic(Ee, R_simkovic)

#### SCHEME C

g_C = np.sqrt((Ee + me)/(2*Ee)) * P_n3[:,idx_R_3] /(p_au * mesh_point_R_au_3)
f_C = np.sqrt((Ee + me)/(2*Ee)) * Q_p3[:,idx_R_3] / (p_au * mesh_point_R_au_3)

Fermi_C = 2/(s+1)*(g_C**2 + f_C **2)



#### SCHEMA A
g_A = np.sqrt((Ee + me)/(2*Ee)) * P_n1[:,idx_R] /(p_au * mesh_point_R_au)
f_A = np.sqrt((Ee - me)/(2*Ee)) * Q_p1[:,idx_R] / (p_au * mesh_point_R_au)
Fermi_A = 2/(s+1) * (g_A**2 + f_A**2)






#### g{-1} Scheme A

Simkovic_gk_A = np.sqrt(Fermi_Analytical(Ee)) * np.sqrt((Ee + me)/(2*Ee))
Simkovic_fk_A = np.sqrt(Fermi_Analytical(Ee)) * np.sqrt((Ee - me)/ (2*Ee))



#### PLOTTING g and f
plt.figure(figsize = (12,8))
#SCHEME B
# plt.plot(T_MeV, gB_analytic, lw = 2.0 ,label = "Simkovic g_{-1}B")
plt.plot(T_MeV, abs(g_B), linestyle = "--", lw= 2.0 ,label = "g_B")
#
# plt.plot(T_MeV, fB_analytic, lw = 2.0 , label = "Simkovic f_{-1}B")
plt.plot(T_MeV, abs(f_B), linestyle = "--" , lw = 2.0 , label = "f_B")

plt.plot(T_MeV, abs(g_B2), lw= 2.0 ,label = "g_B2")
plt.plot(T_MeV, abs(f_B2), lw = 2.0 , label = "f_B2")


#Scheme A
# plt.plot(T_MeV, abs(g_A), linestyle = "--", color = "magenta" , lw = 2.0, label = "g_A")
# plt.plot(T_MeV, Simkovic_gk_A, lw = 2.0, color = "dodgerblue",label = "Simkovic g_{-1}A")

# plt.plot(T_MeV, abs(f_A), ls= "--", lw = 2.0, label = "f_A" )
# plt.plot(T_MeV, Simkovic_fk_A, lw = 2.0, label = "Simkovic f_{+1} A")
# plt.plot(Simkovic_paper_x , Simkovic_paper_y, lw = 2.0, label = "Simkovic paper")

# # SCHEME C
# plt.plot(T_MeV, abs(g_C), linestyle = "--", color = "magenta" , lw = 2.0, label = "g_C")
# plt.plot(Scheme_Cgx, Scheme_Cgy, lw = 2.0, label = "Simkovic g_{-1} C")
# plt.plot(T_MeV, abs(f_C), ls= "--", lw = 2.0, label = "f_C" )
# plt.plot(Scheme_Cfx, Scheme_Cfy, lw = 2.0, label = "Simkovic f_{+1} C")
#



plt.xlim(T_MeV[0], T_MeV[-1])
plt.ylim(0,20)
plt.xscale("log")
plt.title(r"Analytic vs Numeric reduced wavefunction $g_{-1}$ & $f_{+1}$ comparison")
plt.xlabel(r"$T$ (Mev)")
plt.ylabel(r"$g_{\kappa=-1}(R)$")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()



percent_diff_f_func = 100.0 * (abs(g_A) - Simkovic_gk_A)/ Simkovic_gk_A

#### g vs g_simkovic COMPARSION

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


#### FERMI COMPARSION

fig, (ax1, ax2) = plt.subplots(2 , 1, figsize =(12,8) , sharex = True, gridspec_kw ={"height_ratios": [3, 1]})

# ax1.plot(T_MeV[:], Fermi[:], marker = "o", linestyle = "-", label = f"Numerical B @ R = {mesh_point_R_au: .5g}")
ax1.plot(T_MeV[:], F_analytic[:], label = "Saad ")
ax1.plot(T_MeV, Fermi_B, label = f"Fermi_B f&g  @ R = {mesh_point_R_au: .5g}")


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











