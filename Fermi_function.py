#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


NPZ_FILE = "Coulomb_Wavefunctions_beta_decay.npz"
R_NUC_TARGET = 1.88e-5



def load_wavefunctions_by_Z(filename, Z0 , Z_ON ):
    Coulomb_data = np.load(filename)


    psi_Z0 = {}
    psi_Z_ON = {}

    for key in Coulomb_data.files:
        if not key.endswith("_P"):
            continue

        parts = key.split("_")

        E_str = parts[1][1:]
        Z_str = parts[2][1:]

        E_val = float(E_str)
        Z_val = float(Z_str)


        r_key = key[:-2] + "_r"

        r  = Coulomb_data[r_key]
        P = Coulomb_data[key]


        if Z_val == Z0 :
            psi_Z0[E_val] = (r,P)
        elif Z_val== Z_ON:
            psi_Z_ON[E_val] = (r,P)


    return psi_Z0, psi_Z_ON


def pick_nuclear_radius(r_example, R_target = R_NUC_TARGET):

    r_example = np.array(r_example)

    positive = r_example[r_example >0]


    if R_target < min(positive):
        R = min(positive)

    elif R_target > max(positive):
        R = max(positive)
    else:
        idx = np.argmin(abs(positive-R_target))
        R = positive[idx]
    return R

def Compute_Fermi_function(psi_Z0, psi_Z_ON, R_NUC_target = R_NUC_TARGET):


    common_E = sorted(set(psi_Z0.keys()) & set(psi_Z_ON.keys()))


    r0, _ = psi_Z0[common_E[0]]

    R_N = pick_nuclear_radius(r0, R_target = R_NUC_target)


    E_list = []
    F_list = []

    eps = 0
    for E in common_E:

        r0,P0 = psi_Z0[E]
        rZ,PZ = psi_Z_ON[E]

        if not np.allclose(r0,rZ):
            raise RuntimeError(f"r-grid mismatch")

        idx = np.argmin(np.abs(r0-R_N))

        val_P0 = np.abs(P0[idx])
        val_PZ = np.abs(PZ[idx])


        Fermi = val_P0/(val_PZ + eps)

        E_list.append(E)
        F_list.append(Fermi)

    return np.array(E_list), np.array(F_list), R_N

Wave_function_Z0, Wave_function_Z_ON = load_wavefunctions_by_Z(NPZ_FILE, Z0= 0, Z_ON =  1)


E_vals , F_vals, R_N = Compute_Fermi_function(Wave_function_Z0, Wave_function_Z_ON, R_NUC_target= R_NUC_TARGET)


print(f"using nuclear radius R_N = {R_N: .3e} a.u, this is {abs(R_N - R_NUC_TARGET): .3e} a.u away from actuall nuclear radius R_TARGET = {1.88e-5: .3e} a.u ")



order = np.argsort(E_vals)
print(order)
E_vals = E_vals[order]
F_vals = F_vals[order]

np.savez("Fermi_data.npz", x=E_vals, y=F_vals)



print(E_vals)
print(F_vals)

###################
##### Analytical
##################

# N=2000
# E_fine = np.linspace(E_vals[0],E_vals[-1], N)
#
# Z = 1
# k = np.sqrt(2*E_fine)
# eta = Z/(k)
# F_analytical = (2 *np.pi * eta)/(1-np.exp(-2 * np.pi * eta))
#
#
#
# plt.figure(figsize=(12,8))
# plt.plot(E_vals, F_vals, marker = "o", label = "Fermi Numerical")
# plt.plot(E_fine, F_analytical, ls = "--" , label = "Fermi Analytical")
# plt.xlabel("Energy E (1E = 27eV)")
# plt.ylabel("Fermi Function F(E)")
# plt.title(f"Coulomb Fermi function for Z = {Z}, evaluated at R = {R_N :.3e} a.u")
# plt.legend()
# plt.grid(True)
# plt.show()








