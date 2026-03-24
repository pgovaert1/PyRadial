#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt



V_CKM = 0.97435
G_F = 1.1663787e-11 #[1/MeV^2]
g_A = 1.27
m_NEUTRON = 939.56542194 # MeV
m_PROTON = 938.27208943 # MeV
m_ELECTRON = 0.51099895069 # MeV

hbar = 6.5821196654707e-22 # MeV s

##################
##### Fermi Data
##################
data = np.load("Fermi_data.npz")
E_data = data["x"]  ## E in [a.u] = [27eV]
F_data = data["y"]




delta_m_pn = m_NEUTRON - m_PROTON
print(f"delta m = {delta_m_pn}")

N = 101
E_electron = np.linspace(m_ELECTRON, delta_m_pn ,N)

p_electron = np.sqrt(E_electron**2 - m_ELECTRON**2)

Decay_rate = (G_F**2 * V_CKM**2)/ (2*np.pi **3)  * (delta_m_pn - E_electron)**2 * E_electron * (p_electron) * (1+3*g_A**2)




######################
##### Analytical Fermi
######################

E_fine = np.linspace(m_ELECTRON*1e6,delta_m_pn*1e6, N)

Z = 1
k = np.sqrt(2*E_fine)
eta = Z/(k)
F_analytical = (2 *np.pi * eta)/(1-np.exp(-2 * np.pi * eta))

###########################
##### Calculating Half life
#############################

Total_decay = float(np.trapezoid(Decay_rate,E_electron))
Numerical_Fermi_total_decay = float(np.trapezoid(F_data*Decay_rate, E_electron))
Analytical_Fermi_total_decay = float(np.trapezoid(F_analytical*Decay_rate, E_electron))


tau = hbar/Total_decay
Numerical_fermi_tau = hbar/Numerical_Fermi_total_decay
Analytical_fermi_tau = hbar/Analytical_Fermi_total_decay


half_life = tau * np.log(2)
Numerical_fermi_half_life = Numerical_fermi_tau * np.log(2)
Analytical_Fermi_half_life = Analytical_fermi_tau * np.log(2)

print("Pure beta decay:")
print(f"Mean lifetime tau = {tau: .5}s, Half life T = {half_life : .5}s")
print(f"Experimental Half life = 611s. So off by {100* abs(half_life-611)/611: .5}%")
print()
print("Numerical Fermi beta decay:")
print(f"Mean lifetime tau = {Numerical_fermi_tau: .5}s, Half life T = {Numerical_fermi_half_life : .5}s")
print(f"Experimental Half life = 611s. So off by {100* abs(Numerical_fermi_half_life-611)/611: .5}%")
print()
print("Analytical Fermi beta decay:")
print(f"Mean lifetime tau = {Analytical_fermi_tau: .5}s, Half life T = {Analytical_Fermi_half_life : .5}s")
print(f"Experimental Half life = 611s. So off by {100* abs(Analytical_Fermi_half_life-611)/611: .5}%")
print()
print(f"absolute differnece between numerical and Analytical half-life = {abs(Numerical_fermi_half_life - Analytical_Fermi_half_life): .4}s ")



Normalized_decay_rate = Decay_rate / Total_decay


################
##### Plotting
###############

plt.figure(figsize= (12,8))
plt.plot(E_electron, Normalized_decay_rate , label = "Beta-decay")
plt.plot(E_electron, F_data*Normalized_decay_rate, label = "Fermi beta-decay Numerical")
plt.plot(E_electron, F_analytical*Normalized_decay_rate, ls = "--" , label = "Fermi beta-decay Analytical")
plt.xlabel("Energy e [MeV]")
plt.ylabel(r" 1/$\Gamma$ d$\Gamma$ /dE  e [MeV^-1]  ")
plt.title(rf"Normalized Beta Decay Spectrum with total decay time")
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.show()
