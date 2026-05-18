#!/usr/bin/evn python

import numpy as np
import mpmath as mp
from scipy.interpolate import interp1d

mp.mp.dps = 30

##################
### Thomas-Fermi
##################

def solve_thomas_fermi(tsteps, Nmax, dps):
    a = [1, 9 - np.sqrt(73)]


    for m in range(2, Nmax):
        a_temp = sum(a[m - n] * ((n + 1) * a[n + 1] - 2 * (n + 4) * a[n] + (n + 7) * a[n - 1])
                    for n in range(1, m - 1))
        a_temp += a[m - 1] * (m + 7 - 2 * (m + 3) * a[1])
        a_temp += a[m - 2] * (m + 6) * a[1]
        a.append(a_temp / (2 * (m + 8) - (m + 1) * a[1]))

    k = Nmax -2

    b = np.array([a[0]] + [a[n] - a[n - 1] for n in range(1, k + 1)])
    c = np.array([a[0] - a[0]] + [b[n - 1] - b[n] for n in range(1, k + 1)])

    P = np.poly1d(b[::-1])
    Q = np.poly1d(c[::-1])

    def I(t):
        def mp_integrand(x):
            x_float = float(x)
            return mp.mpf(P(x_float)) / mp.mpf(Q(x_float))
        return mp.quad(mp_integrand, [mp.mpf(1) - mp.mpf(t), mp.mpf(1)])

    t_list = np.linspace(0, 0.99, tsteps)
    I_list = np.array([I(t) for t in t_list])
    I_list_float = np.array([float(val) for val in I_list])
    x_vals = (144) ** (1/3) * t_list ** 2 * np.exp(2 * I_list_float)
    phi_vals = np.exp(-6 * I_list_float)

    idx = np.argsort(x_vals)
    x_vals = x_vals[idx]
    phi_vals = phi_vals[idx]

    return x_vals, phi_vals


def make_phi_x(tsteps, Nmax, dps):
    x_vals, phi_vals = solve_thomas_fermi(tsteps, Nmax, dps)

    phi_interp = interp1d(x_vals, phi_vals, kind ="linear", bounds_error=False, fill_value=(1.0, 0.0))

    return phi_interp

def make_phi_r(Z, tsteps=200, Nmax=100, dps=60):
    phi_x = make_phi_x(tsteps, Nmax, dps)

    b_au = 0.8853 * Z **(-1.0/3.0)

    def phi_of_r(r_au):
        r_arr = np.asarray(r_au, dtype=float)
        x = r_arr / b_au
        return phi_x(x)

    return phi_of_r



