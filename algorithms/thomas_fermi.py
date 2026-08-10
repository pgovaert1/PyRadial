#!/usr/bin/evn python

import numpy as np
import mpmath as mp
from scipy.interpolate import interp1d

mp.mp.dps = 30

##################
### Thomas-Fermi
##################

def solve_thomas_fermi(tsteps, Nmax):
    """
    Solve the Thomas-Fermi equation for the universal screening function phi(x)
    via a power-series/rational-approximant expansion, evaluated parametrically
    in a variable t in [0, 1).

    Builds the series coefficients `a` from the TF recursion, forms two
    polynomials P/Q (a Padé-like rational approximant to the series), and uses
    them inside an mpmath quadrature to compute x(t) and phi(t) = exp(-6*I(t))
    on a grid of t values, sorted by increasing x.

    Parameters
    ----------
    tsteps : int
        Number of t-grid points (and output (x, phi) samples).
    Nmax : int
        Number of series coefficients to generate for the P/Q rational approximant.

    Returns
    -------
    x_vals : ndarray
        Dimensionless radial coordinate x = r/b, sorted ascending.
    phi_vals : ndarray
        Thomas-Fermi screening function phi(x) evaluated at `x_vals`.
    """
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
        """Integral of the P/Q rational approximant from (1-t) to 1, used to build x(t) and phi(t)."""
        def mp_integrand(x):
            """Rational approximant P(x)/Q(x) to the Thomas-Fermi series, evaluated at high precision."""
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


def make_phi_x(tsteps, Nmax):
    """
    Build a linear interpolant phi(x) of the Thomas-Fermi screening function
    over the dimensionless coordinate x, from `solve_thomas_fermi`'s samples.

    Parameters
    ----------
    tsteps : int
        Number of samples used to build the interpolant (see `solve_thomas_fermi`).
    Nmax : int
        Number of series coefficients used to build the rational approximant
        (see `solve_thomas_fermi`).

    Returns
    -------
    callable
        Function phi(x) -> screening function value; extrapolates to 1.0 as
        x -> 0 and 0.0 as x -> infinity.
    """
    x_vals, phi_vals = solve_thomas_fermi(tsteps, Nmax)

    phi_interp = interp1d(x_vals, phi_vals, kind ="linear", bounds_error=False, fill_value=(1.0, 0.0))

    return phi_interp

def make_phi_r(Z, tsteps=200, Nmax=100):
    """
    Build the Thomas-Fermi screening function phi(r) in atomic units for a
    given atomic number Z, using the Thomas-Fermi length scale
    b = 0.8853 * Z^(-1/3).

    Parameters
    ----------
    Z : int
        Atomic number, used to set the TF length scale b_au.
    tsteps : int
        Number of samples used to build the underlying phi(x) interpolant.
    Nmax : int
        Number of series coefficients used to build the rational approximant.

    Returns
    -------
    callable
        Function phi(r_au) -> screening function value at radius r_au (a.u.).
    """
    phi_x = make_phi_x(tsteps, Nmax)

    b_au = 0.8853 * Z **(-1.0/3.0)

    def phi_of_r(r_au):
        """Evaluate the Thomas-Fermi screening function at radius r_au (a.u.), via x = r_au / b_au."""
        r_arr = np.asarray(r_au, dtype=float)
        x = r_arr / b_au
        return phi_x(x)

    return phi_of_r



