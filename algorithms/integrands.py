#!usr/bin/env python

import numpy as np
from scipy.integrate import quad
from scipy.constants import physical_constants
from configurations.physics_constants import ME, ALPHA, GF, VUD, HBAR_C, GA, AU, C, E_HARTREE

Gbeta = GF * VUD # Effective weak coupling constant in MeV^-2
c2n = ME * (Gbeta * ME ** 2)**4 / (8 * np.pi**7)
MeVtoyr = (365.25 * 24 * 3600) * 2.998e8 / (1e-15 * HBAR_C)
MGT1, MGT3, MGT5, xi31, xi51 = 0.0104, 0.00403, 0.00126, 0.3867, 0.1207 # Simkovic NMEs for 136Xe


def electron_momentum(E):
    """
    Calcultes the relativistic electron momentum. Returns 0 if imaginary momentum (E^2 < ME^2)

    Parameters
    ----------
    E : ndarray
        Total relativistic energy values

    Returns
    -------
    ndarray
        Relativistic electron momentum values
    """
    p_squared = E * E - ME * ME
    if p_squared <= 0.0:
        return 0.0
    return np.sqrt(p_squared)


def phase_space_polynomial(tag, x, y):
    """
    Compute the phase-space polynomial contribution for a given expansion term.

    Parameters
    ----------
    tag : int
        Identifier specifying which polynomial term to evaluate.
        Supported values are:

        - 0   : leading-order contribution
        - 2   : second-order correction
        - 4   : fourth-order correction
        - 22  : alternate fourth-order correction

    x : float
        Sum of neutrino energies or equivalent kinematic variable.
    y : float
        Electron energy asymmetry variable, Ee1 - Ee2.

    Returns
    -------
    float
        Value of the selected phase-space polynomial.
    """
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
    """
    Evaluate the two-electron phase-space integrand.

    Parameters
    ----------
    Ee1 : float
        Total energy of the first electron.
    Ee2 : float
        Total energy of the second electron.
    Q_value : float
        Q-value of the decay process.
    tag : int
        Polynomial identifier passed to `phase_space_polynomial`.
    fermi_func : callable
        Function returning the Fermi function value for a given
        electron energy.

    Returns
    -------
    float
        Value of the phase-space integrand. Returns 0.0 for
        kinematically forbidden configurations.

    """
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
    """
    Construct a two-dimensional phase-space integrand function.

    Parameters
    ----------
    tag : int
        Polynomial identifier passed to `phase_space_polynomial`.
    Q_value : float
        Q-value of the decay process.
    fermi_func : callable
        Function returning the Fermi correction factor.

    Returns
    -------
    callable
        Function of the form f(Ee2, Ee1).
    """
    return lambda Ee2, Ee1: two_electron_integrand(Ee1, Ee2, Q_value, tag, fermi_func)


def energies_eps_D(eps, D):
    """
    Convert epsilon and asymmetry variables into electron energies.

    Parameters
    ----------
    eps : float
        Sum kinetic-energy variable.
    D : float
        Energy asymmetry parameter.

    Returns
    -------
    Ee1 : float
        Electron energie value for electron 1
    Ee2 : float
        Electron energie value for electron 2
    """
    Ee1 = ME + D + 0.5*eps
    Ee2 = ME - D + 0.5*eps
    return Ee1, Ee2

def phase_combo(Q_minus_eps, D):
    """
    Compute the combined phase-space polynomial expansion.

    Parameters
    ----------
    Q_minus_eps : float
        Difference Q - eps, corresponding to the available
        neutrino energy.
    D : float
        Electron energy asymmetry parameter.

    Returns
    -------
    float
        Weighted sum of phase-space polynomial contributions.
    """
    x = Q_minus_eps
    y = 2*D  # y = Ee1 - Ee2 = 2Δ

    A0  = x**5 / 30.0
    A2  = x**5 * (x*x + 7*y*y) / (420.0 * (2.0*ME)**2)
    A4  = x**5 * (x**4 + 18.0*x*x*y*y + 21.0*y**4) / (5040.0 * (2.0*ME)**4)
    A22 = x**5 * (x**4 - 6.0*x*x*y*y + 21.0*y**4) / (10080.0 * (2.0*ME)**4)

    return A0 + A2*xi31 + A4*(xi31**2/3.0 + xi51) + A22*(xi31**2/3.0)

def epsilon_spectrum_integrand(eps, D, Q_value, fermi_func):
    """
    Evaluate the integrand of the epsilon spectrum distribution.

    Parameters
    ----------
    eps : float
        Sum kinetic-energy variable.
    D : float
        Electron energy asymmetry parameter.
    Q_value : float
        Q-value of the decay process.
    fermi_func : callable
        Function returning the Fermi function value.

    Returns
    -------
    float
        Value of the epsilon-spectrum integrand. Returns 0.0
        outside the physical integration region.
    """

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
    """
    Compute the epsilon spectrum by integrating over the asymmetry variable.

    Parameters
    ----------
    eps : float
        Sum kinetic-energy variable.
    Q_value : float
        Q-value of the decay process.
    fermi_func : callable
        Function returning the Fermi correction factor.

    Returns
    -------
    float
        Differential spectrum value at the specified `eps`.
    """
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
