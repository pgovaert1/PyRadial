#!usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import mpmath as mp
import os
from configurations.physics_constants import ME, ALPHA, HBAR_C, E_HARTREE, AU, C

plt.rcParams.update({
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


def find_mesh_point_R_au(grid_points, rN, verbose=False):
    """
    Obtains the index and value of r_grid which match the nuclear radius or come closest to it.

    Parameters
    ----------
    grid_points : ndarray
        One dimensional ndarray of the radial grid
    rN : float
        Dimensionless nuclear radius defined as R = 1.2 * A^(1/3) / HBAR_C
    verbose : bool
        If True, print the matched mesh point info.

    Returns
    -------
    idx_R : int
        Index point at which grid_points matches nuclear radius or comes closest to it
    grid_point_R_au : float
        Value of grid_points which matches the nuclear radius or comes closest to it
    """
    R_au = rN * HBAR_C * 1e-15 /AU
    idx_R = 0
    grid_point_R_au = 0
    for i in range(0,len(grid_points)-1):
        if grid_points[i] <= R_au  and grid_points[i+1] > R_au:
            grid_point_R_au = grid_points[i]
            idx_R = i
            if verbose:
                print(f"closest mesh point to R_au = {R_au: .5g} is at i ={idx_R} w/ mesh point value ={grid_point_R_au: .5g}")
            break
    return idx_R, grid_point_R_au


def calc_g_and_f(Ee, data):
    """
    Calculate the radial Dirac wavefunction components g with kappa = -1 and f with kappa = +1.

    The functions are computed from the radial wavefunction components
    P and Q following the conventions used in
    "Angular Distributions of Emitted Electrons in the Two-Neutrino
    ββ Decay" by Ovidiu.

    Parameters
    ----------
    Ee : ndarray
        Total relativsitc energy values.
    data : dict
        Dictionary containing the radial wavefunction data with keys:

        - "T" : kinetic energy mesh
        - "P_n" : upper radial wavefunction component
        - "Q_p" : lower radial wavefunction component
        - "idx_R" : index corresponding to the nuclear radius
        - "mesh_point_R_au" : nuclear radius mesh point in atomic units

    Returns
    -------
    g_n1 : ndarray
        Upper component of dirac wavefunction for kappa = -1
    f_p1 : ndarray
        Lower component of dirac wavefunction for kappa = +1
    """
    Ee = np.asarray(Ee, dtype = float)

    T_mesh = data["T"]
    P_n = data["P_n"]
    Q_p = data["Q_p"]
    idx_R = data["idx_R"]
    mesh_point_R_au = data["mesh_point_R_au"]


    T_MeV = T_mesh * E_HARTREE/1e6
    Ee = T_MeV + ME
    p_au = np.sqrt(T_mesh * (T_mesh + 2*C**2))/C


    g_n1 = np.sqrt((Ee + ME)/(2*Ee)) * P_n[:,idx_R] /(p_au * mesh_point_R_au)
    f_p1 = np.sqrt((Ee + ME)/(2*Ee)) * Q_p[:,idx_R] / (p_au * mesh_point_R_au)
    return g_n1, f_p1


def Fermi_numerical_division(Ee,Z, data):
    """
    Calculate the numerical Fermi function from Dirac radial wavefunctions.

    The Fermi function is computed from the radial Dirac components
    g and f evaluated at the nuclear radius,

        F(Z, Ee) =  (g^2 + f^2) / (g^2 + f^2)_{Z=0}

    Parameters
    ----------
    Ee : array_like
        Total relativistic electron energies in MeV.

    Z : int
        Atomic number.

    A : int
        Mass number.

    data : dict
        Dictionary containing the radial wavefunction data with keys:

        - "T" : kinetic energy mesh
        - "P_n" : upper radial wavefunction component
        - "Q_p" : lower radial wavefunction component
        - "idx_R" : index corresponding to the nuclear radius
        - "mesh_point_R_au" : nuclear radius mesh point in atomic units

    Returns
    -------
    ndarray
        Numerical Fermi function evaluated at Ee.
    """

    
    T_mesh = data["T"]
    P_n = data["P_n"]
    Q_p = data["Q_p"]
    P_n_Z0 = data["P_n_Z0"]
    Q_p_Z0 = data["Q_p_Z0"]

    idx_R = data["idx_R"]
    mesh_point_R_au = data["mesh_point_R_au"]

    data_Z = {"T": T_mesh, "P_n": P_n, "Q_p": Q_p, "idx_R": idx_R, "mesh_point_R_au": mesh_point_R_au}
    data_Z0 = {"T": T_mesh, "P_n": P_n_Z0, "Q_p": Q_p_Z0, "idx_R": idx_R, "mesh_point_R_au": mesh_point_R_au}

    T = Ee - ME
    T_MeV = T_mesh * E_HARTREE/1e6

    g, f = calc_g_and_f(Ee, data_Z)

    g0, f0 = calc_g_and_f(Ee, data_Z0)


    Fermi_vals = (g**2 + f**2)/(g0**2 + f0**2)

    return np.interp(T, T_MeV, Fermi_vals)

def Fermi_numerical(Ee, Z, data):
    """
    Calculate the numerical Fermi function from Dirac radial wavefunctions.

    The Fermi function is computed from the radial Dirac components
    g and f evaluated at the nuclear radius,

        F(Z, Ee) = (g^2 + f^2)

    Parameters
    ----------
    Ee : array_like
        Total relativistic electron energies in MeV.

    Z : int
        Atomic number.

    A : int
        Mass number.

    data : dict
        Dictionary containing the radial wavefunction data with keys:

        - "T" : kinetic energy mesh
        - "P_n" : upper radial wavefunction component
        - "Q_p" : lower radial wavefunction component
        - "idx_R" : index corresponding to the nuclear radius
        - "mesh_point_R_au" : nuclear radius mesh point in atomic units

    Returns
    -------
    ndarray
        Numerical Fermi function evaluated at Ee.
    """

    T = Ee - ME
    T_mesh = data["T"]
    T_MeV = T_mesh * E_HARTREE/1e6

    g, f = calc_g_and_f(Ee, data)


    Fermi_vals = (g**2 + f**2)

    return np.interp(T, T_MeV, Fermi_vals)

def E_function(Ee, Z, A, data):
    """
    Calculate the E function from Dirac radial wavefunctions.

    The function is computed from the product of the radial Dirac
    components g and f evaluated at the nuclear radius,

        E(Ee) = 2 g * f.

    Parameters
    ----------
    Ee : array_like
        Total relativistic electron energies in MeV.

    Z : int
        Atomic number.

    A : int
        Mass number.

    data : dict
        Dictionary containing the radial wavefunction data with keys:

        - "T" : kinetic energy mesh
        - "P_n" : upper radial wavefunction component
        - "Q_p" : lower radial wavefunction component
        - "idx_R" : index corresponding to the nuclear radius
        - "mesh_point_R_au" : nuclear radius mesh point in atomic units

    Returns
    -------
    ndarray
        Values of the E function evaluated at Ee.
    """
    T = Ee - ME
    T_mesh = data["T"]
    T_MeV = T_mesh * E_HARTREE/1e6

    g, f = calc_g_and_f(Ee, data)
    E_vals = 2*g * f


    return np.interp(T, T_MeV, E_vals)


def Fermi(Ee, Z, A, rN):
    """
    Calculate the analytic pure Coulomb potential Fermi function.

    Parameters
    ----------
    Ee : ndarray
        Total relativistic energy values.
    Z : int
        Atomic number.
    A : int
        Mass number.
    rN : float
        Dimensionless nuclear radius R = 1.2 * A^(1/3) / HBAR_C

    Returns
    -------
    ndarray
        Values of analytic Fermi function for pure Coulomb potential up to second order correction

    """
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


def _plot_wavefunction_component(Ee, numeric, analytic, ylabel, ylim, name, potential_label, output_dir, isotope):
    """
    Plot one Dirac wavefunction component (g or f) vs kinetic energy, numeric
    only or numeric-vs-analytic with a %-difference subpanel, and save to disk.

    Parameters
    ----------
    Ee : ndarray
        Total energy grid (MeV).
    numeric : ndarray
        Numeric component values on `Ee`.
    analytic : ndarray or None
        Analytic (pure Coulomb) component values on `Ee`, or None to skip the
        comparison panel (e.g. for potentials without an analytic solution).
    ylabel : str
        Y-axis label for the component being plotted.
    ylim : float
        Upper y-axis limit.
    name : str
        Component name ("g" or "f"), used in labels/filenames.
    potential_label : str
        Human-readable potential name, used in the title/filename.
    output_dir : str or Path
        Directory the figure is saved to.
    isotope : str
        Isotope name, used in the title/filename.

    Returns
    -------
    None
        Saves and displays a matplotlib figure as a side effect.
    """
    T_kinetic = Ee - ME
    if analytic is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(T_kinetic, abs(numeric), lw=1.5, label=f"{name} numeric")
        ax1.plot(T_kinetic, analytic,     lw=1.5, label=f"{name} analytic")
        ax1.set_ylabel(ylabel)
        ax1.set_ylim(0, ylim)
        ax1.set_title(f"{isotope}: {name} wavefunction comparison, {potential_label}")
        ax1.legend()
        ax1.grid(True)
        percent_diff = 100.0 * (abs(numeric) - analytic) / analytic
        ax2.plot(T_kinetic, percent_diff, lw=1.5)
        ax2.set_xlabel(r"$T$ (MeV)")
        ax2.set_ylabel("% diff")
        ax2.grid(True)
        ax1.set_xscale("log")
    else:
        plt.figure(figsize=(12, 8))
        plt.plot(T_kinetic, abs(numeric), lw=1.5, label=f"{name} numeric")
        plt.xlabel(r"$T$ (MeV)")
        plt.ylabel(ylabel)
        plt.xlim(T_kinetic[0], T_kinetic[-1])
        plt.xscale("log")
        plt.ylim(0, ylim)
        plt.title(f"{isotope}: {name} wavefunction component comparison, {potential_label}")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_comparison_{potential_label.replace(' ', '_')}_{isotope}.png"), dpi=300, bbox_inches='tight')
    plt.show()


def plot_f_and_g(Ee, potential_index, cnf, data, output_directory_name, isotope):
    """
    Calculates and plots the numeric and analytic g and f Dirac wave function component for comparison

    Analytic g and f components are calculated via methods described in "Angular Distributions of Emitted Electrons in the Two-Neutrino
    ββ Decay" by Ovidiu. Note the Thomas-Fermi potential does not have an analytic solution

    Parameters
    ----------
    Ee : ndarray
        Total relativistic energy values
    potential_index: int
        Index specifying which potential model to use.
    cnf : dict
        Dictionary containing the isotope configurations
        - "Z" : Atomic number.
        - "A" : Mass number.
        - "Q" : MeV Q-value.
    data : dict
        Dictionary containing the radial wavefunction data with keys:

        - "T" : kinetic energy mesh
        - "P_n" : upper radial wavefunction component
        - "Q_p" : lower radial wavefunction component
        - "idx_R" : index corresponding to the nuclear radius
        - "mesh_point_R_au" : nuclear radius mesh point in atomic units
    output_directory_name : string
        Name of directory to which the plots are saved
    isotope : string
        Isotope name, used in plot titles and filenames to avoid overwriting
        plots from different isotopes with the same potential index

    """
    Z = cnf["Z"]
    A = cnf["A"]
    rN = 1.2 * A ** (1 / 3) / HBAR_C

    potential_label = None
    f_analytic = g_analytic = None

    g_numeric, f_numeric = calc_g_and_f(Ee, data)

    if potential_index == 0:
        potential_label = "Pure Coulomb"

        g_analytic = np.empty_like(Ee, dtype = float)
        f_analytic = np.empty_like(Ee, dtype = float)

        kappa_g = -1
        kappa_f = 1

        #### SIMOKVIC ANALYTIC FORMULA
        for i , E in enumerate(Ee):
            p = np.sqrt(E**2 - ME**2)
            eta = ALPHA * Z* E/ p
            gammak = np.sqrt(1 - (ALPHA * Z)**2)

            numerator = np.abs(sp.gamma(1 + gammak + 1j * eta))
            denominator = sp.gamma(1 + 2 * gammak)
            gamma_ratio = numerator/denominator

            sqrt_term_plus = np.sqrt((E + ME) / (2 * E))
            sqrt_term_min = np.sqrt((E - ME) / (2 * E))

            exp_zeta_g = np.sqrt((kappa_g - 1j * eta * ME/E) / (gammak - 1j * eta))
            exp_zeta_f = np.sqrt((kappa_f - 1j * eta * ME/E) / (gammak - 1j * eta))


            hyp = complex(mp.hyp1f1(gammak - 1j*eta, 1 + 2*gammak, -2*1j*p*rN))
            phase = np.exp(1j*p*rN)
            Im_term = np.imag(phase * exp_zeta_g * hyp )
            Re_term = np.real(phase * exp_zeta_f * hyp)

            g_analytic[i] = np.sign(kappa_g) * 1.0/(p*rN) * sqrt_term_plus * gamma_ratio * (2*p*rN)**gammak * np.exp(np.pi * eta/2) * Im_term
            f_analytic[i] = np.sign(kappa_f) * 1.0/(p*rN) * sqrt_term_min * gamma_ratio * (2*p*rN)**gammak * np.exp(np.pi * eta/2) * Re_term


    elif potential_index == 2:
        potential_label = "finite nuclear size"
        g_analytic = np.sqrt(Fermi(Ee, Z, A, rN)) * np.sqrt((Ee + ME)/(2*Ee))
        f_analytic = np.sqrt(Fermi(Ee, Z, A, rN)) * np.sqrt((Ee - ME)/ (2*Ee))

    elif potential_index == 3:
        potential_label = "Thomas-Fermi"



    else:
        raise RuntimeError("No proper potential index has been passed")

    _plot_wavefunction_component(Ee, f_numeric, f_analytic, r"$f_{\kappa=1}(R)$",  4,  "f", potential_label, output_directory_name, isotope)
    _plot_wavefunction_component(Ee, g_numeric, g_analytic, r"$g_{\kappa=-1}(R)$", 20, "g", potential_label, output_directory_name, isotope)

    return None
