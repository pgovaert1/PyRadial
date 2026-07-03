#!usr/bin/env python

import numpy as np
from scipy.integrate import quad
from configurations.physics_constants import ME, GF, VUD, HBAR_C, GA

# ============================================================================
# Constants
# ============================================================================

Gbeta = GF * VUD # Effective weak coupling constant in MeV^-2
c2n = ME * (Gbeta * ME ** 2)**4 / (8 * np.pi**7)
MeVtoyr = (365.25 * 24 * 3600) * 2.998e8 / (1e-15 * HBAR_C)


# ============================================================================
# Helper Functions
# ============================================================================

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
        raise ValueError(f"Unknown tag: {tag}")


# ============================================================================
# Energy Spectrum - Standard Two-Electron Phase Space Integration
# ============================================================================

def two_electron_kernel(Ee1, Ee2, Q_value, tag, fermi_func, nuclear_matrix_elements):
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

    xi31, xi51 = nuclear_matrix_elements["xi31"], nuclear_matrix_elements["xi51"]
    pre_factor = MeVtoyr * c2n / (ME**11 * np.log(2))
    if tag is not None:
        polynomial = phase_space_polynomial(tag, neutrino_energy_sum, energy_assymetry)
        return float(pre_factor * F1 * F2 * p1 * Ee1 * p2 * Ee2 * polynomial)
    else:
        I_0 = phase_space_polynomial(0, neutrino_energy_sum, energy_assymetry)
        I_2 = phase_space_polynomial(2, neutrino_energy_sum, energy_assymetry)
        I_4 = phase_space_polynomial(4, neutrino_energy_sum, energy_assymetry)
        I_22 = phase_space_polynomial(22, neutrino_energy_sum, energy_assymetry)
        polynomial_term = (I_0 + xi31 * I_2 + 1/3 * xi31**2 * I_22 + (1/3 * xi31**2 + xi51) * I_4)
        return float(pre_factor * F1 * F2 * p1 * Ee1 * p2 * Ee2 * polynomial_term)


    


def two_electron_spectrum_integrand(tag, Q_value, fermi_func, nuclear_matrix_elements):
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
    nuclear_matrix_elements : dict
        Nuclear matrix elements for the selected isotope.

    Returns
    -------
    callable
        Function of the form f(Ee2, Ee1).
    """
    return lambda Ee2, Ee1: two_electron_kernel(Ee1, Ee2, Q_value, tag, fermi_func, nuclear_matrix_elements)


def angular_correlation_distribution(Ee1, Ee2, Q_value, fermi_func, E_func, nuclear_matrix_elements):
    '''
    Compute the angular correlation distribution for given electron energies.

    Parameters
    ----------
    Ee1 : float
        Total energy of the first electron.
    Ee2 : float
        Total energy of the second electron.
    Q_value : float
        Q-value of the decay process.
    fermi_func : callable
        Function returning the Fermi function value for a given electron energy.
    E_func : callable
        Function returning the energy-dependent correction factor for a given electron energy.  

    Returns
    -------
    float
        Value of the angular correlation distribution. 
    '''

    a = Q_value + 2*ME - Ee1 - Ee2 # neutrino energy sum
    b = Ee1 - Ee2 # energy asymmetry

    I_tilde2 = 1/14 * 1/(2*ME)**2 *(a**2 + 7 * b**2)
    I_tilde22 = 1/336 * 1/(2*ME)**4 * (a**4 - 6*a**2*b**2 + 21*b**4)
    I_tilde4 = 1/168 * 1/(2*ME)**4 * (a**4 + 18*a**2*b**2 + 21*b**4)

    xi31, xi51 = nuclear_matrix_elements["xi31"], nuclear_matrix_elements["xi51"]
    E_function_numerator = E_func(Ee1) * E_func(Ee2)
    F_function_denominator = fermi_func(Ee1) * fermi_func(Ee2)

    pre_factor = - E_function_numerator / F_function_denominator

    numerator =  (1 + xi31 * I_tilde2 + 5/9 * xi31**2 * I_tilde22 + (2/9 * xi31**2 + xi51) * I_tilde4)
    denominator = (1 + xi31 * I_tilde2 + 1/3 * xi31**2 * I_tilde22 + (1/3 * xi31**2 + xi51) * I_tilde4)

    return float(pre_factor * numerator / denominator)


def angular_correlation_weighted_kernel(Ee1, Ee2, Q_value, fermi_func, E_func, nuclear_matrix_elements):
    """
    Evaluate dΓ/(dEe1 dEe2) · κ²ν(Ee1, Ee2) directly.

    This is the integrand whose double integral over (Ee1, Ee2), divided by
    the double integral of the unweighted two-electron kernel
    (`two_electron_kernel` with `tag=None`) over the same domain, gives the
    decay-rate-weighted average angular correlation coefficient ⟨κ²ν⟩.

    Parameters
    ----------
    Ee1 : float
        Total energy of the first electron.
    Ee2 : float
        Total energy of the second electron.
    Q_value : float
        Q-value of the decay process.
    fermi_func : callable
        Function returning the Fermi function value for a given electron energy.
    E_func : callable
        Function returning the energy-dependent correction factor for a given electron energy.
    nuclear_matrix_elements : dict
        Nuclear matrix elements for the selected isotope.

    Returns
    -------
    float
        Value of dΓ/(dEe1 dEe2) · κ²ν. Returns 0.0 for kinematically
        forbidden configurations.
    """
    base = two_electron_kernel(Ee1, Ee2, Q_value, tag=None,
                               fermi_func=fermi_func,
                               nuclear_matrix_elements=nuclear_matrix_elements)
    if base == 0.0:
        return 0.0

    kappa = angular_correlation_distribution(Ee1, Ee2, Q_value, fermi_func, E_func, nuclear_matrix_elements)
    return base * kappa




# ============================================================================
# Epsilon Spectrum - Energy Sum and Asymmetry Variable Integration
# ============================================================================

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

def phase_combo(Q_minus_eps, D, nuclear_matrix_elements):
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
    xi31, xi51 = nuclear_matrix_elements["xi31"], nuclear_matrix_elements["xi51"]
    x = Q_minus_eps
    y = 2*D  # y = Ee1 - Ee2 = 2Δ

    A0  = phase_space_polynomial(0,  x, y)
    A2  = phase_space_polynomial(2,  x, y)
    A4  = phase_space_polynomial(4,  x, y)
    A22 = phase_space_polynomial(22, x, y)

    return A0 + A2*xi31 + A4*(xi31**2/3.0 + xi51) + A22*(xi31**2/3.0)

def epsilon_spectrum_kernel(eps, D, Q_value, fermi_func, nuclear_matrix_elements):
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

    MGT1 = nuclear_matrix_elements["MGT1"]
    poly = phase_combo(Q_value - eps, D, nuclear_matrix_elements)
    pre_factor = (GA ** 4 * MGT1**2 *MeVtoyr * c2n) / (ME**11)
    return float(pre_factor * F1 * F2 * p1 * Ee1 * p2 * Ee2 * poly)

def spectrum_epsilon_integrand(eps, Q_value, fermi_func, nuclear_matrix_elements):
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
    value, _ = quad(lambda D: epsilon_spectrum_kernel(eps, D, Q_value, fermi_func, nuclear_matrix_elements),
                    -eps/2.0,
                    eps/2.0,
                    epsabs=1e-16,
                    epsrel=1e-14,
                    limit=20000,
                    points=[-eps/2.0, 0.0, eps/2.0])
    return value



def energy_diff_spectrum_integrand(delta_T, Q, fermi_func, nuclear_matrix_elements):
    """
    Compute dΓ/d(|ΔT|) where ΔT = T_{e1} − T_{e2} = Ee1 − Ee2.

    At fixed |ΔT| = delta_T ≥ 0, integrates the two-electron kernel over all
    allowed Ee2, with a factor of 2 because both ΔT = +delta_T and ΔT = −delta_T
    contribute equally (the kernel is even in ΔT).
    """
    if not (0.0 <= delta_T <= Q):
        return 0.0
    Ee2_max = (Q + 2.0 * ME - delta_T) / 2.0
    if Ee2_max <= ME:
        return 0.0
    value, _ = quad(
        lambda Ee2: two_electron_kernel(Ee2 + delta_T, Ee2, Q, tag=None,
                                        fermi_func=fermi_func,
                                        nuclear_matrix_elements=nuclear_matrix_elements),
        ME, Ee2_max,
        epsabs=1e-16, epsrel=1e-14, limit=20000,
        points=[ME, (ME + Ee2_max) / 2.0, Ee2_max]
    )
    return 2.0 * value


def single_electron_spectrum_integrand(Ee1, Q, fermi_func, nuclear_matrix_elements):
    """Compute dΓ/dEe1 by integrating two_electron_kernel over Ee2."""
    if not (ME <= Ee1 <= Q + ME):
        return 0.0
    Ee2_max = Q + 2.0 * ME - Ee1
    if Ee2_max <= ME:
        return 0.0
    value, _ = quad(
        lambda Ee2: two_electron_kernel(Ee1, Ee2, Q, tag=None,
                                        fermi_func=fermi_func,
                                        nuclear_matrix_elements=nuclear_matrix_elements),
        ME, Ee2_max,
        epsabs=1e-16, epsrel=1e-14, limit=20000,
    )
    return value


def double_differential_spectrum_kernel(Ee1, Ee2, cos_theta, Q_value, fermi_func, E_func, nuclear_matrix_elements):
    """
    Compute the triple-differential spectrum dΓ/(dEe1 dEe2 d(cos θ)).
    
    This evaluates the angular-dependent spectrum using:
        dΓ/(dEe1 dEe2 d(cos θ)) = (1/2) * dΓ/(dEe1 dEe2) * (1 + κ²ν * cos(θ))
    
    where:
        - dΓ/(dEe1 dEe2) is the standard two-electron phase space factor
        - κ²ν is the angular correlation coefficient (energy-dependent, only for numeric Fermi)
        - cos(θ) is the angle between the two electrons
    
    Parameters
    ----------
    Ee1 : float
        Total energy of the first electron (MeV).
    Ee2 : float
        Total energy of the second electron (MeV).
    cos_theta : float
        Cosine of the angle between electron momenta (-1 to +1).
    Q_value : float
        Q-value of the decay process (MeV).
    fermi_func : callable
        Function returning the Fermi correction factor F(Ee).
        For this to work, it must be a wrapper that only takes Ee as input
        (pre-bound with Z, A, and other necessary parameters).
    E_func : callable
        Function returning the energy correction factor E(Ee) = 2*g*f.
        For numeric calculations, this is the Dirac wavefunction-based E function.
        For analytic calculations, this can be a dummy function (returns 0).
    
    Returns
    -------
    float
        Value of the triple-differential spectrum. Returns 0.0 for
        kinematically forbidden configurations.
    """
    # ========== Step 1: Get base two-electron spectrum ==========
    # Using tag=0 for leading-order phase space polynomial
    base_spectrum = two_electron_kernel(Ee1, Ee2, Q_value, tag=None, fermi_func=fermi_func, nuclear_matrix_elements=nuclear_matrix_elements)
    
    # Early return if kinematically forbidden
    if base_spectrum == 0.0:
        return 0.0
    
    # ========== Step 2: Get angular correlation coefficient κ²ν ==========
    # This coefficient describes how the spectrum varies with the relative angle
    # between the two electron momenta. It's energy-dependent through the
    # ratio E_func(Ee1)*E_func(Ee2) / (F(Ee1)*F(Ee2))
    # 
    # NOTE: This only has physical meaning when E_func is from Dirac wavefunctions.
    # For analytic Fermi function, E_func will be a dummy that returns 0,
    # resulting in kappa_2nu = 0 (isotropic distribution).
    kappa_2nu = angular_correlation_distribution(Ee1, Ee2, Q_value, fermi_func, E_func, nuclear_matrix_elements)
    
    # ========== Step 3: Apply angular dependence ==========
    # The (1/2) factor comes from normalizing d(cos θ) integral over [-1, +1]
    # The (1 + κ²ν*cos(θ)) describes the anisotropic angular distribution
    angular_factor = 0.5 * (1.0 + kappa_2nu * cos_theta)
    
    # ========== Step 4: Return final spectrum ==========
    result = base_spectrum * angular_factor
    return float(result)


