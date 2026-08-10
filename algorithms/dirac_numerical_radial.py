#!/usr/bin/env python

# ========== Importing libraries ==========
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import mpmath as mp
from tqdm import tqdm
from pathlib import Path
from configurations.physics_constants import ALPHA, C, EPSILON, AU, E_HARTREE, R0
from algorithms.thomas_fermi import make_phi_r
from algorithms.grid_creation import radial_grid, obtain_mesh_range, find_rc

# ========== Set arbitrary precision arithmetic ==========
mp.mp.dps = 30


# ========== Setup Cubic Spline potential V ==========

def potential(r, Z, R_au, potential_index, phi_r):
    """
    Generates the selected atomic potential function.

    Parameters
    ----------
    r : float or ndarray
        Radial coordinate(s) at which potential is evaluated.
    Z : int
        Atomic number.
    R_au : float
        Nuclear radius in Atomic units, defined by R = 1.2e-15 * A^(1/3) / AU.
    potential_index: int
        Index specifying which potential model to use.
    phi_r : callable
        Function phi(r) used in the Thomas-Fermi screening potential.

    Returns
    -------
    float or ndarray
        Evaluated potential value(s).
    """

    if potential_index == 0:
        # Coulomb potential
        return Z/ r

    elif potential_index == 1:
        # Coulomb + exponential perturbation
        V0 = -1
        A = 1
        return Z / r + V0 * np.exp(-A * r)

    elif potential_index == 2:
        # Finite nuclear model (uniform charge distribution)
        r_arr = np.asarray(r, dtype=float)
        V = Z / r_arr

        inside = r_arr < R_au
        V = np.where(inside, Z / (2 * R_au) * (3 - (r_arr / R_au) ** 2), V)

        if np.isscalar(r):
            return float(V)
        else:
            return V

    elif potential_index == 3:
        # Thomas-Fermi screening potential
        r_arr = np.asarray(r, dtype=float)

        V_A = potential(r, Z, R_au, 2, 0)
        phi_vals = phi_r(r_arr)

        V_C = ((r_arr * V_A + 2.0) * phi_vals - 2.0) / r_arr

        if np.isscalar(r):
            return float(V_C)
        return V_C

    else:
        raise RuntimeError("No proper potential function index number was selected")


# ========== Calculate potential parameters u0,u1,u2,u3 for interval [r_a, r_b] ==========

def general_potential_parameters(r_a, r_b, l, T, derivatives):
    """
    Generate cubic splined potential parameters (u0, u1, u2, u3) on range [r_a, r_b]

    Parameters
    ----------
    r_a : float
        Lower radial boundary of the interval [r_a, r_b].
    r_b : float
        Upper radial boundary of the interval [0, r_b].
    l : int
        Orbital angular momentum quantum number.
    T : float
        Kinetic energy in Hartree units.
    derivatives : list
        List made of zero, first, second and third order derivatives of the cubic splined potential

    Returns
    -------
    u0 : float
        First polynomial coefficient of cubic spline.
    u1 : float
        Second polynomial coefficient of cubic spline.
    u2 : float
        Third polynomial coefficient of cubic spline.
    u3 : float
        Fourth polynomial coefficient of cubic spline.

    """
    ra = r_a

    # ========== Extract derivatives at ra ==========
    rv0 = float(derivatives[0](ra))
    rv1 = float(derivatives[1](ra))
    rv2 = 0.5 * float(derivatives[2](ra))
    rv3 = (1.0 / 6.0) * float(derivatives[3](ra))

    # ========== Compute potential parameters via Taylor expansion ==========
    v0 = rv0 - rv1 * r_a + rv2 * (r_a ** 2) - rv3 * (r_a ** 3)
    v1 = rv1 - 2.0 * rv2 * r_a + 3.0 * rv3 * (r_a ** 2)
    v2 = rv2 - 3.0 * rv3 * r_a
    v3 = rv3

    h = r_b - r_a

    # ========== Scale parameters by ALPHA constant ==========
    u0 = ALPHA * (v0 + (v1 - T) * r_a + v2 * r_a ** 2 + v3 * r_a ** 3)
    u1 = ALPHA * h * ((v1 - T) + 2 * v2 * r_a + 3 * v3 * r_a ** 2)
    u2 = ALPHA * h ** 2 * (v2 + 3 * v3 * r_a)
    u3 = ALPHA * v3 * h ** 3

    return u0, u1, u2, u3


# ========== Neumaier compensated summation functions ==========
# ========== Enhances floating-point precision for long accumulation loops ==========

def Neumaier_add(sum_, c, x):
    """
    Perform one step of Neumaier compensated summation.

    This algorithm reduces floating-point truncation errors that
    accumulate during repeated additions by tracking a compensation
    term for lost low-order bits.

    Parameters
    ----------
    sum_ : float
        Current accumulated sum.
    c : float
        Compensation term storing accumulated rounding errors.
    x : float
        New value to add to the sum.

    Returns
    -------
    sum_ : float
        Updated accumulated sum.
    c : float
        Updated compensation term.
    """
    t = sum_ + x
    if abs(sum_) >= abs(x):
        c += (sum_ - t) + x
    else:
        c += (x - t) + sum_
    return t, c


def Neumaier_value(sum_, c):
    """
    Compute the corrected sum from Neumaier compensated summation.

    Parameters
    ----------
    sum_ : float
        Accumulated floating-point sum.
    c : float
        Compensation term containing accumulated rounding corrections.

    Returns
    -------
    float
        Corrected sum including compensation.
    """
    return sum_ + c


# ========== Power series terms for singular interval [0, r_b] ==========

def singular_power_series_terms(u_array, r_b, l ,Z , kappa, sigma):
    """
    Computes the power series coefficients An, Bn on singular radial interval [0,r_b]

    Parameters
    ----------
    u_array : ndarray
        Cubic spline coefficients of the potential (u0,u1,u2,u3).
    r_b : float
        Upper radial coordinate of the interval [0,r_b].
    l : int
        Orbital angular momentum quantum number.
    Z : int
        Atomic number.
    kappa : int
        Relativistic angular momentum quantum number (See eq 2.19b 'RADIAL: a Fortran subroutine by Francesc Salvat').
    sigma : int
        Value of -sign(kappa).

    Returns
    -------
    arr_a : ndarray
        Array of power series coefficients An.
    arr_b : ndarray
        Array of power series coefficients Bn.
    normalized_end_point_P : float
        Normalized value of P(r_b) = P(1)/|P(1)|.
    normalized_end_point_Q : float
        Normalized valued of Q(r_b) = Q(1)/|P(1)|.

    """
    u0 = u_array[0]
    u1 = u_array[1]
    u2 = u_array[2]
    u3 = u_array[3]


    S_a = S_b = 0

    a = []
    b = []

    if abs(u0) > 1e-6:
        t = 0
        if (Z/C)**2 > kappa**2:
            raise RuntimeError("s = sqrt(kappa^2 - u0^2) is imaginary")
        else:
            s = np.sqrt(kappa**2 - (Z/C)**2)

        temp_mult_term = u1 - 2*C*r_b

        def recurance_terms(n,An,Bn):
            an = 1/(n * (2*s + n)) * (-u0 * An - (s + n + sigma * abs(kappa)) * Bn)
            bn = 1/(n * (2*s + n)) * ((s + n - sigma * abs(kappa)) * An - u0 * Bn)

            return an , bn

        a0 = 1.0
        b0 = -(s - sigma * abs(kappa))/u0

        A1 = u1 * a0
        B1 = temp_mult_term * b0
        a1 , b1 = recurance_terms(1, A1, B1)

        A2 = u1 * a1 + u2 * a0
        B2 = temp_mult_term * b1 + u2 * b0

        a2, b2 = recurance_terms(2, A2, B2)

        a += [a0,a1,a2]
        b += [b0,b1, b2]

        S_a, Sa_c = 0.0 , 0.0
        S_b, Sb_c = 0.0 , 0.0

        S_an, San_c = 0.0 , 0.0
        S_bn, Sbn_c = 0.0, 0.0

        for i, ax in enumerate(a):
            S_a, Sa_c = Neumaier_add(S_a, Sa_c, ax)
            if i >= 1:
                S_an, San_c = Neumaier_add(S_an, San_c, i* ax)

        for i , bx in enumerate(b):
            S_b, Sb_c = Neumaier_add(S_b, Sb_c, bx)
            if i >=1:
                S_bn, Sbn_c = Neumaier_add(S_bn, Sbn_c, i * bx)

        n = 2
        while True:
            n +=1

            An = u1*a[n-1] + u2*a[n-2] + u3*a[n-3]
            Bn = temp_mult_term*b[n-1] + u2 * b[n-2] + u3 * b[n-3]

            an, bn = recurance_terms(n, An, Bn)

            a.append(an)
            b.append(bn)

            S_a, Sa_c = Neumaier_add(S_a, Sa_c, an)
            S_b, Sb_c = Neumaier_add(S_b, Sb_c, bn)
            S_an, San_c = Neumaier_add(S_an, San_c, n * an)
            S_bn, Sbn_c = Neumaier_add(S_bn, Sbn_c, n * bn)

            S_a_val = Neumaier_value(S_a, Sa_c)
            S_b_val = Neumaier_value(S_b, Sb_c)
            S_an_val = Neumaier_value(S_an, San_c)
            S_bn_val = Neumaier_value(S_bn, Sbn_c)

            tolerance = EPSILON * max(abs(S_a_val), abs(S_b_val), abs(S_an_val)/n, abs(S_bn_val)/n)
            if max(abs(an) , abs(bn)) < tolerance:

                condition1 = abs(r_b * (s * S_a_val + S_an_val) - sigma * abs(kappa) * r_b *S_a_val + ((u0+u1+u2+u3) - 2*C*r_b) * r_b * S_b_val)
                condition2 = abs(r_b * ((s+t)*S_b_val + S_bn_val) + sigma * abs(kappa) * r_b * S_b_val - (u0+u1+u2+u3)*r_b*S_a_val )

                if max(condition1 , condition2) < tolerance:
                    break

            if n > 499:
                raise RuntimeError(f"No convergence before n = {n}")


    elif sigma == 1:
        s = abs(kappa)
        t = 1

        b_term = 2*abs(kappa) + 1
        a_term = u1 - 2*C*r_b

        a0 = 1.0
        b0 = u1/b_term

        a1 = 0
        b1 = 1/(b_term + 1) * u2 *a0

        a2 = 0.5 * (-a_term * b0)
        b2 = 1/(b_term + 2) * (u1*a2 + u3*a0)

        a3 = 1/3 * (-a_term*b1 - u2*b0)
        b3 = 1/(b_term+3) * (u1*a3 + u2*a2)

        a.append(a0); a.append(a1); a.append(a2); a.append(a3)
        b.append(b0); b.append(b1); b.append(b2); b.append(b3)

        S_a = sum(a)
        S_b = sum(b)

        S_an = a1 + 2*a2 + 3*a3
        S_bn = b1 + 2*b2 + 3*b3

        n=3
        while True:
            n +=1

            an = 1/n * (-a_term * b[n-2] - u2 * b[n-3] - u3 * b[n-4])

            bn = 1/(b_term+n) * (u1 * an + u2*a[n-1] + u3*a[n-2])

            a.append(an)
            b.append(bn)

            S_a += an
            S_b += bn

            S_an += n * an
            S_bn += n * bn

            tolerance = EPSILON * max(abs(S_a), abs(S_b), abs(S_an)/n, abs(S_bn)/n)

            if max(abs(an) , abs(bn)) < tolerance:
                condition1 = abs(r_b * (s * S_a + S_an) - sigma * abs(kappa) * r_b *S_a + ((u0+u1+u2+u3) - 2*C*r_b) * r_b * S_b)
                condition2 = abs(r_b * ((s+t)*S_b + S_bn) + sigma * abs(kappa) * r_b * S_b - (u0+u1+u2+u3)*r_b*S_a )
                if max(condition1 , condition2)  < tolerance:
                    break

            if n > 499:
                raise RuntimeError(f"No convergence before n = {n}")

    else:  # sigma == -1
        s = abs(kappa) + 1
        t = -1

        a_term1 = u1 - 2*C*r_b
        a_term2 = 2*abs(kappa) + 1

        if -a_term1/a_term2 < 0:
            b0 = -1
        else:
            b0 = 1

        a0 = -a_term1/a_term2 * b0

        b1 = 0
        a1 = 1/(a_term2+1) * (-u2 * b0)

        b2 = 0.5 * u1 *a0
        a2 = 1/(a_term2+2) * (-a_term1*b2 - u3*b0)

        b3 = 1/3 * (u1*a1 + u2*a0)
        a3 = 1/(a_term2+3) * (-a_term1*b3 - u2*b2)


        a.append(a0); a.append(a1); a.append(a2); a.append(a3)
        b.append(b0); b.append(b1); b.append(b2); b.append(b3)

        S_a = sum(a)
        S_b = sum(b)

        S_an = a1 + 2*a2 + 3*a3
        S_bn = b1 + 2*b2 + 3*b3

        n = 3
        while True:
            n +=1

            bn = 1/n * (u1*a[n-2] + u2*a[n-3] + u3 * a[n-4])
            an = 1/(a_term2+n) * (-a_term1 * bn - u2 * b[n-1] - u3*b[n-2])

            a.append(an)
            b.append(bn)

            S_a += an
            S_b += bn

            S_an += n * an
            S_bn += n * bn

            tolerance = EPSILON * max(abs(S_a), abs(S_b), abs(S_an)/n, abs(S_bn)/n)

            if max(abs(an) , abs(bn)) < tolerance:
                condition1 = abs(r_b * (s * S_a + S_an) - sigma * abs(kappa) * r_b *S_a + ((u0+u1+u2+u3) - 2*C*r_b) * r_b * S_b)

                condition2 = abs(r_b * ((s+t)*S_b + S_bn) + sigma * abs(kappa) * r_b * S_b - (u0+u1+u2+u3)*r_b*S_a )
                if max(condition1 , condition2)  < tolerance:
                    break

            if n > 499:
                raise RuntimeError(f"No convergence before n = {n}")

    arr_a = np.array(a)
    arr_b = np.array(b)

    normalized_end_point_P = S_a/abs(S_a)
    normalized_end_point_Q = S_b/abs(S_a)

    return arr_a, arr_b, normalized_end_point_P, normalized_end_point_Q




# ========== Power series terms for general interval [r_a, r_b] ==========

def general_power_series_terms(u_array, initial_condition_a, initial_condition_b, r_a, r_b, l, kappa, sigma, verbose=False):
    """
    Computes the power series coefficients An, Bn on regular radial interval [r_a, r_b]

    Parameters
    ----------
    u_array : ndarray
        Cubic spline coefficients of the potential (u0, u1, u2, u3).
    initial_condition_a : float
        Initial condition to assign to A0.
    initial_condition_b : float
        Initial condition to assign to B0.
    r_a : float
        Lower radial coordinate of the interval [r_a, r_b]
    r_b : float
        Upper radial coordinate of the interval [r_a, r_b].
    l : int
        Orbital angular momentum quantum number.
    kappa : int
        Relativistic angular momentum quantum number (See eq 2.19b Radial Fortran subroutine by Francesc Salvat).
    sigma : int
        Value of -sign(kappa).

    Returns
    -------
    arr_a : ndarray
        Array of power series coefficients An.
    arr_b : ndarray
        Array of power series coefficients Bn.
    end_point_P : float
        Calculated P(r_b) value.
    end_point_Q : float
        Calculated Q(r_b) value.

    """
    # ========== Define potential parameters ==========
    u0, u1, u2, u3 = u_array
    u_sum = u0 + u1 + u2 + u3
    h = r_b - r_a

    pre_factor = h / max(r_a, R0)
    mult_term1 = u0 - 2.0 * C * r_a
    mult_term2 = u1 - 2.0 * C * h

    # ========== Initialize coefficient arrays ==========
    a = [initial_condition_a]
    b = [initial_condition_b]

    a0 = a[0]
    b0 = b[0]

    # ========== Calculate initial coefficients a1-a3, b1-b3 ==========
    a1 = -pre_factor * (kappa * a0 + mult_term1 * b0)
    b1 = pre_factor * (kappa * b0 + u0 * a0)

    a2 = -pre_factor / 2.0 * ((kappa + 1.0) * a1 + mult_term1 * b1 + mult_term2 * b0)
    b2 = pre_factor / 2.0 * ((kappa - 1.0) * b1 + u0 * a1 + u1 * a0)

    a3 = -pre_factor / 3.0 * ((kappa + 2.0) * a2 + mult_term1 * b2 + mult_term2 * b1 + u2 * b0)
    b3 = pre_factor / 3.0 * ((kappa - 2.0) * b2 + u0 * a2 + u1 * a1 + u2 * a0)

    a += [a1, a2, a3]
    b += [b1, b2, b3]

    # ========== Initialize sum accumulators with Neumaier compensation ==========
    S_a, Sa_c = 0.0, 0.0
    S_b, Sb_c = 0.0, 0.0
    S_an, San_c = 0.0, 0.0
    S_bn, Sbn_c = 0.0, 0.0

    # ========== Initial sum of coefficients ==========
    for i, ax in enumerate(a):
        S_a, Sa_c = Neumaier_add(S_a, Sa_c, ax)
        if i >= 1:
            S_an, San_c = Neumaier_add(S_an, San_c, i * ax)

    for i, bx in enumerate(b):
        S_b, Sb_c = Neumaier_add(S_b, Sb_c, bx)
        if i >= 1:
            S_bn, Sbn_c = Neumaier_add(S_bn, Sbn_c, i * bx)

    # ========== Iterative recurrence relation ==========
    n = 3
    tol_multiplier = 1.1
    stagnated = False

    while True:
        n += 1
        pf = pre_factor / n

        # Calculate recurrence for series coefficient An
        a_part = (
            (kappa - 1 + n) * a[n - 1]
            + mult_term1 * b[n - 1]
            + mult_term2 * b[n - 2]
            + u2 * b[n - 3]
            + u3 * b[n - 4]
        )
        an = -pf * a_part

        # Calculate recurrence for series coefficient Bn
        b_part = (
            (kappa + 1 - n) * b[n - 1]
            + u0 * a[n - 1]
            + u1 * a[n - 2]
            + u2 * a[n - 3]
            + u3 * a[n - 4]
        )
        bn = pf * b_part

        a.append(an)
        b.append(bn)

        # Accumulate sums with Neumaier compensation
        S_a, Sa_c = Neumaier_add(S_a, Sa_c, an)
        S_b, Sb_c = Neumaier_add(S_b, Sb_c, bn)
        S_an, San_c = Neumaier_add(S_an, San_c, n * an)
        S_bn, Sbn_c = Neumaier_add(S_bn, Sbn_c, n * bn)

        # Extract final sums with Neumaier correction
        S_a_val = Neumaier_value(S_a, Sa_c)
        S_b_val = Neumaier_value(S_b, Sb_c)
        S_an_val = Neumaier_value(S_an, San_c)
        S_bn_val = Neumaier_value(S_bn, Sbn_c)

        # ========== Stagnation detection ==========
        if not stagnated and max(abs(an), abs(bn)) < 1e-15:
            stagnated = True
            if verbose:
                print(
                    f"[series] stagnation at n={n}: "
                    f"|Δa|={abs(an):.2e}, |Δb|={abs(bn):.2e}; "
                    f"tolerance relaxation starts (initial multiplier {tol_multiplier:.2f})"
                )

        # ========== Convergence check ==========
        tolerance = EPSILON * max(
            abs(S_a_val), abs(S_b_val), abs(S_an_val) / n, abs(S_bn_val) / n
        )

        if max(abs(an), abs(bn)) < tolerance:
            # Verify system of equations
            t1 = r_b * S_an_val
            t2 = -sigma * abs(kappa) * h * S_a_val
            t3 = (u_sum - 2 * C * r_b) * h * S_b_val
            condition1 = t1 + t2 + t3

            t4 = r_b * S_bn_val
            t5 = sigma * abs(kappa) * h * S_b_val
            t6 = -u_sum * h * S_a_val
            condition2 = t4 + t5 + t6

            # Evaluate tolerance for system equations
            tol_res = EPSILON * max(abs(S_a_val), abs(S_b_val))
            if stagnated:
                tol_res *= tol_multiplier
                if verbose:
                    print(
                        f"[series] n={n}: tol_res relaxed ×{tol_multiplier:.2f} to {tol_res:.2e}"
                    )
                tol_multiplier += 0.05
          

            if max(abs(condition1), abs(condition2)) < tol_res:
                break

        # ========== Divergence check ==========
        if n > 499:
            raise RuntimeError(f"No convergence before n = {n}")

    # ========== Convert to numpy arrays ==========
    arr_a = np.array([float(x) for x in a], dtype=float)
    arr_b = np.array([float(x) for x in b], dtype=float)

    end_point_P = Neumaier_value(S_a, Sa_c)
    end_point_Q = Neumaier_value(S_b, Sb_c)

    return arr_a, arr_b, end_point_P, end_point_Q




# ========== Assemble series coefficients from all intervals ==========

def calc_series_terms(grid_points, l, T, Z, kappa, sigma, derivatives, ratio_max=0.05, max_subdiv=200, verbose=False):
    """
    Compute power-series coefficients over the full radial grid

    The first interval [0, r_b] is bootstrapped with 'singular_power_series_terms' using potential
    parameters evaluated at the spline's near-origin point. For each subsequent interval [r_a, r_b],
    parameters come from 'general_potential_parameters' and the solution is propagated with
    'general_power_series_terms'. Intervals where h/r_a exceeds ratio_max are automatically
    subdivided.

    Parameters
    ----------
    grid_points : ndarray
        1d grid steps over the total range [0,r]
    l : int
        Orbital angular momentum quantum number.
    T : float
        Kinetic energy in Hartree units.
    Z : int
        Atomic number.
    kappa : int
        Relativistic angular momentum quantum number (See eq 2.19b Radial Fortran subroutine by Francesc Salvat).
    sigma : int
        Value of -sign(kappa).
    derivatives : list
        List made of zero, first, second and third order derivatives of the cubic splined potential.
    ratio_max : float
        Maximum ratio to compare to h/r_a.
    max_subdiv : int
        Maximum amount of times a mesh [r_a,r_b] can be subdivided into smaller meshes.

    Returns
    -------
    Series_terms_list_a : list
        List of all power series coefficients An over range (0,r)
    Series_terms_list_b : list
        List of all power series coefficients Bn over range (0,r)

    """
    # ========== Compute first interval [0, r_b] using spline at R0 ==========
    u_parameters0 = general_potential_parameters(0, grid_points[1], l, T, derivatives)
    (
        Series_terms0a,
        Series_terms0b,
        initial_condition0a,
        initial_condition0b,
    ) = singular_power_series_terms(u_parameters0, grid_points[1], l, Z, kappa, sigma)

    initial_temp_condition_a = initial_condition0a
    initial_temp_condition_b = initial_condition0b

    Series_terms_list_a = [Series_terms0a]
    Series_terms_list_b = [Series_terms0b]

    # ========== Compute general intervals [r_a, r_b] with subdivision if needed ==========
    for i in range(1, len(grid_points) - 1):
        r_a = grid_points[i]
        r_b = grid_points[i + 1]

        h = r_b - r_a
        ratio = h / r_a

        if ratio > ratio_max:
            # Subdivide interval if ratio too large
            m = int(np.ceil(ratio / ratio_max))
            m = min(m, max_subdiv)

            sub_grid = np.linspace(r_a, r_b, m + 1)

            for j in range(m):
                ra = float(sub_grid[j])
                rb = float(sub_grid[j + 1])

                u_temp_parameter = general_potential_parameters(ra, rb, l, T, derivatives)

                (
                    Temp_series_terms_a,
                    Temp_series_terms_b,
                    initial_temp_condition_a,
                    initial_temp_condition_b,
                ) = general_power_series_terms(
                    u_temp_parameter,
                    initial_temp_condition_a,
                    initial_temp_condition_b,
                    ra,
                    rb,
                    l,
                    kappa,
                    sigma,
                    verbose=verbose,
                )

                if rb == r_b:
                    Series_terms_list_a.append(Temp_series_terms_a)
                    Series_terms_list_b.append(Temp_series_terms_b)
        else:
            # Process interval without subdivision
            u_temp_parameter = general_potential_parameters(r_a, r_b, l, T, derivatives)

            (
                Temp_series_terms_a,
                Temp_series_terms_b,
                initial_temp_condition_a,
                initial_temp_condition_b,
            ) = general_power_series_terms(
                u_temp_parameter,
                initial_temp_condition_a,
                initial_temp_condition_b,
                r_a,
                r_b,
                l,
                kappa,
                sigma,
                verbose=verbose,
            )

            Series_terms_list_a.append(Temp_series_terms_a)
            Series_terms_list_b.append(Temp_series_terms_b)

    return Series_terms_list_a, Series_terms_list_b


# ========== Compute wavefunctions from power series coefficients ==========

def solve_power_series(list_of_sequence_terms_a, list_of_sequence_terms_b, grid_points):
    """
    Calculates the wavefunctions P and Q on the entire range [0,r] by calculating all the local
    wavefunctions P and Q on each sub-interval [r_a,r_b] using the power series coefficients
    An, Bn for P and Q respectively obtained through 'calc_series_terms' and stitching the resulting
    wave functions together to obtain P(r) and Q(r) over the total range [0,r]

    Parameters
    ----------
    list_of_sequence_terms_a : list
        List of all power series sequence terms An
    list_of_sequence_terms_b : list
        List of all power series sequence terms Bn
    grid_points : ndarray
        1d grid steps over the total range [0,r]

    Returns
    -------
    P_array : ndarray
        Wavefunction values of P(r)
    Q_array : ndarray
        Wavefunction values of Q(r)
    """

    # ========== Initialize arrays ==========
    r_mesh = np.asarray(grid_points, dtype=float)
    r_size = len(r_mesh)

    P_array = np.zeros(r_size, dtype=float)
    Q_array = np.zeros(r_size, dtype=float)

    P_array[0] = 0.0
    Q_array[0] = 0.0

    # ========== Calculate wavefunction values at grid points ==========
    a0 = np.asarray(list_of_sequence_terms_a[0], dtype=float)
    b0 = np.asarray(list_of_sequence_terms_b[0], dtype=float)

    # At r=r_b for first interval: P(r_b) = sum A_n and Q(r_b) = sum B_n
    P_array[1] = float(np.sum(a0))
    Q_array[1] = float(np.sum(b0))

    # For remaining intervals
    for i in range(1, r_size - 1):
        ai = np.asarray(list_of_sequence_terms_a[i], dtype=float)
        bi = np.asarray(list_of_sequence_terms_b[i], dtype=float)

        P_array[i + 1] = float(np.sum(ai))
        Q_array[i + 1] = float(np.sum(bi))

    return P_array, Q_array

# ========== Calculate normalization constant and phase shift ==========

def Normalization_Constant(grid_points, idx_rc, P_array, Q_array, Analytic_normalization_const, T, k, eta, W, kappa, Z, Lambda, sigma, R_au, potential_index, phi_r, Z_match=None):
    """
    Calculates the normalization constant to normalize the wavefunctions to unity as described in 'RADIAL: a Fortran subroutine by Francesc Salvat'

    For screened potentials (e.g. potential_index 3), the true potential at the matching
    radius r_c is governed by the screened residual charge, not the bare nuclear charge Z.
    `eta`/`Lambda` (and the analytic Coulomb wavefunctions built from them) must therefore
    already be computed from that residual charge by the caller; `Z_match` is the matching
    charge value used in the Dirac spinor mixing coefficients. `Z` itself is still used to
    evaluate the true local potential at r_c (which already accounts for screening via
    `phi_r`), since that part of the matching condition is exact regardless of the
    asymptotic charge assumption.

    Parameters
    ----------
    grid_points : ndarray
        1d grid steps over the total range [0,r].
    idx_rc : int
        Index value at which grid_points ndarray value matches the free electron wave matching radius.
    P_array : ndarray
        Wavefunction values of P(r).
    Q_array : ndarray
        Wavefunction values of Q(r).
    Analytic_normalization_const : float
        Analytic normalization constant as calculated in eq 3.122 in 'RADIAL: a Fortran subroutine by Francesc Salvat'.
    T : float
        Kinetic energy in Hartree units.
    k : float
        Relativistic wave number.
    eta : float
        Relativistic Sommerfeld parameter.
    W : float
        Total relativistic energy (kinetic + rest energy).
    kappa : int
        Relativistic angular momentum quantum number (See eq 2.19b Radial Fortran subroutine by Francesc Salvat).
    Z : int
        Atomic number.
    Lambda :
        Lambda parameter defined as sqrt(kappa^2 - (alpha Z)^2).
    sigma : int
        Value of -sign(kappa).
    R_au : float
        Nuclear radius in Atomic units, defined by R = 1.2e-15 * A^(1/3) / AU.
    potential_index: int
        Index specifying which potential model to use.
    phi_r : callable
        Function phi(r) used in the Thomas-Fermi screening potential.
    Z_match : float, optional
        Charge used in the Dirac spinor mixing coefficients of the asymptotic Coulomb
        matching functions. Defaults to Z. Should be the screened residual charge for
        screened potentials (e.g. potential_index 3), where it differs from Z.

    Returns
    -------
    A : float
        Normalization constant to set P and Q to unit amplitude.
    delta : float
        Phase shift of the wave functions P and Q.
    r_c : float
        Matching radius point on grid_points.

    """
    P = P_array[idx_rc]
    Q = Q_array[idx_rc]

    r_c = grid_points[idx_rc]
    x = k * r_c


    if Z_match is None:
        Z_match = Z

    if abs(Z) == 0:
        l = (-kappa -1) if (kappa < 0) else kappa
        mult_fact = np.sqrt(T/(T + 2*C**2))


        nu = l + 0.5
        pref = mp.sqrt(mp.pi/(2*x))
        jl = pref * mp.besselj(nu, x)
        yl = pref * mp.bessely(nu, x)

        nu1 = (l+1) + 0.5
        jl1 = pref * mp.besselj(nu1, x)
        yl1 = pref * mp.bessely(nu1, x)

        if sigma == 1:
            jlp = jl1
            ylp = yl1
        else:
            nu_m1 = (l-1) + 0.5
            jlm1 = pref * mp.besselj(nu_m1, x)
            ylm1 = pref * mp.bessely(nu_m1, x)
            jlp = jlm1
            ylp = ylm1

        PIA = x * jl
        PIB = -x * yl
        QA = -mult_fact * sigma * x *jlp
        QB = mult_fact * sigma * x * ylp

    else:
        F_l = mp.coulombf(Lambda, eta, x)
        G_l = mp.coulombg(Lambda, eta ,x)
        F_lm = mp.coulombf(Lambda-1, eta, x)
        G_lm = mp.coulombg(Lambda-1 , eta, x)


        mult_term_addition = (np.sqrt(Lambda**2 + eta**2))*k*C
        mult_term_subtraction = (Lambda * C**2 - kappa * W)

        def upper_Dirac_func(Coulomb_func, Coulomb_min_func):
            return Analytic_normalization_const * ((kappa + Lambda) * mult_term_addition* Coulomb_func + Z_match/C * mult_term_subtraction * Coulomb_min_func)


        def lower_Dirac_func(Coulomb_func, Coulomb_min_func):
            return - Analytic_normalization_const * (Z_match/C * mult_term_addition * Coulomb_func + (kappa + Lambda) * mult_term_subtraction * Coulomb_min_func)

        PIA = upper_Dirac_func(F_l, F_lm)
        QA = lower_Dirac_func(F_l, F_lm)

        PIB = upper_Dirac_func(G_l, G_lm)
        QB = lower_Dirac_func(G_l, G_lm)

    V = float(potential(r_c, Z, R_au, potential_index, phi_r))
    FG = (T - V + 2*C**2)/C

    P_prime = -kappa/r_c * P + FG * Q

    PIAP = -kappa * PIA / r_c + FG * QA
    PIBP = -kappa * PIB / r_c + FG * QB

    delta = mp.atan2(P_prime * PIA - P * PIAP, P * PIBP - P_prime * PIB)

    # reduce to (-pi/2, pi/2) intervals
    PIH = mp.pi/2
    TT = abs(delta)
    if TT > PIH:
        delta = delta * (1 - mp.pi/TT)

    COS = mp.cos(delta)
    SIN = mp.sin(delta)

    if abs(P) > EPSILON:
        A = (COS * PIA + SIN * PIB) / P
    else:
        A = (COS * PIAP + SIN * PIBP) / P_prime

    return float(A) , float(delta), r_c




# ========== Visualization of wavefunctions ==========

def Visualize(config,cnf):
    ### DECLARE PARAMETERS
    verbose = config.get("verbose", False)
    A = cnf["A"]
    Zf = cnf["Z"]
    Z = -Zf
    angular_momentum_quantum_number = cnf["angular_momentum_l"]
    kappa = cnf["kappa"]
    sigma = -np.sign(kappa)
    Lambda = np.sqrt(kappa**2 - (Z/C)**2)

    rN = 1.2e-15 * A ** (1 / 3) # Nuclear radius
    R_au = rN/AU # Nuclear radius in au

    potential_index = config["generator"]["potential_index"]
    if potential_index == 3:
        if verbose:
            print("calculating thomas fermi func")
        phi_r = make_phi_r(Zf)
    else:
        phi_r = 0

    Plotting_energy = config["generator"]["T_plot_energy-MeV"]
    T = Plotting_energy * 1e6 / E_HARTREE
    W = T + C**2
    k = np.sqrt(T*(T + 2*C**2))/C
    eta =  ALPHA* Z *W/(k*C)
    Analytic_normalization_const = 1 /Lambda * 1/np.sqrt((Z/C)**2 * (W+C**2)**2 + (kappa + Lambda)**2 *(k*C)**2)


    r_N = config["mesh_grid"]["num_mesh_steps"]
    r2 = config["mesh_grid"]["second_point"]
    DRN = config["mesh_grid"]["upper_limit_step_size"]#Upper limit on distance between points near the end regime (make sure this stays small enough or wavefunc will not converge at larger distances r)


    r_END = obtain_mesh_range(k, Z, R_au, potential_index, phi_r, potential, verbose=verbose)

    if r_N is None:
        r_N = 10000

    A_grid = ((r_END - (r_N - 1) *r2) / r_END) * (DRN / (DRN -r2))

    while not (0.5 < A_grid < 1.0):
        r_N += 5000
        A_grid = ((r_END - (r_N - 1) *r2) / r_END) * (DRN / (DRN -r2))

    if verbose:
        print(f"resolution set to {r_N} mesh points")
    mesh_points = radial_grid(r_END, A_grid, r_N, r2, DRN, R_au)
    rc_idx = find_rc(mesh_points, k, Z, R_au, potential_index, phi_r, potential)

    ###Setting up cubic spline
    N_V = 50000
    r_V = np.linspace(R0,r_END, N_V)

    RV = r_V * potential(r_V,Z, R_au, potential_index, phi_r)

    RV_spline = CubicSpline(r_V, RV)

    RV_spline_prime = RV_spline.derivative(1)
    RV_spline_second = RV_spline.derivative(2)
    RV_spline_third = RV_spline.derivative(3)

    derivatives = [RV_spline, RV_spline_prime, RV_spline_second, RV_spline_third]



    series_terms_upper, series_terms_lower = calc_series_terms(mesh_points, angular_momentum_quantum_number, T, Z, kappa, sigma, derivatives, verbose=verbose)
    wave_function_upper, wave_function_lower = solve_power_series(series_terms_upper, series_terms_lower, mesh_points)

    # The matching radius r_c lies where the *true* potential (screened, for potential_index 3)
    # has converged to its own asymptote, not where it equals the bare nuclear charge. The
    # analytic Coulomb matching functions must therefore be built from that asymptotic charge.
    if potential_index == 3:
        Z_match = -2.0 if Z < 0 else 2.0
        eta_match = ALPHA* Z_match *W/(k*C)
        Lambda_match = np.sqrt(kappa**2 - (Z_match/C)**2)
        Analytic_normalization_const_match = 1 /Lambda_match * 1/np.sqrt((Z_match/C)**2 * (W+C**2)**2 + (kappa + Lambda_match)**2 *(k*C)**2)
    else:
        Z_match = Z
        eta_match = eta
        Lambda_match = Lambda
        Analytic_normalization_const_match = Analytic_normalization_const

    N, delta, r_c = Normalization_Constant(mesh_points, rc_idx ,wave_function_upper,wave_function_lower, Analytic_normalization_const_match, T, k, eta_match, W, kappa, Z, Lambda_match, sigma, R_au, potential_index, phi_r, Z_match)

    plt.figure(figsize=(12,8))
    plt.plot(mesh_points, wave_function_upper*N , label = "P(r) - Normalized")
    plt.plot(mesh_points, wave_function_lower*N , label = "Q(r) - Normalized")
    plt.axvline(x = R_au, color = "gray" , ls = "--", label = f"r = {R_au}")
    plt.axvline(x = r_c, color = "black", ls = "--", label = f"rc = {r_c}")

    plt.xlabel("r in a.u")
    plt.ylabel("P(r)")
    plt.grid(True)
    plt.title(f"E = {T: .2g}, Z = {Z}, A = {N: .5g}, delta = {delta: .3g}, N = {r_N}, DRN = {DRN} , r2 = {r2}, r_end = {r_END}, r_c = {r_c: .3g}")
    plt.legend()
    if verbose:
        plt.show()


def Generate_Fermi_Data(config,cnf):
    ### DECLARE PARAMETERS
    verbose = config.get("verbose", False)
    A = cnf["A"]
    Zf = cnf["Z"]
    Z = -Zf
    angular_momentum_quantum_number = cnf["angular_momentum_l"]

    rN = 1.2e-15 * A ** (1 / 3) # Nuclear radius
    R_au = rN/AU # Nuclear radius in au


    potential_index = config["generator"]["potential_index"]
    if potential_index == 3:
        if verbose:
            print("calculating thomas fermi func")
        phi_r = make_phi_r(Zf)
    else:
        phi_r = 0

    Q = cnf["Q"]
    T_start = config["generator"]["T_start-MeV"]
    n_samples = config["generator"]["num_samples"]

    T_range = np.geomspace(T_start, Q, n_samples) * 1e6 /E_HARTREE



    r_N = config["mesh_grid"]["num_mesh_steps"]
    r2 = config["mesh_grid"]["second_point"]
    DRN = config["mesh_grid"]["upper_limit_step_size"]#Upper limit on distance between points near the end regime (make sure this stays small enough or wavefunc will not converge at larger distances r)

    k_max = np.sqrt(T_range[0]*(T_range[0] + 2*C**2))/C
    r_END = obtain_mesh_range(k_max, Z, R_au, potential_index, phi_r, potential, verbose=verbose)

    if r_N is None:
        r_N = 10000

    A_grid = ((r_END - (r_N - 1) *r2) / r_END) * (DRN / (DRN -r2))

    while not (0.5 < A_grid < 1.0):
        r_N += 10000
        A_grid = ((r_END - (r_N - 1) *r2) / r_END) * (DRN / (DRN -r2))

    if verbose:
        print(f"resolution set to {r_N} mesh points")
    mesh_points = radial_grid(r_END, A_grid, r_N, r2, DRN, R_au)

    if verbose:
        r = mesh_points
        N_r = len(r)
        drs = np.diff(r)
        with open("grid.txt", "w") as gf:
            gf.write(f"Grid: N={N_r}  r_END={r[-1]:.6g} a.u.  r2={r2}  DRN={DRN}  A_grid={A_grid:.6f}\n")
            gf.write(f"{'i':>8}  {'r (a.u.)':>18}  {'dr (a.u.)':>18}\n")
            gf.write("-" * 50 + "\n")

            def _row(i):
                step = drs[i - 1] if i > 0 else float("nan")
                gf.write(f"{i:>8}  {r[i]:>18.8e}  {step:>18.8e}\n")

            gf.write("# first 11 points\n")
            for i in range(11):
                _row(i)
            for frac in [1, 2, 3, 4, 5]:
                mid = N_r * frac // 6
                gf.write(f"# --- ~{frac}/6 of grid (i={mid}) ---\n")
                _row(mid)
            gf.write("# last 11 points\n")
            for i in range(N_r - 11, N_r):
                _row(i)
        print("Grid sample written to grid.txt")

    ###Setting up cubic spline
    N_V = 50000
    N_inner = 20000
    r_V_inner = np.linspace(R0, R_au, N_inner)        # dense inside nucleus
    r_V_outer = np.linspace(R_au, r_END, N_V)         # same outer density as before
    r_V = np.concatenate([r_V_inner, r_V_outer[1:]])  # drop duplicate at R_au

    RV = r_V * potential(r_V,Z, R_au, potential_index, phi_r)

    RV_spline = CubicSpline(r_V, RV)
    


    RV_spline_prime = RV_spline.derivative(1)
    RV_spline_second = RV_spline.derivative(2)
    RV_spline_third = RV_spline.derivative(3)

    derivatives = [RV_spline, RV_spline_prime, RV_spline_second, RV_spline_third]

    kappa = config["parameters"]["kappa"]
    kappa_sign = [1,-1]

    for i , sign in enumerate(kappa_sign):
        kappa *= sign
        if verbose:
            print(f"Performing run ({i+1}/{len(kappa_sign)}) with kappa = {kappa:+d} ")
        sigma = -np.sign(kappa)
        Lambda = np.sqrt(kappa**2 - (Z/C)**2)

        # The matching radius r_c lies where the *true* potential (screened, for potential_index 3)
        # has converged to its own asymptote, not where it equals the bare nuclear charge. The
        # analytic Coulomb matching functions must therefore be built from that asymptotic charge.
        if potential_index == 3:
            Z_match = -2.0 if Z < 0 else 2.0
            Lambda_match = np.sqrt(kappa**2 - (Z_match/C)**2)
        else:
            Z_match = Z
            Lambda_match = Lambda

        P_list = []
        Q_list = []


        for i, T in enumerate(tqdm(T_range, desc = f"Solving Dirac functions over energy range for kappa = {kappa:+d}")):
            W = T + C**2
            k = np.sqrt(T*(T + 2*C**2))/C
            eta = ALPHA* Z*W/(k*C)
            Analytic_normalization_const = 1 /Lambda * 1/np.sqrt((Z/C)**2 * (W+C**2)**2 + (kappa + Lambda)**2 *(k*C)**2)

            if potential_index == 3:
                eta_match = ALPHA* Z_match *W/(k*C)
                Analytic_normalization_const_match = 1 /Lambda_match * 1/np.sqrt((Z_match/C)**2 * (W+C**2)**2 + (kappa + Lambda_match)**2 *(k*C)**2)
            else:
                eta_match = eta
                Analytic_normalization_const_match = Analytic_normalization_const

            if verbose:
                print(f"Finding wave function for T = {T}, iter = {i}, eta = {eta}")
            rc_idx = find_rc(mesh_points, k, Z, R_au, potential_index, phi_r, potential, verbose)
            series_terms_upper, series_terms_lower = calc_series_terms(mesh_points, angular_momentum_quantum_number, T, Z, kappa, sigma, derivatives, verbose=verbose)
            wave_function_upper, wave_function_lower = solve_power_series(series_terms_upper, series_terms_lower, mesh_points)
            N, delta, r_c = Normalization_Constant(mesh_points, rc_idx ,wave_function_upper,wave_function_lower, Analytic_normalization_const_match, T, k, eta_match, W, kappa, Z, Lambda_match, sigma, R_au, potential_index, phi_r, Z_match)

            P_list.append(N*wave_function_upper)
            Q_list.append(N*wave_function_lower)


        P_arr = np.asarray(P_list, dtype = float)
        Q_arr = np.asarray(Q_list, dtype = float)
        r_arr = np.asarray(mesh_points, dtype = float)
        T_arr = np.asarray(T_range, dtype = float)

        filename = f"{config['isotope']}_potential_{potential_index}_kappa_{kappa:+d}_Z{Zf}_A{A}.npz"

        main_output_directory = Path(config["paths"]["output_directory"]) / f"Dirac_{config['isotope']}"
        NPZ_output_directory = main_output_directory / "NPZ_files"
        NPZ_output_directory.mkdir(parents=True, exist_ok=True)
        file_path = NPZ_output_directory / filename

        np.savez_compressed(file_path, T = T_arr, r = r_arr, P = P_arr, Q = Q_arr)

    if config.get("generate_z0", False):
        # ---- Z=0 reference wavefunctions (free particle, V=0) ----
        # Generated with --verbose (unless --no-z0 is set); needed by --mode data
        # for the Fermi division plot.
        main_output_directory_z0 = Path(config["paths"]["output_directory"]) / f"Dirac_{config['isotope']}"
        NPZ_output_directory_z0 = main_output_directory_z0 / "NPZ_files"
        NPZ_output_directory_z0.mkdir(parents=True, exist_ok=True)
        T_arr_z0 = np.asarray(T_range, dtype=float)
        r_arr_z0 = np.asarray(mesh_points, dtype=float)

        RV_zero = np.zeros_like(r_V)
        cs0 = CubicSpline(r_V, RV_zero)
        derivatives_zero = [cs0, cs0.derivative(1), cs0.derivative(2), cs0.derivative(3)]

        kappa = config["parameters"]["kappa"]
        for i, sign in enumerate(kappa_sign):
            kappa *= sign
            if verbose:
                print(f"Z=0 run ({i+1}/{len(kappa_sign)}) with kappa = {kappa:+d}")
            sigma = -np.sign(kappa)
            Lambda_z0 = float(abs(kappa))  # sqrt(kappa^2 - 0)

            P_list_z0 = []
            Q_list_z0 = []

            for _, T in enumerate(tqdm(T_range, desc=f"Z=0 kappa={kappa:+d}")):
                W = T + C**2
                k = np.sqrt(T * (T + 2 * C**2)) / C
                rc_idx = find_rc(mesh_points, k, 0, R_au, 0, 0, potential)
                series_terms_upper, series_terms_lower = calc_series_terms(
                    mesh_points, angular_momentum_quantum_number, T, 0, kappa, sigma, derivatives_zero, verbose=verbose)
                wave_function_upper, wave_function_lower = solve_power_series(
                    series_terms_upper, series_terms_lower, mesh_points)
                N, _, _ = Normalization_Constant(
                    mesh_points, rc_idx, wave_function_upper, wave_function_lower,
                    0.0, T, k, 0.0, W, kappa, 0, Lambda_z0, sigma, R_au, 0, 0)

                P_list_z0.append(N * wave_function_upper)
                Q_list_z0.append(N * wave_function_lower)

            filename_z0 = f"{config['isotope']}_potential_{potential_index}_kappa_{kappa:+d}_Z0_A{A}.npz"
            file_path_z0 = NPZ_output_directory_z0 / filename_z0
            np.savez_compressed(file_path_z0,
                                T=T_arr_z0, r=r_arr_z0,
                                P=np.asarray(P_list_z0, dtype=float),
                                Q=np.asarray(Q_list_z0, dtype=float))



