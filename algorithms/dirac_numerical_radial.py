#!/usr/bin/env python

#Importing libraries
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq
import mpmath as mp
import os
from scipy.special import spherical_jn
from tqdm import tqdm
import time
import math
from pathlib import Path
from configurations.physics_constants import ALPHA, C, EPSILON, AU, E_HARTREE, R0
from algorithms.thomas_fermi import make_phi_r
from algorithms.grid_creation import find_x, radial_grid, obtain_mesh_range, find_rc




mp.mp.dps = 30



###################################
###Setup Cubic Spline potential V
###################################

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


    Z_sing = Z  ###Define the singular potential term which blows up at r ->0

    if potential_index == 0:
        return Z_sing/r

    elif potential_index == 1:
        V0= -0.5
        A = 1
        return Z_sing/r + V0 * np.exp(-A*r)

    elif potential_index == 2:
        r_arr = np.asarray(r,dtype = float)
        V = Z_sing/ r_arr
        inside = r_arr < R_au
        V = np.where(inside, Z_sing/(2*R_au) * (3 - (r_arr/R_au)**2), V)

        if np.isscalar(r):
            return float(V)
        else:
            return V

    elif potential_index == 3:
        r_arr = np.asarray(r, dtype =float)

        V_A =  potential(r,Z,R_au, 2,0)
        phi_vals = phi_r(r_arr)

        V_C = ((r_arr * V_A + 2.0 ) * phi_vals - 2.0) / r_arr

        if np.isscalar(r):
            return float(V_C)
        return V_C

    else:
        raise RuntimeError("No proper potential function idex number was selected")


#############################################################################################################
### Calculate the potential parameters u0,u1,u2,u3 in accordance to RADIAL for the initial mesh step [0,rb]
#############################################################################################################

def singular_potential_parameters(r_b, T, Z, R_au, potential_index, phi_r):
    """
    Generate cubic splined potential parameters (u0,u1,u2,u3) on singular range [0,r_b]

    Parameters
    ----------
    r_b : float
        Final radial coordinate in given segment [0,r_b].
    T : float
        Kinetic energy in Hartree units.
    Z : int
        Atomic number.
    R_au : float
        Nuclear radius in Atomic units, defined by R = 1.2e-15 * A^(1/3) /AU.
    potential_index: int
        Index specifying which potential model to use.
    phi_r : callable
        Function phi(r) used in the Thomas-Fermi screening potential.

    Returns
    -------
    u0 : float
        First polynomial coefficient of cubic spline.
    u1 : float
        Second polynomial coefficient of cubic spline.
    u2 : float
        Thrid polynomial coefficient of cubic spline.
    u3 : float
        Fourth polynomial coefficient of cubic spline.

    """
    h = r_b
    Z_sing = Z

    if potential_index == 0:
        V_reg = lambda r: potential(r, Z, R_au, potential_index, phi_r) - Z_sing/r
        u0 = ALPHA * Z_sing
    elif potential_index in (1,2,3):
        V_reg = lambda r: potential(r, Z, R_au, potential_index, phi_r)
        u0 = 0.0
    else:
        raise RuntimeError("No proper potential potential_index selected")

    v0 = float(V_reg(R0))
    dr = 1e-8
    v1 = (V_reg(R0+dr) - V_reg(R0-dr)) / (2*dr)
    v2 = (V_reg(R0+dr) - 2*V_reg(R0) + V_reg(R0-dr)) / (dr**2)

    u1 = ALPHA * h *(v0-T)
    u2 = ALPHA * h**2 * (v1)
    u3 = ALPHA * h**3 * (0.5*v2)

    return u0, u1, u2, u3

#####################################################################################################################
### Calculate the potential parameters u0,u1,u2,u3 in accordance to RADIAL for the any mesh step [ra,rb] for ra != 0
#####################################################################################################################

def general_potential_parameters(r_a ,r_b, l, T, derivatives):
    """
    Generate cubic splined potential parameters (u0,u1,u2,u3) on range [r_a,r_b]

    Parameters
    ----------
    r_a : float
        Lower radial boundary of the interval [r_a, r_b].
    r_b : float
        Upper radial boundary of the interval [0,r_b].
    l : int
        Orbital angular momentum quantum number.
    T : float
        Kinetic energy in Hartree units.
    derivatives : list
        List made of zero, first, second and thrid order derivatives of the cubic splined potential

    Returns
    -------
    u0 : float
        First polynomial coefficient of cubic spline.
    u1 : float
        Second polynomial coefficient of cubic spline.
    u2 : float
        Thrid polynomial coefficient of cubic spline.
    u3 : float
        Fourth polynomial coefficient of cubic spline.

    """
    ra = max(r_a, R0)
    rb = r_b

    rv0 = float(derivatives[0](ra))
    rv1 = float(derivatives[1](ra))
    rv2 = 0.5 * float(derivatives[2](ra))
    rv3 = (1.0/6.0) * float(derivatives[3](ra))

    v0 = rv0 - rv1*r_a + rv2*(r_a**2) - rv3* (r_a**3)
    v1 = rv1 - 2.0* rv2 *r_a + 3.0 * rv3 * (r_a **2)
    v2 = rv2 - 3.0*rv3 *r_a
    v3 = rv3

    h = r_b-r_a

    u0 = ALPHA * (v0 + (v1-T)*r_a + v2*r_a**2 + v3*r_a**3)
    u1 = ALPHA *h * ((v1-T) + 2*v2*r_a + 3*v3*r_a**2)
    u2 = ALPHA *h**2 * (v2 + 3*v3*r_a)
    u3 = ALPHA *v3*h**3


    return u0, u1, u2, u3

####################################################################################################################
### Setting up Neaumaier addition functions to enhance floating point precision in the final digits during addition
### This is needed for the function Power_series_Terms_Dirac to properly converge
####################################################################################################################

def Neaumaier_add(sum_, c, x):
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
        c += (sum_ -t) + x
    else:
        c += (x-t) + sum_
    return t , c

def Neaumaier_value(sum_, c):
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


##############################################################################################
### Finding the power series terms for the initial mesh step [0,rb] in accordance with RADIAL
##############################################################################################

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

    if u0 != 0:
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

        # S_an = a1 + 2*a2
        # S_bn = b1 + 2*b2

        for i, ax in enumerate(a):
            S_a, Sa_c = Neaumaier_add(S_a, Sa_c, ax)
            if i >= 1:
                S_an, San_c = Neaumaier_add(S_an, San_c, i* ax)

        for i , bx in enumerate(b):
            S_b, Sb_c = Neaumaier_add(S_b, Sb_c, bx)
            if i >=1:
                S_bn, Sbn_c = Neaumaier_add(S_bn, Sbn_c, i * bx)

        n = 2
        while True:
            n +=1

            An = u1*a[n-1] + u2*a[n-2] + u3*a[n-3]
            Bn = temp_mult_term*b[n-1] + u2 * b[n-2] + u3 * b[n-3]

            an, bn = recurance_terms(n, An, Bn)

            a.append(an)
            b.append(bn)

            S_a, Sa_c = Neaumaier_add(S_a, Sa_c, an)
            S_b, Sb_c = Neaumaier_add(S_b, Sb_c, bn)
            S_an, San_c = Neaumaier_add(S_an, San_c, n * an)
            S_bn, Sbn_c = Neaumaier_add(S_bn, Sbn_c, n * bn)

            S_a_val = Neaumaier_value(S_a, Sa_c)
            S_b_val = Neaumaier_value(S_b, Sb_c)
            S_an_val = Neaumaier_value(S_an, San_c)
            S_bn_val = Neaumaier_value(S_bn, Sbn_c)

            # S_a += an
            # S_b += bn
            #
            # S_an += n * an
            # S_bn += n * bn


            tolerance = EPSILON * max(abs(S_a_val), abs(S_b_val), abs(S_an_val)/n, abs(S_bn_val)/n)
            if max(abs(an) , abs(bn)) < tolerance:

                condition1 = abs(r_b * (s * S_a_val + S_an_val) - sigma * abs(kappa) * r_b *S_a_val + ((u0+u1+u2+u3) - 2*C*r_b) * r_b * S_b_val)
                condition2 = abs(r_b * ((s+t)*S_b_val + S_bn_val) + sigma * abs(kappa) * r_b * S_b - (u0+u1+u2+u3)*r_b*S_a_val )

                if max(condition1 , condition2) < tolerance:
                    break

            if n > 499:
                raise RuntimeError(f"No convergenvce before n = {n}")


    elif (u0 == 0 and sigma == 1):
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
                raise RuntimeError(f"No convergenvce before n = {n}")

    elif (u0 == 0 and sigma == -1):
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
                raise RuntimeError(f"No convergenvce before n = {n}")
    else:
        raise RuntimeError("None of the singular u_0, sigma conditions were met")

    arr_a = np.array(a)
    arr_b = np.array(b)

    normalized_end_point_P = S_a/abs(S_a)
    normalized_end_point_Q = S_b/abs(S_a)

    return arr_a, arr_b, normalized_end_point_P, normalized_end_point_Q




######################################################################################################
### Finding the power series terms for the any mesh step [ra,rb] for ra !=0 in accordance with RADIAL
######################################################################################################

def general_power_series_terms(u_array, initial_condition_a, initial_condition_b , r_a, r_b, l, kappa, sigma):
    """
    Computes the power series coefficients An, Bn on regualr radial interval [r_a,r_b]

    Parameters
    ----------
    u_array : ndarray
        Cubic spline coefficients of the potential (u0,u1,u2,u3).
    initial_condition_a : float
        Initial condition to assign to A0.
    initial_condition_b : float
        # Initial condition to assign to B0.
    r_a : float
        Lower radial coordiante of the interval [r_a,r_b]
    r_b : float
        Upper radial coordinate of the interval [r_a,r_b].
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
        # Calculated P(r_b) value.
    end_point_Q : float
        Calculated Q(r_b) value.

    """
    #Define the u terms
    u0, u1, u2, u3 = u_array
    u_sum = u0 + u1 + u2 + u3
    h = r_b-r_a

    pre_factor = h/max(r_a,R0)
    mult_term1 = u0 - 2.0*C*r_a
    mult_term2 = u1 - 2.0*C*h
    #create the a array and define the intial conditions
    a = [initial_condition_a]
    b = [initial_condition_b]

    a0 = a[0] ; b0 = b[0]

    a1 = -pre_factor * (kappa*a0 + mult_term1*b0)
    b1 = pre_factor * (kappa*b0 + u0*a0)

    a2 = -pre_factor/2.0 * ((kappa + 1.0)*a1 + mult_term1*b1 + mult_term2*b0)
    b2 = pre_factor/2.0 * ((kappa - 1.0)*b1 + u0*a1 + u1*a0)

    a3 = -pre_factor/3.0 * ((kappa + 2.0)*a2 + mult_term1*b2 + mult_term2*b1 + u2*b0)
    b3 = pre_factor/3.0 * ((kappa - 2.0)*b2 + u0*a2 + u1*a1 + u2*a0)

    a += [a1, a2, a3]
    b += [b1, b2, b3]

    S_a , Sa_c = 0.0 , 0.0
    S_b , Sb_c = 0.0 , 0.0
    S_an , San_c = 0.0 , 0.0
    S_bn , Sbn_c = 0.0 , 0.0

    for i , ax in enumerate(a):
        S_a, Sa_c = Neaumaier_add(S_a, Sa_c, ax)
        if i >=1:
            S_an, San_c = Neaumaier_add(S_an, San_c, i * ax)


    for i , bx in enumerate(b):
        S_b, Sb_c = Neaumaier_add(S_b, Sb_c, bx)
        if i >=1:
            S_bn, Sbn_c = Neaumaier_add(S_bn, Sbn_c, i * bx)

    n=3
    tol_multiplier = 1.1
    while True:
        n +=1
        pf = pre_factor / n

        a_part = (kappa - 1 + n) * a[n-1] + mult_term1 * b[n-1] + mult_term2 * b[n-2] + u2 * b[n-3] + u3 * b[n-4]
        an = -pf * a_part

        b_part = (kappa + 1 - n) * b[n-1] + u0 * a[n-1] + u1 * a[n-2] + u2 * a[n-3] + u3 * a[n-4]
        bn = pf * b_part

        a.append(an); b.append(bn)

        S_a, Sa_c = Neaumaier_add(S_a, Sa_c, an)
        S_b, Sb_c = Neaumaier_add(S_b, Sb_c, bn)
        S_an, San_c = Neaumaier_add(S_an, San_c, n * an)
        S_bn, Sbn_c = Neaumaier_add(S_bn, Sbn_c, n * bn)

        S_a_val = Neaumaier_value(S_a, Sa_c)
        S_b_val = Neaumaier_value(S_b, Sb_c)
        S_an_val = Neaumaier_value(S_an, San_c)
        S_bn_val = Neaumaier_value(S_bn, Sbn_c)


        tolerance = EPSILON * max(abs(S_a_val), abs(S_b_val), abs(S_an_val)/n, abs(S_bn_val)/n)
        if max(abs(an) , abs(bn)) < tolerance:
            t1 = r_b * S_an_val
            t2 = -sigma*abs(kappa)*h*S_a_val
            t3 = (u_sum - 2*C*r_b) * h * S_b_val
            condition1 = t1 + t2 + t3
            scale1 = abs(t1) + abs(t2) + abs(t3)

            t4 = r_b*S_bn_val
            t5 = sigma*abs(kappa)*h*S_b_val
            t6 = - u_sum*h*S_a_val
            condition2 = t4 + t5 + t6
            scale2 = abs(t4) + abs(t5) + abs(t6)


            # tol_res = EPSILON * max(scale1, scale2, 1.0)
            tol_res = EPSILON * max(abs(S_a_val), abs(S_b_val))
            if n > 20:
                tol_res *= tol_multiplier
                tol_multiplier += 0.05
            if max(abs(condition1) , abs(condition2)) < tol_res:
                if tol_multiplier > 1.1:
                    print(f"For recurance series to converge in range [{r_a}, {r_b}],the tolerance is scaled by x{tol_multiplier:.2f}")
                break

        if n > 499:
            raise RuntimeError(f"No convergence before n = {n}, {max(abs(condition1) , abs(condition2))} < {tol_res} not met")

    arr_a = np.array([float(x) for x in a], dtype = float)
    arr_b = np.array([float(x) for x in b], dtype = float)

    end_point_P = Neaumaier_value(S_a, Sa_c)
    end_point_Q = Neaumaier_value(S_b, Sb_c)

    return arr_a, arr_b, end_point_P, end_point_Q




#################################################################################################
### Calc_Series_Terms function calls the Potential_Parameter and Power_series_Terms functions to
###calculate and stich toghether all the terms needed to solve the power series
#################################################################################################

def calc_series_terms(grid_points, l, T, Z, kappa, sigma, R_au, derivatives, potential_index, phi_r, ratio_max = 0.05, max_subdiv = 200):
    """
    Compute power-series coefficients over the full radial grid

    The singular interval near the origin [0,r_b], is first solved using 'singular_power_series_terms',
    which generates the intiial recurrence coefficients A_n and B_n toghether with the corresponding
    endpoint values of the radial wave function P(r) and Q(r) on the local interval [0,r_b]

    For each subsequent grid interval [r_a,r_b], local recurrence coefficients are computes using
    'general_power_series_terms'. The endpoints values P(r_b) and Q(r_b) from the previous interval
    are used as the initial conditions for the next interval, ensuring continuity of the raidial solution
    across the full grid

    if the interval size ratio (h / r_a) exceeds 'ratio_max', the interval is subdived into smaller subintervals
    before computing the local series expansion. The final output consists of all local coefficient sets A_n and
    B_n stiched oer the complete radial range [0,r]

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
    R_au : float
        Nuclear radius in Atomic units, defined by R = 1.2e-15 * A^(1/3)/ AU.
    derivatives : list
        List made of zero, first, second and thrid order derivatives of the cubic splined potential.
    potential_index: int
        Index specifying which potential model to use.
    phi_r : callable
        Function phi(r) used in the Thomas-Fermi screening potential.
    ratio_max : float
        Maximum ratio to compare to h/r_a.
    max_subdiv : int
        Maximum amount of times a mesh [r_a,r_b] can be subdived into smaller meshes.

    Returns
    -------
    Series_terms_list_a : list
        List of all power series coefficients An over range (0,r)
    Series_terms_list_b : list
        List of all power series coefficients Bn over range (0,r)

    """

    # cnf["Z"]
    #### l = cnf["angular_momentum_l"]
    u_parameters0 = singular_potential_parameters(grid_points[1], T, Z, R_au, potential_index, phi_r)
    Series_terms0a , Series_terms0b , initial_condition0a, initial_condition0b = singular_power_series_terms(u_parameters0 ,grid_points[1] ,l ,Z ,kappa ,sigma)

    initial_temp_condition_a = initial_condition0a
    initial_temp_condition_b = initial_condition0b

    Series_terms_list_a = [Series_terms0a]
    Series_terms_list_b = [Series_terms0b]

    for i in range(1, len(grid_points)-1):
        r_a = grid_points[i]
        r_b = grid_points[i+1]

        h = r_b - r_a
        ratio = h/ r_a

        if ratio > ratio_max:
            m = int(np.ceil(ratio/ratio_max))
            m = min(m, max_subdiv)

            sub_grid = np.linspace(r_a, r_b , m+1)

            for j in range(m):
                ra = float(sub_grid[j])
                rb = float(sub_grid[j+1])

                u_temp_parameter = general_potential_parameters(ra, rb, l, T, derivatives)

                Temp_series_terms_a,Temp_series_terms_b, initial_temp_condition_a, initial_temp_condition_b = general_power_series_terms(u_temp_parameter, initial_temp_condition_a, initial_temp_condition_b, ra , rb, l, kappa, sigma)

                if rb == r_b:
                    Series_terms_list_a.append(Temp_series_terms_a)
                    Series_terms_list_b.append(Temp_series_terms_b)
        else:

            u_temp_parameter = general_potential_parameters(r_a, r_b,l, T, derivatives)

            Temp_series_terms_a, Temp_series_terms_b, initial_temp_condition_a, initial_temp_condition_b = general_power_series_terms(u_temp_parameter, initial_temp_condition_a, initial_temp_condition_b, r_a , r_b, l, kappa, sigma)

            Series_terms_list_a.append(Temp_series_terms_a)
            Series_terms_list_b.append(Temp_series_terms_b)

    return Series_terms_list_a, Series_terms_list_b


######################################################################################################################################
### Solve_Power_Series takes as input the list of power series terms to solve the total power series, returning the ouput in an array
######################################################################################################################################

def solve_power_series(list_of_sequence_terms_a, list_of_sequence_terms_b, grid_points):
    """
    Calculates the wavefunctions P and Q on the entire range [0,r] by calculating all the local
    wavefunctions P and Q on each sub-interval [r_a,r_b] using the power series coefficients
    An, Bn for P and Q respecivelty obtained through 'calc_series_terms' and stiching the resulting
    wave functions toghether to obtain P(r) and Q(r) over the total range [0,r]

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

    r_mesh = np.asarray(grid_points, dtype = float)
    r_size = len(r_mesh)

    P_array = np.zeros(r_size, dtype = float)
    Q_array = np.zeros(r_size ,dtype = float)

    P_array[0] = 0.0
    Q_array[0] = 0.0

    a0 = np.asarray(list_of_sequence_terms_a[0], dtype = float)
    b0 = np.asarray(list_of_sequence_terms_b[0], dtype = float)

    # at x=1 [P(1) = (1)^s sum A_n (1)^n] & [Q(1) = (1)^(s+t) sum B_n (1)^n]
    P_array[1] = float(np.sum(a0))
    Q_array[1] = float(np.sum(b0))

    for i in range(1,r_size - 1):
        ai = np.asarray(list_of_sequence_terms_a[i], dtype = float)
        bi = np.asarray(list_of_sequence_terms_b[i], dtype = float)

        P_array[i+1] = float(np.sum(ai))
        Q_array[i+1] = float(np.sum(bi))

    return P_array, Q_array

###########################################################################################
### Normalization_Constant function finds the normalization constant and phase shift delta
### of wave functions by matching them at rc with their assymptotic behavior
###########################################################################################

def Normalization_Constant(grid_points, idx_rc, P_array, Q_array, Analytic_normalization_const, T, k, eta, W, kappa, Z, Lambda, sigma, R_au, potential_index, phi_r):
    """
    Calculates the normalization constant to normalize the wavefunctions to unity as described in 'RADIAL: a Fortran subroutine by Francesc Salvat'

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
            return Analytic_normalization_const * ((kappa + Lambda) * mult_term_addition* Coulomb_func + Z/C * mult_term_subtraction * Coulomb_min_func)


        def lower_Dirac_func(Coulomb_func, Coulomb_min_func):
            return - Analytic_normalization_const * (Z/C * mult_term_addition * Coulomb_func + (kappa + Lambda) * mult_term_subtraction * Coulomb_min_func)

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




#######################################################################################################################
### Analytic function solves the analytic coulomb wave functions P and Q using mp.math CoulombF and Coulombg functions
#######################################################################################################################

def Analytic(Z,eta, k, W, kappa, Lambda, Analytic_normalization_const, T, delta, N_points, rc):
    """
    Calculates the anlytical pure coulomb solutions of P and Q using mpmath CoulombF and
    CoulombG functions around the matching radius. For the case (Z=0) uses scipy.special
    spherical_jn function instead.

    Parameters
    ----------
    Z : int
        Atomic number.
    eta : float
        Relativistic Sommerfeld parameter
    k : float
        Relativistic wave number.
    W : float
        Total relativistic energy (kinetic + rest energy).
    kappa : int
        Relativistic angular momentum quantum number (See eq 2.19b Radial Fortran subroutine by Francesc Salvat).
    Lambda :
        Lambda parameter defined as sqrt(kappa^2 - (alpha Z)^2).
    Analytic_normalization_const : float
        Analytic normalization constant as calculated in eq 3.122 in 'RADIAL: a Fortran subroutine by Francesc Salvat'.
    T : float
        Kinetic energy in Hartree units.
    delta : float
        Phase shift of the wave functions P and Q.
    N_points : int
        Resolution of the r grid used to calculate P and Q
    rc : float
        Matching radius point on grid_points.

    Returns
    -------
    r : ndarray
        1d linear grid around the matching radius [max(rc-1.5), rc + 1.5)]
    P: ndarray
        Wavefunction values of P(r)
    Q : ndarray
        Wavefunction values of Q(r)

    """

    r = np.linspace(max(0,rc-1.5),rc+1.5,N_points)
    if abs(Z) == 0:
        x = k * r
        mult_fact = np.sqrt(T / (T+2*C**2))
        if kappa < 0:
            l = -kappa - 1
            P = x * spherical_jn(l, x)
            Q = -mult_fact * x * spherical_jn(l+1 , x)
        else:
            l = kappa
            P = x * spherical_jn(l, x)
            Q = mult_fact * x * spherical_jn(l-1, x)
    else:

        def CoulombF_reduced(l,Z,r_array):

            r_array = np.asarray(r_array, dtype= float)
            u_vals = np.empty_like(r_array, dtype =float)

            for i, r in enumerate (tqdm(r_array, desc = f"calculating coulombf for l={l}")):
                u_vals[i] =float(mp.coulombf(l,eta,k*r))

            return u_vals

        def CoulombG_reduced(l,Z,r_array):

            r_array = np.asarray(r_array, dtype= float)
            u_vals = np.empty_like(r_array, dtype =float)

            for i, r in enumerate (tqdm(r_array, desc= f"calculating coulombg for l = {l}")):
                u_vals[i] =float(mp.coulombg(l, eta ,k * r))

            return u_vals

        U_reg_lambda = CoulombF_reduced(Lambda,Z,r)
        U_reg_lambda_min = CoulombF_reduced(Lambda-1 ,Z,r)

        G_lambda = CoulombG_reduced(Lambda,Z,r)
        G_lambda_min = CoulombG_reduced(Lambda -1, Z,r)

        Dirac_upper_analytic = Analytic_normalization_const * ((kappa+Lambda) * np.sqrt(Lambda**2 + eta**2)*k*C*U_reg_lambda + Z/C * (Lambda * C**2 - kappa * W) * U_reg_lambda_min)

        Dirac_lower_analytic = -Analytic_normalization_const * (Z/C * np.sqrt(Lambda**2 + eta**2)*k*C*U_reg_lambda + (kappa + Lambda) * (Lambda* C**2 - kappa * W )* U_reg_lambda_min)

        irreg_Dirac_upper_analytic = Analytic_normalization_const * ((kappa+Lambda) * np.sqrt(Lambda**2 + eta**2)*k*C* G_lambda + Z/C * (Lambda * C**2 - kappa * W) * G_lambda_min)

        irreg_Dirac_lower_analytic = -Analytic_normalization_const * (Z/C * np.sqrt(Lambda**2 + eta**2)*k*C* G_lambda + (kappa + Lambda) * (Lambda* C**2 - kappa * W )* G_lambda_min)


        P = np.cos(delta) * Dirac_upper_analytic + np.sin(delta) * irreg_Dirac_upper_analytic
        Q = np.cos(delta) * Dirac_lower_analytic + np.sin(delta) * irreg_Dirac_lower_analytic

    return r,P, Q

#################################################################################################################
### Visualize function plots the wavefuncs P and Q for the given potential against the analytic coulomb potential
#################################################################################################################

def Visualize(config,cnf):

    ### DECLARE PARAMETERS
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
    r0 = 1e-15 ## to avoid 1/r division by 0
    DRN = config["mesh_grid"]["upper_limit_step_size"]#Upper limit on distance between points near the end regime (make sure this stays small enough or wavefunc will not converge at larger distances r)


    r_END = obtain_mesh_range(k, Z, R_au, potential_index, phi_r, potential)

    if r_N is None:
        r_N = 10000

    A_grid = ((r_END - (r_N - 1) *r2) / r_END) * (DRN / (DRN -r2))

    while not (0.5 < A_grid < 1.0):
        r_N += 5000
        A_grid = ((r_END - (r_N - 1) *r2) / r_END) * (DRN / (DRN -r2))

    print(f"resolution set to {r_N} mesh points")
    mesh_points = radial_grid(r_END, A_grid, r_N, r2, DRN, R_au)
    rc_idx = find_rc(mesh_points, k, Z, R_au, potential_index, phi_r, potential)

    ###Setting up cubic spline
    N_V = 50000
    r_V = np.linspace(r0,r_END, N_V)

    RV = r_V * potential(r_V,Z, R_au, potential_index, phi_r)

    RV_spline = CubicSpline(r_V, RV)

    RV_spline_prime = RV_spline.derivative(1)
    RV_spline_second = RV_spline.derivative(2)
    RV_spline_third = RV_spline.derivative(3)

    derivatives = [RV_spline, RV_spline_prime, RV_spline_second, RV_spline_third]



    series_terms_upper, series_terms_lower = calc_series_terms(mesh_points,angular_momentum_quantum_number, T, Z , kappa, sigma, R_au, derivatives, potential_index, phi_r)
    wave_function_upper, wave_function_lower = solve_power_series(series_terms_upper, series_terms_lower, mesh_points)
    N, delta, r_c = Normalization_Constant(mesh_points, rc_idx ,wave_function_upper,wave_function_lower, Analytic_normalization_const, T, k, eta, W, kappa, Z, Lambda, sigma, R_au, potential_index, phi_r)

    if potential_index == 3:
        Z_match = -2.0
        eta_match = ALPHA* Z_match *W/(k*C)
        Lambda_match = np.sqrt(kappa**2 - (Z_match/C)**2)
        Analytic_normalization_const_match = 1 /Lambda_match * 1/np.sqrt((Z_match/C)**2 * (W+C**2)**2 + (kappa + Lambda_match)**2 *(k*C)**2)


    else:
        Z_match = Z
        eta_match = eta
        Lambda_match = Lambda
        Analytic_normalization_const_match = Analytic_normalization_const

    #
    # r_analytic, P_analytic, Q_analytic = Analytic(Z_match,eta_match, k ,W, kappa, Lambda_match, Analytic_normalization_const_match, T, delta, 100000, mesh_points[rc_idx])

    plt.figure(figsize=(12,8))
    plt.plot(mesh_points, wave_function_upper*N , label = "P(r) - Normalized")
    plt.plot(mesh_points, wave_function_lower*N , label = "Q(r) - Normalized")
    # plt.plot(r_analytic, P_analytic, ls = "--" , label = "Analytical mpmath sol P(r)")
    # plt.plot(r_analytic, Q_analytic, ls = "--" , label = "Analytical mpmath sol Q(r)")
    plt.axvline(x = R_au, color = "gray" , ls = "--", label = f"r = {R_au}")
    plt.axvline(x = r_c, color = "black", ls = "--", label = f"rc = {r_c}")

    plt.xlabel("r in a.u")
    plt.ylabel("P(r)")
    plt.grid(True)
    plt.title(f"E = {T: .2g}, Z = {Z}, A = {N: .5g}, delta = {delta: .3g}, N = {r_N}, DRN = {DRN} , r2 = {r2}, r_end = {r_END}, r_c = {r_c: .3g}")
    plt.legend()
    plt.show()


def Generate_Fermi_Data(config,cnf):
    ### DECLARE PARAMETERS
    A = cnf["A"]
    Zf = cnf["Z"]
    Z = -Zf
    angular_momentum_quantum_number = cnf["angular_momentum_l"]

    rN = 1.2e-15 * A ** (1 / 3) # Nuclear radius
    R_au = rN/AU # Nuclear radius in au


    potential_index = config["generator"]["potential_index"]
    if potential_index == 3:
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
    r_END = obtain_mesh_range(k_max, Z, R_au, potential_index, phi_r, potential)

    if r_N is None:
        r_N = 10000

    A_grid = ((r_END - (r_N - 1) *r2) / r_END) * (DRN / (DRN -r2))

    while not (0.5 < A_grid < 1.0):
        r_N += 10000
        A_grid = ((r_END - (r_N - 1) *r2) / r_END) * (DRN / (DRN -r2))

    print(f"resolution set to {r_N} mesh points")
    mesh_points = radial_grid(r_END, A_grid, r_N, r2, DRN, R_au)

    ###Setting up cubic spline
    N_V = 50000
    r_V = np.linspace(R0,r_END, N_V)

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
        print(f"Performing run ({i+1}/{len(kappa_sign)}) with kappa = {kappa:+d} ")
        sigma = -np.sign(kappa)
        Lambda = np.sqrt(kappa**2 - (Z/C)**2)


        P_list = []
        Q_list = []
        aP_list = []
        aQ_list = []

        for i, T in enumerate(tqdm(T_range, desc = "Solving Dirac functions over energy range")):
            W = T + C**2
            k = np.sqrt(T*(T + 2*C**2))/C
            eta = ALPHA* Z*W/(k*C)
            Analytic_normalization_const = 1 /Lambda * 1/np.sqrt((Z/C)**2 * (W+C**2)**2 + (kappa + Lambda)**2 *(k*C)**2)

            print(f"Finding wave function for T = {T}, iter = {i}, eta = {eta}")
            rc_idx = find_rc(mesh_points, k, Z, R_au, potential_index, phi_r, potential)
            series_terms_upper, series_terms_lower = calc_series_terms(mesh_points,angular_momentum_quantum_number, T, Z , kappa, sigma ,R_au, derivatives, potential_index, phi_r)
            wave_function_upper, wave_function_lower = solve_power_series(series_terms_upper, series_terms_lower, mesh_points)
            N, delta, r_c = Normalization_Constant(mesh_points, rc_idx ,wave_function_upper,wave_function_lower, Analytic_normalization_const, T, k, eta, W, kappa, Z, Lambda, sigma, R_au, potential_index, phi_r)

            P_list.append(N*wave_function_upper)
            Q_list.append(N*wave_function_lower)


        P_arr = np.asarray(P_list, dtype = float)
        Q_arr = np.asarray(Q_list, dtype = float)
        r_arr = np.asarray(mesh_points, dtype = float)
        T_arr = np.asarray(T_range, dtype = float)

        filename = f"{config["isotope"]}_potential_{potential_index}_kappa_{kappa:+d}_Z{Zf}_A{A}.npz"

        if config["paths"]["output_directory"] is None:
            main_output_directory = Path("Dirac_"+config["isotope"])
        else:
            main_output_directory = Path(config["paths"]["output_directory"])

        NPZ_output_directory = main_output_directory / "NPZ_files"
        NPZ_output_directory.mkdir(parents=True, exist_ok=True)
        file_path = NPZ_output_directory / filename

        np.savez_compressed(file_path, T = T_arr, r = r_arr, P = P_arr, Q = Q_arr)





