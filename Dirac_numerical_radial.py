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

mp.mp.dps = 30


########################################################
##### Function to save generated data to a file
########################################################

def save_run_npz(filename, label, r, P):

    if os.path.exists(filename):
        old = np.load(filename)
        data = dict(old)
    else:
        data = {}


    data[f"{label}_r"] = r
    data[f"{label}_P"] = P

    np.savez_compressed(filename, **data)





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


###################################
###Setup Cubic Spline potential V
###################################

def potential(r, Z, R_au, potential_index, phi_r):
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





#####################################################################################################################
### Find_X and Mesh_Grid solve a system of equations given in section 8.4 of RADIAL to create a continues mesh which
### is very fine in the beginning and becomes coarser as it moves further out
#####################################################################################################################
def Find_X(A_grid, x_min = 1e-10):

    def f(x):
        return (x+1.0) * (1.0 - x * np.log((x + 1.0) / x)) - A_grid

    x_max = 1.0/(1.0 - A_grid)

    for i in range(60):
        xm = 0.5*(x_min + x_max)
        if f(xm) == 0: return xm
        if f(x_min) * f(xm) < 0:
            x_max = xm
        else:
            x_min = xm

    return 0.5* (x_min + x_max)


def mesh_grid(r_END, A_grid, N, r2, DRN):


    if not (0.5 < A_grid < 1.0):
        raise ValueError(f"A_grid = {A_grid: .6g}, not in required range (0.5,1). Adjust r2 DRN or 'resolution'" )

    x = Find_X(A_grid)

    c = x * r_END
    b = (x * (c + r_END) * (DRN - r2))/ (DRN * r2)
    a = (c - b * r2) / (c * r2)
    d = 1 - b * np.log(c)

    def G(r):
        return a * r + b * np.log(c + r) + d

    r = np.zeros(N)
    r[0] = 0.0
    r[-1] = r_END

    for i in tqdm(range(1,N-1) , desc = "Constructing mesh grid"):
        target = i + 1
        low = r[i-1]
        high = r_END

        for k in range(60):
            mid = 0.5*(low + high)
            if G(low) > target: raise RuntimeError("Left bracketing failed")
            if G(high) < target: raise RuntimeError("Right bracketing failed")

            if G(mid) < target :
                low = mid
            else:
                high = mid

        r[i] = 0.5*(low+high)
    return r

#############################################################################################################
### Calculate the potential parameters u0,u1,u2,u3 in accordance to RADIAL for the initial mesh step [0,rb]
#############################################################################################################

def Singular_Potential_Parameters_Dirac(r_b, T, Z, alpha, r0, R_au, potential_index, phi_r):
    h = r_b
    Z_sing = Z
    # def V_reg(r,Z):
    #     if potential_index == 0:
    #         return potential(r,Z, R_au, potential_index, phi_r)
    #     elif potential_index == 0 or potential_index == 2:
    #         return potential(r, Z ,R_au, potential_index, phi_r) - Z_sing / r
    #     else:
    #         raise RuntimeError("No proper potential potential_index selected")
    if potential_index == 0:
        V_reg = lambda r: potential(r, Z, R_au, potential_index, phi_r) - Z_sing/r
        u0 = alpha* Z_sing
    elif potential_index in (1,2,3):
        V_reg = lambda r: potential(r, Z, R_au, potential_index, phi_r)
        u0 = 0.0
    else:
        raise RuntimeError("No proper potential potential_index selected")

    v0 = float(V_reg(r0))
    dr = 1e-8
    v1 = (V_reg(r0+dr) - V_reg(r0-dr)) / (2*dr)
    v2 = (V_reg(r0+dr) - 2*V_reg(r0) + V_reg(r0-dr)) / (dr**2)
    # v3 = (V_reg(r0 + 2*dr) - 3*V_reg(r0 + dr) + 3*V_reg(r0) - V_reg(r0 -dr))/ (dr**3)

    # u0 = 0 #alpha * Z_sing
    u1 = alpha * h *(v0-T)
    u2 = alpha * h**2 * (v1)
    u3 = alpha * h**3 * (0.5*v2)

    return u0, u1, u2, u3



#####################################################################################################################
### Calculate the potential parameters u0,u1,u2,u3 in accordance to RADIAL for the any mesh step [ra,rb] for ra != 0
#####################################################################################################################

def General_Potential_Parameters_Dirac(r_a ,r_b, l, T, r0, alpha, derivatives):
    ra = max(r_a, r0)
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

    u0 = alpha * (v0 + (v1-T)*r_a + v2*r_a**2 + v3*r_a**3)
    u1 = alpha*h * ((v1-T) + 2*v2*r_a + 3*v3*r_a**2)
    u2 = alpha*h**2 * (v2 + 3*v3*r_a)
    u3 = alpha*v3*h**3


    return u0, u1, u2, u3

####################################################################################################################
### Setting up Neaumaier addition functions to enhance floating point precision in the final digits during addition
### This is needed for the function Power_series_Terms_Dirac to properly converge
####################################################################################################################

def Neaumaier_add(sum_, c, x):
    t = sum_ + x
    if abs(sum_) >= abs(x):
        c += (sum_ -t) + x
    else:
        c += (x-t) + sum_
    return t , c

def Neaumaier_value(sum_, c):
    return sum_ + c


##############################################################################################
### Finding the power series terms for the initial mesh step [0,rb] in accordance with RADIAL
##############################################################################################

def Singular_Power_Series_Terms_Dirac(u_array, r_b, l ,eps ,Z , kappa, c, sigma):
    u0 = u_array[0]
    u1 = u_array[1]
    u2 = u_array[2]
    u3 = u_array[3]

    s = t = 0
    S_a = S_b = 0

    a = []
    b = []

    if u0 != 0:
        t = 0
        if (Z/c)**2 > kappa**2:
            raise RuntimeError("s = sqrt(kappa^2 - u0^2) is imaginary")
        else:
            s = np.sqrt(kappa**2 - (Z/c)**2)

        temp_mult_term = u1 - 2*c*r_b

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


            tolerance = eps * max(abs(S_a_val), abs(S_b_val), abs(S_an_val)/n, abs(S_bn_val)/n)
            if max(abs(an) , abs(bn)) < tolerance:

                condition1 = abs(r_b * (s * S_a_val + S_an_val) - sigma * abs(kappa) * r_b *S_a_val + ((u0+u1+u2+u3) - 2*c*r_b) * r_b * S_b_val)
                condition2 = abs(r_b * ((s+t)*S_b_val + S_bn_val) + sigma * abs(kappa) * r_b * S_b - (u0+u1+u2+u3)*r_b*S_a_val )

                if max(condition1 , condition2) < tolerance:
                    break

            if n > 499:
                raise RuntimeError(f"No convergenvce before n = {n}")


    elif (u0 == 0 and sigma == 1):
        s = abs(kappa)
        t = 1

        b_term = 2*abs(kappa) + 1
        a_term = u1 - 2*c*r_b

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

            tolerance = eps * max(abs(S_a), abs(S_b), abs(S_an)/n, abs(S_bn)/n)

            if max(abs(an) , abs(bn)) < tolerance:
                condition1 = abs(r_b * (s * S_a + S_an) - sigma * abs(kappa) * r_b *S_a + ((u0+u1+u2+u3) - 2*c*r_b) * r_b * S_b)
                condition2 = abs(r_b * ((s+t)*S_b + S_bn) + sigma * abs(kappa) * r_b * S_b - (u0+u1+u2+u3)*r_b*S_a )
                if max(condition1 , condition2)  < tolerance:
                    break

            if n > 499:
                raise RuntimeError(f"No convergenvce before n = {n}")

    elif (u0 == 0 and sigma == -1):
        s = abs(kappa) + 1
        t = -1

        a_term1 = u1 - 2*c*r_b
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

            tolerance = eps * max(abs(S_a), abs(S_b), abs(S_an)/n, abs(S_bn)/n)

            if max(abs(an) , abs(bn)) < tolerance:
                condition1 = abs(r_b * (s * S_a + S_an) - sigma * abs(kappa) * r_b *S_a + ((u0+u1+u2+u3) - 2*c*r_b) * r_b * S_b)

                condition2 = abs(r_b * ((s+t)*S_b + S_bn) + sigma * abs(kappa) * r_b * S_b - (u0+u1+u2+u3)*r_b*S_a )
                if max(condition1 , condition2)  < tolerance:
                    break

            if n > 499:
                raise RuntimeError(f"No convergenvce before n = {n}")

    arr_a = np.array(a)
    arr_b = np.array(b)

    normalized_end_point_a = S_a/abs(S_a)
    normalized_end_point_b = S_b/abs(S_a)

    return arr_a, arr_b, normalized_end_point_a, normalized_end_point_b, s, t



######################################################################################################
### Finding the power series terms for the any mesh step [ra,rb] for ra !=0 in accordance with RADIAL
######################################################################################################

def Power_Series_Terms_Dirac(u_array, initial_condition_a, initial_condition_b , r_a, r_b, l, eps, kappa, r0, c, sigma):
    #Define the u terms
    u0, u1, u2, u3 = u_array
    u_sum = u0 + u1 + u2 + u3
    h = r_b-r_a

    pre_factor = h/max(r_a,r0)
    mult_term1 = u0 - 2.0*c*r_a
    mult_term2 = u1 - 2.0*c*h
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


        tolerance = eps * max(abs(S_a_val), abs(S_b_val), abs(S_an_val)/n, abs(S_bn_val)/n)
        if max(abs(an) , abs(bn)) < tolerance:
            t1 = r_b * S_an_val
            t2 = -sigma*abs(kappa)*h*S_a_val
            t3 = (u_sum - 2*c*r_b) * h * S_b_val
            condition1 = t1 + t2 + t3
            scale1 = abs(t1) + abs(t2) + abs(t3)

            t4 = r_b*S_bn_val
            t5 = sigma*abs(kappa)*h*S_b_val
            t6 = - u_sum*h*S_a_val
            condition2 = t4 + t5 + t6
            scale2 = abs(t4) + abs(t5) + abs(t6)


            #TODO CHECK WETHER THIS SCALING TOLERANCE IS STILL NEEDED?
            tol_res = eps * max(scale1, scale2, 1.0)
            if max(condition1 , condition2) < tol_res:
                break

        if n > 499:
            raise RuntimeError(f"No convergence before n = {n}")

    arr_a = np.array([float(x) for x in a], dtype = float)
    arr_b = np.array([float(x) for x in b], dtype = float)

    end_point_a = Neaumaier_value(S_a, Sa_c)
    end_point_b = Neaumaier_value(S_b, Sb_c)

    return arr_a, arr_b, end_point_a, end_point_b




#################################################################################################
### Calc_Series_Terms function calls the Potential_Parameter and Power_series_Terms functions to
###calculate and stich toghether all the terms needed to solve the power series
#################################################################################################

def Calc_Series_Terms(mesh_steps, l ,epsilon, T, Z, alpha, kappa, c, sigma, r0, R_au, derivatives, potential_index, phi_r, ratio_max = 0.05, max_subdiv = 200):

    u_parameters0 = Singular_Potential_Parameters_Dirac(mesh_steps[1], T, Z, alpha, r0, R_au, potential_index, phi_r)
    Series_terms0a , Series_terms0b , initial_condition0a, initial_condition0b, s, t = Singular_Power_Series_Terms_Dirac(u_parameters0 ,mesh_steps[1] ,l ,epsilon ,Z ,kappa ,c ,sigma)

    initial_temp_condition_a = initial_condition0a
    initial_temp_condition_b = initial_condition0b

    Series_terms_list_a = [Series_terms0a]
    Series_terms_list_b = [Series_terms0b]

    for i in range(1, len(mesh_steps)-1):
        r_a = mesh_steps[i]
        r_b = mesh_steps[i+1]

        h = r_b - r_a
        ratio = h/ r_a

        if ratio > ratio_max:
            m = int(np.ceil(ratio/ratio_max))
            m = min(m, max_subdiv)

            sub_grid = np.linspace(r_a, r_b , m+1)

            for j in range(m):
                ra = float(sub_grid[j])
                rb = float(sub_grid[j+1])

                u_temp_parameter = General_Potential_Parameters_Dirac(ra, rb, l, T, r0, alpha, derivatives)

                Temp_series_terms_a,Temp_series_terms_b, initial_temp_condition_a, initial_temp_condition_b = Power_Series_Terms_Dirac(u_temp_parameter, initial_temp_condition_a, initial_temp_condition_b, ra , rb, l, epsilon, kappa, r0, c, sigma)

                if rb == r_b:
                    Series_terms_list_a.append(Temp_series_terms_a)
                    Series_terms_list_b.append(Temp_series_terms_b)
        else:

            u_temp_parameter = General_Potential_Parameters_Dirac(r_a, r_b,l, T, r0, alpha, derivatives)

            Temp_series_terms_a, Temp_series_terms_b, initial_temp_condition_a, initial_temp_condition_b = Power_Series_Terms_Dirac(u_temp_parameter, initial_temp_condition_a, initial_temp_condition_b, r_a , r_b, l, epsilon, kappa, r0, c, sigma)

            Series_terms_list_a.append(Temp_series_terms_a)
            Series_terms_list_b.append(Temp_series_terms_b)

    return Series_terms_list_a, Series_terms_list_b


######################################################################################################################################
### Solve_Power_Series takes as input the list of power series terms to solve the total power series, returning the ouput in an array
######################################################################################################################################

def Solve_Power_Series(list_of_sequence_terms_a, list_of_sequence_terms_b, grid_points):

    r_mesh = np.asarray(grid_points, dtype = float)
    r_size = len(r_mesh)

    P_mesh = np.zeros(r_size, dtype = float)
    Q_mesh = np.zeros(r_size ,dtype = float)

    P_mesh[0] = 0.0
    Q_mesh[0] = 0.0

    a0 = np.asarray(list_of_sequence_terms_a[0], dtype = float)
    b0 = np.asarray(list_of_sequence_terms_b[0], dtype = float)

    # at x=1 [P(1) = (1)^s sum A_n (1)^n] & [Q(1) = (1)^(s+t) sum B_n (1)^n]
    P_mesh[1] = float(np.sum(a0))
    Q_mesh[1] = float(np.sum(b0))

    for i in range(1,r_size - 1):
        ai = np.asarray(list_of_sequence_terms_a[i], dtype = float)
        bi = np.asarray(list_of_sequence_terms_b[i], dtype = float)

        P_mesh[i+1] = float(np.sum(ai))
        Q_mesh[i+1] = float(np.sum(bi))

    return P_mesh, Q_mesh

###################################################################################
### Find mminimum range up to which to extend the mesh range used to solve P and Q
###################################################################################

def obtain_mesh_range(k,epsilon, Z, R_au, potential_index, phi_r):
    kr_min = 20.0
    r_max = 0.01

    RV = r_max* potential(r_max, Z, R_au, potential_index, phi_r)
    r_infty = 1e5
    Z_inf = r_infty * potential(r_infty, Z, R_au, potential_index, phi_r)


    if potential_index == 3:
        TAS = 10e-4 * abs(Z_inf)
    else:
        TAS = max(1e-11, epsilon) * abs(Z_inf)

    while abs(RV - Z_inf) <= TAS and (k * r_max <= kr_min):
        r_max *= 2
        RV = r_max* potential(r_max, Z, R_au, potential_index, phi_r)

    print(f"mesh range set to {r_max}")

    return r_max





#################################################################################################################################################
### Find_rc is a function finding the matching radius rc at which one can normalize the power series by matching it witht he asymptotic behavior
#################################################################################################################################################

def Find_rc(r_mesh,k,epsilon, Z , R_au, potential_index, phi_r):
    kr_min = 20.0
    r_grid = np.asarray(r_mesh, dtype = float)


    r_safe = np.maximum(r_grid, 1e-16)
    RV = r_safe * potential(r_safe, Z, R_au, potential_index, phi_r)

    r_infty = 10000
    Z_inf = r_infty * potential(r_infty, Z, R_au, potential_index, phi_r)

    if potential_index == 3:
        TAS = 10e-4 * abs(Z_inf)
    else:
        TAS = max(1e-11, epsilon) * abs(Z_inf)

    idx_rc = None
    for idx in range(len(r_grid)-1 ,2, -1):
        if abs(RV[idx] - Z_inf) <= TAS and (k * r_grid[idx] >= kr_min):
            idx_rc = idx
        else:
            break


    if idx_rc is None:
        raise RuntimeError("No rc found, extend distance")
    print(f"rc succesfully found at rc = {r_grid[idx_rc]}")
    return idx_rc

###########################################################################################
### Normalization_Constant function finds the normalization constant and phase shift delta
### of wave functions by matching them at rc with their assymptotic behavior
###########################################################################################

def Normalization_Constant(r_mesh, idx_rc, P_mesh, Q_mesh, Analytic_normalization_const, T, k, eta, W, kappa, Z, c, Lambda, sigma, epsilon, R_au, potential_index, phi_r):
    P = P_mesh[idx_rc]
    Q = Q_mesh[idx_rc]

    r_c = r_mesh[idx_rc]
    x = k * r_c


    if abs(Z) < 1:
        l = (-kappa -1) if (kappa < 0) else kappa
        mult_fact = np.sqrt(T/(T + 2*c**2))


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


        mult_term_addition = (np.sqrt(Lambda**2 + eta**2))*k*c
        mult_term_subtraction = (Lambda * c**2 - kappa * W)

        def upper_Dirac_func(Coulomb_func, Coulomb_min_func):
            return Analytic_normalization_const * ((kappa + Lambda) * mult_term_addition* Coulomb_func + Z/c * mult_term_subtraction * Coulomb_min_func)


        def lower_Dirac_func(Coulomb_func, Coulomb_min_func):
            return - Analytic_normalization_const * (Z/c * mult_term_addition * Coulomb_func + (kappa + Lambda) * mult_term_subtraction * Coulomb_min_func)

        PIA = upper_Dirac_func(F_l, F_lm)
        QA = lower_Dirac_func(F_l, F_lm)

        PIB = upper_Dirac_func(G_l, G_lm)
        QB = lower_Dirac_func(G_l, G_lm)

    V = float(potential(r_c, Z, R_au, potential_index, phi_r))
    FG = (T - V + 2*c**2)/c

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

    if abs(P) > epsilon:
        A = (COS * PIA + SIN * PIB) / P
    else:
        A = (COS * PIAP + SIN * PIBP) / P_prime

    return float(A) , float(delta), r_c




#######################################################################################################################
### Analytic function solves the analytic coulomb wave functions P and Q using mp.math CoulombF and Coulombg functions
#######################################################################################################################

def Analytic(Z,eta, k, W, kappa, c, Lambda, Analytic_normalization_const, T, delta, N_points, rc):

    r = np.linspace(max(0,rc-1.5),rc+1.5,N_points)
    if abs(Z) < 1:
        x = k * r
        mult_fact = np.sqrt(T / (T+2*c**2))
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

        Dirac_upper_analytic = Analytic_normalization_const * ((kappa+Lambda) * np.sqrt(Lambda**2 + eta**2)*k*c*U_reg_lambda + Z/c * (Lambda * c**2 - kappa * W) * U_reg_lambda_min)

        Dirac_lower_analytic = -Analytic_normalization_const * (Z/c * np.sqrt(Lambda**2 + eta**2)*k*c*U_reg_lambda + (kappa + Lambda) * (Lambda* c**2 - kappa * W )* U_reg_lambda_min)

        irreg_Dirac_upper_analytic = Analytic_normalization_const * ((kappa+Lambda) * np.sqrt(Lambda**2 + eta**2)*k*c* G_lambda + Z/c * (Lambda * c**2 - kappa * W) * G_lambda_min)

        irreg_Dirac_lower_analytic = -Analytic_normalization_const * (Z/c * np.sqrt(Lambda**2 + eta**2)*k*c* G_lambda + (kappa + Lambda) * (Lambda* c**2 - kappa * W )* G_lambda_min)


        P = np.cos(delta) * Dirac_upper_analytic + np.sin(delta) * irreg_Dirac_upper_analytic
        Q = np.cos(delta) * Dirac_lower_analytic + np.sin(delta) * irreg_Dirac_lower_analytic

    return r,P, Q

#################################################################################################################
### Visualize function plots the wavefuncs P and Q for the given potential against the analytic coulomb potential
#################################################################################################################

def Visualize(config):

    ### DECLARE PARAMETERS
    A = config["parameters"]["mass_number"]
    Zf = config["parameters"]["atomic_number"]
    Z = -Zf
    angular_momentum_quantum_number = config["parameters"]["angular_momentum_l"]
    kappa = config["parameters"]["kappa"]
    sigma = -np.sign(kappa)
    c = 137.03599
    alpha = 1.0/c
    epsilon = 1e-15
    Lambda = np.sqrt(kappa**2 - (Z/c)**2)


    rN = 1.2e-15 * A ** (1 / 3) # Nuclear radius
    au = 5.29177210903e-11 #Bohr radius
    R_au = rN/au # Nuclear radius in au
    E_hatree = 27.211386 #eV

    potential_index = config["generator"]["potential_index"]
    if potential_index == 3:
        print("calculating thomas fermi func")
        phi_r = make_phi_r(Zf)
    else:
        phi_r = 0

    Plotting_energy = config["generator"]["T_plot_energy-MeV"]
    T = Plotting_energy * 1e6 / E_hatree
    W = T + c**2
    k = np.sqrt(T*(T + 2*c**2))/c
    eta =  alpha* Z *W/(k*c)
    Analytic_normalization_const = 1 /Lambda * 1/np.sqrt((Z/c)**2 * (W+c**2)**2 + (kappa + Lambda)**2 *(k*c)**2)


    r_N = config["mesh_grid"]["num_mesh_steps"]
    r2 = config["mesh_grid"]["second_point"]
    r0 = 1e-15 ## to avoid 1/r division by 0
    DRN = config["mesh_grid"]["upper_limit_step_size"]#Upper limit on distance between points near the end regime (make sure this stays small enough or wavefunc will not converge at larger distances r)


    r_END = obtain_mesh_range(k, epsilon, Z, R_au, potential_index, phi_r)

    if r_N is None:
        r_N = 10000

    A_grid = ((r_END - (r_N - 1) *r2) / r_END) * (DRN / (DRN -r2))

    while not (0.5 < A_grid < 1.0):
        r_N += 5000
        A_grid = ((r_END - (r_N - 1) *r2) / r_END) * (DRN / (DRN -r2))

    print(f"resolution set to {r_N} mesh points")
    mesh_points = mesh_grid(r_END, A_grid, r_N, r2, DRN)
    rc_idx = Find_rc(mesh_points, k, epsilon, Z, R_au, potential_index, phi_r)

    ###Setting up cubic spline
    N_V = 50000
    r_V = np.linspace(r0,r_END, N_V)

    RV = r_V * potential(r_V,Z, R_au, potential_index, phi_r)

    RV_spline = CubicSpline(r_V, RV)

    RV_spline_prime = RV_spline.derivative(1)
    RV_spline_second = RV_spline.derivative(2)
    RV_spline_third = RV_spline.derivative(3)

    derivatives = [RV_spline, RV_spline_prime, RV_spline_second, RV_spline_third]



    series_terms_upper, series_terms_lower = Calc_Series_Terms(mesh_points,angular_momentum_quantum_number, epsilon, T, Z , alpha, kappa, c, sigma, r0 ,R_au, derivatives, potential_index, phi_r)
    wave_function_upper, wave_function_lower = Solve_Power_Series(series_terms_upper, series_terms_lower, mesh_points)
    N, delta, r_c = Normalization_Constant(mesh_points, rc_idx ,wave_function_upper,wave_function_lower, Analytic_normalization_const, T, k, eta, W, kappa, Z, c, Lambda, sigma, epsilon, R_au, potential_index, phi_r)

    if potential_index == 3:
        Z_match = -2.0
        eta_match = alpha* Z_match *W/(k*c)
        Lambda_match = np.sqrt(kappa**2 - (Z_match/c)**2)
        Analytic_normalization_const_match = 1 /Lambda_match * 1/np.sqrt((Z_match/c)**2 * (W+c**2)**2 + (kappa + Lambda_match)**2 *(k*c)**2)


    else:
        Z_match = Z
        eta_match = eta
        Lambda_match = Lambda
        Analytic_normalization_const_match = Analytic_normalization_const


    r_analytic, P_analytic, Q_analytic = Analytic(Z_match,eta_match, k ,W, kappa, c, Lambda_match, Analytic_normalization_const_match, T, delta, 100000, mesh_points[rc_idx])

    plt.figure(figsize=(12,8))
    plt.plot(mesh_points, wave_function_upper*N , label = "P(r) - Normalized")
    plt.plot(mesh_points, wave_function_lower*N , label = "Q(r) - Normalized")
    plt.plot(r_analytic, P_analytic, ls = "--" , label = "Analytical mpmath sol P(r)")
    plt.plot(r_analytic, Q_analytic, ls = "--" , label = "Analytical mpmath sol Q(r)")
    plt.axvline(x = R_au, color = "gray" , ls = "--", label = f"r = {R_au}")
    plt.axvline(x = r_c, color = "black", ls = "--", label = f"rc = {r_c}")

    plt.xlabel("r in a.u")
    plt.ylabel("P(r)")
    plt.grid(True)
    plt.title(f"E = {T: .2g}, Z = {Z}, A = {N: .3g}, delta = {delta: .3g}, N = {r_N}, DRN = {DRN} , r2 = {r2}, r_end = {r_END}, r_c = {r_c: .3g}")
    plt.legend()
    plt.show()


def Generate_Fermi_Data(config):
    ### DECLARE PARAMETERS
    A = config["parameters"]["mass_number"]
    Zf = config["parameters"]["atomic_number"]
    Z = -Zf
    angular_momentum_quantum_number = config["parameters"]["angular_momentum_l"]
    kappa = config["parameters"]["kappa"]
    sigma = -np.sign(kappa)
    c = 137.03599
    alpha = 1.0/c
    epsilon = 1e-15
    Lambda = np.sqrt(kappa**2 - (Z/c)**2)


    rN = 1.2e-15 * A ** (1 / 3) # Nuclear radius
    au = 5.29177210903e-11 #Bohr radius
    R_au = rN/au # Nuclear radius in au
    E_hatree = 27.211386 #eV


    potential_index = config["generator"]["potential_index"]
    if potential_index == 3:
        print("calculating thomas fermi func")
        phi_r = make_phi_r(Zf)
    else:
        phi_r = 0

    Q = config["generator"]["T_end-MeV"]
    T_start = config["generator"]["T_start-MeV"]
    n_samples = config["generator"]["num_samples"]

    T_range = np.geomspace(T_start, Q, n_samples) * 1e6 /E_hatree



    r_N = config["mesh_grid"]["num_mesh_steps"]
    r2 = config["mesh_grid"]["second_point"]
    r0 = 1e-15 ## to avoid 1/r division by 0
    DRN = config["mesh_grid"]["upper_limit_step_size"]#Upper limit on distance between points near the end regime (make sure this stays small enough or wavefunc will not converge at larger distances r)

    k_max = np.sqrt(T_range[0]*(T_range[0] + 2*c**2))/c
    r_END = obtain_mesh_range(k_max, epsilon, Z, R_au, potential_index, phi_r)

    if r_N is None:
        r_N = 10000

    A_grid = ((r_END - (r_N - 1) *r2) / r_END) * (DRN / (DRN -r2))

    while not (0.5 < A_grid < 1.0):
        r_N += 10000
        A_grid = ((r_END - (r_N - 1) *r2) / r_END) * (DRN / (DRN -r2))

    print(f"resolution set to {r_N} mesh points")
    mesh_points = mesh_grid(r_END, A_grid, r_N, r2, DRN)

    ###Setting up cubic spline
    N_V = 50000
    r_V = np.linspace(r0,r_END, N_V)

    RV = r_V * potential(r_V,Z, R_au, potential_index, phi_r)

    RV_spline = CubicSpline(r_V, RV)

    RV_spline_prime = RV_spline.derivative(1)
    RV_spline_second = RV_spline.derivative(2)
    RV_spline_third = RV_spline.derivative(3)

    derivatives = [RV_spline, RV_spline_prime, RV_spline_second, RV_spline_third]


    P_list = []
    Q_list = []
    aP_list = []
    aQ_list = []

    for i, T in enumerate(tqdm(T_range, desc = "Solving Dirac functions over energy range")):
        W = T + c**2
        k = np.sqrt(T*(T + 2*c**2))/c
        eta = alpha* Z*W/(k*c)
        Analytic_normalization_const = 1 /Lambda * 1/np.sqrt((Z/c)**2 * (W+c**2)**2 + (kappa + Lambda)**2 *(k*c)**2)

        print(f"Finding wave function for T = {T}, iter = {i}, eta = {eta}")
        rc_idx = Find_rc(mesh_points, k, epsilon, Z, R_au, potential_index, phi_r)
        series_terms_upper, series_terms_lower = Calc_Series_Terms(mesh_points,angular_momentum_quantum_number, epsilon, T, Z , alpha, kappa, c, sigma, r0 ,R_au, derivatives, potential_index, phi_r)
        wave_function_upper, wave_function_lower = Solve_Power_Series(series_terms_upper, series_terms_lower, mesh_points)
        N, delta, r_c = Normalization_Constant(mesh_points, rc_idx ,wave_function_upper,wave_function_lower, Analytic_normalization_const, T, k, eta, W, kappa, Z, c, Lambda, sigma, epsilon, R_au, potential_index, phi_r)

        P_list.append(N*wave_function_upper)
        Q_list.append(N*wave_function_lower)


    P_arr = np.asarray(P_list, dtype = float)
    Q_arr = np.asarray(Q_list, dtype = float)
    r_arr = np.asarray(mesh_points, dtype = float)
    T_arr = np.asarray(T_range, dtype = float)

    filename = f"potential_{potential_index}_kappa_{kappa:+d}_Z{Z}_A{A}.npz"

    output_directory = Path(config["paths"]["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    file_path = output_directory / filename

    np.savez_compressed(file_path, T = T_arr, r = r_arr, P = P_arr, Q = Q_arr)





