#!/usr/bin/env python

#Importing libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import mpmath as mp
import os
from scipy.special import spherical_jn
from tqdm import tqdm
import time
import math

mp.mp.dps = 22
eps_machine = np.finfo(float).eps
print(f"Machine precision : {eps_machine}")

A = 136
Zf = 56
Z = -Zf
angular_momentum_quantum_number = 0
kappa = 1
sigma = -np.sign(kappa)
c = 137.03599
alpha = 1.0/c
epsilon = 1e-15
Lambda = np.sqrt(kappa**2 - (Z/c)**2)


rN = 1.2e-15 * A ** (1 / 3) # Nuclear radius
au = 5.29177210903e-11 #Bohr radius
R_au = rN/au

print(R_au)

# #
E_hatree = 27.211386 #eV
# # m_ELECTRON = 0.51099895069e6/E_hatree #eV
# # E_max = 1.2933325099999138e6/27.211386 #eV
# #
# # T = 1e-4 * 1e9 / E_hatree
# T = 5000
# print(f"T = {T}")
# # E = m_ELECTRON
#
#
# W = T + c**2
#
# k = np.sqrt(T*(T + 2*c**2))/c
# eta = Z/k
# Lambda = np.sqrt(kappa**2 - (Z/c)**2)
#
#
# Analytic_normalization_const = 1/Lambda * 1/np.sqrt((Z/c)**2 * (W+c**2)**2 + (kappa + Lambda)**2 *(k*c)**2)



r_steps = 300000
r_END = 25
r2 = 1e-7
r0 = 1e-15 ## to avoid 1/r division by 0
DRN = 0.0001 #Upper limit on distance between points near the end regime (make sure this stays small enough or wavefunc will not converge at larger distances r)


methods = ["Coulomb" , "Spline"]


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


###################################
###Setup Cubic Spline potential V
###################################
N_V = 50000
V0 = 1
A =  1
Z_sing = Z  ###Define the singular potential term which blows up at r ->0
def potential(r):
    r_arr = np.asarray(r,dtype = float)
    V = Z_sing/ r_arr
    inside = r_arr < R_au
    V = np.where(inside, Z_sing/(2*R_au) * (3 - (r_arr/R_au)**2), V)

    if np.isscalar(r):
        return float(V)
    return V

r_V = np.linspace(r0,r_END, N_V)

RV = r_V * potential(r_V)

RV_spline = CubicSpline(r_V, RV)

RV_spline_prime = RV_spline.derivative(1)
RV_spline_second = RV_spline.derivative(2)
RV_spline_third = RV_spline.derivative(3)

def Singular_potential_parameters_Dirac(r_b, T):
    print("SINGULAR POTENTIAL PARAMS IS BEING CALLED")
    h = r_b

    def V_reg(r):
        return potential(r) #- Z_sing / r

    v0 = float(V_reg(r0))
    dr = 1e-8
    v1 = (V_reg(r0+dr) - V_reg(r0-dr)) / (2*dr)
    v2 = (V_reg(r0+dr) - 2*V_reg(r0) + V_reg(r0-dr)) / (dr**2)
    # v3 = (V_reg(r0 + 2*dr) - 3*V_reg(r0 + dr) + 3*V_reg(r0) - V_reg(r0 -dr))/ (dr**3)

    u0 = 0 #alpha * Z_sing
    u1 = alpha * h *(v0-T)
    u2 = alpha * h**2 * (v1)
    u3 = alpha * h**3 * (0.5*v2)

    return u0, u1, u2, u3




def Potential_parameters_Dirac(name, r_a ,r_b, l, T):
    ra = max(r_a, r0)
    rb = r_b

    if name == "Coulomb":
        v0 = Z/ra
        v1 = -Z/ra**2
        v2 = Z/ra**3
        v3 = -Z/ra**5

    elif name == "Spline":

        rv0 = float(RV_spline(ra))
        rv1 = float(RV_spline_prime(ra))
        rv2 = 0.5 * float(RV_spline_second(ra))
        rv3 = (1.0/6.0) * float(RV_spline_third(ra))
        #
        # v0 = rv0 / ra
        # v1 = (rv1 * ra - rv0) / (ra**2)
        # v2 = (rv2 * ra**2 - rv1 * ra + rv0) / (ra**3)
        # v3 = (rv3 * ra**3 - rv2 * ra**2 + rv1 * ra - rv0) / (ra**4)

        v0 = rv0 - rv1*r_a + rv2*(r_a**2) - rv3* (r_a**3)
        v1 = rv1 - 2.0* rv2 *r_a + 3.0 * rv3 * (r_a **2)
        v2 = rv2 - 3.0*rv3 *r_a
        v3 = rv3
    else:
        raise ValueError("No valid potential type was given, choose between 'Coulomb' or 'Spline' ")


    # print(f"v0 = {v0}")
    # print(f"v1 = {v1}")
    # print(f"v2 = {v2}")
    # print(f"v3 = {v3}")

    h = r_b-r_a

    u0 = alpha * (v0 + (v1-T)*r_a + v2*r_a**2 + v3*r_a**3)
    u1 = alpha*h * ((v1-T) + 2*v2*r_a + 3*v3*r_a**2)
    u2 = alpha*h**2 * (v2 + 3*v3*r_a)
    u3 = alpha*v3*h**3


    return u0, u1, u2, u3




def Singular_term_power_series_Dirac(u_array, r_b, l ,eps):
    u0 = u_array[0]
    u1 = u_array[1]
    u2 = u_array[2]
    u3 = u_array[3]

    s = t = 0

    a = []
    b = []

    S_a = S_b = 0 #Purely to avoid python flags of unbounded terms later on

    # CASE 1: (u0 != 0)  , (t = 0)

    if u0 != 0:
        # print("u0 != 0")
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




        a.append(a0)
        a.append(a1)
        a.append(a2)


        b.append(b0)
        b.append(b1)
        b.append(b2)


        S_a = sum(a)
        S_b = sum(b)

        S_an = a1 + 2*a2
        S_bn = b1 + 2*b2

        n = 2
        while True:
            n +=1

            An = u1*a[n-1] + u2*a[n-2] + u3*a[n-3]
            Bn = temp_mult_term*b[n-1] + u2 * b[n-2] + u3 * b[n-3]

            an, bn = recurance_terms(n, An, Bn)


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

                if max(condition1 , condition2) < tolerance:
                    break

            if n > 499:
                raise RuntimeError(f"No convergenvce before n = {n}")


    elif (u0 == 0 and sigma == 1):
        print("u0 = 1 , sigma = +1")
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

        a.append(a0)
        a.append(a1)
        a.append(a2)
        a.append(a3)
        b.append(b0)
        b.append(b1)
        b.append(b2)
        b.append(b3)

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
        print("u0 = 0, sigma = -1")
        s = abs(kappa) + 1
        t = -1

        a = []
        b = []

        a_term1 = u1 - 2*c*r_b
        a_term2 = 2*abs(kappa)+1



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


        a.append(a0)
        a.append(a1)
        a.append(a2)
        a.append(a3)
        b.append(b0)
        b.append(b1)
        b.append(b2)
        b.append(b3)

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




def Neaumaier_add(sum_, c, x):
    t = sum_ + x
    if abs(sum_) >= abs(x):
        c += (sum_ -t) + x
    else:
        c += (x-t) + sum_
    return t , c

def Neaumaier_value(sum_, c):
    return sum_ + c

def Power_series_Dirac(u_array, initial_condition_a, initial_condition_b , r_a, r_b, l, eps):
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



    # #Create the Sum terms to check for convergence critaria
    # S_a = a0 + a1 + a2 + a3 #Sum a_n from n= 0 to j
    # S_b = b0 + b1 + b2 + b3
    # S_an = a1 + 2.0*a2 + 3.0*a3
    # S_bn = b1 + 2.0*b2 + 3.0*b3



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





    # ra = mp.mpf(r_a)
    # rb = mp.mpf(r_b)
    # hh = mp.mpf(h)
    #
    # two = mp.mpf(2)
    #
    #
    # U0 = mp.mpf(u0); U1 = mp.mpf(u1); U2 = mp.mpf(u2); U3 = mp.mpf(u3)
    # K = mp.mpf(kappa)
    #
    # pre_factor = hh/ra
    # mult_term1 = U0 - two*mp.mpf(c)*ra
    # mult_term2 = U1 - two*mp.mpf(c)*hh
    # #create the a array and define the intial conditions
    # a = [mp.mpf(initial_condition_a)]
    # b = [mp.mpf(initial_condition_b)]
    #
    #
    # a0 = a[0]
    # b0 = b[0]
    #
    # a1 = -pre_factor * (K*a0 + mult_term1*b0)
    # b1 = pre_factor * (K*b0 + U0*a0)
    #
    # a2 = -pre_factor/two * ((K + mp.mpf(1))*a1 + mult_term1*b1 + mult_term2*b0)
    # b2 = pre_factor/two * ((K - mp.mpf(1))*b1 + U0*a1 + U1*a0)
    #
    # a3 = -pre_factor/mp.mpf(3) * ((K + two)*a2 + mult_term1*b2 + mult_term2*b1 + U2*b0)
    # b3 = pre_factor/mp.mpf(3) * ((K - two)*b2 + U0*a2 + U1*a1 + U2*a0)
    #
    # a += [a1, a2, a3]
    # b += [b1, b2, b3]
    #
    #  #Create the Sum terms to check for convergence critaria
    # S_a = mp.fsum(a)  #Sum a_n from n= 0 to j
    # S_b = mp.fsum(b)
    # S_an = a1 + two*a2 + mp.mpf(3)*a3
    # S_bn = b1 + two*b2 + mp.mpf(3)*b3


    # print(f" a terms: {a}")
    # print(f" b terms; {b}")

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


        # S_a += an
        # S_b += bn
        #
        # S_an += n * an
        # S_bn += n * bn
        # print()
        # print(f"n = {n}, r_a = {r_a}, r_b = {r_b}")
        # print(f"S_a = {S_a_val}, S_b = {S_b_val} , S_an = {S_an_val} , S_bn = {S_bn_val}")
        #


        tolerance = eps * max(abs(S_a_val), abs(S_b_val), abs(S_an_val)/n, abs(S_bn_val)/n)

        # n +=1
        # n_mp = mp.mpf(n)
        # pf = pre_factor / n_mp


        # a_part = (K - 1 + n_mp) * a[n-1] + mult_term1 * b[n-1] + mult_term2 * b[n-2] + U2 * b[n-3] + U3 * b[n-4]
        # an = -pf * a_part
        #
        # b_part = (K + 1 - n_mp) * b[n-1] + U0 * a[n-1] + U1 * a[n-2] + U2 * a[n-3] + U3 * a[n-4]
        # bn = pf * b_part


        # S_a += an
        # S_b += bn
        #
        # S_an += n * an
        # S_bn += n * bn

        #
        # tolerance = mp.mpf(eps) * max(abs(S_a), abs(S_b), abs(S_an)/n_mp, abs(S_bn)/n_mp)

        #
        #
        # print(f"1st condition: {max(abs(an) , abs(bn))} < {tolerance}")

        # condition1 = abs(r_b*S_an - sigma*abs(kappa)*h*S_a + ((u0+u1+u2+u3) - 2*c*r_b) * h * S_b)
        # condition2 = abs(r_b*S_bn + sigma*abs(kappa)*h*S_b - (u0+u1+u2+u3)*h*S_a )
        # print(f"2nd condition: {max(condition1 , condition2)} < {tolerance}")
        # print()

        # U_sum = U0 + U1 + U2 + U3
        if max(abs(an) , abs(bn)) < tolerance:
            # condition1 = abs(r_b*S_an_val - sigma*abs(kappa)*h*S_a_val + (u_sum - 2*c*r_b) * h * S_b_val)
            # condition2 = abs(r_b*S_bn_val + sigma*abs(kappa)*h*S_b_val - u_sum*h*S_a_val )

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



            tol_res = eps * max(scale1, scale2, 1.0)


            # condition1 = abs(rb*S_an - mp.mpf(int(sigma))*abs(K)*hh*S_a + (U_sum - two*mp.mpf(c)*rb) * hh * S_b)
            # condition2 = abs(rb*S_bn + mp.mpf(int(sigma))*abs(K)*hh*S_b - (U_sum)*hh*S_a )
            #
            # print(f"2nd condition: {max(condition1 , condition2)} < {tol_res}")



            # if (S_a == S_a_prev) and (S_b == S_b_prev):
            #     S_a, S_b, S_an, S_bn = recompute_sums_longdouble(a,b)
            #     condition1 = abs(r_b*S_an - sigma*abs(kappa)*h*S_a + (u_sum - 2*c*r_b) * h * S_b)
            #     condition2 = abs(r_b*S_bn + sigma*abs(kappa)*h*S_b - u_sum*h*S_a )
            #     print(f"Recomputed in double precision")
            #     print(f"S_a = {S_a}, S_b = {S_b} , S_an = {S_an} , S_bn = {S_bn}")
            #     print(f"1st condition: {max(abs(an) , abs(bn))} < {tolerance}")
            #     print(f"2nd condition: {max(condition1 , condition2)} < {tolerance}")
            #     print()


            if max(condition1 , condition2) < tol_res:
                break



        if n > 499:
            raise RuntimeError(f"No convergence before n = {n}")



    arr_a = np.array([float(x) for x in a], dtype = float)
    arr_b = np.array([float(x) for x in b], dtype = float)

    # end_point_a = float(S_a)
    # end_point_b = float(S_b)
    end_point_a = Neaumaier_value(S_a, Sa_c)
    end_point_b = Neaumaier_value(S_b, Sb_c)

    return arr_a, arr_b, end_point_a, end_point_b



def find_x(A_grid, x_min = 1e-10):
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







def mesh_grid(r_N, N , r2, DRN):

    A_grid = ((r_N - (N-1) *r2) / r_N) * (DRN / (DRN -r2))

    # print(f"A_grid = {A_grid}")
    if not (0.5 < A_grid < 1.0):
        raise ValueError(f"A_grid = {A_grid: .6g}, not in required range (0.5,1). Adjust r2, DRN or N" )

    x = find_x(A_grid)
    # print(f"x = {x}")

    c = x * r_N
    b = (x * (c + r_N) * (DRN - r2))/ (DRN * r2)
    a = (c - b * r2) / (c * r2)
    d = 1 - b * np.log(c)

    def G(r):
        return a * r + b * np.log(c + r) + d


    r = np.zeros(N)
    r[0] = 0.0
    r[-1] = r_N

    for i in tqdm(range(1,N-1) , desc = "Constructing mesh grid"):
        target = i + 1
        low = r[i-1]
        high = r_N

        for k in range(60):
            mid = 0.5*(low + high)
            if G(low) > target: raise RuntimeError("Left bracketing failed")
            if G(high) < target: raise RuntimeError("Right bracketing failed")

            if G(mid) < target :
                low = mid
            else:
                high = mid

        r[i] = 0.5*(low+high)

    # print(f"1st 10 r terms: {r[:10]}")
    return r


def calc_series_terms(mesh_steps,l, potential: int, T , ratio_max = 0.05, max_subdiv = 200):

    if (potential != 0) and (potential != 1):
        raise RuntimeError(f"No valid potential function was choosen")


    u_parameters0 = Singular_potential_parameters_Dirac(mesh_steps[1], T)

    Series_terms0a , Series_terms0b , initial_condition0a, initial_condition0b, s, t = Singular_term_power_series_Dirac(u_parameters0, mesh_steps[1],l,epsilon)

    initial_temp_condition_a = initial_condition0a
    initial_temp_condition_b = initial_condition0b


    Series_terms_list_a = [Series_terms0a]
    Series_terms_list_b = [Series_terms0b]

    # for i in tqdm(range(1, len(mesh_steps)-1), desc="Calculating Series Terms"):
    for i in range(1, len(mesh_steps)-1):
        r_a = mesh_steps[i]
        r_b = mesh_steps[i+1]

        h = r_b - r_a
        ratio = h/ r_a
        #
        # print(f"h = {h}, r_b = {r_b} , r_a = {r_a} , h/r_a = {ratio}")
        # print(f"condition ratio > ratio_max: {ratio} > {ratio_max}")

        if ratio > ratio_max:
            m = int(np.ceil(ratio/ratio_max))
            m = min(m, max_subdiv)


            sub_grid = np.linspace(r_a, r_b , m+1)

            for j in range(m):
                ra = float(sub_grid[j])
                rb = float(sub_grid[j+1])

                u_temp_parameter = Potential_parameters_Dirac(methods[potential], ra, rb, l, T)
                # print(f"calling regular with ra - rb = {ra, rb}, init: {initial_temp_condition_a, initial_temp_condition_b}, u: {u_temp_parameter}")
                Temp_series_terms_a,Temp_series_terms_b, initial_temp_condition_a, initial_temp_condition_b = Power_series_Dirac(u_temp_parameter, initial_temp_condition_a, initial_temp_condition_b, ra , rb, l, epsilon)

                if rb == r_b:
                    Series_terms_list_a.append(Temp_series_terms_a)
                    Series_terms_list_b.append(Temp_series_terms_b)


        else:

            u_temp_parameter = Potential_parameters_Dirac(methods[potential], r_a, r_b,l, T)



            Temp_series_terms_a,Temp_series_terms_b, initial_temp_condition_a, initial_temp_condition_b = Power_series_Dirac(u_temp_parameter, initial_temp_condition_a, initial_temp_condition_b, r_a , r_b, l, epsilon)

            Series_terms_list_a.append(Temp_series_terms_a)
            Series_terms_list_b.append(Temp_series_terms_b)


    # Series_terms_arr_a = np.concatenate(Series_terms_list_a)
    # Series_terms_arr_b = np.concatenate(Series_terms_list_b)


    return Series_terms_list_a, Series_terms_list_b


def Find_rc(r_mesh,k):
    kr_min = 20.0
    r_grid = np.asarray(r_mesh, dtype = float)


    r_safe = np.maximum(r_grid, 1e-16)
    RV = r_safe * potential(r_safe)

    r_infty = 1000
    Z_inf = r_infty * potential(r_infty)

    print(f"Z_inf = {Z_inf}")
    TAS = max(1e-11, epsilon) * abs(Z_inf)

    #scanning inward istead of outward for efficiency
    idx_rc = None
    for idx in range(len(r_grid)-1 ,2, -1):
        # print(f"idx = {idx}: rc condition: {abs(RV[idx] - Z_inf)} < {TAS} and k*r condition: {k * r_grid[idx]} => {kr_min}")
        if abs(RV[idx] - Z_inf) <= TAS and (k * r_grid[idx] >= kr_min):
            idx_rc = idx
            # if r_grid[idx_rc] < 20:
            #break
        else:
            break


    if idx_rc is None:
        #### Include a loop to automatically extend the distance further
        raise RuntimeError("No rc found, extend distance")
    print(f"rc succesfully found at rc = {r_grid[idx_rc]}")
    return idx_rc


def Normalization_constant(r_mesh, idx_rc, P_mesh, Q_mesh, Analytic_normalization_const, T, k, eta, W):
    P = P_mesh[idx_rc]
    Q = Q_mesh[idx_rc]

    r_c = r_mesh[idx_rc]
    x = k * r_c


    if abs(Z) < 1:
        print("potential is OFF")
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
        print("potential is ON")


        # Q_prime = kappa/r_c * Q - ((T - V)/c) * P



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


    V = float(potential(r_c))
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

    # print(f"P = {P}, P_prime = {P_prime}")

    if abs(P) > epsilon:
        A = (COS * PIA + SIN * PIB) / P
    else:
        A = (COS * PIAP + SIN * PIBP) / P_prime

    # print(f"A = {A}")
    # print(f"delta = {delta}")
    #
    # print()
    # print(f"r_c = {r_c}, idx = {idx_rc}")
    # print(f"PO = {P}, QO = {Q}")
    # print(f"V(r_c) = {V}, FG = {FG}")
    # print(f"PA = {float(PIA): .6f}, QA = {float(QA): .6f}, PB = {float(PIB): .6f}, QB = {float(QB): .6f}, PIAP = {float(PIAP): .6f}, PIBP = {float(PIBP): .6f} ")
    #




    return float(A) , float(delta), r_c


def radial_dirac_wave_function(list_of_sequence_terms_a, list_of_sequence_terms_b, grid_points):

    r_mesh = np.asarray(grid_points, dtype = float)
    r_size = len(r_mesh)

    P_mesh = np.zeros(r_size, dtype = float)
    Q_mesh = np.zeros(r_size ,dtype = float)

    # Initialize wavefunc at r=0
    P_mesh[0] = 0.0
    Q_mesh[0] = 0.0

    a0 = np.asarray(list_of_sequence_terms_a[0], dtype = float)
    b0 = np.asarray(list_of_sequence_terms_b[0], dtype = float)

    # at x=1 [P(1) = (1)^s sum A_n (1)^n] & [Q(1) = (1)^(s+t) sum B_n (1)^n]
    P_mesh[1] = float(np.sum(a0))
    Q_mesh[1] = float(np.sum(b0))

    # Now regular intervals
    # for i in tqdm(range(1,r_size -1), desc = "Calculating wave functions P and Q"):
    for i in range(1,r_size - 1):
        ai = np.asarray(list_of_sequence_terms_a[i], dtype = float)
        bi = np.asarray(list_of_sequence_terms_b[i], dtype = float)

        P_mesh[i+1] = float(np.sum(ai))
        Q_mesh[i+1] = float(np.sum(bi))

    return P_mesh, Q_mesh





## Analytic Dirac-Coulomg function
def Analytic(eta, k, W, Analytic_normalization_const, T, delta, end_point, N_points):
    r = np.linspace(0,end_point,N_points)
    if abs(Z) < 1:
        print(f"Z low: {Z}")
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
        print(f"|Z| > 0: {Z}")
        # if kappa < 0:
        #     l = -kappa -1
        #     lp = l + 1
        # else:
        #     l = kappa
        #     lp = l-1


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




        # print(f"lambda - 1 = {Lambda - 1}")
        U_reg_lambda = CoulombF_reduced(Lambda,Z,r)

        U_reg_lambda_min = CoulombF_reduced(Lambda-1 ,Z,r)

        G_lambda = CoulombG_reduced(Lambda,Z,r)
        G_lambda_min = CoulombG_reduced(Lambda -1, Z,r)


        #### Remove this
        delta_eff =delta
        norm_eff = Analytic_normalization_const



        Dirac_upper_analytic = norm_eff* ((kappa+Lambda) * np.sqrt(Lambda**2 + eta**2)*k*c*U_reg_lambda + Z/c * (Lambda * c**2 - kappa * W) * U_reg_lambda_min)

        Dirac_lower_analytic = -norm_eff*(Z/c * np.sqrt(Lambda**2 + eta**2)*k*c*U_reg_lambda + (kappa + Lambda) * (Lambda* c**2 - kappa * W )* U_reg_lambda_min)

        irreg_Dirac_upper_analytic = norm_eff* ((kappa+Lambda) * np.sqrt(Lambda**2 + eta**2)*k*c* G_lambda + Z/c * (Lambda * c**2 - kappa * W) * G_lambda_min)

        irreg_Dirac_lower_analytic = -norm_eff*(Z/c * np.sqrt(Lambda**2 + eta**2)*k*c* G_lambda + (kappa + Lambda) * (Lambda* c**2 - kappa * W )* G_lambda_min)


        P = np.cos(delta_eff) * Dirac_upper_analytic + np.sin(delta_eff) * irreg_Dirac_upper_analytic
        Q = np.cos(delta_eff) * Dirac_lower_analytic + np.sin(delta_eff) * irreg_Dirac_lower_analytic




    return r,P, Q




def Visualize(name, Analytic_normalization_const, T, k, eta, W):

    if name == "Coulomb":
        mesh_points = mesh_grid(r_END, r_steps , r2 , DRN)
        idx_rc = Find_rc(mesh_points)
        series_terms_upper, series_terms_lower = calc_series_terms(mesh_points,angular_momentum_quantum_number,0, T)
        wave_function_upper, wave_function_lower = radial_dirac_wave_function(series_terms_upper, series_terms_lower, mesh_points)
        N, delta, r_c = Normalization_constant(mesh_points, idx_rc,  wave_function_upper, wave_function_lower)

    elif name == "Spline":
        mesh_points = mesh_grid(r_END, r_steps , r2 , DRN)
        idx_rc = Find_rc(mesh_points, k)
        t_start = time.perf_counter()
        series_terms_upper, series_terms_lower = calc_series_terms(mesh_points,angular_momentum_quantum_number,1, T)
        t_end = time.perf_counter()
        wave_function_upper, wave_function_lower = radial_dirac_wave_function(series_terms_upper, series_terms_lower, mesh_points)
        N, delta, r_c = Normalization_constant(mesh_points, idx_rc ,wave_function_upper,wave_function_lower, Analytic_normalization_const, T,k,eta,W)
        print(f"TOTAL RUNTIME = {t_end - t_start: .3f} s")
        print(f"Normalization_constant = {N}")

    else:
        raise RuntimeError("No valid potential choosen")



    # #
    # r = np.linspace(0,r_END,2000)
    # # # analytical_solution = CoulombF_reduced(angular_momentum_quantum_number,Z,T,r)
    # #
    r_analytic, P_analytic, Q_analytic = Analytic(eta, k ,W, Analytic_normalization_const, T, delta, 3, 100000)
    #

    # #
    # np.savez_compressed("analytic_dirac.npz", r=r_analytic, P = P_analytic, Q = Q_analytic)

    # print("Paremeters used during Analytcic setup")
    # print(f"k = {k}")
    # print(f"eta = {eta}")
    # print(f"Lambda = {Lambda}")
    # print(f"r_max numerica = {r_analytic[-1]}")
    # print(f"len(P_grid) = {len(f__upper_analytic)}, len(Q_grid) = {len(f_lower_analytic)}")
    # print(f"r_grid (mesh) first/last: no mesh used, just r = np.linspace() functions")

    # #
    # data = np.load("analytic_dirac.npz")
    # r_analytic = data["r"]
    # P_analytic = data["P"]
    # Q_analytic = data["Q"]
    #
    plt.figure(figsize=(12,8))
    plt.plot(mesh_points, wave_function_upper*N , label = "P(r) - Normalized")
    plt.plot(mesh_points, wave_function_lower*N , label = "Q(r) - Normalized")
    plt.plot(r_analytic, P_analytic, ls = "--" , label = "Analytical mpmath sol P(r)")
    plt.plot(r_analytic, Q_analytic, ls = "--" , label = "Analytical mpmath sol Q(r)")
    plt.axvline(x = R_au, color = "gray" , ls = "--")
    # print(f"Normalized P and Q around R are P = {N*wave_function_upper[978]} and Q = {N*wave_function_lower[978]}")
    # print(f"potential V(r): 1st 15 points: {potential(mesh_points[:15])}, last 15 points: {potential(mesh_points[-15:])}")


    plt.xlabel("r in a.u")
    plt.ylabel("P(r)")
    plt.grid(True)
    plt.title(f"E = {T: .2g}, Z = {Z}, A = {N: .3g}, delta = {delta: .3g}, N = {r_steps}, DRN = {DRN} , r2 = {r2}, r_end = {r_END}, r_c = {r_c: .3g}")
    plt.legend()
    plt.show()


T =  1e-4 * 1e6 / E_hatree
W = T + c**2
k = np.sqrt(T*(T + 2*c**2))/c
eta =  alpha* Z *W/(k*c)
Analytic_normalization_const = 1 /Lambda * 1/np.sqrt((Z/c)**2 * (W+c**2)**2 + (kappa + Lambda)**2 *(k*c)**2)

print(f"T = {T:.3g}, E = W = T + c^2 = {W: .5g}, k = {k: .5g}, eta = {eta:.5g}")

Visualize(methods[1] ,Analytic_normalization_const, T, k, eta, W)


def Obtain_Fermi_Data(Energy_range):


    print(f"Checking for assymptotic solution on current r_grid")
    mesh_points = mesh_grid(r_END, r_steps , r2 , DRN)
    # rc_idx = Find_rc(mesh_points, 1)


    P_list = []
    Q_list = []
    aP_list = []
    aQ_list = []

    kr_c_list = []

    for i, T in enumerate(tqdm(Energy_range, desc = "Solving Dirac functions over energy range")):
        W = T + c**2
        k = np.sqrt(T*(T + 2*c**2))/c
        eta = alpha* Z*W/(k*c)
        Analytic_normalization_const = 1 /Lambda * 1/np.sqrt((Z/c)**2 * (W+c**2)**2 + (kappa + Lambda)**2 *(k*c)**2)
        # print(f"1st term (Z/c)**2 * (W+c**2)**2 = {(Z/c)**2 * (W+c**2)**2 } ")
        # print(f"2nd term (kappa + Lambda)**2 *(k*c)**2 = {(kappa + Lambda)**2 *(k*c)**2} ")


        print(f"Finding wave function for T = {T}, iter = {i}, eta = {eta}")
        rc_idx = Find_rc(mesh_points, k)
        series_terms_upper, series_terms_lower = calc_series_terms(mesh_points,angular_momentum_quantum_number,1, T)
        wave_function_upper, wave_function_lower = radial_dirac_wave_function(series_terms_upper, series_terms_lower, mesh_points)
        N, delta, r_c = Normalization_constant(mesh_points, rc_idx , wave_function_upper,wave_function_lower, Analytic_normalization_const, T,k, eta, W)
        print(f"Normalization_constant = {N}")
        print(f"k = {k: .6g}, r_c = {r_c: .6g}, k*r_c = {k*r_c}")
        kr_c_list.append(k*r_c)

        P_list.append(N*wave_function_upper)
        Q_list.append(N*wave_function_lower)



        # end_pt = 0.1
        # N_grid = 1000
        # r_analytic, P_analytic, Q_analytic = Analytic(eta, k ,W, Analytic_normalization_const, T, delta, end_pt, N_grid )
        #
        # aP_list.append(P_analytic)
        # aQ_list.append(Q_analytic)

        # if (i==0) or (i== len(Energy_range)-1):
        #     plt.figure(figsize=(12,8))
        #     plt.plot(mesh_points, wave_function_upper*N , label = "P(r) - Normalized")
        #     plt.plot(mesh_points, wave_function_lower*N , label = "Q(r) - Normalized")
        #     plt.plot(r_analytic, P_analytic, label = "P(r) - Analytic")
        #     plt.plot(r_analytic, Q_analytic, label = "Q(r) - analytic" )
        #     plt.xlabel("r in a.u")
        #     plt.ylabel("P(r)")
        #     plt.grid(True)
        #     plt.title(f"E = {T: .5g}, Z = {Z}, A = {N: .5}, delta = {delta: .5}, N = {r_steps}, DRN = {DRN} , r2 = {r2}, r_end = {r_END}, r_c = {r_c: .5}")
        #     plt.legend()
        #     plt.show()

    P_arr = np.asarray(P_list, dtype = float)
    Q_arr = np.asarray(Q_list, dtype = float)
    r_arr = np.asarray(mesh_points, dtype = float)
    T_arr = np.asarray(Energy_range, dtype = float)
    #
    # aP_arr = np.asarray(aP_list, dtype = float)
    # aQ_arr = np.asarray(aQ_list, dtype = float)

    # plt.figure()
    # plt.plot(T_arr, kr_c_list, marker = "o")
    # plt.grid(True)
    # plt.ylabel(" k*r_c")
    # plt.xlabel("T")
    # plt.show()
    print(np.asarray(kr_c_list, dtype=float))


    np.savez_compressed("Dirac_run_kappa_n_schemeA_V_Z56.npz", T = T_arr, r = r_arr, P = P_arr, Q = Q_arr)
    # np.savez_compressed("Analytic_Coulomb_kappa_p_V_Z56.npz", T = T_arr, r = r_analytic, P = aP_arr, Q = aQ_arr)
#
#
#
# Q = 2.45791
# T_range = np.geomspace(1e-4, Q, 30) * 1e6 /E_hatree
# t_start = time.perf_counter()
# Obtain_Fermi_Data(T_range)
# t_end = time.perf_counter()
# print(f"TOTAL RUNTIME = {t_end - t_start: .3f} s, {(t_end - t_start)/60: .3f} minutes")


