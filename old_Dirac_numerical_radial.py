#!/usr/bin/env python

#Importing libraries
from mpmath.libmp.libintmath import MAX_EULER_CACHE
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import mpmath as mp
import os
from tqdm import tqdm

mp.mp.dps = 50


Z = 1
angular_momentum_quantum_number = 0
kappa = -1
sigma = -np.sign(kappa)
c = 137.03599
epsilon = 1e-15


#
# E_hatree = 27.211386 #eV
# m_ELECTRON = 0.51099895069e6/E_hatree #eV
# E_max = 1.2933325099999138e6/27.211386 #eV

T = 1e-4
# E = m_ELECTRON


W = T + c**2

k = np.sqrt(T*(T + 2*c**2))/c
eta = Z/k
Lambda = np.sqrt(kappa**2 - (Z/c)**2)

Analytic_normalization_const = 1/Lambda * 1/np.sqrt((Z/c)**2 * (W+c**2)**2 + (kappa + Lambda)**2 *(k*c)**2)



r_steps = 80000
r_END = 3
r2 = 1e-7
r0 = 1e-16 ## to avoid 1/r division by 0
DRN = 0.0001 #Upper limit on distance between points near the end regime (make sure this stays small enough or wavefunc will not converge at larger distances r)



N=10 # steps in between r_a and r_b

methods = ["Coulomb" , "Spline"]


###################################
###Setup Cubic Spline potential V
###################################
N_V = 20000
V0 = -0.1
A =  1.0
def potential(r):
    return Z/r + V0* np.exp(-A*r)

r_V = np.linspace(r0,r_END, N_V)


V = potential(r_V)
RV = r_V * V

V_spline = CubicSpline(r_V, RV)

V_spline_prime = V_spline.derivative(1)
V_spline_second = V_spline.derivative(2)
V_spline_third = V_spline.derivative(3)


def Potential_parameters_Dirac(name, r_a ,r_b, l):

    if name == "Coulomb":
        v0 = Z
        v1 = v2= v3 = 0.0


    elif name == "Spline":

        rv0 = float(V_spline(r_a))
        rv1 = float(V_spline_prime(r_a))
        rv2 = 0.5 * float(V_spline_second(r_a))
        rv3 = (1.0/6.0) * float(V_spline_third(r_a))

        v0 = rv0 / r_a
        v1 = (rv1 * r_a - rv0) / (r_a**2)
        v2 = (rv2 * r_a**2 - rv1 * r_a + rv0) / (r_a**3)
        v3 = (rv3 * r_a**3 - rv2 * r_a**2 + rv1 * r_a - rv0) / (r_a**4)
    else:
        raise ValueError("No valid potential type was given, choose between 'Coulomb' or 'Spline' ")


    # print(f"v0 = {v0}")
    # print(f"v1 = {v1}")
    # print(f"v2 = {v2}")
    # print(f"v3 = {v3}")

    h = r_b-r_a
    alpha = 1.0/c

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
        print("u0 != 0")
        t = 0
        if u0**2 > kappa**2:
            raise RuntimeError("s = sqrt(kappa^2 - u0^2) is imaginary")
        else:
            s = np.sqrt(kappa**2 - u0**2)


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




def Power_series_Dirac(u_array, initial_condition_a, initial_condition_b , r_a, r_b, l, eps):
    #Define the u terms
    u0, u1, u2, u3 = u_array
    h = r_b-r_a

    print(f"r_a = {r_a} , r_b = {r_b} , h = {h}")

    ra = mp.mpf(r_a)
    rb = mp.mpf(r_b)
    hh = mp.mpf(h)

    U0 = mp.mpf(u0); U1 = mp.mpf(u1); U2 = mp.mpf(u2); U3 = mp.mpf(u3)
    K = mp.mpf(kappa)

    pre_factor = hh/ra
    mult_term1 = U0 - mp.mpf(2)*mp.mpf(c)*ra
    mult_term2 = U1 - mp.mpf(2)*mp.mpf(c)*hh
    #create the a array and define the intial conditions
    a = [mp.mpf(initial_condition_a)]
    b = [mp.mpf(initial_condition_b)]


    a0 = a[0]
    b0 = b[0]

    a1 = -pre_factor * (K*a0 + mult_term1*b0)
    b1 = pre_factor * (K*b0 + U0*a0)

    a2 = -pre_factor/2 * ((K + 1)*a1 + mult_term1*b1 + mult_term2*b0)
    b2 = pre_factor/2 * ((K - 1)*b1 + U0*a1 + U1*a0)

    a3 = -pre_factor/3 * ((K + 2)*a2 + mult_term1*b2 + mult_term2*b1 + U2*b0)
    b3 = pre_factor/3 * ((K - 2)*b2 + U0*a2 + U1*a1 + U2*a0)

    a += [a1, a2, a3]
    b += [b1, b2, b3]



    #Create the Sum terms to check for convergence critaria
    S_a = mp.fsum(a)  #Sum a_n from n= 0 to j
    S_b = mp.fsum(b)
    S_an = a1 + mp.mpf(2)*a2 + mp.mpf(3)*a3
    S_bn = b1 + mp.mpf(2)*b2 + mp.mpf(3)*b3


    print(f" a terms: {a}")
    print(f" b terms; {b}")

    n=3




    while True:
        n +=1
        n_mp = mp.mpf(n)
        pf = pre_factor / n_mp

#         an = -pre_factor/n * ((kappa - 1 + n)*a[n-1] + mult_term1*b[n-1] + mult_term2*b[n-2] + u2*b[n-3] + u3*b[n-4])
#
#         bn = pre_factor/n * ((kappa + 1 - n)*b[n-1] + u0*a[n-1] + u1*a[n-2] + u2*a[n-3] + u3*a[n-4])
        a_part = (K - 1 + n_mp) * a[n-1] + mult_term1 * b[n-1] + mult_term2 * b[n-2] + U2 * b[n-3] + U3 * b[n-4]
        an = -pf * a_part

        b_part = (K + 1 - n_mp) * b[n-1] + U0 * a[n-1] + U1 * a[n-2] + U2 * a[n-3] + U3 * a[n-4]
        bn = pf * b_part
        #
        # an = float(an_mp)
        # bn = float(bn_mp)

        # print(f"n = {n}")
        # print(f"an components = {pre_factor/n} x ({(kappa - 1 + n)*a[n-1]} + {mult_term1*b[n-1] } + {mult_term2*b[n-2]} + {u2*b[n-3]} + {u3*b[n-4]})")

        # print(f"a = {an}")
        # print(f"b = {bn} ")


        a.append(an)
        b.append(bn)

        S_a += an
        S_b += bn

        S_an += n_mp * an
        S_bn += n_mp * bn


        # print(f"S_a = {S_a}, S_b = {S_b} , S_an = {S_an} , S_bn = {S_bn}")



        tolerance = mp.mpf(eps) * max(abs(S_a), abs(S_b), abs(S_an)/n_mp, abs(S_bn)/n_mp)
        # print(f" {max(abs(an) , abs(bn))} < {tolerance}")

        # condition1 = abs(r_b*S_an - sigma*abs(kappa)*h*S_a + ((u0+u1+u2+u3) - 2*c*r_b) * h * S_b)
        # condition2 = abs(r_b*S_bn + sigma*abs(kappa)*h*S_b - (u0+u1+u2+u3)*h*S_a )
        # print(f"{max(condition1 , condition2)} < {tolerance}")
        # print()

        if max(abs(an) , abs(bn)) < tolerance:
            condition1 = abs(rb*S_an - mp.mpf(sigma)*abs(K)*hh*S_a + ((U0+U1+U2+U3) - mp.mpf(2)*mp.mpf(c)*rb) * hh * S_b)
            condition2 = abs(rb*S_bn + mp.mpf(sigma)*abs(K)*hh*S_b - (U0+U1+U2+U3)*hh*S_a )
            # print(f"{max(condition1 , condition2)} < {tolerance}")
            # print()
            if max(condition1 , condition2) < tolerance:
                break



        if n > 499:
            raise RuntimeError(f"No convergence before n = {n}")



    arr_a = np.array([float(x) for x in a], dtype = float)
    arr_b = np.array([float(x) for x in b], dtype = float)

    end_point_a = float(S_a)
    end_point_b = float(S_b)

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

    print(f"A_grid = {A_grid}")
    if not (0.5 < A_grid < 1.0):
        raise ValueError(f"A_grid = {A_grid: .6g}, not in required range (0.5,1). Adjust r2, DRN or N" )

    x = find_x(A_grid)
    print(f"x = {x}")

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

    print(f"1st 10 r terms: {r[:10]}")
    return r


def calc_series_terms(distance,N_steps,l, potential: int, ratio_max = 0.1, max_subdiv = 200):

    if (potential != 0) and (potential != 1):
        raise RuntimeError(f"No valid potential function was choosen")


    # mesh_steps = np.linspace(0,r, N_steps+1)
    mesh_steps = mesh_grid(distance, N_steps , r2 , DRN)


    r_index = np.linspace(0, N_steps-1, N_steps)
    print(r_index[0:10])
    print(r_index[75000:75010])

    plt.figure(figsize = (12,8))
    plt.plot(r_index, mesh_steps, label = "mesh points")
    plt.xlabel("index i")
    plt.ylabel("distance r")
    plt.title("mesh points plot")
    plt.show()


    # print(mesh_steps)
    u_parameters0 = Potential_parameters_Dirac(methods[potential], mesh_steps[0], mesh_steps[1], l)

    Series_terms0a , Series_terms0b , initial_condition0a, initial_condition0b, s, t = Singular_term_power_series_Dirac(u_parameters0, mesh_steps[1],l,epsilon)
    #
    initial_temp_condition_a = initial_condition0a
    initial_temp_condition_b = initial_condition0b


    Series_terms_list_a = [Series_terms0a]
    Series_terms_list_b = [Series_terms0b]
    mesh_steps_effective = [mesh_steps[0], mesh_steps[1]]

    for i in tqdm(range(1, len(mesh_steps)-1), desc="Calculating Series Terms"):
        print()
        print(f"i = {i}")
        # print(i)
        # print(f"r range from {mesh_steps[i]} to {mesh_steps[i+1]}")
        # print(f"Using intial conditions a0 = {initial_temp_condition0}, a1 = {initial_temp_condition1}")
        r_a = float(mesh_steps[i])
        r_b = float(mesh_steps[i+1])

        h = r_b - r_a
        ratio = h/ max(r_a, r0)

        print(f"h/r_a = {ratio}")

        if ratio > ratio_max:
            m = int(np.ceil(ratio/ratio_max))
            m = min(m, max_subdiv)


            sub_grid = np.linspace(r_a, r_b , m+1)

            for j in range(m):
                ra = float(sub_grid[j])
                rb = float(sub_grid[j+1])

                u_temp_parameter = Potential_parameters_Dirac(methods[potential], ra, rb, l)
                Temp_series_terms_a,Temp_series_terms_b, initial_temp_condition_a, initial_temp_condition_b = Power_series_Dirac(u_temp_parameter, initial_temp_condition_a, initial_temp_condition_b, ra , rb, l, epsilon)

                Series_terms_list_a.append(Temp_series_terms_a)
                Series_terms_list_b.append(Temp_series_terms_b)

                mesh_steps_effective.append(rb)

        else:

            u_temp_parameter = Potential_parameters_Dirac(methods[potential], r_a, r_b,l)



            Temp_series_terms_a,Temp_series_terms_b, initial_temp_condition_a, initial_temp_condition_b = Power_series_Dirac(u_temp_parameter, initial_temp_condition_a, initial_temp_condition_b, r_a , r_b, l, epsilon)

            Series_terms_list_a.append(Temp_series_terms_a)
            Series_terms_list_b.append(Temp_series_terms_b)

            mesh_steps_effective.append(r_b)

    # Series_terms_arr_a = np.concatenate(Series_terms_list_a)
    # Series_terms_arr_b = np.concatenate(Series_terms_list_b)
    mesh_steps_effective = np.array(mesh_steps_effective, dtype = float)

    return Series_terms_list_a, Series_terms_list_b, mesh_steps_effective, s, t







# def Normalization_constant(method,distance, list_of_sequence_terms_a , list_of_sequence_terms_b, grid_points, unnormalized_wave_func_a = np.empty(1), unnormalized_wave_func_b = np.empty(1), k=k):
#
#
    # def find_rc():
    #     r_grid = np.asarray(grid_points, dtype = float)
    #
    #     r_safe = np.maximum(r_grid, 1e-16)
    #     RV = r_safe * potential(r_safe)
    #
    #     Z_inf = RV[-1]
    #     print(f"Z_inf = {Z_inf}")
    #     TAS = max(1e-11, epsilon) * abs(Z_inf)
    #
    #     #scanning inward istead of outward for efficiency
    #     for idx in range(len(r_grid)-1 ,2, -1):
    #         if abs(RV[idx] - Z_inf) <= TAS:
    #             return r_grid[idx], idx
    #     raise RuntimeError("No rc found: extend grid")
    #
    # r_c , idx_rc = find_rc()
    #
    # print(f"r_c = {r_c}")
#
#
#     def eval_series_at_rc(series_terms, x):
#         y = 0.0
#         for a in reversed(series_terms):
#             y = y*x + a
#         return y
#     # elif method == "Coulomb":
#     #
#     #     fraction = 0.98
#     #     # last_quartile = int(fraction*len(unnormalized_wave_func_a))
#     #     # last_quartile_b = int(fraction*len(unnormalized_wave_func_b))
#     #     #
#     #     # idx = np.argmax(unnormalized_wave_func_a[last_quartile:])
#     #     # idx_b = np.argmax(unnormalized_wave_func_b[last_quartile_b:])
#     #     #
#     #     # idx_rc_b = last_quartile_b + idx_b
#     #     # idx_rc = last_quartile + idx
#     #     #
#     #     # r_c = r_function[idx_rc]
#     #     # r_c_b = r_function[idx_rc_b]
#     #     #
#     #     # print(f"r_c = {r_c}")
#     #     # print(f"r_c_b = {r_c_b}")
#     #
#     #
#     #     tail0 = int(fraction * len(r_function))
#     #     R = np.sqrt(unnormalized_wave_func_a*unnormalized_wave_func_a + unnormalized_wave_func_b * unnormalized_wave_func_b)
#     #     R_tail = R[tail0:]
#     #
#     #     # dR = np.abs(np.diff(R_tail))
#     #     # j = np.argmin(dR)
#     #     #
#     #     # idx_rc = tail0+j
#     #     # r_c = r_function[idx_rc]
#     #     # print(f"index of argmin = {j} with r_c = {r_c}")
#     #
#     #     idx = np.argmax(R_tail)
#     #     r_c = r_function[idx + tail0]
#     #
#     # else:
#     #     raise RuntimeError("No valid potential was choosen, specify between 'Coulomb' or 'Spline'")
#
#
#
#     print(f"r_c = {r_c}")
#
#
#     # print("Checks")
#     # print(len(r_function) , len(unnormalized_wave_func_a))
#     # print(r_function[0], r_function[-1])
#     # print(f"tail start r ~ { r_function[int(fraction*len(r_function))]}")
#     #
#     # last_section = int(fraction * len(unnormalized_wave_func_a))
#     # r_tail = r_function[last_section:]
#     # P_tail = unnormalized_wave_func_a[last_section:]
#     #
#     # s = P_tail[:-1] * P_tail[1:]
#     # idxs = np.where(s<0)[0]
#     #
#     # if len(idxs) > 0:
#     #     i = idxs[0]
#     #     r0, r1 = r_tail[i], r_tail[i+1]
#     #     p0, p1 = P_tail[i], P_tail[i+1]
#     #     r_c = r0 - p0 * (r1-r0)/(p1-p0)
#     # else:
#     #     dP = np.gradient(P_tail, r_tail)
#     #     i = np.argmax(np.abs(dP))
#     #     r_c = float(r_tail[i])
#     #
#
#
#
#
#
#     #
#     # edge_mesh_idx = 0
#     # for k in range(len(grid_points)-1):
#     #     if (r_c >= grid_points[k]) & (r_c <= grid_points[k+1]):
#     #         print(f" r_c = {r_c} and {grid_points[k]} <= {r_c} <= {grid_points[k+1]}")
#     #         edge_mesh_idx = k
#     #         break
#
#
#
#
#
#     series_terms_a = np.array(list_of_sequence_terms_a[i])
#     series_terms_b = np.array(list_of_sequence_terms_b[edge_mesh_idx])
#
#
#
#     x_c = (r_c - grid_points[edge_mesh_idx]) / (grid_points[edge_mesh_idx+1] - grid_points[edge_mesh_idx])
#
#
#
#     def Power_series(series_terms, x_c, local_idx):
#         Power_series = 0
#         Power_series_prime = 0
#         for j in range(len(series_terms)-1):
#             Power_series += series_terms[j] * x_c**j
#         #     Power_series_prime += (j+1) * series_terms[j+1] *  x_c**j  ## j+1 to avoid x^j-1 term at j=0
#         # Power_series_prime /= (grid_points[local_idx + 1] - grid_points[local_idx])
#         return Power_series
#
#     P  = Power_series(series_terms_a, x_c, edge_mesh_idx)
#     Q  = Power_series(series_terms_b, x_c, edge_mesh_idx)
#
#     P_prime = -kappa/r_c * P + (W- (Z/r_c) + c**2)/c * Q
#     Q_prime = kappa/r_c * Q - (W - (Z/r_c) - c**2)/c * P
#
#     def Coulomb_funcs(index, r_c):
#         Coulombf= mp.coulombf(index, eta, k*r_c)
#         Coulombg = mp.coulombg(index, eta, k*r_c)
#
#         deriv_Coulombf= k * mp.diff(lambda t: mp.coulombf(index, eta, t), k*r_c)
#         deriv_Coulombg = k * mp.diff(lambda t: mp.coulombg(index, eta, t), k*r_c)
#
#         return Coulombf, Coulombg, deriv_Coulombf, deriv_Coulombg
#     # UPPER
#     F_upper_lambda, G_upper_lambda, deriv_F_upper_lambda , deriv_G_upper_lambda = Coulomb_funcs(Lambda, r_c)
#
#     F_upper_lambda_min, G_upper_lambda_min, deriv_F_upper_lambda_min, deriv_G_upper_lambda_min = Coulomb_funcs(Lambda-1 ,r_c)
#     # LOWER
#     F_lower_lambda, G_lower_lambda, deriv_F_lower_lambda , deriv_G_lower_lambda = Coulomb_funcs(Lambda, r_c)
#
#     F_lower_lambda_min, G_lower_lambda_min, deriv_F_lower_lambda_min, deriv_G_lower_lambda_min = Coulomb_funcs(Lambda-1 ,r_c)
#
#
#
#     mult_term_addition = (np.sqrt(Lambda**2 + eta**2))*k*c
#     mult_term_subtraction = (Lambda * c**2 - kappa * W)
#
#     def Dirac_upper_func(Coulomb_func,Coulomb_min_func):
#
#         upper = Analytic_normalization_const * ((kappa + Lambda) * mult_term_addition* Coulomb_func + Z/c * mult_term_subtraction * Coulomb_min_func)
#
#         return upper
#
#     def Dirac_lower_func(Coulomb_func, Coulomb_min_func):
#
#        lower = -Analytic_normalization_const* (Z/c * mult_term_addition * Coulomb_func + (kappa + Lambda) * mult_term_subtraction * Coulomb_min_func)
#
#        return lower
#
#     # DF = 'Dirac Function'
#     upper_regular_DF = Dirac_upper_func(F_upper_lambda, F_upper_lambda_min) #F^u
#     deriv_upper_regular_DF = Dirac_upper_func(deriv_F_upper_lambda, deriv_F_upper_lambda_min)
#
#     upper_irregular_DF = Dirac_upper_func(G_upper_lambda, G_upper_lambda_min) #G^u
#     deriv_upper_irregular_DF = Dirac_upper_func(deriv_G_upper_lambda, deriv_G_upper_lambda_min)
#
#     lower_regular_DF = Dirac_lower_func(F_lower_lambda, F_lower_lambda_min) #F^l
#     deriv_lower_regular_DF = Dirac_lower_func(deriv_F_lower_lambda, deriv_F_lower_lambda_min)
#
#     lower_irregular_DF = Dirac_lower_func(G_lower_lambda, G_lower_lambda_min) #G^l
#     deriv_lower_irregular_DF = Dirac_lower_func(deriv_G_lower_lambda, deriv_G_lower_lambda_min)
#
#
#     # print(f"Pprime: via formula {P_prime_formula}, via np.diff {P_prime}")
#     # print(f"Qprime: via formula {Q_prime_formula}, via np.diff {Q_prime}")
#
#
#     delta = mp.atan2((P * lower_regular_DF - Q * upper_regular_DF),(Q * upper_irregular_DF - P * lower_irregular_DF))
#
#
#     if abs(P) < epsilon:
#         A_upper = (deriv_upper_regular_DF + mp.tan(delta) * deriv_upper_irregular_DF)/P_prime
#
#     else:
#          A_upper = (upper_regular_DF + mp.tan(delta) * upper_irregular_DF)/P
#
#
#     if abs(Q) < epsilon:
#         A_lower = (deriv_lower_regular_DF + mp.tan(delta) * deriv_lower_irregular_DF)/Q_prime
#
#     else:
#          A_lower = (lower_regular_DF + mp.tan(delta) * lower_irregular_DF) / Q
#
#
#     A_upper = float(A_upper)
#     A_lower = float(A_lower)
#     delta = float(delta)
#
#     print(f"A^u = {A_upper}, A^l = {A_lower}")
#     print(f"delta = {delta}")
#
#
#     print("Numerical stability check")
#     print(f"P = {P}")
#     print(f"Q = {Q}")
#     print(f"F^u = {upper_regular_DF}, F^l = {lower_regular_DF}")
#     print(f"G^u = {upper_irregular_DF} G^l = {lower_irregular_DF}")
#     print(f"Denominator = {(Q * upper_irregular_DF - P * lower_irregular_DF)}")
#     print()
#
#
#
#
#
#     return A_upper, A_lower, delta, r_c

def Normalization_constant(r_mesh, P_mesh, Q_mesh):

    def find_rc():
        r_grid = np.asarray(r_mesh, dtype = float)

        r_safe = np.maximum(r_grid, 1e-16)
        RV = r_safe * potential(r_safe)

        Z_inf = RV[-1]
        print(f"Z_inf = {Z_inf}")
        TAS = max(1e-11, epsilon) * abs(Z_inf)

        #scanning inward istead of outward for efficiency
        for idx in range(len(r_grid)-1 ,2, -1):
            if abs(RV[idx] - Z_inf) <= TAS:
                return r_grid[idx], idx
        raise RuntimeError("No rc found: extend grid")

    r_c , idx_rc = find_rc()

    print(f"r_c = {r_c}")

    P = P_mesh[idx_rc]
    Q = Q_mesh[idx_rc]
    
    V = float(V_spline(r_c))
    
    FG = (T - V +2*c**2)/c
    
    Pp = -kappa/r_c *P + FG * Q
    
    F = mp.coulombf(Lambda, )
    

# def radial_dirac_wave_function(list_of_sequence_terms_a, list_of_sequence_terms_b, grid_points,Nt , s, t):
#
#
#     singular_terms_a = np.array(list_of_sequence_terms_a[0])
#     singular_terms_b = np.array(list_of_sequence_terms_b[0])
#
#
#     r_singular_range = np.linspace(grid_points[0],grid_points[1], Nt) # P(r) from [0, r_b]
#     x_singular = r_singular_range/grid_points[1]
#
#
#     # print(f"Calculting P(r) for [{grid_points[0]} , {grid_points[1]}]]")
#     def singular_wave_function_component(singular_terms):
#         singular_wave_function= np.zeros(len(r_singular_range), dtype = float)
#         for i in range(len(singular_terms)):
#             singular_wave_function += singular_terms[i] * (x_singular)**i
#
#         return singular_wave_function
#
#     singular_wave_function_a = singular_wave_function_component(singular_terms_a)
#     singular_wave_function_a *= (x_singular)**s
#
#     singular_wave_function_b = singular_wave_function_component(singular_terms_b)
#     singular_wave_function_b *= (x_singular)**(s+t)
#
#
#     wave_function_upper = singular_wave_function_a
#     wave_function_lower = singular_wave_function_b
#
#     r_total_range = r_singular_range
#
#
#
#
#
#     for i in tqdm(range(1,len(grid_points)-1), desc="Calculating wave function"):
#         # print(f"Calculting P(r) for [{grid_points[i]} , {grid_points[i+1]}]]")
#
#         r_regular_range = np.linspace(grid_points[i], grid_points[i+1],Nt) #from [r[i], r[i+1]]
#         x_regular_range = (r_regular_range - grid_points[i])/(grid_points[i+1] - grid_points[i])
#
#         temp_terms_upper = np.array(list_of_sequence_terms_a[i])
#         temp_terms_lower = np.array(list_of_sequence_terms_b[i])
#
#         temp_wave_func_upper = np.zeros(len(r_regular_range), dtype = float)
#         temp_wave_func_lower = np.zeros(len(r_regular_range), dtype = float)
#
#         for j in range(len(temp_terms_upper)):
#             temp_wave_func_upper += temp_terms_upper[j] * (x_regular_range)**j
#         for k in range(len(temp_terms_lower)):
#             temp_wave_func_lower += temp_terms_lower[k] * (x_regular_range)**k
#
#
#
#         wave_function_upper = np.concatenate((wave_function_upper, temp_wave_func_upper[1:]))
#         wave_function_lower = np.concatenate((wave_function_lower, temp_wave_func_lower[1:]))
#         r_total_range = np.concatenate((r_total_range, r_regular_range[1:]))
#
#
#
#
#
#
#     return wave_function_upper, wave_function_lower , r_total_range

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
    for i in tqdm((1,r_size -1), desc = "Calculating wave functions P and Q"):
        ai = np.asarray(list_of_sequence_terms_a[i], dtype = float)
        bi = np.asarray(list_of_sequence_terms_b[i], dtype = float)

        P_mesh[i+1] = float(np.sum(ai))
        Q_mesh[i+1] = float(np.sum(bi))

    return P_mesh, Q_mesh





## Analytic Dirac-Coulomg function
def Analytic():


    def CoulombF_reduced(l,Z,E,r_array):

        r_array = np.asarray(r_array, dtype= float)
        u_vals = np.empty_like(r_array, dtype =float)

        for i, r in enumerate (r_array):
            u_vals[i] =float(mp.coulombf(l,eta,k*r))

        return u_vals

    def CoulombG_reduced(l,Z,E,r_array):

        r_array = np.asarray(r_array, dtype= float)
        u_vals = np.empty_like(r_array, dtype =float)

        for i, r in enumerate (r_array):
            u_vals[i] =float(mp.coulombg(l, eta ,k * r))

        return u_vals

    r = np.linspace(0,r_END,10000)


    print(f"lambda - 1 = {Lambda - 1}")
    U_reg_lambda = CoulombF_reduced(Lambda,Z,W,r)

    U_reg_lambda_min = CoulombF_reduced(Lambda - 1,Z,W,r)

    G_lambda = CoulombG_reduced(Lambda,Z,W,r)
    G_lambda_min = CoulombG_reduced(Lambda-1, Z, W,r)




    Dirac_upper_analytic = Analytic_normalization_const* ((kappa+Lambda) * np.sqrt(Lambda**2 + eta**2)*k*c*U_reg_lambda + Z/c * (Lambda * c**2 - kappa * W) * U_reg_lambda_min)


    Dirac_lower_analytic = -Analytic_normalization_const*(Z/c * np.sqrt(Lambda**2 + eta**2)*k*c*U_reg_lambda + (kappa + Lambda) * (Lambda* c**2 - kappa * W )* U_reg_lambda_min)

    irreg_Dirac_upper_analytic = Analytic_normalization_const* ((kappa+Lambda) * np.sqrt(Lambda**2 + eta**2)*k*c* G_lambda + Z/c * (Lambda * c**2 - kappa * W) * G_lambda_min)

    irreg_Dirac_lower_analytic = -Analytic_normalization_const*(Z/c * np.sqrt(Lambda**2 + eta**2)*k*c* G_lambda + (kappa + Lambda) * (Lambda* c**2 - kappa * W )* G_lambda_min)




    # plt.figure()
    # # plt.plot(r, U_reg, ls = "--",  label = "Coulombf")
    # # plt.plot(r,U_ireg, label = "Coulombg")
    # plt.plot(r,Dirac_upper_analytic, label = "dirac upper")
    # plt.plot(r,Dirac_lower_analytic, label = "dirac lower")
    # plt.xlabel("r")
    # plt.ylabel("wavefunc")
    # plt.title("analytic plot")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return r,Dirac_upper_analytic, Dirac_lower_analytic, irreg_Dirac_upper_analytic,irreg_Dirac_lower_analytic




def Visualize(name, r_start = 0 , r_end = r_END):

    if name == "Coulomb":
        series_terms_upper, series_terms_lower, mesh_points, s, t = calc_series_terms(r_end,r_steps,angular_momentum_quantum_number,0)
        wave_function_upper, wave_function_lower = radial_dirac_wave_function(series_terms_upper, series_terms_lower, mesh_points)
        N_upper, N_lower, delta, r_c = Normalization_constant(name, r_end, series_terms_upper, series_terms_lower, mesh_points, wave_function_upper, wave_function_lower)


    elif name == "Spline":
        series_terms_upper, series_terms_lower, mesh_points, s, t = calc_series_terms(r_END,r_steps,angular_momentum_quantum_number,1)
        wave_function_upper, wave_function_lower, r_range = radial_dirac_wave_function(series_terms_upper, series_terms_lower, mesh_points, N, s ,t)
        N_upper, N_lower, delta, r_c = Normalization_constant(name, r_end, series_terms_upper, series_terms_lower, mesh_points, wave_function_upper, wave_function_lower, r_range)

    else:
        raise RuntimeError("No valid potential choosen")




    print()
    print(f"r_function first 1-:")
    rP = r_range
    P = wave_function_upper

    zc = np.where(np.diff(np.sign(P)) != 0 )[0]
    zc = zc[zc > int(0.2*len(zc))]

    if len(zc) > 5:
        dr = np.diff(rP[zc])
        mean_half_period = np.mean(dr)
        k_est = np.pi / mean_half_period
        print(f"k_est from numeric = {k_est}")
        print(f"k from formula = {k}")


    print()

    print("Paremeters used during Numerical setup")
    print(f"k = {k}")
    print(f"eta = {eta}")
    print(f"Lambda = {Lambda}")
    print(f"r_max numerica = {r_range[-1]}")
    print(f"len(P_grid) = {len(wave_function_upper)}, len(Q_grid) = {len(wave_function_lower)}")
    print(f"r_grid (mesh) first/last = {mesh_points[0], mesh_points[-1]} , N_mesh = {len(mesh_points)}")
    # print(f"N_upper = {N_upper}, N_lower = {N_lower}")
    # A = 0.004290560928654226

    #
    # r = np.linspace(0,r_END,2000)
    # analytical_solution = CoulombF_reduced(angular_momentum_quantum_number,Z,E,r)
    #
    # r_analytic, f__upper_analytic, f_lower_analytic, g_upper_analytic, g_lower_analytic = Analytic()
    #
    # P_analytic = f__upper_analytic + 0.0 * g_upper_analytic
    # Q_analytic = f_lower_analytic + 0.0 * g_lower_analytic
    # #
    # np.savez_compressed("analytic_dirac.npz", r=r_analytic, P = P_analytic, Q = Q_analytic)

    # print("Paremeters used during Analytcic setup")
    # print(f"k = {k}")
    # print(f"eta = {eta}")
    # print(f"Lambda = {Lambda}")
    # print(f"r_max numerica = {r_analytic[-1]}")
    # print(f"len(P_grid) = {len(f__upper_analytic)}, len(Q_grid) = {len(f_lower_analytic)}")
    # print(f"r_grid (mesh) first/last: no mesh used, just r = np.linspace() functions")

    #
    data = np.load("analytic_dirac.npz")
    r_analytic = data["r"]
    P_analytic = data["P"]
    Q_analytic = data["Q"]

    plt.figure(figsize=(12,8))
    plt.plot(r_range, wave_function_upper * N_upper, label = "P(r) - Normalized")
    plt.plot(r_range, wave_function_lower * N_upper, label = "Q(r) - Normalized")
    plt.plot(r_analytic, P_analytic, ls = "--" , label = "Analytical mpmath sol P(r)")
    plt.plot(r_analytic, Q_analytic, ls = "--" , label = "Analytical mpmath sol Q(r)")



    plt.xlabel("r in a.u")
    plt.ylabel("P(r)")
    plt.grid(True)
    plt.title(f"E = {E: .5}, Z = {Z}, A = {N_upper: .5}, delta = {delta: .5}, N = {r_steps}, DRN = {DRN} , r2 = {r2}, r_end = {r_END}, r_c = {r_c: .5}")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(r_range, np.sqrt(N_upper*wave_function_upper*N_upper*wave_function_upper + N_upper**2 * wave_function_lower * wave_function_lower))
    plt.grid(True)
    plt.show()



Visualize(methods[1], r_END)


