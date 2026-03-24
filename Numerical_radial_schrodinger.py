#!/usr/bin/env python

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
import mpmath as mp
import os
from tqdm import tqdm

mp.mp.dps = 50
#Parameters
E_hatree = 27.211386 #eV
m_ELECTRON = 0.51099895069e6/E_hatree #eV
E_max = 1.2933325099999138e6/27.211386 #eV

Z = -1
E = E_max
# E = 1.2933325099999138e6/27.211386


angular_momentum_quantum_number=0
epsilon = 1e-15
s = angular_momentum_quantum_number + 1
#For mpmath
# npp = 200  #points per wavelength
# k = np.sqrt(2*E)
# wavelength = 2*np.pi/k
# dr_plot = wavelength/npp


r_steps = 60000
r_END = 5
r2 = 1e-6
DRN = 0.0001



N=10 # steps in between r_a and r_b


methods = ["Coulomb" , "Spline"]

# label = f"Coulomb_E{E}_Z{Z}_l{angular_momentum_quantum_number}"

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

########################################################
### Setup the analyitcal solution given through mpmath
########################################################

def CoulombF_reduced_scalar(l,Z,E,r):
    k = mp.sqrt(2*E)
    eta = Z/k
    rho = k*r

    return mp.coulombf(l,eta,rho)

def Coulomb_funcs_and_derivs(l,Z,E,r):
    k = mp.sqrt(2*E)
    eta = Z/k
    rho = k*r

    F = mp.coulombf(l,eta,rho)
    G = mp.coulombg(l,eta,rho)

    dF_drho = mp.diff(lambda t: mp.coulombf(l,eta,t), rho)
    dG_drho = mp.diff(lambda t: mp.coulombg(l,eta,t), rho)

    F_prime = k * dF_drho
    G_prime = k * dG_drho

    return F, F_prime , G , G_prime

def CoulombF_reduced(l,Z,E,r_array):

    r_array = np.asarray(r_array, dtype= float)
    u_vals = np.empty_like(r_array, dtype =float)

    for i, r in enumerate (r_array):
        u_vals[i] =float(CoulombF_reduced_scalar(l,Z,E,float(r)))

    return u_vals





###################################
###Setup Cubic Spline potential V
###################################
N_V = 2000
V0 = -0.5
def potential(r):
    return Z + V0* np.exp(-r)

r_V = np.linspace(0.0,r_END, N_V)


V = potential(r_V)

V_spline = CubicSpline(r_V, V)

V_spline_prime = V_spline.derivative(1)
V_spline_second = V_spline.derivative(2)
V_spline_third = V_spline.derivative(3)

##################################################################################
### Function which returns potential parameters needed for the numerical solution
### As we use (r_a + hx)^2 P''(x) - h^2 U(x) P(x) = 0
### With U(x) = 2r^2 [V(r) - E] + l(l+1) = u0 + u1 x + u2 x^2 + u3 x^3 + u4 x^4
##################################################################################
def Potential_parameters(name, r_a ,r_b, l):

    if name == "Coulomb":
        v0 = Z
        v1 = v2= v3 = 0


    elif name == "Spline":

        v0 = float(V_spline(r_a))
        v1 = float(V_spline_prime(r_a))
        v2 = 0.5 * float(V_spline_second(r_a))
        v3 = (1.0/6.0) * float(V_spline_third(r_a))

    else:
        raise ValueError("No valid potential type was given, choose between 'Coulomb' or 'Spline' ")


    # print(f"v0 = {v0}")
    # print(f"v1 = {v1}")
    # print(f"v2 = {v2}")
    # print(f"v3 = {v3}")

    h = r_b-r_a

    u0 = l*(l+1) + 2*v0*r_a + 2*(v1 - E)*r_a**2 + 2*v2*r_a**3 + 2*v3*r_a**4
    u1 = 2*h * (v0 + 2*(v1 -E)*r_a + 3*v2*r_a**2 + 4*v3*r_a**3)
    u2 = 2*h**2 * ((v1-E) + 3*v2*r_a + 6*v3*r_a**2)
    u3 = 2*h**3 * (v2 + 4*v3*r_a)
    u4 = 2*v3*h**4

    return u0, u1, u2, u3, u4



######################################################################################################################################
### Function which returns an array of the individual series terms An for [0,r_b] and the terms P(r_b) and P'(r_b) wich are the initial
### conditions for the concurrent grid point [r_a,r_b].  This function uses P(x) = x^s Sum An x^n
######################################################################################################################################

def Singular_term_power_series(u_array,r_b, l ,eps):
    #Define the u terms
    u0 = u_array[0]
    u1 = u_array[1]
    u2 = u_array[2]
    u3 = u_array[3]
    u4 = u_array[4]




    #create the a array and define the intial conditions
    # a = []
    a = []

    #calc 1st terms by hand
    a0 = 1.0
    a1 = u1*a0 / ((s+1)*(l+1)-u0)
    a2 = (u1*a1 + u2*a0) / ((s+2) * (l+2) -u0)
    a3 = (u1*a2 + u2*a1 + u3*a0) / ((s+3) * (l+3) -u0)

    a.append(a0)
    a.append(a1)
    a.append(a2)
    a.append(a3)

    #Create the Sum terms to check for convergence critaria
    S = sum(a)  # x^s * Sum a_n from n= 0 to j
    Sn = a1 + 2*a2 + 3*a3  #Sum n* a_n from n=0 to j, therefor to start only need n=1 a_1 term
    Snn = 2*a2 + 6*a3   #Sum n * (n-1) * a_n

    n=3
    # print(f"singular loop for range [0, {r_b}]")

    while True:
        n +=1

        an = (u1 *a[n-1] + u2 * a[n-2] + u3 * a[n-3] + u4 * a[n-4])/((s+n) * (l+n) - u0)

        #append terms
        a.append(an)
        S += an
        Sn += n*an
        Snn += n*(n-1) * an


        tolerance = eps * max(abs(S), abs(s*S + Sn)/n)

        if abs(an) < tolerance:
            second_deriv_P = s*(s-1) * S + 2* s* Sn  + Snn
            residual = abs(r_b**2 *( second_deriv_P - (u0+u1+u2+u3+u4) * S))
            if residual < tolerance:
                # print(f"residual = {residual} < {tolerance}")
                # print(f"final number of iterations n= {n}")
                break


        if n > 99:
            raise RuntimeError(f"No convergence before n = {n}")



    arr = np.array(a)

    # arr /= S #Normalzing the waveunction to set P(r) from 0 to 1. This will also normalize the rest of the entire wave function for all r

    # P(r_2) = P(1) = sum An
    # P'(r_2) = P'(1)/r_2 = (s*sum(A_n) + sum(n*A_n) )/r_2
    return arr, S/abs(S), (s*S +Sn)/(r_b*abs(S))






######################################################################################################################################
### Function which returns an array of the individual series terms An from [r_a,r_b] and the terms P(r_b) and P'(r_b) wich are
### the initial conditions for the concurrent grid point [r_a,r_b].  This function uses P(x) = Sum An x^n
######################################################################################################################################


def Power_series_schrodinger(u_array, initial_condition0, initial_condition1 , r_a, r_b, l, eps):
    #Define the u terms
    u0 = u_array[0]
    u1 = u_array[1]
    u2 = u_array[2]
    u3 = u_array[3]
    u4 = u_array[4]

    h = r_b-r_a

    # print(f"h = {h}")

    #create the a array and define the intial conditions
    a = []


    a0 = initial_condition0
    a1 = h*initial_condition1

    d = h/r_a #division term to safe computing time
    a2 = d/2 * d*u0*a0
    a3 = d/6 * (-4*a2 + d * (u0*a1 + u1*a0))
    a4 = d/12 * (-12*a3 + d * ((u0 - 2)*a2 + u1*a1 + u2*a0))
    a5 = d/20 * (-24*a4 + d * ((u0 -6)*a3 + u1*a2 + u2*a1 + u3*a0))




    a.append(a0)
    a.append(a1)
    a.append(a2)
    a.append(a3)
    a.append(a4)
    a.append(a5)


    #Create the Sum terms to check for convergence critaria
    S = sum(a)  #Sum a_n from n= 0 to j
    Sn = a1 + 2*a2 + 3*a3 + 4*a4 + 5*a5   #Sum n* a_n from n=0 to j, therefor to start only need n=1 a_1 term
    Snn =  2*a2 + 6*a3 + 12*a4 + 20*a5   #Sum n*(n-1) a_n for n=0 to j



    n=5

    while True:
        n +=1


        #recuurance relation
        term1 = -2*(n-1) * (n-2) * a[n-1]
        term2 = (u0 - (n-2) * (n-3)) * a[n-2] + u1 * a[n-3] + u2 * a[n-4] + u3 * a[n-5] + u4 * a[n-6]

        an = d / (n*(n-1)) * (term1 + d * term2 )

        #append terms
        a.append(an)
        S += an
        Sn += n*an
        Snn += n*(n-1) * an



        tolerance1 = eps * max(abs(S),  abs(Sn)/n)
        # if h > 0.0061966:
        #     print(f"Condition 1: S = {abs(S)}, Sn/n = {abs(Sn)/n}")

        if abs(an) < tolerance1:

            second_deriv_P = Snn
            residual = abs(r_b**2 * second_deriv_P - h**2 *(u0+u1+u2+u3+u4) * S)

            tolerance2 = eps* max(abs(S) , abs(Sn))
            # if h > 0.0061966:
            #     print(f"Condition 2: S = {abs(S)}, Sn = {abs(Sn)}")
            if residual < tolerance2:

                break




        if n > 499:
            print(f"r_a = {r_a}, r_b = {r_b}, h= {h: .4} , d = {d: .4}")
            print(f"E , l , Z = {E} , {l}, {Z}")
            print(f"u0....u4 = {u0}, {u1}, {u2} , {u3} , {u4}")
            raise RuntimeError(f"No convergence before n = {n}")

    # print("")
    arr = np.array(a)


    return arr, S, Sn/h

#########################################################################################
### Function to create a mesh grid which scales logarithmically neer r = 0
### and continuously at larger r where we expect wave function to act as a free wave
#########################################################################################
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

#### this below function can increase speed (not sure if needed) but we now always sample 10 terms between r_a and r_b while this function would scale the number of
# Sampples needed witht the distance between r_a and r_b
# def number_of_steps(r_a,r_b, Energy):
#     k = np.sqrt(2*Energy)
#     wavelength = 2*np.pi/(k)
#     dr = wavelength/npp
#
#     h = r_b -r_a
#     N_min = N
#
#     Nt_block = max(N_min, int(np.ceil(h/dr)))
#
#     return Nt_block

############################################################################################################################################
###Function which inputs a range from [0,r] and the number of steps the user wants to sub-divide the grid into.
### After this the set of grid points [0,r_b1] and [r_ai,r_bi] for (i = 2,3,4....) are passed onto their respective functions
### to obtain the series terms and the initial conditions for the next grid step. The obtained series terms are stiched into one long array
### The fucntion outputs this stichted array of all series terms and the r points used.
###########################################################################################################################################

def calc_series_terms(r,N_steps,l, potential: int):

    if (potential != 0) and (potential != 1):
        raise RuntimeError(f"No valid potential function was choosen")


    # mesh_steps = np.linspace(0,r, N_steps+1)
    mesh_steps = mesh_grid(r, N_steps , r2 , DRN)


    # print(mesh_steps)
    u_parameters0 = Potential_parameters(methods[potential], mesh_steps[0], mesh_steps[1], l)

    Series_terms0 , initial_condition0_next_term, initial_condition1_next_term = Singular_term_power_series(u_parameters0, mesh_steps[1],l,epsilon)

    initial_temp_condition0 = initial_condition0_next_term
    initial_temp_condition1 = initial_condition1_next_term


    Series_terms_list = []
    Series_terms_list.append(Series_terms0)

    for i in tqdm(range(1, len(mesh_steps)-1), desc="Calculating Series Terms"):
        # print(i)
        # print(f"r range from {mesh_steps[i]} to {mesh_steps[i+1]}")
        # print(f"Using intial conditions a0 = {initial_temp_condition0}, a1 = {initial_temp_condition1}")

        u_temp_parameter = Potential_parameters(methods[potential], mesh_steps[i], mesh_steps[i+1],l)



        Temp_series_terms, initial_temp_condition0, initial_temp_condition1 = Power_series_schrodinger(u_temp_parameter, initial_temp_condition0, initial_temp_condition1, mesh_steps[i],mesh_steps[i+1], l, epsilon)

        Series_terms_list.append(Temp_series_terms)

    # Series_terms_arr = np.concatenate(Series_terms_list)

    return Series_terms_list, mesh_steps




def Normalization_constant(method,distance, list_of_sequence_terms, grid_points, unnormalized_wave_func = np.empty(1), r_function =np.empty(1)):
    i = 0

    if method == "Spline":
        r_range = np.linspace(0,distance,100*distance)

        V = potential(r_range)
        Z_infty = potential(10000)

        while abs( V[i] - Z_infty) >= epsilon:
            i+=1

            if i > len(r_range)-2:
                raise RuntimeError(f"No convergence, increase distance to obtain assympotic behavior. last result found: {abs( V[i-1] - Z_infty)} < {epsilon}")
        r_c = r_range[i]
        print(f"Converged at i = {i}, with {abs( V[i-1] - Z_infty)} < {epsilon}")
        print(f"r_c = {r_c}")
    elif method == "Coulomb":
        fraction = 0.95
        last_quartile = int(fraction*len(unnormalized_wave_func))

        idx = np.argmax(abs(unnormalized_wave_func[last_quartile:]))

        idx_rc = last_quartile + idx
        r_c = r_function[idx_rc]
    else:
        raise RuntimeError("No valid potential method has been chosen")



    edge_mesh_idx = 0
    for k in range(len(grid_points)-1):
        if (r_c >= grid_points[k]) & (r_c <= grid_points[k+1]):
            # print(f" r_c = {r_c} and {grid_points[k]} <= {r_c} <= {grid_points[k+1]}")
            edge_mesh_idx = k
            break



    series_terms = np.array(list_of_sequence_terms[edge_mesh_idx])
    P = 0
    P_prime = 0

    x_c = (r_c - grid_points[edge_mesh_idx]) / (grid_points[edge_mesh_idx+1] - grid_points[edge_mesh_idx])

    for j in range(len(series_terms)-1):
        P += series_terms[j] * x_c**j
        P_prime += (j+1) * series_terms[j+1] *  x_c**j  ## j+1 to avoid x^j-1 term at j=0

    P_prime /= (grid_points[1] - grid_points[0]) # P(x) = P(r) but P´(x) = dp/dx = dr/dx dp/dr = h * P'(r)

    F_rc , F_prime_rc, G_rc, G_prime_rc = Coulomb_funcs_and_derivs(angular_momentum_quantum_number,Z,E,r_c)

    delta = mp.atan( ((P_prime * F_rc) - (P * F_prime_rc)) / ( (P * G_prime_rc) - (P_prime * G_rc)) )
    delta = float(delta)

    if abs(P) < epsilon:
        A = (mp.cos(delta) * F_prime_rc  + mp.sin(delta) * G_prime_rc) / (P_prime)
    else:
        A = (mp.cos(delta) * F_rc + mp.sin(delta) * G_rc) / P


    A = float(A)
    # print(f"Normalization constant A = {A}")
    # print(f"phase shift delta = {delta}")



    return A , delta








#################################################################################################################################################
### This function takes as input the stiched array of series terms from [0,r] and the subdivided grid points from function: "calc_series_terms"
### It then uses the obtained series terms and grid points to sitch toghether the respective wave functions where the intial wave function
### from point [0,r_b] is given by P(x) = x^s Sum An x^n and all series terms and grid points after this use P(x) = Sum An x^n.
### note in both case x = r-r_a / (r_b -r_a)
#################################################################################################################################################

def radial_wave_function(list_of_sequence_terms, grid_points,Nt):


    singular_terms = np.array(list_of_sequence_terms[0])


    r_singular_range = np.linspace(grid_points[0],grid_points[1], Nt) # P(r) from [0, r_b]
    x_singular = r_singular_range/grid_points[1]

    singular_wave_function= np.zeros(len(r_singular_range), dtype = float)
    # print(f"Calculting P(r) for [{grid_points[0]} , {grid_points[1]}]]")
    for i in range(len(singular_terms)):
        singular_wave_function += singular_terms[i] * (x_singular)**i

    singular_wave_function *= (x_singular)**s

    wave_function = singular_wave_function
    r_total_range = r_singular_range





    for i in tqdm(range(1,len(grid_points)-1), desc="Calculating wave function"):
        # print(f"Calculting P(r) for [{grid_points[i]} , {grid_points[i+1]}]]")

        r_regular_range = np.linspace(grid_points[i], grid_points[i+1],Nt) #from [r[i], r[i+1]]
        x_regular_range = (r_regular_range - grid_points[i])/(grid_points[i+1] - grid_points[i])

        temp_terms = np.array(list_of_sequence_terms[i])

        temp_wave_func = np.zeros(len(r_regular_range), dtype = float)

        for j in range(len(temp_terms)):
            temp_wave_func += temp_terms[j] * (x_regular_range)**j



        wave_function = np.concatenate((wave_function, temp_wave_func[1:]))
        r_total_range = np.concatenate((r_total_range, r_regular_range[1:]))






    return wave_function , r_total_range







#######################
###### Plotting
######################

def Visualize(name, r_start = 0 , r_end = r_END):
#TODO Rewrite this as 1 function not 2 if statemnts. Just make everything in terms of "mesh_points_" and just add the name at the endd by string additions
    plt.figure(figsize=(12,8))
    if name == "Coulomb":


        series_terms_coulomb, mesh_points_coulomb = calc_series_terms(r_end,r_steps,angular_momentum_quantum_number,0)
        wave_function_coulomb, r_coulomb = radial_wave_function(series_terms_coulomb,mesh_points_coulomb, N)
        A , delta = Normalization_constant(name, r_end, series_terms_coulomb, mesh_points_coulomb, wave_function_coulomb, r_coulomb)



        r = np.linspace(0,r_end,40000)
        analytical_solution = CoulombF_reduced(angular_momentum_quantum_number,Z,E,r)

        plt.plot(r_coulomb, wave_function_coulomb*A, label = "P(r) - Normalized")
        plt.plot(r,analytical_solution, ls = "--" , label = "Analytical mpmath sol")

    elif name == "Spline" :
        series_terms_spline , mesh_points_spline = calc_series_terms(r_END,r_steps,angular_momentum_quantum_number, 1)
        Wave_function_spline , r_spline = radial_wave_function(series_terms_spline, mesh_points_spline, N)
        A , delta = Normalization_constant(name, r_end, series_terms_spline, mesh_points_spline)




        #plt.plot(r_spline, Wave_function_spline/max(Wave_function_spline), label = "P(r) - Cubic Spline potential")
        plt.plot(r_spline, Wave_function_spline*A, label = "P(r) - Normalized")
        # plt.plot(r_tail, P_analytic, ls = "--" ,  label = "P(r) - analytical")



    plt.xlabel("r in a.u")
    plt.ylabel("P(r)")
    plt.grid(True)
    plt.title(f"Normalized wave function E = {E}, Z = {Z}, Potential = '{name}' , A = {A: .5}, delta = {delta: .5}'")
    plt.legend()
    plt.show()






#
Visualize(methods[1])






###################################
##### Fermi Function data collection
##################################

# E_hatree = 27.211386 #eV
# m_ELECTRON = 0.51099895069e6/E_hatree #eV
# E_max = 1.2933325099999138e6/E_hatree #eV
# size = 101
# E = np.linspace(m_ELECTRON,E_max, size)
# print(f"E = {E}")
# print(int(len(E)))
#
# for i, E in enumerate (E):
#
#     print(f"Finding wave function for E = {E}, iter ={i}")
#     label = f"Coulomb_E{E}_Z{Z}_l{angular_momentum_quantum_number}"
#
#     series_terms_coulomb, mesh_points_coulomb = calc_series_terms(r_END,r_steps,angular_momentum_quantum_number,0)
#
#     wave_function_coulomb, r_coulomb = radial_wave_function(series_terms_coulomb,mesh_points_coulomb, N)
#
#     A , delta = Normalization_constant("Coulomb", r_END, series_terms_coulomb, mesh_points_coulomb, wave_function_coulomb, r_coulomb)
#
#     if (i==0) or (i == size-1):
#         print(f"Plotting wave function for i = {i} and E = {E} to ensure normalization of the wave function was done properly throughout")
#         r_analytical = np.linspace(0,r_END, 10000)
#         analytical_solution = CoulombF_reduced(angular_momentum_quantum_number,Z,E,r_analytical)
#
#         plt.figure(figsize=(12,8))
#
#         plt.plot(r_coulomb, wave_function_coulomb*A, label = "P(r) - Normalized")
#         plt.plot(r_analytical,analytical_solution, ls = "--" , label = "Analytical mpmath sol")
#         plt.xlabel("r")
#         plt.ylabel("P(r)")
#         plt.tight_layout()
#         plt.grid(True)
#         plt.title(f"Normalized wave function E = {E}, Z = {Z}, Potential = '{methods[0]}'")
#         plt.legend()
#         plt.show()
#
#     save_run_npz("Coulomb_Wavefunctions_beta_decay.npz", label, r_coulomb, A*wave_function_coulomb)
#




##################
### SINGULAR PLOT
#################
#u_parameters =  Potential_parameters(methods[0], r_START, r_END, angular_momentum_quantum_number)
# series_terms, A,B = Singular_term_power_series(u_parameters, r_END, angular_momentum_quantum_number, epsilon)
#
# # print(f"An = {series_terms}")
# print(f"a0 = {A}")
# print(f"a1 = {B}")
#
#
# # Finding series solution P(r) = sum a_n ((r-ra)/h)^n
# r = np.linspace(r_START,r_END, 1000)
# #
# summation_of_terms = np.zeros(len(r), dtype= float)
# for i in range(len(series_terms)):
#
#     summation_of_terms += series_terms[i] * (r/r_END)**i
#
# solution_r = (r/r_END)**s * summation_of_terms
#
# analytical_solution = CoulombF_reduced(angular_momentum_quantum_number,Z,E,r)
# analytical_solution = analytical_solution
#
# alpha = analytical_solution[-1]/solution_r[-1]
# analytical_solution_scaled = analytical_solution/alpha


#################
### Regular plot
################

# a0= 0.9140058599831451
# a1= 1.5026949026035312
#
# series_terms, A , B = Power_series_schrodinger(u_parameters,a0,a1,r_START, r_END, angular_momentum_quantum_number,epsilon)
#
#
# r = np.linspace(r_START,r_END, 1000)
# #
# summation_of_terms = 0
# for i in range(len(series_terms)):
#     summation_of_terms += series_terms[i] * (r-r_START/r_END - r_START)**i
#
#
# solution_r = summation_of_terms
#
#
# analytical_solution = CoulombF_reduced(angular_momentum_quantum_number,Z,E,r)
#
#
# alpha = analytical_solution[-1]/solution_r[-1]
# analytical_solution_scaled = analytical_solution/alpha




#
# plt.figure(figsize=(12,8))
# plt.plot(r,r**(angular_momentum_quantum_number+1)/r_END , ls= '--' ,  label = "P(r) near 0")
#
# plt.plot(r,solution_r, label = "P(r)")
# plt.plot(r,analytical_solution_scaled, ls = "--" , label = "Analytical mpmath sol")
# plt.xlabel("r- points")
# plt.ylabel("P(r)")
# plt.title("Normalized wave function")
# plt.legend()
# plt.show()
# #
