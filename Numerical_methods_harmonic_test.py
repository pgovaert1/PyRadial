#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--I", type = float, default = 1, help = "Initial position")
parser.add_argument("--w", type= float, default = 2.0, help = "omega")
parser.add_argument("--T", type = int, default = 5, help = "time of osciallation in integer unit")
parser.add_argument("--num_periods" , type = int, default = 10, help= "Number of discrete time steps")

# parser.add_argument("--Terms", type = int, default = 1, help = "number of terms for the power series")

args = parser.parse_args()

initial_pos, omega, T , Nt = args.I, args.w, args.T , args.num_periods




time = np.linspace(0,T, Nt+1)
time_step = time[0]-time[1]


def exact(x0, time):
    return x0 * np.cos(omega * time)

def numerical(x0, t,dt):

    u = np.zeros(len(t))

    u[0] = x0 #set intial posistion
    u[1] = u[0] + 0.5 * (dt * omega)**2 * u[0] #find 1st function steps

    for i in range(1,Nt):
        u[i+1] = 2 * u[i] - u[i-1] - (dt * omega)**2 * u[i]


    return u



def power_series(x0,T,omega,tol = 1e-40):
    solution = 0
    dt = min(1/abs(omega), T/10, 0.25)  #chooses a small time step
    print(f"dt = {dt}")
    m = int(T/dt) # number of steps
    print(f"number of iterations = {m}")

    Omega_squared= omega**2
    while True:

        t = np.linspace(0,T,m+1)
        print(t)
        print(f"size t = {len(t)}")
        a = np.zeros(m+3) #extra 3 terms cuz we are working with a[i+2]
        print(f"size a = {len(a)}")
        a[0] = x0
        a[1] = 0

        for i in range(0,m+1, 2):
            print(i)
            a[i+2] = -Omega_squared * a[i] / ((i+2) * (i+1))

        tail_remainder = np.abs(a[-1] * t[-1] /(1 - t[-3]))
        print(f"caluclated tail remaineder = {tail_remainder}")
        if tail_remainder < tol:
            print("Tail is below the tol")
            break
        print("Tail is larger than tol and time step is halved")
        dt = 0.5 * dt
        m = int(T/dt)


    t_power = np.ones(len(t))
    for i in range(0,m+1,2):
        print(f"iter = {i}")
        if i>0:
            t_power *= t*t
        solution += a[i] * t_power

    return solution, t




def visualize(u,t,x0,omega, numerial_type):

    plt.figure()

    t_fine = np.linspace(0,t[-1], 1001) #creates a very fine mesh
    u_e = exact(x0, t_fine)
    plt.plot(t_fine, u_e, 'b-', label = "exact")

    if numerial_type == 0:
        plt.plot(t, u, 'r--o', label = "Forward euler")

    elif numerial_type == 1:
        plt.plot(t, u, 'r--o', label = "power series")

    plt.legend()
    plt.xlabel('t')
    plt.ylabel('f(t)')
    dt = t[1]-t[0]
    plt.title(f'dt = {dt:.2f}, w = {omega}')
    plt.show()


#Forward_euler = numerical(initial_pos, time , time_step)
Series, time_series = power_series(initial_pos,T,omega)

#visualize(Forward_euler,time,initial_pos, 0)
visualize(Series, time_series, initial_pos, omega,  1)


