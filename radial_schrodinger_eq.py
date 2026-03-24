#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


N_steps = 100
radius = 5
phase_shift = 0

eta = 0.5
Energy = 1

epsilon = 0.000001

radius = np.linspace(0,radius,N_steps+1)
radius[0] = epsilon


def analyticall(r,k,eta,d):
    return np.sin(k*r - eta * np.log(k*r) + d)


def visualize(r):
    fine_radius = np.linspace(r[0],r[-1],1001)
    u_e  = analyticall(fine_radius, Energy , eta, phase_shift)

    plt.figure(figsize=(12,8))
    plt.plot(fine_radius,u_e, "b-",  label = "analyticall solution")

    plt.xlabel("r")
    plt.ylabel("P(r)")
    plt.legend()
    plt.show()

visualize(radius)
