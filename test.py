#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import mpmath as mp


R_au = 1.5
Z_sing = 1

def potential(r):
    r_arr = np.asarray(r,dtype = float)
    V = Z_sing/ r_arr
    inside = r_arr < R_au
    V = np.where(inside, Z_sing/(2*R_au) * (3 - (r_arr/R_au)**2), V)

    if np.isscalar(r):
        return float(V)
    return V


def smooth(r):
    return Z_sing/ r


r = np.linspace(0,5,1000)

plt.figure()
plt.plot(r,potential(r), label = "piecewise")
plt.plot(r, smooth(r), ls="-", label = "smooth")
plt.axvline(x = R_au, linestyle = "--", color = "grey")
plt.legend()
plt.show()
