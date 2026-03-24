#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d

Nt = 30
Z = 2


def function(x):
    return -Z/x





x1 = np.linspace(0.1,1,5)
x2 = np.arange(1,10)
x = np.concatenate((x1[:-1],x2))

y = function(x)

cs = CubicSpline(x,y)

prime = cs.derivative(1)
print(prime)
print(prime(0.1))
xs = np.linspace(0.1,10,1000)

# interpolated_function, time_points = linear_spline(y,x)



plt.figure(figsize=(12,8))

# plt.plot(time_points, interpolated_function, ".", label = "spline")

plt.plot(xs, function(xs), label = "function")
plt.plot(xs,cs(xs), label = "S")
plt.plot(xs, cs(xs,1) , label = "S'")
# plt.plot(xs,cs(xs, 2), label = "S''")
# plt.plot(xs,cs(xs, 3), label = "S'''")

plt.plot(x, y, 'o', label = "data points")
# plt.plot(x, y, 'red',  lw= 1, label = "spline via matplotlib")
plt.xlabel("time points")
plt.ylabel("y points")
plt.legend()
plt.show()







# def linear_spline(y, time):
#     grid_size= 20
#     quadrant = 0
#     interpolated_y = np.zeros((grid_size )* (len(y)-1))
#     stiched_time = np.zeros((grid_size) * (len(y)-1))
#
#     for i in range(len(y)-1):
#
#         temporary_time_grid = np.linspace(time[i],time[i+1],grid_size)
#
#         interpolated_y[quadrant] = y[i]
#         interpolated_y[quadrant+grid_size-1] = y[i+1]
#         stiched_time[quadrant] = temporary_time_grid[0]
#         stiched_time[quadrant+grid_size -1] = time[-1]
#
#         interpolated_y[quadrant: quadrant + grid_size] = y[i] + (y[i+1] - y[i])/(time[i+1] - time[i]) * (temporary_time_grid[:] - time[i])
#         stiched_time[quadrant: quadrant+grid_size] = temporary_time_grid[:]
#
#         quadrant += grid_size
#
#
#
#     return interpolated_y, stiched_time










