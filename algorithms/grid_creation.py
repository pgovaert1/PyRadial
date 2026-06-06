#!/usr/bin/env python

import numpy as np
from tqdm import tqdm
from configurations.physics_constants import EPSILON

#####################################################################################################################
### Find_X and Mesh_Grid solve a system of equations given in section 8.4 of RADIAL to create a continues mesh which
### is very fine in the beginning and becomes coarser and linear as it moves further out
#####################################################################################################################
def find_x(A_grid, x_min = 1e-10):
    """
    Find the positive root of eq 8.10 from 'RADIAL: a Fortran subroutine by Francesc Salvat'
    using a bisection based root finding method

    Parameters
    ----------
    A_grid : float
        Quantity defined in eq 8.8 of 'RADIAL: a Fortran subroutine by Francesc Salvat'
    x_min : float
        Initial mminimum x value for root finding interval. Default is 1e-10

    Returns
    -------
    float
        Positve root of f(x)
    """

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


def radial_grid(r_END, A_grid, N, r2, DRN, nuc_radius):
    """
    Construct 1d radial grid with high resolution near the nuclear radius.

    The grid spacing increases logarithmically near the origin and gradually approaches
    a linear grid spacing with upper resolution boundary set by DRN.

    Parameters
    ----------
    r_END : float
        Grid end point.
    A_grid : float
        Quantity defined in eq 8.8 of 'RADIAL: a Fortran subroutine by Francesc Salvat'.
    N : int
        Number of grid points.
    r2 : float
        First point after zero.
    DRN : float
        Upper limit in resolution.
    nuc_radius :
        Nuclear radius in Atomic units, defined by R = 1.2e-15 * A^(1/3) / AU.

    Returns
    -------
    r : ndarray
        one dimensional radial grid
    """

    if not (0.5 < A_grid < 1.0):
        raise ValueError(f"A_grid = {A_grid: .6g}, not in required range (0.5,1). Adjust r2 DRN or 'resolution'" )

    x = find_x(A_grid)

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


    for i in range(len(r)-1):
        if r[i] < nuc_radius and r[i+1] > nuc_radius:
            r = np.insert(r,i+1,nuc_radius)
            break
        elif r[i+1] > nuc_radius:
            break

    return r




###################################################################################
### Find mminimum range up to which to extend the mesh range used to solve P and Q
###################################################################################

def obtain_mesh_range(k, Z, R_au, potential_index, phi_r, potential_function, verbose=False):
    """
    Find the minimum range needed such that a matching radius rc is included in the radial grid.
    Extending the radial point r_max until asymptotic condition |rV(r) - Z| < EPSILON is satisfied
    and k * r_max > kr_min.

    Parameters
    ----------
    k : float
        Relativistic wave number.
    Z : int
        Atomic number.
    R_au : float
        Nuclear radius in Atomic units, defined by R = 1.2e-15 * A^(1/3) / AU.
    potential_index: int
        Index specifying which potential model to use.
    phi_r : callable
        Function phi(r) used in the Thomas-Fermi screening potential.
    potential_function : callable
        Function returning the potential corresponding to potential_index

    Returns
    -------
    r_max : float
        Maximum radial coordinate used for constructing the radial grid
    """
    kr_min = 20.0
    r_max = 0.01

    RV = r_max* potential_function(r_max, Z, R_au, potential_index, phi_r)
    r_infty = 1e5
    Z_inf = r_infty * potential_function(r_infty, Z, R_au, potential_index, phi_r)


    if potential_index == 3:
        TAS = 10e-4 * abs(Z_inf)
    else:
        TAS = max(1e-11, EPSILON) * abs(Z_inf)

    while abs(RV - Z_inf) >= TAS or (k * r_max <= kr_min):
        r_max *= 2
        RV = r_max* potential_function(r_max, Z, R_au, potential_index, phi_r)

    if verbose:
        print(f"mesh range set to {r_max}")

    return r_max



#################################################################################################################################################
### Find_rc is a function finding the matching radius rc at which one can normalize the power series by matching it witht he asymptotic behavior
#################################################################################################################################################

def find_rc(r_grid, k, Z , R_au, potential_index, phi_r, potential_function, verbose=False):
    """
    Obtains the ndarray index where the value of r_grid satisfies |rV(r) - Z| < EPSILON
    and k*r_grid > k_min.

    Parameters
    ----------
    r_grid : ndarray
        One dimensional radial grid
    k : float
        Relativistic wave number.
    Z : int
        Atomic number.
    R_au : float
        Nuclear radius in Atomic units, defined by R = 1.2e-15 * A^(1/3) / AU.
    potential_index: int
        Index specifying which potential model to use.
    phi_r : callable
        Function phi(r) used in the Thomas-Fermi screening potential.
    potential_function : callable
        Function returning the potential corresponding to potential_index

    Returns:
    idx_rc : int
        Index value at which r_grid satisfied matching radius criteria
    --------


    """
    kr_min = 20.0
    r_grid = np.asarray(r_grid, dtype = float)


    r_safe = np.maximum(r_grid, 1e-16)
    RV = r_safe * potential_function(r_safe, Z, R_au, potential_index, phi_r)

    r_infty = 10000
    Z_inf = r_infty * potential_function(r_infty, Z, R_au, potential_index, phi_r)

    if potential_index == 3:
        TAS = 10e-4 * abs(Z_inf)
    else:
        TAS = max(1e-11, EPSILON) * abs(Z_inf)

    idx_rc = None
    for idx in range(len(r_grid)-1 ,2, -1):
        if abs(RV[idx] - Z_inf) <= TAS and (k * r_grid[idx] >= kr_min):
            idx_rc = idx
        else:
            break


    if idx_rc is None:
        raise RuntimeError("No rc found, extend distance")
    if verbose:
        print(f"rc succesfully found at rc = {r_grid[idx_rc]}")
    return idx_rc
