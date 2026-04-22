import numpy as np
from scipy.special import spence

# effectively the Sirlin function from https://arxiv.org/pdf/2603.23592
me = 0.511
Q = 2.45783
E0 = Q + 2 * me


def dilog(x):
    """Dilogarithm Li_2(x), using scipy.special.spence."""
    return spence(1 - x)


def f(beta):
    """The f function from equation 16."""
    return dilog(2 * beta / (1 + beta)) / beta + np.log((1 + beta) / (1 - beta)) ** 2 / (4 * beta)


def l(beta):
    """The l function from equation 16."""
    return np.log((1 + beta) / (1 - beta)) / beta


def f3(beta12):
    """The f3 function from equation 17."""
    x12 = (beta12 - 1) / (beta12 + 1)
    return (
        -2 * dilog(-x12)
        - 2 * np.log(-x12) * np.log(1 + x12)
        + 0.5 * np.log(-x12) ** 2
        - 2 * np.pi**2 / 3
    ) / beta12


def K(z, zp, zm):
    """
    The K function from equation 19.
    """
    term1 = 0.5 * np.log(((z - zm) * (zp - z)) / ((zp + z) * (zm + z))) ** 2
    term2 = 2 * dilog(2 * zm * (zp - z) / ((zp - zm) * (zm + z)))
    term3 = 2 * dilog(-2 * zp * (zm + z) / ((zp - zm) * (zp - z)))

    return -term1 - term2 - term3


def ieps(beta1, beta2, y12):
    """
    The ieps function from equation 18.
    """
    a = beta1**2 + beta2**2 - 2 * beta1 * beta2 * y12
    b = (beta1**2 * beta2**2 - (beta1 * beta2 * y12) ** 2) / a
    c = np.sqrt(b / (4 * a))

    zp = (1 + np.sqrt(1 - b)) / np.sqrt(b)
    zm = (1 - np.sqrt(1 - b)) / np.sqrt(b)

    x1 = (beta1**2 - beta1 * beta2 * y12) / a
    x2 = (beta2**2 - beta1 * beta2 * y12) / a

    z1 = (np.sqrt(x1**2 + 4 * c**2) - x1) / (2 * c)
    z2 = (np.sqrt(x2**2 + 4 * c**2) + x2) / (2 * c)

    return (K(z2, zp, zm) - K(z1, zp, zm)) * (1 - beta1 * beta2 * y12)/ np.sqrt(a * (1 - b))


def double_sirlin_function(Ee1, Ee2, y12, mu):
    """
    Calculate the double Sirlin function for given electron energies and their relative angle.
    This is the expression given (omega_n/epsilon_K,L)^2 is a large number which is true in most instances
    To include this expression, multiply the G0 phase space term of with (1+alpha/2pi * double_sirlin_function)
    """
    l_mu = np.log(mu**2 / me**2)

    beta1 = np.sqrt(Ee1**2 - me**2) / Ee1
    beta2 = np.sqrt(Ee2**2 - me**2) / Ee2

    s = 2 * me**2 + 2 * Ee1 * Ee2 * (1 - beta1 * beta2 * y12)
    beta12 = np.sqrt(1 - 4 * me**2 / s)

    Eloss = E0 - Ee1 - Ee2          # energy "lost" to neutrinos
    A0    = 1 - beta1 * beta2 * y12 # tree-level angular dependence

    part1 = (
        3 * l_mu
        - 8 * (f(beta1) + f(beta2))
        + 3 * (l(beta1) + l(beta2))
        + 2 * (s - 2 * me**2) / s * f3(beta12)
        + ieps(beta1, beta2, y12)
    )

    part2 = (
        6 - 2 * (l(beta1) + l(beta2))
        + 2 * (1 - 2 * me**2 / s) * l(beta12)
    ) * (np.log(me**2 / (4 * Eloss**2)) + 137 / 30)

    part3 = 1/A0 * (
        -2 * me**2 * (
            beta12**2 * l(beta12) / (Ee1 * Ee2)
            + l(beta1) / Ee1**2 + l(beta2) / Ee2**2
        )
    )

    part4 = (
        Eloss / (6 * A0) * (
            l(beta1) / Ee1 * (2 + me**2 / (Ee1 * Ee2) - (2 + me**2 / Ee1**2) * beta2 / beta1 * y12)
            + l(beta2) / Ee2 * (2 + me**2 / (Ee1 * Ee2)- (2 + me**2 / Ee2**2) * beta1 / beta2 * y12)
            - 6 * (1 / Ee1 + 1 / Ee2 - y12 * (beta1 / (Ee2 * beta2) + beta2 / (Ee1 * beta1)))
        )
    )

    part5 = (
        Eloss**2 / (42 * A0) * (
            (beta1 - y12 * beta2) * l(beta1) / (Ee1**2 * beta1)
            + (beta2 - y12 * beta1) * l(beta2) / (Ee2**2 * beta2) - 4 / (Ee1 * Ee2)
            + 2 * y12 / (beta1 * beta2) * (1 / Ee1**2 + 1 / Ee2**2 - 2 * me**2 / (Ee1**2 * Ee2**2))
        )
    )

    return part1 + part2 + part3 + part4 + part5

print(double_sirlin_function(.62171,1.252521,.6123631,0.511))