import numpy as np
from numba import njit
from numpy import dot

from pypde.tests.gpr.misc import mg
from pypde.tests.gpr.misc.functions import dev, gram
from pypde.tests.gpr.params import cs2, cα2


@njit
def E_1(ρ, p):
    """ Returns the microscale energy
    """
    return mg.internal_energy(ρ, p)


@njit
def E_2A(A):
    """ Returns the mesoscale energy dependent on the distortion
    """
    G = gram(A)
    devG = dev(G)
    return cs2 / 4 * np.sum(devG * devG)


@njit
def E_2J(J):
    """ Returns the mesoscale energy dependent on the thermal impulse
    """
    return cα2 / 2 * dot(J, J)


@njit
def E_3(v):
    """ Returns the macroscale kinetic energy
    """
    return dot(v, v) / 2


@njit
def total_energy(ρ, p, v, A, J):
    """ Returns the total energy
    """

    return E_1(ρ, p) + E_2A(A) + E_2J(J) + E_3(v)
