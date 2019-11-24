from numba import njit
from numpy import dot

from examples.gpr.misc import mg
from examples.gpr.misc.eos import E_2A, E_2J, E_3
from examples.gpr.misc.functions import dev, gram
from examples.gpr.params import cs2, cα2


@njit
def dEdA(A):
    G = gram(A)
    return cs2 * dot(A, dev(G))


@njit
def dEdJ(J):
    return cα2 * J


@njit
def pressure(ρ, E, v, A, J):

    E1 = E - E_3(v) - E_2A(A) - E_2J(J)
    return mg.pressure(ρ, E1)


@njit
def temperature(ρ, p):

    return mg.temperature(ρ, p)


@njit
def heat_flux(T, J):

    H = dEdJ(J)
    return H * T


@njit
def sigma(ρ, A):
    """ Returns the symmetric viscous shear stress tensor
    """
    ψ = dEdA(A)
    return -ρ * dot(A.T, ψ)
