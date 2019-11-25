from numba import njit
from numpy import inf
from numpy.linalg import det

from pypde.tests.gpr.misc.functions import sigma_norm
from pypde.tests.gpr.misc.state import sigma
from pypde.tests.gpr.params import PLASTIC, T0, cs2, cα2, n, ρ0, σY, τ1, τ2


@njit
def theta1inv(ρ, A):
    """ Returns 1/θ1
    """
    if τ1 == inf:
        return 0

    # if PLASTIC:
    #     σ = sigma(ρ, A)
    #     sn = sigma_norm(σ)
    #     sn = min(sn, 1e8)  # Hacky fix
    #     return 3 * det(A)**(5 / 3) / (cs2 * τ1) * (sn / σY)**n

    return 3 * det(A)**(5 / 3) / (cs2 * τ1)


@njit
def theta2inv(ρ, T):
    """ Returns 1/θ2
    """
    return 1 / (cα2 * τ2 * (ρ / ρ0) * (T0 / T))
