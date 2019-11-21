from numba import njit
from numpy import dot, eye, sqrt


@njit
def sigma_norm(σ):
    """ Returns the norm defined in Boscheri et al
    """
    tmp1 = (σ[0, 0] - σ[1, 1])**2 + \
           (σ[1, 1] - σ[2, 2])**2 + \
           (σ[2, 2] - σ[0, 0])**2

    tmp2 = σ[0, 1]**2 + σ[1, 2]**2 + σ[2, 0]**2

    return sqrt(0.5 * tmp1 + 3 * tmp2)


@njit
def dev(G):
    """ Returns the deviator of G
    """
    return G - (G[0, 0] + G[1, 1] + G[2, 2]) / 3 * eye(3)


@njit
def gram(A):
    """ Returns the Gram matrix for A
    """
    return dot(A.T, A)
