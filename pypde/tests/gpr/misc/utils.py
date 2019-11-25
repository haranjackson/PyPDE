from pypde.tests.gpr.misc.eos import total_energy
from numba import njit
from numpy import eye, zeros


@njit
def get_variables(Q):

    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2:5] / ρ
    A = Q[5:14].copy().reshape((3, 3))
    J = Q[14:17] / ρ

    return ρ, E, v, A, J


def Cvec(ρ, p, v, A, J):
    """ Returns the vector of conserved variables, given the primitive variables
    """
    Q = zeros(17)
    Q[0] = ρ
    Q[1] = ρ * total_energy(ρ, p, v, A, J)
    Q[2:5] = ρ * v
    Q[5:14] = A.ravel()
    Q[14:17] = ρ * J
    return Q


def riemann_IC(n, ρL, pL, vL, ρR, pR, vR):
    """ Constructs the riemann problem corresponding to the parameters given
    """
    AL = ρL**(1 / 3) * eye(3)
    JL = zeros(3)
    AR = ρR**(1 / 3) * eye(3)
    JR = zeros(3)

    QL = Cvec(ρL, pL, vL, AL, JL)
    QR = Cvec(ρR, pR, vR, AR, JR)

    u = zeros([n, 17])

    for i in range(n):

        if i / n < 0.5:
            u[i] = QL
        else:
            u[i] = QR

    return u
