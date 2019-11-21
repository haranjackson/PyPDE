import matplotlib.pyplot as plt
from numpy import array, eye, linspace, sqrt, zeros
from scipy.special import erf

from pypde import ader_solver
from testing.gpr.misc.eos import total_energy
from testing.gpr.params import γ, μ
from testing.gpr.system import B_gpr, F_gpr, S_gpr


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


def first_stokes_problem_IC(n=100, v0=0.1):

    tf = 1

    ρL = 1
    pL = 1 / γ
    vL = array([0, -v0, 0])

    ρR = 1
    pR = 1 / γ
    vR = array([0, v0, 0])

    L = [1]

    u = riemann_IC(n, ρL, pL, vL, ρR, pR, vR)

    ret = ader_solver(u,
                      tf,
                      L,
                      F=F_gpr,
                      B=B_gpr,
                      S=S_gpr,
                      STIFF=False,
                      CFL=0.6)

    # plt.plot(ret[-1, :, 3])
    # plt.show()

    return ret


def first_stokes_problem_exact(n=100, v0=0.1, t=1):
    """ Returns the exact solution of the y-velocity in the x-axis for Stokes'
        First Problem
    """
    dx = 1 / n
    x = linspace(-0.5 + dx / 2, 0.5 - dx / 2, num=n)
    return v0 * erf(x / (2 * sqrt(μ * t)))
