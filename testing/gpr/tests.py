import matplotlib.pyplot as plt
from numpy import arange, array, exp, eye, linspace, sqrt, zeros
from scipy.optimize import brentq
from scipy.special import erf

from pypde import ader_solver
from testing.gpr.misc.eos import total_energy
from testing.gpr.params import TEST, γ, μ
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


def stokes_exact(n=100, v0=0.1, t=1):
    """ Returns the exact solution of the y-velocity in the x-axis for Stokes'
        First Problem
    """
    dx = 1 / n
    x = linspace(-0.5 + dx / 2, 0.5 - dx / 2, num=n)
    return v0 * erf(x / (2 * sqrt(μ * t)))


def stokes_test(n=100, v0=0.1):

    assert (TEST == 'stokes')

    tf = 1

    ρL = 1
    pL = 1 / γ
    vL = array([0, -v0, 0])

    ρR = 1
    pR = 1 / γ
    vR = array([0, v0, 0])

    L = [1.]

    u = riemann_IC(n, ρL, pL, vL, ρR, pR, vR)

    ret = ader_solver(u,
                      tf,
                      L,
                      F=F_gpr,
                      B=B_gpr,
                      S=S_gpr,
                      STIFF=False,
                      CFL=0.9)

    plt.plot(ret[-1, :, 3] / ret[-1, :, 0])
    plt.show()

    return ret


def heat_conduction_test(n=200):

    assert (TEST == 'heat_conduction')

    tf = 1

    ρL = 2
    pL = 1
    vL = zeros(3)

    ρR = 0.5
    pR = 1
    vR = zeros(3)

    L = [1.]

    u = riemann_IC(n, ρL, pL, vL, ρR, pR, vR)

    ret = ader_solver(u,
                      tf,
                      L,
                      F=F_gpr,
                      B=B_gpr,
                      S=S_gpr,
                      STIFF=False,
                      CFL=0.9)

    return ret


def viscous_shock_exact(x, Ms=2):
    """ Returns the density, pressure, and velocity of the viscous shock
        (Mach number Ms) at x
    """
    ρ0 = 1
    p0 = 1 / γ

    L = 0.3

    x = min(x, L)
    x = max(x, -L)

    c0 = sqrt(γ * p0 / ρ0)
    a = 2 / (Ms**2 * (γ + 1)) + (γ - 1) / (γ + 1)
    Re = ρ0 * c0 * Ms / μ
    c1 = ((1 - a) / 2)**(1 - a)
    c2 = 3 / 4 * Re * (Ms**2 - 1) / (γ * Ms**2)

    def f(z):
        return (1 - z) / (z - a)**a - c1 * exp(c2 * -x)

    vbar = brentq(f, a + 1e-16, 1)
    p = p0 / vbar * (1 + (γ - 1) / 2 * Ms**2 * (1 - vbar**2))
    ρ = ρ0 / vbar
    v = Ms * c0 * vbar
    v = Ms * c0 - v  # Shock travelling into fluid at rest

    return ρ, p, v


def viscous_shock_test(nx=200):
    """ 10.1016/j.jcp.2016.02.015
        4.13 Viscous shock profile
    """
    assert (TEST == 'viscous_shock')

    tf = 0.2
    nx = 200

    x = arange(-0.5, 0.5, 1 / nx)
    ρ = zeros(nx)
    p = zeros(nx)
    v = zeros(nx)
    for i in range(nx):
        ρ[i], p[i], v[i] = viscous_shock_exact(x[i])

    v -= v[0]  # Velocity in shock 0

    u = zeros([nx, 17])
    for i in range(nx):
        A = (ρ[i])**(1 / 3) * eye(3)
        J = zeros(3)
        u[i] = Cvec(ρ[i], p[i], array([v[i], 0, 0]), A, J)

    L = [1.]

    ret = ader_solver(u, tf, L, F=F_gpr, B=B_gpr, S=S_gpr, CFL=0.6)

    plt.plot(ret[-1, :, 2] / ret[-1, :, 0])
    plt.show()

    return ret
