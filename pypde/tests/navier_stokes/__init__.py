import matplotlib.pyplot as plt
from numpy import arange, array, dot, eye, zeros
from pypde import ader_solver
from pypde.tests.exact import viscous_shock_exact
from pypde.tests.navier_stokes.system import F_navier_stokes, total_energy, γ


def Cvec(ρ, p, v):
    """ Returns the vector of conserved variables, given the primitive variables
    """
    Q = zeros(5)
    Q[0] = ρ
    Q[1] = ρ * total_energy(ρ, p, v)
    Q[2:5] = ρ * v
    return Q


def viscous_shock_ns():
    """ 10.1016/j.jcp.2016.02.015
        4.13 Viscous shock profile
    """
    tf = 0.2
    nx = 200

    x = arange(-0.5, 0.5, 1 / nx)
    ρ = zeros(nx)
    p = zeros(nx)
    v = zeros(nx)
    for i in range(nx):
        ρ[i], p[i], v[i] = viscous_shock_exact(x[i])

    v -= v[0]  # Velocity in shock 0

    u = zeros([nx, 5])
    for i in range(nx):
        u[i] = Cvec(ρ[i], p[i], array([v[i], 0, 0]))

    L = [1.]

    ret = ader_solver(u, tf, L, F=F_navier_stokes, CFL=0.6)

    plt.plot(ret[-1, :, 2] / ret[-1, :, 0])
    plt.show()

    return ret
