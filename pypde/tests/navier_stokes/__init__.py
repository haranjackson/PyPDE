import matplotlib.pyplot as plt
from numpy import arange, array, cos, dot, eye, linspace, pi, sin, zeros

from pypde import pde_solver
from pypde.tests.exact import viscous_shock_exact
from pypde.tests.navier_stokes.system import F_navier_stokes, total_energy, γ


def make_Q(ρ, p, v):
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
        u[i] = make_Q(ρ[i], p[i], array([v[i], 0, 0]))

    L = [1.]

    ret = pde_solver(u, tf, L, F=F_navier_stokes, cfl=0.6)

    plt.plot(ret[-1, :, 2] / ret[-1, :, 0])
    plt.show()

    return ret


def taylor_green_vortex_2d():
    """ 10.1016/j.jcp.2016.02.015
        4.10 2D Taylor-Green Vortex
    """
    L = [2 * pi, 2 * pi]

    nx = 50
    ny = 50
    tf = 1

    C = 100 / γ
    ρ = 1
    v = zeros(3)

    u = zeros([nx, ny, 5])
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) * L[0] / nx
            y = (j + 0.5) * L[1] / ny
            v[0] = sin(x) * cos(y)
            v[1] = -cos(x) * sin(y)
            p = C + (cos(2 * x) + cos(2 * y)) / 4
            u[i, j] = make_Q(ρ, p, v)

    ret = pde_solver(u,
                     tf,
                     L,
                     F=F_navier_stokes,
                     cfl=0.9,
                     order=2,
                     boundaryTypes='periodic')

    x = linspace(0, L[0], nx)
    y = linspace(0, L[1], ny)

    ut = ret[-1, :, :, 2] / ret[-1, :, :, 0]
    vt = ret[-1, :, :, 3] / ret[-1, :, :, 0]
    plt.streamplot(x, y, ut, vt)
    plt.show()

    return ret
