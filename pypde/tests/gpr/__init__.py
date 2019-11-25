import matplotlib.pyplot as plt
from numpy import arange, array, eye, zeros
from pypde import ader_solver
from pypde.tests.exact import viscous_shock_exact
from pypde.tests.gpr.misc.utils import Cvec
from pypde.tests.gpr.system import B_gpr, F_gpr, S_gpr


def viscous_shock_gpr():
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
