import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import arange, array, meshgrid, zeros

from pypde import ader_solver


def F_euler(Q):

    γ = 1.4

    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2] / ρ

    e = E - v**2 / 2
    p = (γ - 1) * ρ * e

    return array([ρ * v, ρ * E * v + p * v, ρ * v**2 + p])


def energy(ρ, p, v):

    γ = 1.4
    return p / ((γ - 1) * ρ) + v**2 / 2


def test_euler(multiDimensional=True):

    nx = 200

    ρL = 1
    pL = 1
    vL = 0

    ρR = 0.125
    pR = 0.1
    vR = 0

    QL = ρL * array([1., energy(ρL, pL, vL), vL])
    QR = ρR * array([1., energy(ρR, pR, vR), vR])

    u = zeros([nx, 3])
    for i in range(nx):
        if i / nx < 0.5:
            u[i] = QL
        else:
            u[i] = QR

    tf = 0.2

    if multiDimensional:
        u_ = u.copy()
        u = zeros([nx, 5, 3])
        for i in range(5):
            u[:, i] = u_
        L = [1., .2]
        boundaryTypes = ['transitive', 'periodic']
    else:
        L = [1.]
        boundaryTypes = ['transitive']

    ret = ader_solver(u,
                      tf,
                      L,
                      F=F_euler,
                      STIFF=False,
                      boundaryTypes=boundaryTypes)

    if multiDimensional:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = arange(0, L[0], L[0] / nx)
        y = arange(0, L[1], L[1] / 5)
        X, Y = meshgrid(y, x)
        Z = ret[-1, :, :, 0]
        ax.plot_surface(X, Y, Z)
    else:
        plt.plot(ret[-1, :, 0])

    plt.show()

    return ret
