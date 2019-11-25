import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import arange, array, meshgrid, zeros
from pypde import ader_solver
from pypde.tests.euler.system import F_euler, energy


def plot_2d(L, Z):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    nx, ny = Z.shape[:2]
    x = arange(0, L[0], L[0] / nx)
    y = arange(0, L[1], L[1] / ny)
    X, Y = meshgrid(y, x)

    ax.plot_surface(X, Y, Z)
    plt.show()


def sod_shock():

    nx = 200
    ny = 5

    ρL = 1
    pL = 1
    vL = 0

    ρR = 0.125
    pR = 0.1
    vR = 0

    QL = ρL * array([1., energy(ρL, pL, vL), vL])
    QR = ρR * array([1., energy(ρR, pR, vR), vR])

    u_ = zeros([nx, 3])
    for i in range(nx):
        if i / nx < 0.5:
            u_[i] = QL
        else:
            u_[i] = QR

    tf = 0.2

    u = zeros([nx, ny, 3])
    for i in range(ny):
        u[:, i] = u_
    L = [1., .2]
    boundaryTypes = ['transitive', 'periodic']

    ret = ader_solver(u,
                      tf,
                      L,
                      F=F_euler,
                      STIFF=False,
                      boundaryTypes=boundaryTypes)

    plot_2d(L, ret[-1, :, :, 0])

    return ret
