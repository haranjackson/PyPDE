import matplotlib.pyplot as plt
from numpy import array, zeros

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


def test_euler():

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
    L = [1.]

    ret = ader_solver(u, tf, L, F=F_euler, STIFF=False, CFL=0.6)

    plt.plot(ret[-1, :, 0])
    plt.show()
