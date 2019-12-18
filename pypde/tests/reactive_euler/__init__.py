import matplotlib.pyplot as plt
from numpy import array, zeros

from pypde import pde_solver
from pypde.tests.reactive_euler.system import (F_reactive_euler,
                                               S_reactive_euler, energy)


def detonation_wave():

    nx = 400

    ρL = 1.4
    pL = 1
    vL = [0, 0, 0]
    λL = 0
    EL = energy(ρL, pL, vL, λL)

    ρR = 0.887565
    pR = 0.191709
    vR = [-0.57735, 0, 0]
    λR = 1
    ER = energy(ρR, pR, vR, λR)

    QL = ρL * array([1, EL] + vL + [λL])
    QR = ρR * array([1, ER] + vR + [λR])

    u = zeros([nx, 6])
    for i in range(nx):
        if i / nx < 0.25:
            u[i] = QL
        else:
            u[i] = QR

    tf = 0.5
    L = [1.]

    ret = pde_solver(u,
                     tf,
                     L,
                     F=F_reactive_euler,
                     S=S_reactive_euler,
                     stiff=False,
                     cfl=0.6,
                     flux='roe',
                     order=3)

    plt.plot(ret[-1, :, 0])
    plt.show()

    return ret
