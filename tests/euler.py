import matplotlib.pyplot as plt
from numba import jit
from numpy import array, concatenate, inner, zeros

from pypde import ader_solver


@jit
def internal_energy(E, v, λ):

    Qc = 1

    return E - (v[0]**2 + v[1]**2 + v[2]**2) / 2 - Qc * (λ - 1)


def F_reactive_euler(Q, d):

    γ = 1.4

    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2:5] / ρ
    λ = Q[5] / ρ

    # internal energy
    e = internal_energy(E, v, λ)

    # pressure
    p = (γ - 1) * ρ * e

    ret = v[d] * Q
    ret[1] += p * v[d]
    ret[2 + d] += p

    return ret


@jit
def reaction_rate(E, v, λ):
    """ Returns the rate of reaction according to discrete ignition temperature
        reaction kinetics
    """

    cv = 2.5
    Ti = 0.25
    K0 = 250

    e = internal_energy(E, v, λ)
    T = e / cv
    return K0 if T > Ti else 0


def S_reactive_euler(Q):

    ret = zeros(6)

    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2:5] / ρ
    λ = Q[5] / ρ

    ret[5] = -ρ * λ * reaction_rate(E, v, λ)

    return ret


def energy(ρ, p, v, λ):

    Qc = 1
    γ = 1.4

    return p / ((γ - 1) * ρ) + inner(v, v) / 2 + Qc * (λ - 1)


def _test_euler():

    nx = 100

    ρL = 1.4
    pL = 1
    vL = zeros(3)
    λL = 0
    EL = energy(ρL, pL, vL, λL)

    ρR = 0.887565
    pR = 0.191709
    vR = array([-0.57735, 0, 0])
    λR = 1
    ER = energy(ρR, pR, vR, λR)

    QL = ρL * concatenate([array([1, EL]), vL, array([λL])])
    QR = ρR * concatenate([array([1, ER]), vR, array([λR])])

    u = zeros([nx, 6])
    for i in range(nx):
        if i / nx < 0.25:
            u[i] = QL
        else:
            u[i] = QR

    tf = 0.5
    L = [1.]

    return u, tf, L


def test_euler():

    u, tf, L = _test_euler()
    tf = 1
    return ader_solver(u, tf, L, F=F_reactive_euler, S=None)
