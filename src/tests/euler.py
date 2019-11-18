from numba import jit
from numpy import array, concatenate, inner, zeros


@jit
def internal_energy(E, v, λ, Qc):

    return E - (v[0]**2 + v[1]**2 + v[2]**2) / 2 - Qc * (λ - 1)


def F_reactive_euler(Q, d):

    γ = 1.4
    Qc = 1

    ρ = Q[0]
    E = Q[1] / ρ
    v = [Q[2] / ρ, Q[3] / ρ, Q[4] / ρ]
    λ = Q[5] / ρ

    # internal energy
    e = internal_energy(E, v, λ, Qc)

    # pressure
    p = (γ - 1) * ρ * e

    ret = v[d] * Q
    ret[1] += p * v[d]
    ret[2 + d] += p

    return ret


@jit
def reaction_rate(E, v, λ, Qc, cv, Ti, K0):
    """ Returns the rate of reaction according to discrete ignition temperature
        reaction kinetics
    """
    e = internal_energy(E, v, λ, Qc)
    T = e / cv
    return K0 if T > Ti else 0


def S_reactive_euler(Q):

    ret = zeros(6)

    Qc = 1
    cv = 2.5
    Ti = 0.25
    K0 = 250

    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2:5] / ρ
    λ = Q[5] / ρ

    ret[5] = -ρ * λ * reaction_rate(E, v, λ, Qc, cv, Ti, K0)

    return ret


def energy(ρ, p, v, λ, γ, Qc):
    return p / ((γ - 1) * ρ) + inner(v, v) / 2 + Qc * (λ - 1)


def _test_euler():

    γ = 1.4
    Qc = 1

    nx = 20

    ρL = 1.4
    pL = 1
    vL = array([0, 0, 0])
    λL = 0
    EL = energy(ρL, pL, vL, λL, γ, Qc)

    ρR = 0.887565
    pR = 0.191709
    vR = array([-0.57735, 0, 0])
    λR = 1
    ER = energy(ρR, pR, vR, λR, γ, Qc)

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
