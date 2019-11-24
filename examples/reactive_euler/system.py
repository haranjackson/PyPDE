from numba import njit
from numpy import inner, zeros


@njit
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


@njit
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
