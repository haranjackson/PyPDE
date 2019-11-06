from numpy import array, inner, zeros

from numba import jit


@jit
def sqnorm(x):
    """ Returns the squared norm of a 3-vector
    """
    return inner(x, x)


@jit
def pressure(ρ, E, v, λ, γ, Qc):
    e = E - sqnorm(v) / 2 - Qc * (λ - 1)
    return (γ - 1) * ρ * e


@jit
def temperature(ρ, E, v, λ, Qc, cv):
    e = E - sqnorm(v) / 2 - Qc * (λ - 1)
    return e / cv


@jit
def energy(ρ, p, v, λ, γ, Qc):
    return p / ((γ - 1) * ρ) + sqnorm(v) / 2 + Qc * (λ - 1)


@jit
def reaction_rate(ρ, E, v, λ, Qc, cv, Ti, K0):
    """ Returns the rate of reaction according to discrete ignition temperature
        reaction kinetics
    """
    T = temperature(ρ, E, v, λ, Qc, cv)
    return K0 if T > Ti else 0


def F_reactive_euler(Q, d):

    γ = 1.4
    Qc = 1

    ρ = Q[0]
    E = Q[1] / ρ
    v = [Q[2] / ρ, Q[3] / ρ, Q[4] / ρ]
    λ = Q[5] / ρ

    p = pressure(ρ, E, v, λ, γ, Qc)

    # convert to list to avoid:
    #   TypeError: 'ArrayBox' object does not support item assignment
    ret = list(v[d] * Q)
    ret[1] += p * v[d]
    ret[2 + d] += p

    return array(ret)


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

    ret[5] = -ρ * λ * reaction_rate(ρ, E, v, λ, Qc, cv, Ti, K0)

    return ret
