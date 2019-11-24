from numpy import array


def energy(ρ, p, v):

    γ = 1.4
    return p / ((γ - 1) * ρ) + v**2 / 2


def F_euler(Q):

    γ = 1.4

    ρ = Q[0]
    E = Q[1] / ρ
    v = Q[2] / ρ

    e = E - v**2 / 2
    p = (γ - 1) * ρ * e

    return array([ρ * v, ρ * E * v + p * v, ρ * v**2 + p])
