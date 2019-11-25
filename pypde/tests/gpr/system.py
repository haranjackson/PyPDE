from pypde.tests.gpr.misc.sources import theta1inv, theta2inv
from pypde.tests.gpr.misc.state import (dEdA, dEdJ, heat_flux, pressure, sigma,
                                     temperature)
from pypde.tests.gpr.misc.utils import get_variables
from numpy import dot, zeros


def F_gpr(Q, d):

    ret = zeros(17)

    ρ, E, v, A, J = get_variables(Q)

    p = pressure(ρ, E, v, A, J)
    T = temperature(ρ, p)
    σ = sigma(ρ, A)
    q = heat_flux(T, J)

    vd = v[d]
    ρvd = ρ * vd

    ret[0] = ρvd
    ret[1] = ρvd * E + p * vd
    ret[2:5] = ρvd * v
    ret[2 + d] += p

    σd = σ[d]
    ret[1] -= dot(σd, v)
    ret[2:5] -= σd

    Av = dot(A, v)
    ret[5 + d] = Av[0]
    ret[8 + d] = Av[1]
    ret[11 + d] = Av[2]

    ret[1] += q[d]
    ret[14:17] = ρvd * J
    ret[14 + d] += T

    return ret


def S_gpr(Q):

    ret = zeros(17)

    ρ, E, v, A, J = get_variables(Q)

    p = pressure(ρ, E, v, A, J)
    T = temperature(ρ, p)

    ψ = dEdA(A)
    ret[5:14] = -ψ.ravel() * theta1inv(ρ, A)

    H = dEdJ(J)
    ret[14:17] = -ρ * H * theta2inv(ρ, T)

    return ret


def B_gpr(Q, d):

    ret = zeros((17, 17))

    ρ = Q[0]
    v = Q[2:5] / ρ
    vd = v[d]

    for i in range(5, 14):
        ret[i, i] = vd
    ret[5 + d, 5 + d:8 + d] -= v
    ret[8 + d, 8 + d:11 + d] -= v
    ret[11 + d, 11 + d:14 + d] -= v

    return ret
