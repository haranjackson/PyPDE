from numpy import exp, linspace, sqrt
from scipy.optimize import brentq
from scipy.special import erf


def stokes_exact(n=100, v0=0.1, t=1):
    """ Returns the exact solution of the y-velocity in the x-axis for Stokes'
        First Problem
    """
    μ = 1e-2
    dx = 1 / n
    x = linspace(-0.5 + dx / 2, 0.5 - dx / 2, num=n)
    return v0 * erf(x / (2 * sqrt(μ * t)))


def viscous_shock_exact(x, Ms=2):
    """ Returns the density, pressure, and velocity of the viscous shock
        (Mach number Ms) at x
    """
    γ = 1.4
    μ = 2e-2

    ρ0 = 1
    p0 = 1 / γ

    L = 0.3

    x = min(x, L)
    x = max(x, -L)

    c0 = sqrt(γ * p0 / ρ0)
    a = 2 / (Ms**2 * (γ + 1)) + (γ - 1) / (γ + 1)
    Re = ρ0 * c0 * Ms / μ
    c1 = ((1 - a) / 2)**(1 - a)
    c2 = 3 / 4 * Re * (Ms**2 - 1) / (γ * Ms**2)

    def f(z):
        return (1 - z) / (z - a)**a - c1 * exp(c2 * -x)

    vbar = brentq(f, a + 1e-16, 1)
    p = p0 / vbar * (1 + (γ - 1) / 2 * Ms**2 * (1 - vbar**2))
    ρ = ρ0 / vbar
    v = Ms * c0 * vbar
    v = Ms * c0 - v  # Shock travelling into fluid at rest

    return ρ, p, v
