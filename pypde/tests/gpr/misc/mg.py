from numba import njit

from pypde.tests.gpr.params import (EOS, STIFFENED_GAS, Γ0, Tref, c02, cv, pINF, s,
                                γ, ρ0)


@njit
def Γ_MG(ρ):
    """ Returns the Mie-Gruneisen parameter
    """
    if EOS == STIFFENED_GAS:
        return γ - 1

    # SHOCK_MG
    return Γ0 * ρ0 / ρ


@njit
def p_ref(ρ):
    """ Returns the reference pressure in the Mie-Gruneisen EOS
    """
    if EOS == STIFFENED_GAS:
        return -pINF

    # SHOCK_MG
    if ρ > ρ0:
        return c02 * (1 / ρ0 - 1 / ρ) / (1 / ρ0 - s * (1 / ρ0 - 1 / ρ))**2

    return c02 * (ρ - ρ0)


@njit
def e_ref(ρ):
    """ Returns the reference energy for the Mie-Gruneisen EOS
    """
    if EOS == STIFFENED_GAS:
        return pINF / ρ

    # SHOCK_MG
    pr = p_ref(ρ)
    if ρ > ρ0:
        return 0.5 * pr * (1 / ρ0 - 1 / ρ)
    return 0


@njit
def internal_energy(ρ, p):
    """ Returns the Mie-Gruneisen internal energy
    """
    Γ = Γ_MG(ρ)
    pr = p_ref(ρ)
    er = e_ref(ρ)
    return er + (p - pr) / (ρ * Γ)


@njit
def pressure(ρ, e):
    """ Returns the Mie-Gruneisen pressure, given the density and internal
        energy
    """
    Γ = Γ_MG(ρ)
    pr = p_ref(ρ)
    er = e_ref(ρ)
    return (e - er) * ρ * Γ + pr


@njit
def temperature(ρ, p):
    """ Returns the Mie-Gruneisen temperature, given the density and pressure
    """
    Γ = Γ_MG(ρ)
    pr = p_ref(ρ)
    return Tref + (p - pr) / (ρ * Γ * cv)
