TEST = 'viscous_shock'

STIFFENED_GAS = 0
SHOCK_MG = 1

EOS = STIFFENED_GAS

γ = 1.4
ρ0 = 1
p0 = 1 / γ
Tref = 0

if TEST == 'stokes':

    cv = 1
    cs = 1
    cα = 1e-16

    μ = 1e-2
    Pr = 0.75
    κ = μ * γ * cv / Pr

elif TEST == 'heat_conduction':

    cv = 2.5
    cs = 1
    cα = 2

    μ = 1e-2
    κ = 1e-2

elif TEST == 'viscous_shock':

    cv = 2.5
    cs = 5
    cα = 5

    μ = 2e-2
    Pr = 0.75
    κ = μ * γ * cv / Pr

T0 = p0 / (ρ0 * (γ - 1) * cv)

τ1 = 6 * μ / (ρ0 * cs**2)
τ2 = κ * ρ0 / (T0 * cα**2)

cs2 = cs**2
cα2 = cα**2

pINF = 0

Γ0 = None
s = None
c02 = None

PLASTIC = None
n = None
σY = None
