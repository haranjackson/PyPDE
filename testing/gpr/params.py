STIFFENED_GAS = 0
SHOCK_MG = 1

EOS = STIFFENED_GAS

γ = 1.4
ρ0 = 1
p0 = 1 / γ
cv = 1
Tref = 0

cs = 1
cα = 1e-16

μ = 1e-2
Pr = 0.75

T0 = p0 / (ρ0 * (γ - 1) * cv)

τ1 = 1
τ2 = 1

cs2 = cs**2
cα2 = cα**2

pINF = 0
Γ0 = None
s = None
c02 = None

PLASTIC = False
n = None
σY = None
