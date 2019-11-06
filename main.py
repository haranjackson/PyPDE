import ctypes

from numba import carray, cfunc, jit
from numba.types import CPointer, double, intc, void
from numpy import array, concatenate, inner, ones, zeros

from src.tests import F_reactive_euler, S_reactive_euler

F_sig = void(CPointer(double), CPointer(double), intc)
B_sig = void(CPointer(double), CPointer(double), intc)
S_sig = void(CPointer(double), CPointer(double))

FLUXES = {'rusanov': 0, 'roe': 1, 'osher': 2}


def main(u,
         tf,
         L,
         ndt=100,
         flux='rusanov',
         STIFF=True,
         N=2,
         F=None,
         B=None,
         S=None,
         boundaryTypes='transitive',
         CFL=0.9):

    nX = array(u.shape[:-1])
    ndim = len(nX)
    V = u.shape[-1]
    dX = array(L[i] / nX[i] for i in range(len(L)))

    if boundaryTypes == 'transitive':
        boundaryTypes = zeros(ndim)
    elif boundaryTypes == 'periodic':
        boundaryTypes = ones(ndim)

    if F is None:
        F = lambda Q, d: zeros(V)

    if B is None:
        B = lambda Q, d: zeros((V, V))

    if S is None:
        S = lambda Q: zeros(V)

    F_jit = jit(cache=False)(F)
    B_jit = jit(cache=False, nopython=True)(B)
    S_jit = jit(cache=False)(S)

    @cfunc(F_sig)
    def _F(inArr, outArr, d):
        Q = carray(inArr, (V, ))
        ret = carray(outArr, (V, ))
        FQ = F_jit(Q, d)
        for i in range(V):
            ret[i] = FQ[i]

    @cfunc(B_sig)
    def _B(inArr, outArr, d):
        Q = carray(inArr, (V, ))
        ret = carray(outArr, (V, V))
        BQ = B_jit(Q, d)
        for i in range(V):
            for j in range(V):
                ret[i, j] = BQ[i, j]

    @cfunc(S_sig)
    def _S(inArr, outArr):
        Q = carray(inArr, (V, ))
        ret = carray(outArr, (V, ))
        SQ = S_jit(Q)
        for i in range(V):
            ret[i] = SQ[i]

    libader = ctypes.CDLL('build/libader.dylib')
    solver = libader.ader_solver
    solver.restype = None
    ret = zeros(ndt * u.size)

    solver(_F.ctypes, _B.ctypes, _S.ctypes, u.ctypes.data, tf, nX.ctypes.data,
           ndim, dX.ctypes.data, CFL, boundaryTypes.ctypes.data, STIFF,
           FLUXES[flux], N, V, ndt, ret.ctypes.data)


def energy(ρ, p, v, λ, γ, Qc):
    return p / ((γ - 1) * ρ) + inner(v, v) / 2 + Qc * (λ - 1)


def test_euler():

    γ = 1.4
    Qc = 1

    nx = 400

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

    #main(u, tf, L, F=F_reactive_euler, S=S_reactive_euler)
    main(u, tf, L)
