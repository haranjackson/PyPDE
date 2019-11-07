from ctypes import CDLL, POINTER, c_bool, c_double, c_int, c_void_p

from numba import carray, cfunc, jit
from numba.types import CPointer, double, intc, void
from numpy import array, concatenate, dtype, inner, ones, zeros

from src.tests import F_reactive_euler, S_reactive_euler

F_sig = void(CPointer(double), CPointer(double), intc)
B_sig = void(CPointer(double), CPointer(double), intc)
S_sig = void(CPointer(double), CPointer(double))

FLUXES = {'rusanov': 0, 'roe': 1, 'osher': 2}


def c_ptr(arr):

    if arr.dtype == dtype('float64'):
        return arr.ctypes.data_as(POINTER(c_double))
    elif arr.dtype == dtype('int64'):
        return arr.ctypes.data_as(POINTER(c_int))


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
    boundaryTypes = array(boundaryTypes, dtype=int)

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

    libader = CDLL('build/libader.dylib')
    solver = libader.ader_solver

    solver.argtypes = [
        c_void_p, c_void_p, c_void_p,
        POINTER(c_double), c_double,
        POINTER(c_int), c_int,
        POINTER(c_double), c_double,
        POINTER(c_int), c_bool, c_int, c_int, c_int, c_int,
        POINTER(c_double)
    ]

    solver.restype = None

    ret = zeros(ndt * u.size)

    solver(_F.ctypes, _B.ctypes, _S.ctypes, c_ptr(u), tf, c_ptr(nX), ndim,
           c_ptr(dX), CFL, c_ptr(boundaryTypes), STIFF, FLUXES[flux], N, V,
           ndt, c_ptr(ret))


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
