from ctypes import CDLL

from numba import carray, cfunc, njit
from numpy import array, zeros

from pypde.utils import (BOUNDARIES, FLUXES, SOLVER_ARGTYPES, B_Csig, F_Csig,
                         S_Csig, c_ptr, parse_boundary_types)


def ader_solver(u,
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
    dX = array([L[i] / nX[i] for i in range(len(L))])

    boundaryTypes = parse_boundary_types(boundaryTypes, ndim)

    if F is None:
        F = lambda Q, d: zeros(V)
    useF = False if F is None else True

    if B is None:
        B = lambda Q, d: zeros((V, V))
    useB = False if B is None else True

    if S is None:
        S = lambda Q: zeros(V)
    useS = False if S is None else True

    F_Sig = 'double[:](double[:], intc)'
    B_Sig = 'double[:,:](double[:], intc)'
    S_Sig = 'double[:](double[:])'

    F_jit = njit(F_Sig)(F)
    B_jit = njit(B_Sig)(B)
    S_jit = njit(S_Sig)(S)

    @cfunc(F_Csig)
    def _F(outArr, inArr, d):
        Q = carray(inArr, (V, ))
        ret = carray(outArr, (V, ))
        FQ = F_jit(Q, d)
        for i in range(V):
            ret[i] = FQ[i]

    @cfunc(B_Csig)
    def _B(outArr, inArr, d):
        Q = carray(inArr, (V, ))
        ret = carray(outArr, (V, V))
        BQ = B_jit(Q, d)
        for i in range(V):
            for j in range(V):
                ret[i, j] = BQ[i, j]

    @cfunc(S_Csig)
    def _S(outArr, inArr):
        Q = carray(inArr, (V, ))
        ret = carray(outArr, (V, ))
        SQ = S_jit(Q)
        for i in range(V):
            ret[i] = SQ[i]

    libader = CDLL('build/libader.dylib')
    solver = libader.ader_solver

    solver.argtypes = SOLVER_ARGTYPES
    solver.restype = None

    ret = zeros(ndt * u.size)
    ur = u.ravel()

    solver(_F.ctypes, _B.ctypes, _S.ctypes, useF, useB, useS, c_ptr(ur), tf,
           c_ptr(nX), ndim, c_ptr(dX), CFL, c_ptr(boundaryTypes), STIFF,
           FLUXES[flux], N, V, ndt, c_ptr(ret))

    return ret.reshape((ndt, ) + u.shape)
