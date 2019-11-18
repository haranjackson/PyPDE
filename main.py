import sys
from ctypes import CDLL, POINTER, c_bool, c_double, c_int, c_void_p

from numba import carray, cfunc, jit
from numba.types import CPointer, double, intc, void
from numpy import array, dtype, zeros

from src.tests import F_reactive_euler, S_reactive_euler, _test_euler

F_Csig = void(CPointer(double), CPointer(double), intc)
B_Csig = void(CPointer(double), CPointer(double), intc)
S_Csig = void(CPointer(double), CPointer(double))

FLUXES = {'rusanov': 0, 'roe': 1, 'osher': 2}

BOUNDARIES = {'transitive': 0, 'periodic': 1}


def c_ptr(arr):

    if arr.dtype == dtype('int64'):
        ptr = POINTER(c_int)
    else:
        ptr = POINTER(c_double)

    return arr.ctypes.data_as(ptr)


def parse_boundary_types(boundaryTypes, ndim):

    errMsg = ('boundaryTypes must be a string from {"transitive", "periodic"} '
              'or a list of such strings, of length equal to the number of '
              'dimensions of the domain.')

    if isinstance(boundaryTypes, str):
        try:
            ret = [BOUNDARIES[boundaryTypes]]
        except KeyError:
            print(errMsg)
            sys.exit(1)

    elif isinstance(boundaryTypes, list):

        if not len(boundaryTypes) == ndim:
            print(errMsg)
            sys.exit(1)

        try:
            ret = [BOUNDARIES[b] for b in boundaryTypes]
        except KeyError:
            print(errMsg)
            sys.exit(1)

    else:
        print(errMsg)
        sys.exit(1)

    return array(ret, dtype=int)


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
    dX = array([L[i] / nX[i] for i in range(len(L))])

    boundaryTypes = parse_boundary_types(boundaryTypes, ndim)

    if F is None:
        F = lambda Q, d: zeros(V)

    if B is None:
        B = lambda Q, d: zeros((V, V))

    if S is None:
        S = lambda Q: zeros(V)

    F_Sig = 'double[:](double[:], intc)'
    B_Sig = 'double[:,:](double[:], intc)'
    S_Sig = 'double[:](double[:])'

    F_jit = jit(F_Sig, nopython=True)(F)
    B_jit = jit(B_Sig, nopython=True)(B)
    S_jit = jit(S_Sig, nopython=True)(S)

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

    ur = u.ravel()

    solver(_F.ctypes, _B.ctypes, _S.ctypes, c_ptr(ur), tf, c_ptr(nX), ndim,
           c_ptr(dX), CFL, c_ptr(boundaryTypes), STIFF, FLUXES[flux], N, V,
           ndt, c_ptr(ret))


def test_euler():

    u, tf, L = _test_euler()
    main(u, tf, L, F=F_reactive_euler, S=S_reactive_euler)
