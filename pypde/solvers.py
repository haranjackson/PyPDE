from ctypes import POINTER, c_double, c_int

from numpy import array, concatenate, int32, zeros

from pypde.cfuncs import generate_cfuncs
from pypde.utils import (c_ptr, create_solver, get_cdll, nargs,
                         parse_boundary_types)

FLUXES = {'rusanov': 0, 'roe': 1, 'osher': 2}


def pde_solver(u,
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

    nX = array(u.shape[:-1], dtype='int32')
    ndim = len(nX)
    V = u.shape[-1]
    dX = array([L[i] / nX[i] for i in range(len(L))])

    boundaryTypes = parse_boundary_types(boundaryTypes, ndim)

    useF = False if F is None else True
    if F is None:
        F = lambda Q, dQ, d: zeros(V)

    useB = False if B is None else True
    if B is None:
        B = lambda Q, d: zeros((V, V))

    useS = False if S is None else True
    if S is None:
        S = lambda Q: zeros(V)

    secondOrder = nargs(F) == 3

    print('compiling functions...')

    _F, _B, _S = generate_cfuncs(F, B, S, ndim, V)

    solver = create_solver()

    ret = zeros(ndt * u.size)
    ur = u.ravel()

    solver(_F.ctypes, _B.ctypes, _S.ctypes, useF, useB, useS, c_ptr(ur), tf,
           c_ptr(nX), ndim, c_ptr(dX), CFL, c_ptr(boundaryTypes), STIFF,
           FLUXES[flux], N, V, ndt, secondOrder, c_ptr(ret))

    return ret.reshape((ndt, ) + u.shape)


def weno_solver(u, N=2):

    nX = array(u.shape[:-1], dtype=int32)
    ndim = len(nX)
    V = u.shape[-1]

    nXret = nX - 2 * (N - 1)

    libpypde = get_cdll()
    solver = libpypde.weno_solver

    solver.argtypes = [
        POINTER(c_double),
        POINTER(c_double),
        POINTER(c_int),
        c_int,
        c_int,
        c_int,
    ]
    solver.restype = None

    ncellRet = nXret.prod()
    ret = zeros(ncellRet * N**ndim * V)
    ur = u.ravel()

    solver(c_ptr(ret), c_ptr(ur), c_ptr(nX), ndim, N, V)

    return ret.reshape(concatenate([nXret, [N] * ndim, [V]]))
