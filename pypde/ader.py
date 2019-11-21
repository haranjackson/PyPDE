from numpy import array, zeros

from pypde.cfuncs import generate_cfuncs
from pypde.utils import c_ptr, create_solver, nargs, parse_boundary_types

FLUXES = {'rusanov': 0, 'roe': 1, 'osher': 2}


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
