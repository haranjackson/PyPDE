from ctypes import CDLL, POINTER, c_double, c_int

from numpy import array, concatenate, int32, zeros
from pypde.utils import c_ptr, get_cdll


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
