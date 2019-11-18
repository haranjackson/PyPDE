import sys
from ctypes import POINTER, c_bool, c_double, c_int, c_void_p

from numba.types import CPointer, double, intc, void
from numpy import array, dtype

F_Csig = void(CPointer(double), CPointer(double), intc)
B_Csig = void(CPointer(double), CPointer(double), intc)
S_Csig = void(CPointer(double), CPointer(double))

FLUXES = {'rusanov': 0, 'roe': 1, 'osher': 2}

BOUNDARIES = {'transitive': 0, 'periodic': 1}

SOLVER_ARGTYPES = [
    c_void_p, c_void_p, c_void_p, c_bool, c_bool, c_bool,
    POINTER(c_double), c_double,
    POINTER(c_int), c_int,
    POINTER(c_double), c_double,
    POINTER(c_int), c_bool, c_int, c_int, c_int, c_int,
    POINTER(c_double)
]


def c_ptr(arr):

    if arr.dtype == dtype('int64'):
        arr = arr.astype('int32')

    if arr.dtype == dtype('int32'):
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
