import inspect
import sys
from ctypes import CDLL, POINTER, c_bool, c_double, c_int, c_void_p
from importlib.util import find_spec
from sys import platform

from numpy import array

ADER_ARGTYPES = [
    c_void_p, c_void_p, c_void_p, c_bool, c_bool, c_bool,
    POINTER(c_double), c_double,
    POINTER(c_int), c_int,
    POINTER(c_double), c_double,
    POINTER(c_int), c_bool, c_int, c_int, c_int, c_int, c_bool,
    POINTER(c_double)
]

BOUNDARIES = {'transitive': 0, 'periodic': 1}


def nargs(func):
    return len(inspect.signature(func).parameters)


def c_ptr(arr):

    if arr.dtype == 'int32':
        ptr = POINTER(c_int)
    elif arr.dtype == 'float64':
        ptr = POINTER(c_double)
    else:
        print('invalid array type')

    return arr.ctypes.data_as(ptr)


def parse_boundary_types(boundaryTypes, ndim):

    errMsg = ('boundaryTypes must be a string from {"transitive", "periodic"} '
              'or a list of such strings, of length equal to the number of '
              'dimensions of the domain.')

    if isinstance(boundaryTypes, str):
        try:
            ret = [BOUNDARIES[boundaryTypes]] * ndim
        except KeyError:
            print(errMsg)
            sys.exit(1)

    elif isinstance(boundaryTypes, list):

        if len(boundaryTypes) != ndim:
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

    return array(ret, dtype='int32')


def get_cdll():

    loc = find_spec('pypde').submodule_search_locations[0]

    if platform == "linux" or platform == "linux2":
        end = 'so'
    if platform == "darwin":
        end = 'dylib'
    if platform == "win32":
        end = 'dll'

    return CDLL(loc + '/build/libpypde.' + end)


def create_solver():

    libpypde = get_cdll()
    solver = libpypde.ader_solver

    solver.argtypes = ADER_ARGTYPES
    solver.restype = None

    return solver
