import ctypes

from numpy import array

from numba import carray, cfunc, jit
from numba.types import CPointer, double, void

c_sig = void(CPointer(double), CPointer(double))


@jit
def f_(Q):
    return array([Q[2], Q[1], Q[0]])


def F(Q):
    return f_(Q)


def main(func):
    F_ = jit(cache=False)(func)

    @cfunc(c_sig)
    def my_callback(inArray, outArray):
        Q = carray(inArray, (3, ))
        ret = carray(outArray, (3, ))
        FQ = F_(Q)
        for i in range(3):
            ret[i] = FQ[i]

    _test = ctypes.CDLL('test.so')
    test_func = _test.test_func
    test_func.restype = ctypes.c_double
    print(test_func(my_callback.ctypes))


main(F)
