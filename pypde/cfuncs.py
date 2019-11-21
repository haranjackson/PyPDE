from numba import carray, cfunc, njit
from numba.types import CPointer, double, intc, void

from pypde.utils import nargs

F_SIG = void(CPointer(double), CPointer(double), CPointer(double), intc)
B_SIG = void(CPointer(double), CPointer(double), intc)
S_SIG = void(CPointer(double), CPointer(double))


def generate_signatures(nFargs, nBargs):

    if nFargs == 1:
        Fsig = 'double[:](double[:])'
    elif nFargs == 2:
        Fsig = 'double[:](double[:], intc)'
    elif nFargs == 3:
        Fsig = 'double[:](double[:], double[:,:], intc)'

    if nBargs == 1:
        Bsig = 'double[:,:](double[:])'
    elif nBargs == 2:
        Bsig = 'double[:,:](double[:], intc)'

    Ssig = 'double[:](double[:])'

    return Fsig, Bsig, Ssig


def generate_cfuncs(F, B, S, ndim, V):

    nFargs = nargs(F)
    nBargs = nargs(B)

    Fsig, Bsig, Ssig = generate_signatures(nFargs, nBargs)

    Fjit = njit(Fsig)(F)
    Bjit = njit(Bsig)(B)
    Sjit = njit(Ssig)(S)

    @cfunc(F_SIG)
    def _F(outArr, q, dq, d):
        Q = carray(q, (V, ))
        dQ = carray(dq, (ndim, V))
        ret = carray(outArr, (V, ))
        if nFargs == 1:
            FQ = Fjit(Q)
        elif nFargs == 2:
            FQ = Fjit(Q, d)
        elif nFargs == 3:
            FQ = Fjit(Q, dQ, d)
        for i in range(V):
            ret[i] = FQ[i]

    @cfunc(B_SIG)
    def _B(outArr, q, d):
        Q = carray(q, (V, ))
        ret = carray(outArr, (V, V))
        if nBargs == 1:
            BQ = Bjit(Q)
        elif nBargs == 2:
            BQ = Bjit(Q, d)
        for i in range(V):
            for j in range(V):
                ret[i, j] = BQ[i, j]

    @cfunc(S_SIG)
    def _S(outArr, q):
        Q = carray(q, (V, ))
        ret = carray(outArr, (V, ))
        SQ = Sjit(Q)
        for i in range(V):
            ret[i] = SQ[i]

    return _F, _B, _S
