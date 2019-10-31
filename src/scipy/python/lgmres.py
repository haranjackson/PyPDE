import numpy as np
from numpy import asanyarray, asarray, asmatrix, array, matrix, zeros
from numpy.linalg import LinAlgError
from scipy._lib.six import xrange
from scipy.linalg import get_blas_funcs, qr_insert, lstsq
from scipy.sparse.linalg.interface import aslinearoperator, LinearOperator, \
    IdentityOperator

_coerce_rules = {('f', 'f'): 'f', ('f', 'd'): 'd', ('f', 'F'): 'F',
                 ('f', 'D'): 'D', ('d', 'f'): 'd', ('d', 'd'): 'd',
                 ('d', 'F'): 'D', ('d', 'D'): 'D', ('F', 'f'): 'F',
                 ('F', 'd'): 'D', ('F', 'F'): 'F', ('F', 'D'): 'D',
                 ('D', 'f'): 'D', ('D', 'd'): 'D', ('D', 'F'): 'D',
                 ('D', 'D'): 'D'}


def _fgmres(matvec, v0, m, atol, lpsolve=None, rpsolve=None, cs=(), outer_v=(),
            prepend_outer_v=False):

    if lpsolve is None:
        def lpsolve(x): return x
    if rpsolve is None:
        def rpsolve(x): return x

    axpy, dot, scal, nrm2 = get_blas_funcs(
        ['axpy', 'dot', 'scal', 'nrm2'], (v0,))

    vs = [v0]
    zs = []
    y = None

    m = m + len(outer_v)

    B = np.zeros((len(cs), m), dtype=v0.dtype)
    Q = np.ones((1, 1), dtype=v0.dtype)
    R = np.zeros((1, 0), dtype=v0.dtype)

    eps = np.finfo(v0.dtype).eps

    breakdown = False

    for j in xrange(m):

        if prepend_outer_v and j < len(outer_v):
            z, w = outer_v[j]
        elif prepend_outer_v and j == len(outer_v):
            z = rpsolve(v0)
            w = None
        elif not prepend_outer_v and j >= m - len(outer_v):
            z, w = outer_v[j - (m - len(outer_v))]
        else:
            z = rpsolve(vs[-1])
            w = None

        if w is None:
            w = lpsolve(matvec(z))
        else:
            w = w.copy()

        w_norm = nrm2(w)

        for i, c in enumerate(cs):
            alpha = dot(c, w)
            B[i, j] = alpha
            w = axpy(c, w, c.shape[0], -alpha)

        hcur = np.zeros(j + 2, dtype=Q.dtype)
        for i, v in enumerate(vs):
            alpha = dot(v, w)
            hcur[i] = alpha
            w = axpy(v, w, v.shape[0], -alpha)
        hcur[i + 1] = nrm2(w)

        with np.errstate(over='ignore', divide='ignore'):
            alpha = 1 / hcur[-1]

        if np.isfinite(alpha):
            w = scal(alpha, w)

        if not (hcur[-1] > eps * w_norm):
            breakdown = True

        vs.append(w)
        zs.append(z)

        Q2 = np.zeros((j + 2, j + 2), dtype=Q.dtype, order='F')
        Q2[:j + 1, :j + 1] = Q
        Q2[j + 1, j + 1] = 1

        R2 = np.zeros((j + 2, j), dtype=R.dtype, order='F')
        R2[:j + 1, :] = R

        Q, R = qr_insert(Q2, R2, hcur, j, which='col',
                         overwrite_qru=True, check_finite=False)

        res = abs(Q[0, -1])

        if res < atol or breakdown:
            break

    if not np.isfinite(R[j, j]):
        raise LinAlgError()

    y, _, _, _, = lstsq(R[:j + 1, :j + 1], Q[0, :j + 1].conj())

    B = B[:, :j + 1]

    return Q, R, B, vs, zs, y


def coerce(x, y):
    if x not in 'fdFD':
        x = 'd'
    if y not in 'fdFD':
        y = 'd'
    return _coerce_rules[x, y]


def id(x):
    return x


def make_system(A, M, x0, b):

    A_ = A
    A = aslinearoperator(A)

    if A.shape[0] != A.shape[1]:
        raise ValueError(
            'expected square matrix, but got shape=%s' % (A.shape,))

    N = A.shape[0]

    b = asanyarray(b)

    if not (b.shape == (N, 1) or b.shape == (N,)):
        raise ValueError('A and b have incompatible dimensions')

    if b.dtype.char not in 'fdFD':
        b = b.astype('d')  # upcast non-FP types to double

    def postprocess(x):
        if isinstance(b, matrix):
            x = asmatrix(x)
        return x.reshape(b.shape)

    if hasattr(A, 'dtype'):
        xtype = A.dtype.char
    else:
        xtype = A.matvec(b).dtype.char
    xtype = coerce(xtype, b.dtype.char)

    b = asarray(b, dtype=xtype)  # make b the same type as x
    b = b.ravel()

    if x0 is None:
        x = zeros(N, dtype=xtype)
    else:
        x = array(x0, dtype=xtype)
        if not (x.shape == (N, 1) or x.shape == (N,)):
            raise ValueError('A and x have incompatible dimensions')
        x = x.ravel()

    # process preconditioner
    if M is None:
        if hasattr(A_, 'psolve'):
            psolve = A_.psolve
        else:
            psolve = id
        if hasattr(A_, 'rpsolve'):
            rpsolve = A_.rpsolve
        else:
            rpsolve = id
        if psolve is id and rpsolve is id:
            M = IdentityOperator(shape=A.shape, dtype=A.dtype)
        else:
            M = LinearOperator(A.shape, matvec=psolve, rmatvec=rpsolve,
                               dtype=A.dtype)
    else:
        M = aslinearoperator(M)
        if A.shape != M.shape:
            raise ValueError('matrix and preconditioner have different shapes')

    return A, M, x, b, postprocess


def lgmres(A, b, x0=None, tol=1e-5, maxiter=1000, M=None, callback=None,
           inner_m=30, outer_k=3, outer_v=None, store_outer_Av=True,
           prepend_outer_v=False):

    A, M, x, b, postprocess = make_system(A, M, x0, b)

    if not np.isfinite(b).all():
        raise ValueError("RHS must contain only finite numbers")

    matvec = A.matvec
    psolve = M.matvec

    if outer_v is None:
        outer_v = []

    axpy, dot, scal = None, None, None
    nrm2 = get_blas_funcs('nrm2', [b])

    b_norm = nrm2(b)
    if b_norm == 0:
        b_norm = 1

    for k_outer in xrange(maxiter):
        r_outer = matvec(x) - b

        # -- callback
        if callback is not None:
            callback(x)

        # -- determine input type routines
        if axpy is None:
            if np.iscomplexobj(r_outer) and not np.iscomplexobj(x):
                x = x.astype(r_outer.dtype)
            axpy, dot, scal, nrm2 = get_blas_funcs(['axpy', 'dot', 'scal', 'nrm2'],
                                                   (x, r_outer))

        # -- check stopping condition
        r_norm = nrm2(r_outer)
        if r_norm <= tol * b_norm or r_norm <= tol:
            break

        # -- inner LGMRES iteration
        v0 = -psolve(r_outer)
        inner_res_0 = nrm2(v0)

        if inner_res_0 == 0:
            rnorm = nrm2(r_outer)
            raise RuntimeError("Preconditioner returned a zero vector; "
                               "|v| ~ %.1g, |M v| = 0" % rnorm)

        v0 = scal(1.0 / inner_res_0, v0)

        try:
            Q, R, B, vs, zs, y = _fgmres(matvec,
                                         v0,
                                         inner_m,
                                         lpsolve=psolve,
                                         atol=tol * b_norm / r_norm,
                                         outer_v=outer_v,
                                         prepend_outer_v=prepend_outer_v)
            y *= inner_res_0
            if not np.isfinite(y).all():
                # Overflow etc. in computation. There's no way to
                # recover from this, so we have to bail out.
                raise LinAlgError()
        except LinAlgError:
            # Floating point over/underflow, non-finite result from
            # matmul etc. -- report failure.
            return postprocess(x), k_outer + 1

        # -- GMRES terminated: eval solution
        dx = zs[0] * y[0]
        for w, yc in zip(zs[1:], y[1:]):
            dx = axpy(w, dx, dx.shape[0], yc)  # dx += w*yc

        # -- Store LGMRES augmentation vectors
        nx = nrm2(dx)
        if nx > 0:
            if store_outer_Av:
                q = Q.dot(R.dot(y))
                ax = vs[0] * q[0]
                for v, qc in zip(vs[1:], q[1:]):
                    ax = axpy(v, ax, ax.shape[0], qc)
                outer_v.append((dx / nx, ax / nx))
            else:
                outer_v.append((dx / nx, None))

        # -- Retain only a finite number of augmentation vectors
        while len(outer_v) > outer_k:
            del outer_v[0]

        # -- Apply step
        x += dx
    else:
        # didn't converge ...
        return postprocess(x), maxiter

    return postprocess(x), 0
