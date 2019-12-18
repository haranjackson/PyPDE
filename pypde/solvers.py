from ctypes import POINTER, c_double, c_int
from multiprocessing import cpu_count

from numpy import array, concatenate, int32, zeros

from pypde.cfuncs import generate_cfuncs
from pypde.utils import (c_ptr, create_solver, get_cdll, nargs,
                         parse_boundary_types)

FLUXES = {'rusanov': 0, 'roe': 1, 'osher': 2}


def pde_solver(Q0,
               tf,
               L,
               F=None,
               B=None,
               S=None,
               boundaryTypes='transitive',
               cfl=0.9,
               order=2,
               ndt=100,
               flux='rusanov',
               stiff=True,
               nThreads=-1):
    """

    |

    Solves PDEs of the following form:

    .. math::

     \\frac{\partial\mathbf{Q}}{\partial t}+\\frac{\partial}{\partial x_{1}}\mathbf{F}_{1}\left(\mathbf{Q},\\frac{\partial\mathbf{Q}}{\partial x_{1}},\ldots,\\frac{\partial\mathbf{Q}}{\partial x_{n}}\\right)+\cdots+\\frac{\partial}{\partial x_{n}}\mathbf{F}_{n}\left(\mathbf{Q},\\frac{\partial\mathbf{Q}}{\partial x_{1}},\ldots,\\frac{\partial\mathbf{Q}}{\partial x_{n}}\\right)+B_{1}\left(\mathbf{Q}\\right)\\frac{\partial\mathbf{Q}}{\partial x_{1}}+\cdots+B_{n}\left(\mathbf{Q}\\right)\\frac{\partial\mathbf{Q}}{\partial x_{n}}=\mathbf{S}\left(\mathbf{Q}\\right)

    or, more succinctly:

    .. math::

     \\frac{\partial\mathbf{Q}}{\partial t}+\\nabla\mathbf{F}\left(\mathbf{Q},\\nabla\mathbf{Q}\\right)+B\left(\mathbf{Q}\\right)\cdot\\nabla\mathbf{Q}=\mathbf{S}\left(\mathbf{Q}\\right)

    where :math:`\mathbf{Q},\mathbf{F}_i,\mathbf{S}` are vectors of
    :math:`n_{var}` variables and :math:`B_i` are matrices of shape
    :math:`\left(n_{var}\\times n_{var}\\right)`.
    Of course, :math:`\mathbf{F},\mathbf{B},\mathbf{S}` can be 0.

    Define
    :math:`\Omega=\left[0,L_1\\right]\\times\cdots\\times\left[0,L_n\\right]`.
    Given :math:`\mathbf{Q}\left(\mathbf{x},0\\right)` for
    :math:`\mathbf{x}\in\Omega`, ``pde_solver`` finds
    :math:`\mathbf{Q}\left(\mathbf{x},t\\right)` for any :math:`t>0`.

    Taking integers :math:`m_1,\ldots,m_n>0` we split :math:`\Omega` into
    :math:`\left(m_1\\times\cdots\\times m_n\\right)` cells with volume
    :math:`dx_1 dx_2 \ldots dx_n` where :math:`dx_i=\\frac{L_i}{m_i}`.

    |

    Parameters
    ----------
    Q0 : ndarray

        An array with shape :math:`\left(m_1,\ldots m_n,n_{var}\\right)`.

        ``Q0[i1, i2, ..., in, j]`` is equal to:

        .. math::

         \mathbf{Q}_{j+1}\left(\\frac{\left(i_1+0.5\\right) dx_1}{m_1},\ldots,\\frac{\left(i_n+0.5\\right) dx_n}{m_n}, 0\\right)

    tf : double

        The final time at which to return the value of
        :math:`\mathbf{Q}\left(\mathbf{x},t\\right)`.

    L : ndarray or list

        An array of length :math:`n`, ``L[i]`` is equal to :math:`L_{i+1}`.

    F : callable, optional

        The flux terms, with signature ``F(Q, DQ, d) -> ndarray``,
        corresponding to
        :math:`\mathbf{F}_{d+1}\left(\mathbf{Q},\\nabla\mathbf{Q}\\right)`.

        If :math:`\mathbf{F}` has no :math:`\\nabla\mathbf{Q}` dependence, the
        signature may be ``F(Q, d) -> ndarray``, corresponding to
        :math:`\mathbf{F}_{d+1}\left(\mathbf{Q}\\right)`.

        If :math:`n=1` and :math:`\mathbf{F}` has no :math:`\\nabla\mathbf{Q}`
        dependence, the signature may be ``F(Q) -> ndarray``, corresponding
        to :math:`\mathbf{F}_1\left(\mathbf{Q}\\right)`.

            - ``Q`` is a 1-D array with shape :math:`\left(n_{var},\\right)`
            - ``DQ`` is an 2-D array with shape :math:`\left(n,n_{var}\\right)` \
              (as ``DQ[i]`` is equal to \
              :math:`\\frac{\partial\mathbf{Q}}{\partial x_{i}}`)
            - ``d`` is an integer (ranging from :math:`0` to :math:`n-1`)
            - the returned array has shape :math:`\left(n_{var},\\right)`

    B : callable, optional

        The non-conservative terms, with signature ``B(Q, d) -> ndarray``,
        corresponding to :math:`\mathbf{B}_{d+1}\left(\mathbf{Q}\\right)`.

        If :math:`n=1`, the signature may be ``B(Q) -> ndarray``, corresponding
        to :math:`\mathbf{B}_1\left(\mathbf{Q}\\right)`.

            - ``Q`` is a 1-D array with shape :math:`\left(n_{var},\\right)`
            - ``d`` is an integer (ranging from :math:`0` to :math:`n-1`)
            - the returned array has shape :math:`\left(n_{var},n_{var}\\right)`

    S : callable, optional

        The source terms, with signature ``B(Q, d) -> ndarray``, corresponding
        to :math:`\mathbf{S}\left(\mathbf{Q}\\right)`.

            - ``Q`` is a 1-D array with shape :math:`\left(n_{var},\\right)`
            - the returned array has shape :math:`\left(n_{var},\\right)`

    boundaryTypes : string or list, optional

        - If a string, must be one of ``'transitive'``, ``'periodic'``. In this
          case, all boundaries will take the stated form.

        - If a list, must have length :math:`n`, containing strings taken from
          ``'transitive'``, ``'periodic'``. In this case, the first element of
          the list describes the boundaries at :math:`x_1=0,L_1`, the second
          describes the boundaries at :math:`x_2=0,L_2`, etc.

    cfl : double, optional

        The CFL number: :math:`0<CFL<1` (default 0.9).

    order : int, optional

        The order of the polynomial reconstructions used in the solver, must be
        an integer greater than 0 (default 2)

    ndt : int, optional

        The number of timesteps at which to return the value of the grid
        (default 100) e.g. if ``tf=5``, and ``ndt=4`` then the value of the grid
        will be returned at ``t=1.25,2.5,3.75,5``.

    flux : string, optional

        The kind of flux to use at cell boundaries (default ``'rusanov'``)
        Must be one of ``'rusanov'``, ``'roe'``, ``'osher'``.

    stiff : bool, optional

        Whether to use a stiff solver for the Discontinuous Galerkin step (default True)
        If the equations are stiff (i.e. the source terms are much larger than
        the other terms), then this is probably required to make the method
        converge. Otherwise, it can be turned off to improve speed.

    nThreads : int, optional

        The number of threads to use in the solver (default -1). If less than 1,
        the thread count will default to (number of cores) - 1.

    Returns
    -------
    out : ndarray

        An array with shape :math:`\left(ndt,m_1,\ldots m_n,n_{var}\\right)`.

        Defining :math:`dt=\\frac{tf}{ndt}`, then ``out[i, i1, i2, ..., in, j]``
        is equal to:

        .. math::

         \mathbf{Q}_{j+1}\left(\\frac{\left(i_1+0.5\\right) dx_1}{m_1},\ldots,\\frac{\left(i_n+0.5\\right) dx_n}{m_n}, \left(i+1\\right) dt\\right)
    """

    nX = array(Q0.shape[:-1], dtype='int32')
    ndim = len(nX)
    V = Q0.shape[-1]
    dX = array([L[i] / nX[i] for i in range(len(L))])

    boundaryTypes = parse_boundary_types(boundaryTypes, ndim)

    useF = False if F is None else True
    if F is None:
        F = lambda Q, DQ, d: zeros(V)

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

    ret = zeros(ndt * Q0.size)
    ur = Q0.ravel()

    if nThreads < 1:
        nThreads = cpu_count() - 1

    solver(_F.ctypes, _B.ctypes, _S.ctypes, useF, useB, useS, c_ptr(ur), tf,
           c_ptr(nX), ndim, c_ptr(dX), cfl, c_ptr(boundaryTypes), stiff,
           FLUXES[flux], order, V, ndt, secondOrder, c_ptr(ret), nThreads)

    return ret.reshape((ndt, ) + Q0.shape)


def weno_solver(u, order=2):

    nX = array(u.shape[:-1], dtype=int32)
    ndim = len(nX)
    V = u.shape[-1]

    nXret = nX - 2 * (order - 1)

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
    ret = zeros(ncellRet * order**ndim * V)
    ur = u.ravel()

    solver(c_ptr(ret), c_ptr(ur), c_ptr(nX), ndim, order, V)

    return ret.reshape(concatenate([nXret, [order] * ndim, [V]]))
