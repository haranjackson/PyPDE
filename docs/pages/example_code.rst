Example Code
============

See more examples `here
<https://github.com/haranjackson/PyPDE/tree/master/pypde/tests>`_.

Reactive Euler (1D, hyperbolic, S≠0)
------------------------------------

We must define our fluxes and source vector:

.. code-block:: python

    from numba import njit
    from numpy import zeros

    # material constants
    Qc = 1
    cv = 2.5
    Ti = 0.25
    K0 = 250
    gam = 1.4

    @njit
    def internal_energy(E, v, lam):
        return E - (v[0]**2 + v[1]**2 + v[2]**2) / 2 - Qc * (lam - 1)

    def F(Q, d):

        r = Q[0]
        E = Q[1] / r
        v = Q[2:5] / r
        lam = Q[5] / r

        e = internal_energy(E, v, lam)

        # pressure
        p = (gam - 1) * r * e

        F_ = v[d] * Q
        F_[1] += p * v[d]
        F_[2 + d] += p

        return F_

    @njit
    def reaction_rate(E, v, lam):

        e = internal_energy(E, v, lam)
        T = e / cv

        return K0 if T > Ti else 0

    def S(Q):

        S_ = zeros(6)

        r = Q[0]
        E = Q[1] / r
        v = Q[2:5] / r
        lam = Q[5] / r

        S_[5] = -r * lam * reaction_rate(E, v, lam)

        return S_

Under the Reactive Euler model, :math:`\mathbf{F}_i` has no
:math:`\nabla\mathbf{Q}` dependence, thus ``F`` here has call signature
``(Q, d)``. Note that any functions called by ``F`` or ``S`` must be decorated
with ``@nit``, as must any functions that they subsequently call.

The numba library is used to compile ``F`` and ``S`` before solving the system.
numba is able to compile `some numpy functions
<https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html>`_. As a
general rule though, you should aim to write your functions in pure Python, with
no classes. This is guaranteed to compile. It will not produce the performance
hit usually associated with Python loops and other features.

We now set out the initial conditions for the 1D detonation wave test. We use
400 cells, with a domain length of 1. The test is run to a final time of 0.5.

.. code-block:: python

    from numpy import inner, array

    def energy(r, p, v, lam):
        return p / ((gam - 1) * r) + inner(v, v) / 2 + Qc * (lam - 1)

    nx = 400
    L = [1.]
    tf = 0.5

    rL = 1.4
    pL = 1
    vL = [0, 0, 0]
    lamL = 0
    EL = energy(rL, pL, vL, lamL)

    rR = 0.887565
    pR = 0.191709
    vR = [-0.57735, 0, 0]
    lamR = 1
    ER = energy(rR, pR, vR, lamR)

    QL = rL * array([1, EL] + vL + [lamL])
    QR = rR * array([1, ER] + vR + [lamR])

    Q0 = zeros([nx, 6])
    for i in range(nx):
        if i / nx < 0.25:
            Q0[i] = QL
        else:
            Q0[i] = QR

We now solve the system. ``pde_solver`` returns an array ``out`` of shape
:math:`100\times nx\times 6`. ``out[j]`` corresponds to the domain at :math:`\left(j+1\right)\%` through
the simulation. We plot the final state of the domain for variable 0 (density):

.. code-block:: python

    import matplotlib.pyplot as plt

    from pypde import pde_solver

    out = pde_solver(Q0, tf, L, F=F, S=S, stiff=False, flux='roe', order=3)

    plt.plot(out[-1, :, 0])
    plt.show()

The plot is found below, in accordance with accepted numerical results:

.. image:: https://github.com/haranjackson/PyPDE/raw/master/docs/images/ReactiveEulerDetonation.png
   :width: 360px
   :alt: Reactive Euler detonation wave
   :align: center


Navier-Stokes (2D, parabolic)
-----------------------------

We must define our fluxes and source vector:

.. code-block:: python

    from numba import njit
    from numpy import dot, eye, zeros

    # material constants
    gam = 1.4
    mu = 1e-2


    @njit
    def sigma(dv):
        return mu * (dv + dv.T - 2 / 3 * (dv[0, 0] + dv[1, 1] + dv[2, 2]) * eye(3))


    @njit
    def pressure(r, E, v):
        return r * (gam - 1) * (E - dot(v, v) / 2)


    def F(Q, DQ, d):

        F_ = zeros(5)

        r = Q[0]
        E = Q[1] / r
        v = Q[2:5] / r

        dr_dx = DQ[0, 0]
        drv_dx = DQ[0, 2:5]
        dv_dx = (drv_dx - dr_dx * v) / r

        dv = zeros((3, 3))
        dv[0] = dv_dx

        p = pressure(r, E, v)
        sig = sigma(dv)

        vd = v[d]
        rvd = r * vd

        F_[0] = rvd
        F_[1] = rvd * E + p * vd
        F_[2:5] = rvd * v
        F_[2 + d] += p

        sigd = sig[d]
        F_[1] -= dot(sigd, v)
        F_[2:5] -= sigd

        return F_

Under the Navier-Stokes model, :math:`\mathbf{F}_i` has a
:math:`\nabla\mathbf{Q}` dependence, thus ``F`` here has call signature
``(Q, DQ, d)``. Note that any functions called by ``F`` must be decorated
with ``@nit``, as must any functions that they subsequently call.

The numba library is used to compile ``F`` before solving the system.
numba is able to compile `some numpy functions
<https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html>`_. As a
general rule though, you should aim to write your functions in pure Python, with
no classes. This is guaranteed to compile. It will not produce the performance
hit usually associated with Python loops and other features.

We now set out the initial conditions for the 2D Taylor-Green vortex test. We
use 50x50 cells, with a domain length of :math:`2\pi`. The test is run to a
final time of 1.

.. code-block:: python

    from numpy import cos, pi, sin

    def total_energy(r, p, v):
        return p / (r * (gam - 1)) + dot(v, v) / 2


    def make_Q(r, p, v):
        """ Returns the vector of conserved variables, given the primitive variables
        """
        Q = zeros(5)
        Q[0] = r
        Q[1] = r * total_energy(r, p, v)
        Q[2:5] = r * v
        return Q


    L = [2 * pi, 2 * pi]

    nx = 50
    ny = 50
    tf = 1

    C = 100 / gam
    r = 1
    v = zeros(3)

    u = zeros([nx, ny, 5])
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) * L[0] / nx
            y = (j + 0.5) * L[1] / ny
            v[0] = sin(x) * cos(y)
            v[1] = -cos(x) * sin(y)
            p = C + (cos(2 * x) + cos(2 * y)) / 4
            u[i, j] = make_Q(r, p, v)


We now solve the system. ``pde_solver`` returns an array ``out`` of shape
:math:`100\times nx\times ny\times 5`. ``out[j]`` corresponds to the domain at :math:`\left(j+1\right)\%` through
the simulation. We plot the final state of the domain for velocity:

.. code-block:: python

    import matplotlib.pyplot as plt
    from numpy import linspace

    from pypde import pde_solver

    out = pde_solver(u,
                    tf,
                    L,
                    F=F,
                    cfl=0.9,
                    order=2,
                    boundaryTypes='periodic')

    x = linspace(0, L[0], nx)
    y = linspace(0, L[1], ny)

    ut = out[-1, :, :, 2] / out[-1, :, :, 0]
    vt = out[-1, :, :, 3] / out[-1, :, :, 0]
    plt.streamplot(x, y, ut, vt)
    plt.show()

The plot is found below, in accordance with accepted numerical results:

.. image:: https://github.com/haranjackson/PyPDE/raw/master/docs/images/NavierStokesTaylorGreen.png
   :width: 360px
   :alt: Taylor-Green vortex under the Navier-Stokes model
   :align: center