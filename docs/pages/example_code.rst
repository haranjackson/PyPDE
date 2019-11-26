Example Code
============

See more examples `here
<https://github.com/haranjackson/PyPDE/tree/master/pypde/tests>`_.

Reactive Euler
--------------

We must define our fluxes and source vector:

.. code-block::

    from numba import njit
    from numpy import zeros

    # material constants
    Qc = 1
    cv = 2.5
    Ti = 0.25
    K0 = 250
    γ = 1.4

    def F(Q, i):

        ρ = Q[0]
        E = Q[1] / ρ
        v = Q[2:5] / ρ
        λ = Q[5] / ρ

        # internal energy
        e = E - (v[0]**2 + v[1]**2 + v[2]**2) / 2 - Qc * (λ - 1)

        # pressure
        p = (γ - 1) * ρ * e

        ret = v[i] * Q
        ret[1] += p * v[i]
        ret[2 + i] += p

        return ret

    @njit
    def reaction_rate(E, v, λ):

        e = internal_energy(E, v, λ)
        T = e / cv

        return K0 if T > Ti else 0

    def S(Q):

        ret = zeros(6)

        ρ = Q[0]
        E = Q[1] / ρ
        v = Q[2:5] / ρ
        λ = Q[5] / ρ

        ret[5] = -ρ * λ * reaction_rate(E, v, λ)

        return ret

Under the Reactive Euler model, :math:`\mathbf{F}_i` has no
:math:`\nabla\mathbf{Q}` dependence, thus ``F`` here has call signature
``(Q, i)``. Note that any functions called by ``F`` or ``S`` must be decorated
with ``@nit``, as must any functions that they subsequently call.

The numba library is used to compile ``F`` and ``S`` before solving the system.
numba is able to compile `some numpy functions
<https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html>`_. As a
general rule though, you should aim to write your functions in pure Python, with
no classes. This is guaranteed to compile. It will not produce the performance
hit usually associated with Python loops and other features.

We now set out the initial conditions for the 1D detonation wave test. We use
400 cells, with a domain length of 1. The test is run to a final time of 0.5.

.. code-block::

    def energy(ρ, p, v, λ):
        return p / ((γ - 1) * ρ) + inner(v, v) / 2 + Qc * (λ - 1)

    nx = 400
    L = [1.]
    tf = 0.5

    ρL = 1.4
    pL = 1
    vL = [0, 0, 0]
    λL = 0
    EL = energy(ρL, pL, vL, λL)

    ρR = 0.887565
    pR = 0.191709
    vR = [-0.57735, 0, 0]
    λR = 1
    ER = energy(ρR, pR, vR, λR)

    QL = ρL * ([1, EL] + vL)
    QR = ρR * ([1, ER] + vR)

    u = zeros([nx, 6])
    for i in range(nx):
        if i / nx < 0.25:
            u[i] = QL
        else:
            u[i] = QR

We now solve the system. ``pde_solver`` returns an array ``ret`` of shape
:math:`100\times nx\times 5`. ``ret[j]`` corresponds to the domain at j% through
the simulation. We plot the final state of the domain for variable 0 (density):

.. code-block::

    import matplotlib.pyplot as plt

    from pypde import pde_solver

    ret = pde_solver(u, tf, L, F=F, S=S)

    plt.plot(ret[-1, :, 0])
    plt.show()

The plot is found below, in accordance with accepted numerical results:

.. image:: https://github.com/haranjackson/PyPDE/raw/master/docs/images/ReactiveEulerDetonation.png
   :width: 360px
   :alt: Reactive Euler detonation wave
   :align: center