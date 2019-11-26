Background
==========

We can solve any system of PDEs of the form:

.. math::

    \frac{\partial\mathbf{Q}}{\partial t} & +\frac{\partial}{\partial x_{1}}\mathbf{F}_{1}\left(\mathbf{Q},\frac{\partial\mathbf{Q}}{\partial x_{1}},\ldots,\frac{\partial\mathbf{Q}}{\partial x_{n}}\right)+\cdots+\frac{\partial}{\partial x_{n}}\mathbf{F}_{n}\left(\mathbf{Q},\frac{\partial\mathbf{Q}}{\partial x_{1}},\ldots,\frac{\partial\mathbf{Q}}{\partial x_{n}}\right)\\
    & +B_{1}\left(\mathbf{Q}\right)\frac{\partial\mathbf{Q}}{\partial x_{1}}+\cdots+B_{n}\left(\mathbf{Q}\right)\frac{\partial\mathbf{Q}}{\partial x_{n}}\\
    & =\mathbf{S}\left(\mathbf{Q}\right)

or, more succinctly:

.. math::

    \frac{\partial\mathbf{Q}}{\partial t}+\nabla\mathbf{F}\left(\mathbf{Q},\nabla\mathbf{Q}\right)+B\left(\mathbf{Q}\right)\cdot\nabla\mathbf{Q}=\mathbf{S}\left(\mathbf{Q}\right)

See :doc:`examples of such systems<example_pdes>` .

If you give the values of :math:`\mathbf{Q}` at time :math:`t=0` on a
rectangular domain in :math:`\mathbb{R}^n`, then PyPDE will calculate
:math:`\mathbf{Q}` on the domain at a later time :math:`t_f` that you specify.

The boundary conditions at the edges of the domain can be either transitive or
periodic.