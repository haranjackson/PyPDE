Example PDEs
============

The following PDEs all have the form solvable by PyPDE:

.. math::

    \frac{\partial\mathbf{Q}}{\partial t}+\nabla\mathbf{F}\left(\mathbf{Q},\nabla\mathbf{Q}\right)+B\left(\mathbf{Q}\right)\cdot\nabla\mathbf{Q}=\mathbf{S}\left(\mathbf{Q}\right)

3D Navier-Stokes
----------------

.. math::

    \mathbf{Q}=\left(\begin{array}{c}
    \rho\\
    \rho E\\
    \rho v_{1}\\
    \rho v_{2}\\
    \rho v_{3}
    \end{array}\right)\quad\mathbf{F}_{i}=\left(\begin{array}{c}
    \rho v_{i}\\
    \rho Ev_{i}+\mathbf{\Sigma_{i}}\cdot\mathbf{v}\\
    \rho v_{i}v_{1}+\mathbf{\Sigma_{i1}}\\
    \rho v_{i}v_{2}+\mathbf{\Sigma_{i2}}\\
    \rho v_{i}v_{3}+\mathbf{\Sigma_{i3}}
    \end{array}\right)\quad B_{i}=0\quad\mathbf{S}=0

where:

.. math::

    \Sigma=pI-\mu\left(\nabla\mathbf{v}+\nabla\mathbf{v}^{T}-\frac{2}{3}tr\left(\nabla\mathbf{v}\right)I\right)

2D Reactive Euler
-----------------

.. math::

    \mathbf{Q}=\left(\begin{array}{c}
    \rho\\
    \rho E\\
    \rho v_{1}\\
    \rho v_{2}\\
    \rho\lambda
    \end{array}\right)\quad\mathbf{F}_{i}=\left(\begin{array}{c}
    \rho v_{i}\\
    \left(\rho E+p\right)v_{i}\\
    \rho v_{i}v_{1}+\delta_{i1}p\\
    \rho v_{i}v_{2}+\delta_{i2}p\\
    \rho v_{i}\lambda
    \end{array}\right)\quad B_{i}=0\quad\mathbf{S}=\left(\begin{array}{c}
    0\\
    0\\
    0\\
    0\\
    -\rho\lambda K\left(T\right)
    \end{array}\right)

where :math:`K` is a (potentially large) function depending on temperature
:math:`T`.

3D Godunov-Romenski
-------------------

.. math::

    \mathbf{Q}=\begin{pmatrix}\rho\\
    \rho E\\
    \rho v_{1}\\
    \rho v_{2}\\
    \rho v_{3}\\
    A_{11}\\
    A_{12}\\
    A_{13}\\
    A_{21}\\
    A_{22}\\
    A_{23}\\
    A_{31}\\
    A_{32}\\
    A_{33}
    \end{pmatrix}\quad\mathbf{F}_{i}=\begin{pmatrix}\rho v_{i}\\
    \rho Ev_{i}+\mathbf{\Sigma_{i}}\cdot\mathbf{v}\\
    \rho v_{i}v_{1}+\mathbf{\Sigma_{i1}}\\
    \rho v_{i}v_{2}+\mathbf{\Sigma_{i2}}\\
    \rho v_{i}v_{3}+\mathbf{\Sigma_{i3}}\\
    \delta_{i1}\mathbf{A_{1}}\cdot\mathbf{v}\\
    \delta_{i2}\mathbf{A_{1}}\cdot\mathbf{v}\\
    \delta_{i3}\mathbf{A_{1}}\cdot\mathbf{v}\\
    \delta_{i1}\mathbf{A_{2}}\cdot\mathbf{v}\\
    \delta_{i2}\mathbf{A_{2}}\cdot\mathbf{v}\\
    \delta_{i3}\mathbf{A_{2}}\cdot\mathbf{v}\\
    \delta_{i1}\mathbf{A_{3}}\cdot\mathbf{v}\\
    \delta_{i2}\mathbf{A_{3}}\cdot\mathbf{v}\\
    \delta_{i3}\mathbf{A_{3}}\cdot\mathbf{v}
    \end{pmatrix}\quad B_{i}=v_{i}I_{14}-\left(\begin{array}{cccc}
    0_{5} & 0_{3} & 0_{3} & 0_{3}\\
    0_{3} & \delta_{i1}v_{1}I_{3} & \delta_{i1}v_{2}I_{3} & \delta_{i1}v_{3}I_{3}\\
    0_{3} & \delta_{i2}v_{1}I_{3} & \delta_{i2}v_{2}I_{3} & \delta_{i2}v_{3}I_{3}\\
    0_{3} & \delta_{i3}v_{1}I_{3} & \delta_{i3}v_{2}I_{3} & \delta_{i3}v_{3}I_{3}
    \end{array}\right)\quad\mathbf{S}=-\frac{1}{\theta_{1}}\left(\begin{array}{c}
    \mathbf{0_{5}}\\
    \mathbf{\frac{\partial E}{\partial A}_{1}}\\
    \mathbf{\frac{\partial E}{\partial A}_{2}}\\
    \mathbf{\frac{\partial E}{\partial A}_{3}}
    \end{array}\right)

where :math:`\theta` is a (potentially very small) function of :math:`A`, and
now:

.. math::

    \Sigma=pI+\rho A^{T}\frac{\partial E}{\partial A}
