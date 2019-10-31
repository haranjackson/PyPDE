import numpy as np
from scipy.linalg import norm
from scipy.sparse.linalg import lgmres


def maxnorm(x):
    return np.absolute(x).max()


class KrylovJacobian(object):

    def __init__(self, x, f, func):

        self.method_kw = {}
        self.method_kw['outer_k'] = 10
        self.method_kw['maxiter'] = 1
        self.method_kw.setdefault('outer_v', [])
        self.method_kw.setdefault('prepend_outer_v', True)
        self.method_kw.setdefault('store_outer_Av', False)

        self.func = func
        self.shape = (f.size, x.size)
        self.dtype = f.dtype

        self.x0 = x
        self.f0 = f

        self.rdiff = np.finfo(x.dtype).eps ** (1. / 2)

        self._update_diff_step()

    def _update_diff_step(self):
        mx = abs(self.x0).max()
        mf = abs(self.f0).max()
        self.omega = self.rdiff * max(1, mx) / max(1, mf)

    def matvec(self, v):
        nv = norm(v)
        if nv == 0:
            return 0 * v
        sc = self.omega / nv
        r = (self.func(self.x0 + sc * v) - self.f0) / sc
        return r

    def solve(self, rhs, tol):
        sol, info = lgmres(self, rhs, tol=tol, **self.method_kw)
        return sol

    def update(self, x, f):
        self.x0 = x
        self.f0 = f
        self._update_diff_step()


def _nonlin_line_search(func, x, Fx, dx):

    x = x + dx
    Fx = func(x)
    Fx_norm = norm(Fx)
    return x, Fx, Fx_norm


class TerminationCondition(object):

    def __init__(self, f_tol, f_rtol, x_tol, x_rtol):

        if f_tol is None:
            f_tol = np.finfo(np.float_).eps ** (1. / 3)
        if f_rtol is None:
            f_rtol = np.inf
        if x_tol is None:
            x_tol = np.inf
        if x_rtol is None:
            x_rtol = np.inf

        self.x_tol = x_tol
        self.x_rtol = x_rtol
        self.f_tol = f_tol
        self.f_rtol = f_rtol

        self.f0_norm = None

    def check(self, f, x, dx):

        f_norm = maxnorm(f)
        x_norm = maxnorm(x)
        dx_norm = maxnorm(dx)

        if self.f0_norm is None:
            self.f0_norm = f_norm

        if f_norm == 0:
            return True

        return f_norm <= self.f_tol and f_norm / self.f_rtol <= self.f0_norm and \
            dx_norm <= self.x_tol and dx_norm / self.x_rtol <= x_norm


def nonlin_solve(F, x, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None):

    condition = TerminationCondition(f_tol, f_rtol, x_tol, x_rtol)

    dx = np.inf
    Fx = F(x)
    Fx_norm = norm(Fx)

    jacobian = KrylovJacobian(x.copy(), Fx, F)

    maxiter = 100 * (x.size + 1)

    # Solver tolerance selection
    gamma = 0.9
    eta_max = 0.9999
    eta_treshold = 0.1
    eta = 1e-3

    for n in range(maxiter):

        if condition.check(Fx, x, dx):
            break

        tol = min(eta, eta * Fx_norm)
        dx = -jacobian.solve(Fx, tol=tol)

        x, Fx, Fx_norm_new = _nonlin_line_search(F, x, Fx, dx)

        jacobian.update(x.copy(), Fx)

        eta_A = gamma * Fx_norm_new * Fx_norm_new / (Fx_norm * Fx_norm)
        if gamma * eta * eta < eta_treshold:
            eta = min(eta_max, eta_A)
        else:
            eta = min(eta_max, max(eta_A, gamma * eta * eta))

        Fx_norm = Fx_norm_new

    return x
