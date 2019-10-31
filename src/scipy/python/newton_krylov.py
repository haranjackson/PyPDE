from scipy.linalg import norm
from scipy.linalg import qr, lstsq
from numpy import abs, array, dot, finfo, inf, insert, ones, sqrt, zeros

from numpy import cos, sin, tan


EPS = finfo('d').eps


def qr_insert2(A, u, i):
    A1 = insert(A, i, u, 1)
    return qr(A1)

def lgmres(matvec, b, tol):

    inner_m = 30
    maxiter = 1
    outer_k = 10

    N = b.size
    outer_v = []
    x = zeros(N)

    b_norm = norm(b)
    if b_norm == 0:
        b_norm = 1

    for k_outer in range(maxiter):
        r_outer = matvec(x) - b

        r_norm = norm(r_outer)
        if r_norm <= tol * b_norm or r_norm <= tol:
            break

        vs0 = -r_outer
        inner_res_0 = norm(vs0)

        vs0 /= inner_res_0
        vs = [vs0]
        ws = []

        Q = ones((1, 1))
        R = zeros((1, 0))

        for j in range(1, 1 + inner_m + len(outer_v)):

            if j < len(outer_v) + 1:
                z = outer_v[j-1]
            elif j == len(outer_v) + 1:
                z = vs0
            else:
                z = vs[-1]

            v_new = matvec(z)
            v_new_norm = norm(v_new)

            hcur = zeros(j+1)
            for i in range(len(vs)):
                v = vs[i]
                alpha = dot(v, v_new)
                hcur[i] = alpha
                v_new = v_new - alpha * v
            hcur[-1] = norm(v_new)

            v_new /= hcur[-1]

            vs.append(v_new)
            ws.append(z)

            Q2 = zeros((j+1, j+1), order='F')
            Q2[:j,:j] = Q
            Q2[j,j] = 1

            R2 = zeros((j+1, j-1), order='F')
            R2[:j,:] = R

            Q, R = qr_insert2(dot(Q2,R2), hcur, j-1)

            inner_res = abs(Q[0,-1]) * inner_res_0

            if (inner_res <= tol * inner_res_0) or (hcur[-1] <= EPS * v_new_norm):
                break

        y, _, _, _, = lstsq(R[:j,:j], Q[0,:j])
        y *= inner_res_0

        dx = ws[0]*y[0]
        for i in range(1,len(y)):
            dx += y[i] * ws[i]

        nx = norm(dx)
        if nx > 0:
            outer_v.append(dx/nx)

        outer_v = outer_v[-outer_k:]

        x += dx

    return x

def maxnorm(x):
    return abs(x).max()

def nonlin_solve(F, x0, f_tol=EPS**(1./3), f_rtol=inf, x_tol=inf, x_rtol=inf,
                 norm=maxnorm):

    condition = TerminationCondition(f_tol, f_rtol, x_tol, x_rtol, norm)

    gamma = 0.9
    eta_max = 0.9999
    eta_treshold = 0.1
    eta = 1e-3

    func = F
    x = x0.flatten()

    dx = inf
    Fx = func(x)
    Fx_norm = norm(Fx)

    jacobian = KrylovJacobian(x.copy(), Fx, func)

    maxiter = 100*(x.size+1)

    for n in range(maxiter):

        print(n)

        if condition.check(Fx, x, dx):
            break

        tol = min(eta, eta*Fx_norm)
        dx = -jacobian.solve(Fx, tol=tol)

        s, x, Fx, Fx_norm_new = _nonlin_line_search(func, x, Fx, dx)

        jacobian.update(x.copy(), Fx)

        # Adjust forcing parameters for inexact methods
        eta_A = gamma * Fx_norm_new**2 / Fx_norm**2
        if gamma * eta**2 < eta_treshold:
            eta = min(eta_max, eta_A)
        else:
            eta = min(eta_max, max(eta_A, gamma*eta**2))

        Fx_norm = Fx_norm_new

    return x

def phi(s, tmp_s, tmp_phi, tmp_Fx, func, x, dx):
    if s == tmp_s[0]:
        return tmp_phi[0]
    xt = x + s*dx
    v = func(xt)
    p = norm(v)**2
    tmp_s[0] = s
    tmp_phi[0] = p
    tmp_Fx[0] = v
    return p

def scalar_search_armijo(phi0, derphi0, tmp_s, tmp_phi, tmp_Fx, func, x, dx):

    c1 = 1e-4

    amin = 1e-2
    phi_a0 = phi(1, tmp_s, tmp_phi, tmp_Fx, func, x, dx)
    if phi_a0 <= phi0 + c1*derphi0:
        return 1, phi_a0

    alpha1 = -(derphi0) / 2.0 / (phi_a0 - phi0 - derphi0)
    phi_a1 = phi(alpha1, tmp_s, tmp_phi, tmp_Fx, func, x, dx)

    if (phi_a1 <= phi0 + c1*alpha1*derphi0):
        return alpha1, phi_a1

    while alpha1 > amin:
        factor = alpha1**2 * (alpha1-1)
        a = phi_a1 - phi0 - derphi0*alpha1 - alpha1**2 * (phi_a0 - phi0 - derphi0)
        a = a / factor
        b = -(phi_a1 - phi0 - derphi0*alpha1) + alpha1**3 * (phi_a0 - phi0 - derphi0)
        b = b / factor

        alpha2 = (-b + sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2, tmp_s, tmp_phi, tmp_Fx, func, x, dx)

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    return 1.0, phi_a1

def _nonlin_line_search(func, x, Fx, dx):

    tmp_s = [0]
    tmp_Fx = [Fx]
    tmp_phi = [norm(Fx)**2]

    s, phi1 = scalar_search_armijo(tmp_phi[0], -tmp_phi[0],
                                   tmp_s, tmp_phi, tmp_Fx, func, x, dx)

    x = x + s*dx
    if s == tmp_s[0]:
        Fx = tmp_Fx[0]
    else:
        Fx = func(x)
    Fx_norm = norm(Fx)

    return s, x, Fx, Fx_norm


class TerminationCondition(object):

    def __init__(self, f_tol, f_rtol, x_tol, x_rtol, norm):

        self.x_tol = x_tol
        self.x_rtol = x_rtol
        self.f_tol = f_tol
        self.f_rtol = f_rtol
        self.norm = norm

        self.f0_norm = None

    def check(self, f, x, dx):
        f_norm = self.norm(f)
        x_norm = self.norm(x)
        dx_norm = self.norm(dx)

        if self.f0_norm is None:
            self.f0_norm = f_norm

        if f_norm == 0:
            return 1

        # NB: condition must succeed for rtol=inf even if norm == 0
        return int((f_norm <= self.f_tol
                    and f_norm/self.f_rtol <= self.f0_norm)
                   and (dx_norm <= self.x_tol
                        and dx_norm/self.x_rtol <= x_norm))

class KrylovJacobian(object):

    def __init__(self, x, f, func):
        self.func = func
        self.shape = (f.size, x.size)
        self.x0 = x
        self.f0 = f
        self.rdiff = EPS ** (1./2)
        self._update_diff_step()

    def _update_diff_step(self):
        mx = abs(self.x0).max()
        mf = abs(self.f0).max()
        self.omega = self.rdiff * max(1, mx) / max(1, mf)

    def matvec(self, v):
        nv = norm(v)
        if nv == 0:
            return 0*v
        sc = self.omega / nv
        r = (self.func(self.x0 + sc*v) - self.f0) / sc
        return r

    def solve(self, rhs, tol=0):
        return lgmres(self.matvec, rhs, tol)

    def update(self, x, f):
        self.x0 = x
        self.f0 = f
        self._update_diff_step()


if __name__ == "__main__":

    def F(x):
        return array([sin(x[0]), cos(x[1]), tan(x[2]),
                         x[3]**2-x[3]])

    x0 = array([1., 2., 3., 5.])
    print(nonlin_solve(F, x0))