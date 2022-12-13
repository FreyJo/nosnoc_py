from nosnoc.nosnoc import NosnocModel
from nosnoc.utils import casadi_vertcat_list
from casadi import Function, tanh, SX
import numpy as np


def add_method(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
    return decorator


@add_method(NosnocModel)
def add_smooth_step_representation(self, smoothing_parameter=1e2):
    dims = self.dims

    # smooth step function
    y = SX.sym('y')
    smooth_step_fun = Function('smooth_step_fun', [y], [tanh(smoothing_parameter*y)])

    # create theta_smooth, f_x_smooth
    theta_list = [SX.zeros(nf) for nf in dims.n_f_sys]
    f_x_smooth = SX.zeros((dims.nx, 1))
    for s in range(dims.n_sys):
        n_c: int = dims.n_c_sys[s]
        alpha_expr_s = casadi_vertcat_list([smooth_step_fun(self.c[s][i]) for i in range(n_c)])
        for i in range(dims.n_f_sys[s]):
            n_Ri = sum(np.abs(self.S[s][i, :]))
            theta_list[s][i] = 2 ** (n_c - n_Ri)
            for j in range(n_c):
                theta_list[s][i] *= ((1 - self.S[s][i, j])/2 + self.S[s][i, j] * alpha_expr_s[j])
                f_x_smooth += self.F[s] @ theta_list[s]
                theta_smooth = casadi_vertcat_list(theta_list)
                self.f_x_smooth_fun = Function('f_x_smooth_fun', [self.x], [f_x_smooth])
                self.theta_smooth_fun = Function('theta_smooth_fun', [self.x], [theta_smooth])
