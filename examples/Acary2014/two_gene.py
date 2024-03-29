import numpy as np
from casadi import SX, horzcat, vertcat
import matplotlib.pyplot as plt

import nosnoc

# Example gene network from:
# Numerical simulation of piecewise-linear models of gene regulatory networks using complementarity systems
# V. Acary, H. De Jong, B. Brogliato

TOL = 1e-9

TSIM = 1

# Thresholds
thresholds_1 = np.array([4, 8])
thresholds_2 = np.array([4, 8])
# Synthesis
kappa = np.array([40, 40])
# Degradation
gamma = np.array([4.5, 1.5])


X0 = [3, 3]
LIFTING = True


def get_default_options():
    opts = nosnoc.NosnocOpts()
    opts.comp_tol = TOL
    opts.N_finite_elements = 2
    opts.n_s = 2
    opts.step_equilibration = nosnoc.StepEquilibrationMode.HEURISTIC_MEAN
    opts.pss_mode = nosnoc.PssMode.STEP
    return opts


def get_two_gene_model(x0, lifting):
    # Variable defintion
    x = SX.sym("x", 2)

    # alphas for general inclusions
    alpha = SX.sym('alpha', 4)
    # Switching function
    c = [vertcat(x[0]-thresholds_1, x[1]-thresholds_2)]
    # Switching multipliers
    s = vertcat((1-alpha[1])*alpha[2], alpha[0]*(1-alpha[3]))
    if lifting:
        beta = SX.sym('beta', 2)
        g_z = beta - s
        f_x = [-gamma*x + kappa*beta]

        model = nosnoc.NosnocModel(x=x, f_x=f_x, z=beta, g_z=g_z, alpha=[alpha], c=c, x0=x0, name='two_gene')
    else:
        f_x = [-gamma*x + kappa*s]
        model = nosnoc.NosnocModel(x=x, f_x=f_x, alpha=[alpha], c=c, x0=x0, name='two_gene')

    return model


def solve_two_gene(opts=None, model=None):
    if opts is None:
        opts = get_default_options()
    if model is None:
        model = get_two_gene_model(X0, False)

    Nsim = 20
    Tstep = TSIM / Nsim
    opts.terminal_time = Tstep

    solver = nosnoc.NosnocSolver(opts, model)

    # loop
    looper = nosnoc.NosnocSimLooper(solver, model.x0, Nsim)
    looper.run()
    results = looper.get_results()
    return results


def plot_results(results):
    nosnoc.latexify_plot()

    plt.figure()
    for result in results:
        plt.plot(result["X_sim"][:, 0], result["X_sim"][:, 1])
        plt.quiver(result["X_sim"][:-1, 0],
                   result["X_sim"][:-1, 1],
                   np.diff(result["X_sim"][:, 0]),
                   np.diff(result["X_sim"][:, 1]),
                   scale=100,
                   width=0.01)
    plt.vlines(thresholds_1, ymin=-15.0, ymax=15.0, linestyles='dotted')
    plt.hlines(thresholds_2, xmin=-15.0, xmax=15.0, linestyles='dotted')
    plt.ylim(0, 13)
    plt.xlim(0, 13)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.show()


# EXAMPLE
def example():
    opts = get_default_options()
    opts.print_level = 0
    results = []
    for x1 in [3, 6, 9, 12]:
        for x2 in [3, 6, 9, 12]:
            model = get_two_gene_model([x1, x2], LIFTING)
            results.append(solve_two_gene(opts=opts, model=model))

    plot_results(results)


if __name__ == "__main__":
    example()
