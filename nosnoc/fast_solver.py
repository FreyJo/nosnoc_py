from typing import Optional, List
from abc import ABC, abstractmethod
import time
from copy import copy
from dataclasses import dataclass, field

import numpy as np
import casadi as ca

from nosnoc.problem import NosnocModel, NosnocOcp
from nosnoc.solver import NosnocSolverBase, get_results_from_primal_vector
from nosnoc.nosnoc_opts import NosnocOpts
from nosnoc.nosnoc_types import MpccMode, InitializationStrategy, CrossComplementarityMode, StepEquilibrationMode, PssMode, IrkRepresentation, HomotopyUpdateRule, ConstraintHandling
from nosnoc.utils import casadi_vertcat_list, casadi_sum_list, print_casadi_vector, casadi_length

def casadi_inf_norm_nan(x: ca.DM):
    norm = 0
    x = x.full().flatten()
    for i in range(len(x)):
        norm = max(norm, x[i])
    return norm


class NosnocFastSolver(NosnocSolverBase):
    def __init__(self, opts: NosnocOpts, model: NosnocModel, ocp: Optional[NosnocOcp] = None):
        super().__init__(opts, model, ocp)
        prob = self.problem

        # assume SCHOLTES_INEQ, or any mpcc_mode that transforms complementarities into inequalites.
        # complements
        G1 = casadi_vertcat_list([casadi_sum_list(x[0]) for x in prob.comp_list])
        G2 = casadi_vertcat_list([x[1] for x in prob.comp_list])
        # equalities
        H = casadi_vertcat_list( [prob.g[i] for i in range(len(prob.lbg)) if prob.lbg[i] == prob.ubg[i]] )

        n_comp = casadi_length(G1)
        n_H = casadi_length(H)

        # setup primal dual variables:
        lam_H = ca.SX.sym('lam_H', n_H)
        lam_comp = ca.SX.sym('lam_comp', n_comp)
        lam_pd = ca.vertcat(lam_H, lam_comp)
        self.lam_pd_0 = np.zeros((casadi_length(lam_pd),))

        mu_G1 = ca.SX.sym('mu_G1', n_comp)
        mu_G2 = ca.SX.sym('mu_G2', n_comp)
        mu_s = ca.SX.sym('mu_s', n_comp)
        mu_pd = ca.vertcat(mu_G1, mu_G2, mu_s)
        self.mu_pd_0 = np.ones((casadi_length(mu_pd),))

        slack = ca.SX.sym('slack', n_comp)
        w_pd = ca.vertcat(prob.w, lam_pd, slack, mu_pd)

        # slack = - G1 * G2 + sigma
        self.slack0_expr = -ca.diag(G1) @ G2 + prob.sigma
        self.slack0_fun = ca.Function('slack0_fun', [prob.w, prob.p], [self.slack0_expr])

        # barrier parameter
        tau = prob.tau
        sigma = prob.sigma

        # collect KKT system
        slacked_complementarity = slack - self.slack0_expr
        stationarity_w = ca.jacobian(H, prob.w).T @ lam_H + \
             ca.jacobian(slacked_complementarity, prob.w).T @ lam_comp \
             - ca.jacobian(G1, prob.w).T @ mu_G1 \
             - ca.jacobian(G2, prob.w).T @ mu_G2 \
             - ca.jacobian(slack, prob.w).T @ mu_s

        stationarity_s = ca.jacobian(H, slack).T @ lam_H + \
             ca.jacobian(slacked_complementarity, slack).T @ lam_comp \
             - ca.jacobian(G1, slack).T @ mu_G1 \
             - ca.jacobian(G2, slack).T @ mu_G2 \
             - ca.jacobian(slack, slack).T @ mu_s

        kkt_eq = stationarity_w
        kkt_eq = ca.vertcat(kkt_eq, stationarity_s)

        kkt_eq = ca.vertcat(kkt_eq, H)

        kkt_eq = ca.vertcat(kkt_eq, slacked_complementarity)
        # treat IP complementarities:
        #  (G1, mu_1), (G2, mu_2), (s, mu_s) via FISCHER_BURMEISTER
        for i in range(n_comp):
            kkt_eq = ca.vertcat(kkt_eq, G1[i] + mu_G1[i] - ca.sqrt(G1[i]**2 + mu_G1[i]**2 + 2*tau))
        for i in range(n_comp):
            kkt_eq = ca.vertcat(kkt_eq, G2[i] + mu_G2[i] - ca.sqrt(G2[i]**2 + mu_G2[i]**2 + 2*tau))
        for i in range(n_comp):
            kkt_eq = ca.vertcat(kkt_eq, slack[i] + mu_s[i] - ca.sqrt(slack[i]**2 + mu_s[i]**2 + 2*tau))


        self.kkt_eq = kkt_eq
        kkt_eq_jac = ca.jacobian(kkt_eq, w_pd)
        self.kkt_eq_jac_fun = ca.Function('kkt_eq_jac_fun', [w_pd, prob.p], [kkt_eq, kkt_eq_jac])
        self.kkt_eq_fun = ca.Function('kkt_eq_fun', [w_pd, prob.p], [kkt_eq])

        self.nw = casadi_length(prob.w)
        self.nw_pd = casadi_length(w_pd)
        self.n_comp = n_comp
        self.kkt_eq_offsets = [0, self.nw]  + [n_H+self.nw + i * n_comp for i in range(5)]

        print(f"created primal dual problem with {casadi_length(w_pd)} variables and {casadi_length(kkt_eq)} equations, {n_comp=}, {self.nw=}, {n_H=}")

        return

    def plot_newton_matrix(self, matrix, title='', fig_filename=None):
        import matplotlib.pyplot as plt
        from nosnoc.plot_utils import latexify_plot
        latexify_plot()
        fig, axs = plt.subplots(1, 1)
        self.add_scatter_spy_magnitude_plot(axs, fig, matrix)
        axs.set_title(title)

        # Q, R = np.linalg.qr(matrix)
        # axs[1].spy(Q)
        # axs[1].set_title('Q')
        # axs[2].spy(R)
        # axs[2].set_title('R')

        if fig_filename is not None:
            plt.savefig(fname=fig_filename)
            print(f"saved figure as {fig_filename}")
        plt.show()
        # breakpoint()

    def add_scatter_spy_magnitude_plot(self, ax, fig, matrix):
        import matplotlib
        rows, cols = matrix.nonzero()
        values = np.abs(matrix[rows, cols])

        cmap = matplotlib.colormaps['inferno']
        # scatter spy
        sc = ax.scatter(cols, rows, c=values, cmap=cmap,
                           norm=matplotlib.colors.LogNorm(vmin=np.min(values), vmax=np.max(values)),
                           marker='s', s=20)
        ax.set_xlim((-0.5, matrix.shape[1] - 0.5))
        ax.set_ylim((matrix.shape[0] - 0.5, -0.5))
        ax.set_aspect('equal')
        fig.colorbar(sc, ax = ax, ticks=matplotlib.ticker.LogLocator())

        ax.set_xticks(self.kkt_eq_offsets)
        ax.set_xticklabels([r'$w$', r'$\lambda_H$', r'$\lambda_{\mathrm{comp}}$', r'$s$', r'$\mu_1$', r'$\mu_2$', r'$\mu_s$'])
        # ax.set_xticklabels(['w', 'lam_H', 'lam_comp', 'slack', 'mu_G1', r'$\mu_G2', r'$\mu_s$'])
        ax.set_yticks(self.kkt_eq_offsets)
        ax.set_yticklabels(['stat w', 'stat s', '$H$', 'slacked comp', 'comp $G_1$', 'comp $G_2$', 'comp $s$'])
        return



    def solve(self) -> dict:
        """
        Solves the NLP with the currently stored parameters.

        :return: Returns a dictionary containing ... TODO document all fields
        """
        opts = self.opts
        prob = self.problem

        # initialize
        self.initialize()
        # slack0
        x0 = prob.model.x0
        p0 = prob.model.p_val_ctrl_stages[0]
        lambda00 = self.model.lambda00_fun(x0, p0).full().flatten()
        tau_val = opts.sigma_0
        p_val = np.concatenate(
                (prob.model.p_val_ctrl_stages.flatten(),
                 np.array([opts.sigma_0, tau_val]), lambda00, x0))
        # slack0 = self.slack0_fun(prob.w0, p_val).full().flatten()
        slack0 = np.zeros((self.n_comp,))
        w_pd_0 = np.concatenate((prob.w0, slack0, self.mu_pd_0, self.lam_pd_0))

        w_current = w_pd_0

        w_all = [w_current.copy()]
        n_iter_polish = opts.max_iter_homotopy + 1
        complementarity_stats = n_iter_polish * [None]
        cpu_time_nlp = n_iter_polish * [None]
        nlp_iter = n_iter_polish * [None]

        # if opts.print_level:
        #     print('-------------------------------------------')
        #     print('sigma \t\t compl_res \t nlp_res \t cost_val \t CPU time \t iter \t status')

        sigma_k = opts.sigma_0

        # lambda00 initialization
        x0 = prob.model.x0
        p0 = prob.model.p_val_ctrl_stages[0]
        lambda00 = self.model.lambda00_fun(x0, p0).full().flatten()

        # if opts.fix_active_set_fe0 and opts.pss_mode == PssMode.STEWART:

        max_gn_iter = 12
        # homotopy loop
        for ii in range(opts.max_iter_homotopy):
            tau_val = sigma_k
            p_val = np.concatenate(
                (prob.model.p_val_ctrl_stages.flatten(),
                 np.array([sigma_k, tau_val]), lambda00, x0))

            print(f"sigma = {sigma_k:.2e}")
            print("alpha \t step norm \t kkt res \t cond(A)")
            t = time.time()
            for gn_iter in range(max_gn_iter):

                # compute step step
                kkt_val, jac_kkt_val = self.kkt_eq_jac_fun(w_current, p_val)
                newton_matrix = jac_kkt_val.full()

                # regularize
                newton_matrix[-3*self.n_comp:, -3*self.n_comp:] += 1e-6 * np.diag(np.ones(3*self.n_comp,))

                try:
                    step = -np.linalg.solve(newton_matrix, kkt_val.full().flatten())
                except:
                    cond = np.linalg.cond(newton_matrix)
                    print(f"failed to solve system with conditioning {cond:.2e}")
                    self.plot_newton_matrix(newton_matrix, title=f'Newton matrix: sigma = {sigma_k:.2e} cond = {cond:.2e}',
                            # fig_filename=f'newton_spy_{prob.model.name}_{ii}_{gn_iter}.pdf'
                    )
                    breakpoint()
                step_norm = np.linalg.norm(step)
                nlp_res = casadi_inf_norm_nan(kkt_val)
                if step_norm < sigma_k or nlp_res < sigma_k / 10:
                    break

                # line search:
                alpha = 1.0
                rho = 0.8
                gamma = .2
                line_search_max_iter = 7
                for line_search_iter in range(line_search_max_iter):
                    w_candidate = w_current + alpha * step
                    step_res_norm = casadi_inf_norm_nan(self.kkt_eq_fun(w_candidate, p_val))
                    if step_res_norm < (1-gamma*alpha) * nlp_res:
                        break
                    else:
                        alpha *= rho

                maxA = np.max(np.abs(newton_matrix))
                cond = np.linalg.cond(newton_matrix)
                tmp = newton_matrix.copy()
                tmp[tmp==0] = 1
                minA = np.min(np.abs(tmp))
                print(f"{alpha:.3f} \t {step_norm:.2e} \t {nlp_res:.2e} \t {cond:.2e} \t {maxA:.2e} \t {minA:.2e}")
                # self.plot_newton_matrix(newton_matrix, title=f'regularized matrix: sigma = {sigma_k:.2e} cond = {cond:.2e}',
                #         fig_filename=f'newton_spy_reg_{prob.model.name}_{ii}_{gn_iter}.pdf'
                #  )

                print(f"{alpha:.3f} \t {step_norm:.2e} \t {nlp_res:.2e} \t")
                w_current = w_candidate

            cpu_time_nlp[ii] = time.time() - t

            # do line search
            # print and process solution
            status = 1
            nlp_iter[ii] = gn_iter # TODO
            cost_val = 0
            # nlp_res = ca.norm_inf(sol['g']).full()[0][0]
            # cost_val = ca.norm_inf(sol['f']).full()[0][0]
            w_all.append(w_current)

            complementarity_residual = prob.comp_res(w_current[:self.nw], p_val).full()[0][0]
            complementarity_stats[ii] = complementarity_residual
            # if opts.print_level:
            #     self._print_iter_stats(sigma_k, complementarity_residual, nlp_res, cost_val,
            #                            cpu_time_nlp[ii], nlp_iter[ii], status)
            if complementarity_residual < opts.comp_tol:
                break
            if sigma_k <= opts.sigma_N:
                break

            # Update the homotopy parameter.
            if opts.homotopy_update_rule == HomotopyUpdateRule.LINEAR:
                sigma_k = opts.homotopy_update_slope * sigma_k
            elif opts.homotopy_update_rule == HomotopyUpdateRule.SUPERLINEAR:
                sigma_k = max(
                    opts.sigma_N,
                    min(opts.homotopy_update_slope * sigma_k,
                        sigma_k**opts.homotopy_update_exponent))

        # if opts.do_polishing_step:
        #     w_current, cpu_time_nlp[n_iter_polish - 1], nlp_iter[n_iter_polish -
        #                                                      1] = self.polish_solution(
        #                                                          w_current, lambda00, x0)

        # collect results
        results = get_results_from_primal_vector(prob, w_current)

        if opts.initialization_strategy == InitializationStrategy.ALL_XCURRENT_WOPT_PREV:
            prob.w0[:] = w_current[:]
        # stats
        results["cpu_time_nlp"] = cpu_time_nlp
        results["nlp_iter"] = nlp_iter
        results["w_all"] = w_all
        results["w_sol"] = w_current
        results["cost_val"] = 0.0

        sum_iter = sum([i for i in nlp_iter if i is not None])
        total_time = sum([i for i in cpu_time_nlp if i is not None])
        # total_time = 0.0
        # for i in nlp_iter:
        #     if i is not None:
        #         sum_iter += i
        print(f"total iterations {sum_iter}, CPU time {total_time:.3f}")

        # for i in range(len(w_current)):
        #     print(f"w{i}: {prob.w[i]} = {w_current[i]}")
        return results