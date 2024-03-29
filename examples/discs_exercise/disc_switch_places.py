# Disc OCP

import nosnoc as ns
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from functools import partial


def get_disc_ocp(opts, r1, r2, q_target1, q_target2):
    # model vars
    q = ca.SX.sym('q', 4)
    q1 = q[0:2]
    q2 = q[2:4]
    v = ca.SX.sym('v', 4)
    v1 = v[0:2]
    v2 = v[2:4]
    t = ca.SX.sym('t')
    x = ca.vertcat(q, v)
    u = ca.SX.sym('u', 2)

    alpha = ca.SX.sym('alpha', 2)
    theta = ca.SX.sym('theta', 2)
    z = theta
    z0 = np.ones((2,))

    # state equations
    m1 = 2
    m2 = 1
    cv = 2
    eps = 1e-1
    M = np.diag([m1, m1, m2, m2])
    a_n = 10

    # Box constraints
    ubx = np.array([10, 10, 10, 10, 5, 5, 5, 5, np.inf])
    lbx = -ubx
    x0 = np.array([-1, 0, 1, 0, 0, 0, 0, 0])
    ubu = np.array([20, 20])
    lbu = np.array([-20, -20])
    u_guess = np.array([0, 0])

    Q = np.diag([10, 10, 20, 20, 0, 0, 0, 0])
    Q_terminal = np.diag([10, 10, 10, 10, 10, 10, 10, 10])*100  # np.diag([10000, 100000, 20000, 20000, 10000, 10000, 10000, 10000])
    R = np.diag([0.1, 0.1])

    x_ref = np.concatenate((q_target1, q_target2, np.zeros(4)))
    f_q = (ca.transpose(ca.vertcat(q, v) - x_ref)@Q@(ca.vertcat(q, v)-x_ref) + ca.transpose(u)@R@u)
    f_q_T = ca.transpose(x - x_ref)@Q_terminal@(x - x_ref)

    # hand crafted time freezing :)
    inv_M = ca.inv(M)
    f_c = ca.norm_2(q1-q2)**2-(r1+r2)**2
    J_normal = ca.transpose(ca.jacobian(f_c, q))
    v_normal = ca.transpose(J_normal)@v
    c = [ca.vertcat(f_c, v_normal)]

    f_drag = cv*ca.vertcat(v1/ca.norm_2(v1+eps), v2/ca.norm_2(v2+eps))
    f_v = ca.vertcat(u, np.array([0, 0])) - f_drag
    f_x = ca.vertcat(theta[0]*v,
                     theta[0]*inv_M@f_v + theta[1]*inv_M@J_normal*a_n,
                     theta[0])

    g_z = theta-ca.vertcat(alpha[0]+alpha[1]-(alpha[0]*alpha[1]),
                           (1-alpha[0])*(1-alpha[1]))

    model = ns.NosnocModel(x=ca.vertcat(x, t), f_x=[f_x], alpha=[alpha], c=c, x0=np.concatenate((x0, [0])),
                           u=u, t_var=t,
                           z=z, z0=z0, g_z=g_z)
    ocp = ns.NosnocOcp(lbu=lbu, ubu=ubu, f_q=f_q, f_terminal=f_q_T, lbx=lbx, ubx=ubx, u_guess=u_guess)

    v_normal_fun = ca.Function('v_normal_fun', [x], [v_normal])
    f_c_fun = ca.Function('f_c_fun', [x], [f_c])
    return model, ocp, x_ref, v_normal_fun, f_c_fun


def get_default_options_step():
    opts = ns.NosnocOpts()
    opts.pss_mode = ns.PssMode.STEP
    opts.use_fesd = True
    opts.homotopy_update_slope = 0.1
    opts.comp_tol = 1e-6
    opts.sigma_0 = 1.
    opts.homotopy_update_rule = ns.HomotopyUpdateRule.LINEAR
    opts.n_s = 1
    opts.step_equilibration = ns.StepEquilibrationMode.HEURISTIC_MEAN
    opts.cross_comp_mode = ns.CrossComplementarityMode.SUM_LAMBDAS_COMPLEMENT_WITH_EVERY_THETA
    opts.mpcc_mode = ns.MpccMode.ELASTIC_INEQ
    opts.objective_scaling_direct = True
    opts.print_level = 1

    opts.opts_casadi_nlp['ipopt']['print_level'] = 5
    opts.opts_casadi_nlp['ipopt']['max_iter'] = 2000

    opts.time_freezing = True
    opts.speed_of_time_variables = ns.SpeedOfTimeVariableMode.LOCAL
    opts.equidistant_control_grid = True
    opts.N_stages = 30
    opts.N_finite_elements = 3
    opts.max_iter_homotopy = 7
    return opts


def solve_ocp_step(opts=None, plot=True):
    if opts is None:
        opts = get_default_options_step()
        opts.terminal_time = 3.0
        opts.N_stages = 30

    r1 = 0.3
    r2 = 0.2
    q_target1 = np.array([1, 0])
    q_target2 = np.array([-1, 0])
    model, ocp, x_ref, v_normal_fun, f_c_fun = get_disc_ocp(opts, r1, r2, q_target1, q_target2)

    solver = ns.NosnocSolver(opts, model, ocp)
    results = solver.solve()
    if plot:
        plot_results(results, opts, r1, r2, q_target1, q_target2)

    return results


def animate_discs(state, disc1, disc2):
    global frame_cnt
    disc1.set(center=(state[0], state[1]))
    disc2.set(center=(state[2], state[3]))
    return disc1, disc2


def plot_results(results, opts, r1, r2, q_target1, q_target2):
    x_traj = np.array(results['x_traj'])
    u_traj = np.array(results['u_traj'])
    theta = np.array(results['z_list'])
    t_num = results['t_grid']
    t = x_traj[:, -1]
    x1 = x_traj[:, 0]
    y1 = x_traj[:, 1]
    x2 = x_traj[:, 2]
    y2 = x_traj[:, 3]
    vx1 = x_traj[:, 4]
    vy1 = x_traj[:, 5]
    vx2 = x_traj[:, 6]
    vy2 = x_traj[:, 7]
    ux = u_traj[:, 0]
    uy = u_traj[:, 1]

    fig, ax = plt.subplots()

    ax.axis('equal')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    disc1 = patches.Circle((x1[0], y1[0]), radius=r1, animated=True, color='k', fill=False)
    disc2 = patches.Circle((x2[0], y2[0]), radius=r2, animated=True, color='r', fill=False)
    ax.add_patch(disc1)
    ax.add_patch(disc2)

    target1 = patches.Circle((q_target1[0], q_target1[1]), radius=r1, alpha=0.5, color='k', fill=False)
    target2 = patches.Circle((q_target2[0], q_target2[1]), radius=r2, alpha=0.5, color='r', fill=False)
    ax.add_patch(target1)
    ax.add_patch(target2)

    ani = FuncAnimation(fig, partial(animate_discs, disc1=disc1, disc2=disc2),
                        frames=results['x_traj'], blit=True, repeat=False)
    try:
        ani.save('discs.gif', writer='imagemagick', fps=10)
    except Exception:
        print("install imagemagick to save as gif")

    # Plot Trajectory
    plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(t, x1, label=r"$x_1(t)$")
    plt.plot(t, y1, label=r"$y_1(t)$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$q_1(t)$")
    plt.legend(loc='best', framealpha=0.1)
    plt.grid()
    plt.subplot(2, 2, 2)
    plt.plot(t, vx1, label=r"$v_{x1}(t)$")
    plt.plot(t, vy1, label=r"$v_{y1}(t)$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$v_1(t)$")
    plt.legend(loc='best', framealpha=0.1)
    plt.grid()
    plt.subplot(2, 2, 3)
    plt.plot(t, x2, label=r"$x_2(t)$")
    plt.plot(t, y2, label=r"$x_2(t)$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$q_2(t)$")
    plt.legend(loc='best', framealpha=0.1)
    plt.grid()
    plt.subplot(2, 2, 4)
    plt.plot(t, vx2, label=r"$v_{x2}(t)$")
    plt.plot(t, vy2, label=r"$v_{y2}(t)$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$v_2(t)$")
    plt.legend(loc='best', framealpha=0.1)
    plt.grid()
    plt.tight_layout()

    # Plot Controls
    plt.figure()
    plt.stairs(ux, results['t_grid_u'], label=r"$u_x$")
    plt.stairs(uy, results['t_grid_u'], label=r"$u_y$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$u$")
    plt.grid()

    # Plot thetas
    plt.figure()
    plt.plot(t_num[1:], theta[:, 0], label=r"$\theta_1$")
    plt.plot(t_num[1:], theta[:, 1], label=r"$\theta_2$")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$u$")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    opts = get_default_options_step()
    opts.terminal_time = 3.0
    opts.N_stages = 30
    opts.time_freezing_tolerance = 0.0
    solve_ocp_step(opts=opts, plot=True)
