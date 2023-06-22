import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from gekko import GEKKO
import math
from collections import Counter

from params import *


def sys(x, u):
    """
    Represent system dynamics.

    :param x: system state := [x_1(m), y_1(m), vx_1(m/s), vy_1(m/s), x_2(m), y_2(m), vx_2(m/s), vy_2(m/s)]
    :type x: list or numpy array
    :param u: control input
    :type u: list or numpy array
    :return: the new system state after applying the given control input
    """
    return (np.matmul(A_mul, np.reshape(x, (-1, 1))) + np.matmul(B_mul, np.reshape(u, (-1, 1)))).flatten()


def beta_binomial(a, b, n, size):
    """
    Sample some points from a beta binomial distribution with specified characters.

    :param a: parameter 'a' in the definition of the beta-binomial distribution
    :type a: float
    :param b: parameter 'b' in the definition of the beta-binomial distribution
    :type b: float
    :param n: parameter 'n' in the definition of the beta-binomial distribution
    :type n: int
    :param size: number of sample points
    :type size: int
    :return: a list of points sampled from the specified bet-binomial distribution
    """
    np.random.seed(42)
    p = np.random.beta(a, b, size=size)
    r = np.random.binomial(n, p)
    r = ((r/n)*uncer_diff) - (uncer_diff/2)
    return r[0]


def list_decompose(data, inds_list):
    """
    Cast the specified columns of a numpy array to a list of lists.

    :param data: the input numpy array
    :type data: numpy array
    :param inds_list: list of desired columns
    :type inds_list: list
    :return: a list of lists each of them are a row of the input array
    """
    return np.array(data)[:, inds_list].tolist()


def get_obs_pos(obs_pos):
    """
    Add a random value (derived from a beta-binomial distribution) to the position of the obstacle.

    :param obs_pos: position of the obstacle
    :type obs_pos: numpy array
    :return: the new position of the obstacle
    """
    uncer = beta_binomial(15, 10, uncer_support-1, 1)
    return uncer, obs_pos + (np.array([1, -1]) * (dt*uncer)) / np.sqrt(2)


def get_multi_obs_pos(obs_pos_list):
    """
    Add random values to the position of all obstacles.

    :param obs_pos_list: list of the obstacles position
    :type obs_pos_list: list
    :return: a list of obstacles new positions
    """
    new_pos_list = []
    uncer_list = []
    for obs_pos in obs_pos_list:
        uncer, pos = get_obs_pos(obs_pos)
        uncer_list.append(uncer)
        new_pos_list.append(pos)
    return uncer_list, new_pos_list


def dist_to_obs(pos, obs_pos):
    """
    Measure the distance between a circular agent from a rectangular obstacle.

    :param pos: position of the agent's center
    :type pos: list or numpy array
    :param obs_pos: position of the obstacle's center
    :type obs_pos: list or numpy array
    :return: the distance between the agent and the obstacle as float
    """
    rect_min_x = obs_pos[0]-obs_radius
    rect_max_x = obs_pos[0]+obs_radius
    rect_min_y = obs_pos[1]-obs_radius
    rect_max_y = obs_pos[1]+obs_radius
    dx = max(rect_min_x - pos[0], 0, pos[0] - rect_max_x)
    dy = max(rect_min_y - pos[1], 0, pos[1] - rect_max_y)
    return np.sqrt(dx*dx + dy*dy)-agent_radius


def collision_check(agent_state, obs_pos_list):
    """
    Check if an agent collides with any of the obstacles.

    :param agent_state: position of the agent
    :type agent_state list or numpy array
    :param obs_pos_list: list of all obstacles' positions
    :type obs_pos_list: list
    :return: a flag showing that if a collision happened as bool
    """
    agent_pos_list = [agent_state[:2], agent_state[4:6]]
    for agent_pos in agent_pos_list:
        for obs_pos in obs_pos_list:
            if dist_to_obs(agent_pos, obs_pos) <= 0:
                return True
    return False


def new_dist_det(pos, uncer):
    """
    Calculate the deterministic distance between agent and the obstacle.

    :param pos: agent position
    :type pos: numpy array
    :param uncer: a random value
    :type uncer: numpy array
    :return: distance between the agent and obstacle
    """
    return d_min - np.max(((A_o @ ((pos - uncer).reshape(-1, 1))) - b_o))


def new_dist(agent_pos, uncer, eta):
    """
    Calculate the parametric approximation distance between agent and the obstacle.

    :param agent_pos: agent position
    :type agent_pos: numpy array
    :param uncer: a random value
    :type uncer: numpy array
    :param eta: regulation term
    :param eta: array
    :return: parametric expression of the distance between the agent and obstacle
    """
    return d_min - (((A_o@((agent_pos-uncer).reshape(-1, 1))) - b_o).T @ eta.reshape(-1, 1))[0, 0]


def plot_arrow(x, y, yaw, length=agent_radius, width=0.1):
    """
    Plot agent's yaw vector as an arrow.

    :param x: position of the obstacle in x axis
    :type x: float
    :param y: position of the agent in y axis
    :type y: float
    :param yaw: agent's yaw angel in radian
    :type yaw: float
    :param length: agent's radius
    :type length: float
    :param width: agent's radius
    :type width: float
    :return: None
    """
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)


def plot_agent(x, y, yaw):
    """
    Plot a circular agent showing its yaw angel with an arrow.

    :param x: position of the obstacle in x axis
    :type x: float
    :param y: position of the agent in y axis
    :type y: float
    :param yaw: agent's yaw angel in radian
    :type yaw: float
    :return: None
    """
    circle = plt.Circle((x, y), agent_radius, color="b")
    plt.gcf().gca().add_artist(circle)
    out_x, out_y = (np.array([x, y]) +
                    np.array([np.cos(yaw), np.sin(yaw)]) * agent_radius)
    plt.plot([x, out_x], [y, out_y], "-k")


def show_animation(agent_state, obs_pos_list):
    """
    Plot the environment evolutions for the two-agent problem.

    :param agent_state: position of the agent
    :type agent_state: numpy array
    :param obs_pos_list: list of the obstacles' positions
    :type obs_pos_list: list
    :return: None
    """
    # decompose the position of the agents
    agent_pos_list = [agent_state[:4], agent_state[4:]]

    # clear the figure for the new plot
    plt.cla()

    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    # plot each agent
    for agent_pos in agent_pos_list:
        plt.plot(agent_pos[0], agent_pos[1], "xr")
        if agent_pos[3] <= 1e-10:
            yaw = 0
        elif agent_pos[2] <= 1e-10:
            yaw = math.pi/2
        else:
            yaw = math.atan(agent_pos[3] / agent_pos[2])
        plot_agent(agent_pos[0], agent_pos[1], yaw)
        plot_arrow(agent_pos[0], agent_pos[1], yaw)

    # plot the target points
    plt.plot(x_f_mul[0], x_f_mul[1], "xb")
    plt.plot(x_f_mul[4], x_f_mul[5], "xr")

    # plot the rectangular obstacles
    for obs_pos in obs_pos_list:
        plt.plot(obs_pos[0], obs_pos[1], "ok")
        obs = Rectangle((obs_pos[0]-obs_radius, obs_pos[1]-obs_radius), obs_radius*2, obs_radius*2, facecolor='r')
        plt.gcf().gca().add_artist(obs)

    # set plot configurations
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.0001)


def calc_costs_mul(states_traj, inputs_traj):
    """
    Calculate Q (cost) value for each state in the two-agent setup.

    :param states_traj: list of all states within the trajectory
    :type states_traj: list
    :param inputs_traj: list of the inputs applied to the agents during the trajectory
    :type inputs_traj: list
    :return: list of each state's cost as a list
    """
    costs = [0]*len(states_traj)
    cost_t = 0
    for i in range(len(states_traj)-2, -1, -1):
        cost_t += calc_cost_mul(states_traj[i], inputs_traj[i])
        costs[i] = cost_t
    return costs


def calc_cost_mul(x, u):
    """
    Calculate the cost of a single state.

    :param x: the cost for which the cost should be calculated
    :type x: list
    :param u: the control input applied to the agent when it's in the state `x`
    :type u: list
    :return: cost of state x as float
    """
    cost = 0
    u = np.array(u).reshape(-1, 1)
    cost += np.matmul(np.matmul(u.T, R_mul), u)[0][0]
    cost += np.matmul(np.matmul((x_f_mul-np.array(x)).T, Q_mul), (x_f_mul-np.array(x)))
    return cost


def get_new_val(u):
    """
    Get the first element of control input vector calculated during a horizon of MPC problem.

    :param u: control input vector
    :type u: GEKKO array
    :return: the value of the first element of the control input vector
    """
    return u.NEWVAL


def new_dist_l1_single_agent(agent_state, ss, q_func, observations, theta, x_f_sing):
    """
    Solve the optimization problem for the whole length of the MPC horizon.

    :param agent_state: position of the agent
    :type agent_state: numpy array
    :param ss: safe set
    :type ss: list
    :param q_func: list of states' costs
    :type q_func: list
    :param observations: list of all observed random samples added to the obstacle position
    :type observations: list
    :param theta: ambiguity set radius
    :type theta: float
    :param x_f_sing: the target point of current agent
    :type x_f_sing: numpy array
    :return: the calculated control input and its respective cost
    """
    # Observation process
    N_t = len(observations)
    N_k = uncer_support
    emp_dist = dict(Counter(observations))
    emp_dist = {k: v/N_t for k, v in emp_dist.items()}
    emp_vals = np.fromiter(emp_dist.values(), dtype=float)
    emp_dist_keys = np.array(list(emp_dist.keys())).tolist()
    emp_dist_vals = [0] * N_k
    for ind, w in enumerate(emp_dist_keys):
        emp_dist_vals[dist_range.index(w)] = emp_vals[ind]
    # #################################################################################################################
    # Initializing solver, cost and constraints
    objectives = 0
    equations = []
    m = GEKKO(remote=False)
    m.options.SOLVER = 1
    # #################################################################################################################
    # Initializing decision variables
    x = m.Array(m.CV, (len(agent_state), mpc_horizon + 1))
    u = m.Array(m.MV, (2, mpc_horizon), lb=-1*max_accel, ub=max_accel)
    z = m.Array(m.Var, mpc_horizon)
    lambda0 = m.Array(m.Var, mpc_horizon, lb=0)
    nu = m.Array(m.Var, mpc_horizon)
    lambda1 = m.Array(m.Var, (mpc_horizon, N_k), lb=0)
    lambda2 = m.Array(m.Var, (mpc_horizon, N_k), lb=0)
    lmbd = m.Array(m.Var, (len(ss)), lb=0, ub=1, integer=True)
    eta_u = m.Array(m.Var, (mpc_horizon, N_k, len(b_o)), lb=0)
    # eta_l = m.Array(m.Var, (mpc_horizon, N_k, len(b_o)), lb=0)
    # #################################################################################################################
    # State initial condition constraints
    for i in range(len(agent_state)):
        equations.append(x[i, 0] == agent_state[i])
    # #################################################################################################################
    for t in range(mpc_horizon):
        # Finite time cost
        if t != 0:
            objectives += u[:, t].T @ R @ u[:, t]
            objectives += (x_f_sing - x[:, t]).T @ Q_lin @ (x_f_sing - x[:, t])
        # #############################################################################################################
        # Input limit constraints
        for iiii in range(2):
            u[iiii, t].STATUS = 1
        # #############################################################################################################
        # Constraint 19a (CVaR)
        equations.append(((lambda0[t]*theta) + z[t] + nu[t] + ((lambda1[t, :]-lambda2[t, :])@emp_dist_vals)) <= 0)
        # #############################################################################################################
        for samp in range(N_k):
            # Constraint 19B
            equations.append(lambda1[t, samp] - lambda2[t, samp] + nu[t] >= 0)
            equations.append(lambda1[t, samp] - lambda2[t, samp] + nu[t] >=
                             (1/alpha)*(new_dist(C_lin@x[:, t], emp_dist_support[samp], eta_u[t, samp, :])
                                        - z[t]))
            temp_val_1 = A_o.T @ eta_u[t, samp, :]
            equations.append((temp_val_1[0] ** 2 + temp_val_1[1] ** 2) <= 1)

            # equations.append(lambda1[t, samp] - lambda2[t, samp] + nu[t] + (z[t]/alpha) >= 0)
            # #########################################################################################################
            # Constraint 19C
            equations.append(lambda1[t, samp] + lambda2[t, samp] <= lambda0[t])
            # #########################################################################################################
        # System dynamics constraints
        equations.extend([x[ind, t + 1] == (A @ x[:, t] + B @ u[:, t])[ind] for ind in range(len(agent_state))])
        # #############################################################################################################
    # In safe set constraint
    equations.append(m.sum(lmbd) == 1)
    for iji in range(len(agent_state)):
        equations.append(x[iji, mpc_horizon] == (lmbd @ np.array(ss))[iji])
    # #########################################################################################################
    # Terminal cost
    objectives += lmbd@np.array(q_func)
    # #########################################################################################################
    m.Equation(equations)
    m.Minimize(objectives)
    # #########################################################################################################
    # Solve optimization problem
    try:
        # IPOPT solver
        m.solve(disp=False)
        # #####################################################################################################
    except Exception as e:
        # APOPT (Mixed-integer) solver
        print(e)
        m.options.SOLVER = 3
        m.solve(disp=False)
        # #####################################################################################################

    return list(map(get_new_val, u[:, 0])), m.options.OBJFCNVAL


def calc_cvar_new_dist_multi_agent(pos, observations, theta):
    """
    Calculate the risk value of the input state.

    :param pos: position of the agents
    :type pos: numpy array
    :param observations: list of all observed random samples added to the obstacle position
    :type observations: list
    :param theta: ambiguity set radius
    :type theta: float
    :return: risk value of the input state
    """
    # Observation process
    N_t = len(observations)
    N_k = uncer_support
    emp_dist = dict(Counter(observations))
    emp_dist = {k: v / N_t for k, v in emp_dist.items()}
    emp_vals = np.fromiter(emp_dist.values(), dtype=float)
    emp_dist_keys = np.array(list(emp_dist.keys())).tolist()
    emp_dist_vals = [0] * N_k
    for ind, w in enumerate(emp_dist_keys):
        emp_dist_vals[dist_range.index(w)] = emp_vals[ind]

    # #################################################################################################################
    # Initializing solver, cost and constraints
    objectives = 0
    equations = []
    m = GEKKO(remote=False)
    m.options.SOLVER = 1
    # #################################################################################################################
    # Initializing decision variables
    z = m.Var()
    lambda0 = m.Var(lb=0)
    nu = m.Var()
    lambda1 = m.Array(m.Var, N_k, lb=0)
    lambda2 = m.Array(m.Var, N_k, lb=0)
    # #################################################################################################################
    objectives += lambda0*theta + z + nu + (lambda1[:]-lambda2[:])@emp_dist_vals
    for l in range(N_k):
        equations.append(lambda1[l] - lambda2[l] + nu >= (1/alpha)*(new_dist_det(C_lin@pos[:4], emp_dist_support[l])
                                                                    - z))
        equations.append(lambda1[l] - lambda2[l] + nu >= (1 / alpha) * (new_dist_det(C_lin@pos[4:], emp_dist_support[l])
                                                                        - z))
        equations.append(lambda1[l] - lambda2[l] + nu >= 0)
        equations.append(lambda1[l] - lambda2[l] + nu + (z/alpha) >= 0)
        equations.append(lambda1[l] + lambda2[l] <= lambda0)
    m.Equation(equations)
    m.Minimize(objectives)
    m.solve(disp=False)
    return m.options.OBJFCNVAL
