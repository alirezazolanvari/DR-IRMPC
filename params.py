import numpy as np

# ### Robot ###
agent_radius = 0.2  # [m] for collision check

# ### state and input constraints ###
max_speed = 0.5  # [m/s]
max_accel = 0.25  # [m/ss]

# ### resolutions ###
dt = 1  # [s] Time tick for motion prediction

# ### MPC params ###
R = np.diag([0.01, 0])  # input cost matrix
mpc_horizon = 5  # [steps]
max_iter_len = 30  # [steps]
max_iter = 20  # [steps]
final_iter_cost = 1e-2

# ### obstacle ###
obses_init_pos_list = [np.array([3, 1.2])]
obs_init_pos = np.array([3, 1.2])
obs_radius = 0.9
A_o = np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])
b_o = np.array([obs_init_pos[0]+(obs_radius + agent_radius), -obs_init_pos[1] + (obs_radius + agent_radius),
               - obs_init_pos[0] + (obs_radius + agent_radius), obs_init_pos[1]
                + (obs_radius + agent_radius)]).reshape(-1, 1)

# theta
d = [1e-3, 0.1, 0.2, 0.4]

# ### LMPC options ###
alpha = 0.2
beta = 1-alpha
d_min = 0.02


# Distribution params
observations_set = []
uncer_diff = 1.8
uncer_support = 9
dist_range = np.array(range(uncer_support))
dist_range = (dist_range-min(dist_range))/max(dist_range)
dist_range = list((dist_range*uncer_diff) - (uncer_diff/2))
emp_dist_support = [(np.array([1, -1])*sample)/np.sqrt(2) for sample in dist_range]
a_init = 9
b_init = 10
p_init = np.random.beta(a_init, b_init, size=5)
r_init = np.random.binomial(uncer_support-1, p_init)
r_init = ((r_init / (uncer_support-1)) * uncer_diff) - (uncer_diff / 2)
# initial observations set
init_observations = list(r_init)

# System dynamics matrices
# ### single agent ###
A = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
C_lin = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
# ### double agent ###
A_mul = np.kron(np.eye(2, dtype=int), A)
B_mul = np.kron(np.eye(2, dtype=int), B)

# Cost matrices
Q_lin = np.diag([1, 1, 1, 1])  # state cost matrix
Q_mul = np.kron(np.eye(2, dtype=int), Q_lin)
R_mul = np.diag(0.1*np.ones(4))

# X_s (initial state) and X_f (target state)
x_s_mul = np.array([0.0, 1, 0.0, 0.0, 0.0, -1, 0.0, 0.0])
x_f_mul = np.array([5, 3, 0, 0, 6, 2, 0, 0])

# initial safe set
s0_states_multi = [[0.,  1.,  0.,  0.,  0., -1.,  0.,  0.],  [0.,  1.,  0.,  0.25,  0., -1.,  0.25,  0.],
                   [0.,  1.25,  0.,  0.5,  0.25, -1.,  0.5,  0.],  [0.,  1.75,  0.,  0.5,  0.75, -1.,  0.5,  0.],
                   [0.,  2.25,  0.,  0.5,  1.25, -1.,  0.5,  0.],  [0.,  2.75,  0.,  0.25,  1.75, -1.,  0.5,  0.],
                   [0.,  3.,  0.,  0.,  2.25, -1.,  0.5,  0.],  [0.,  3.,  0.25,  0.,  2.75, -1.,  0.5,  0.],
                   [0.25,  3.,  0.5,  0.,  3.25, -1.,  0.5,  0.],  [0.75,  3.,  0.5,  0.,  3.75, -1.,  0.5,  0.],
                   [1.25,  3.,  0.5,  0.,  4.25, -1.,  0.5,  0.],  [1.75,  3.,  0.5,  0.,  4.75, -1.,  0.5,  0.],
                   [2.25,  3.,  0.5,  0.,  5.25, -1.,  0.5,  0.],  [2.75,  3.,  0.5,  0.,  5.75, -1.,  0.25,  0.],
                   [3.25,  3.,  0.5,  0.,  6., -1.,  0.,  0.],  [3.75,  3.,  0.5,  0.,  6., -1.,  0.,  0.25],
                   [4.25,  3.,  0.5,  0.,  6., -0.75,  0.,  0.5],  [4.75,  3.,  0.25,  0.,  6., -0.25,  0.,  0.5],
                   [5., 3., 0., 0., 6., 0.25, 0., 0.5],  [5., 3., 0., 0., 6., 0.75, 0., 0.5],
                   [5., 3., 0., 0., 6., 1.25, 0., 0.5],  [5., 3., 0., 0., 6., 1.75, 0., 0.25],
                   [5., 3., 0., 0., 6., 2., 0., 0.]]

# the respective inputs of the initial safe set
s0_inputs_multi = []
s0_inputs_multi.extend([np.array([0, max_accel, max_accel, 0]) for _ in range(int(max_speed/max_accel))])
s0_inputs_multi.extend([np.array([0, 0, 0, 0]) for _ in range(2)])
s0_inputs_multi.extend([np.array([0, -max_accel, 0, 0]) for _ in range(int(max_speed/max_accel))])
s0_inputs_multi.extend([np.array([max_accel, 0, 0, 0]) for _ in range(int(max_speed/max_accel))])
s0_inputs_multi.extend([np.array([0, 0, 0, 0]) for _ in range(4)])
s0_inputs_multi.extend([np.array([0, 0, -max_accel, 0]) for _ in range(int(max_speed/max_accel))])
s0_inputs_multi.extend([np.array([0, 0, 0, max_accel]) for _ in range(int(max_speed/max_accel))])
s0_inputs_multi.extend([np.array([-max_accel, 0, 0, 0]) for _ in range(int(max_speed/max_accel))])
s0_inputs_multi.extend([np.array([0, 0, 0, 0]) for _ in range(2)])
s0_inputs_multi.extend([np.array([0, 0, 0, -max_accel]) for _ in range(int(max_speed/max_accel))])
s0_inputs_multi.extend([np.array([0, 0, 0, 0])])
