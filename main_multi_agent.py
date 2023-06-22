from utils_multi_agent import *
import os
from collections import defaultdict
import copy

# Start a loop to check the results for different ambiguity set radii
for theta in d:
    # Initializations for the first iteration
    num_collision = 0
    observations = copy.copy(init_observations)
    safe_set = copy.copy(s0_states_multi)
    safe_inputs = copy.copy(s0_inputs_multi)
    q_func = calc_costs_mul(safe_set, safe_inputs)
    objectives = []
    ss_tup = [(0, copy.copy(s0_states_multi), copy.copy(q_func))]
    data_dir = f'output_radius{theta}_d_min{d_min}_uncer{uncer_diff}'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    # Start a loop for simulating a specified number of iterations for one ambiguity set radius
    for j in range(max_iter):
        # Initializations for the first step within an iteration
        agent_pos = copy.copy(x_s_mul)
        obs_pos_list = copy.copy(obses_init_pos_list)
        iter_traj_states = []
        iter_traj_inputs = []
        iter_observations = []
        iter_obj = []
        print(f"Iteration {j} is started!")

        # Start a loop for simulating one iteration with a specified maximum number of steps
        for i in range(max_iter_len):
            show_animation(agent_pos, obs_pos_list)
            print(f'\tStep {i} of iteration {j} ...')

            # Check if the agents reached their targets
            if np.linalg.norm(x_f_mul[[0, 1, 4, 5]]-agent_pos[[0, 1, 4, 5]]) <= final_iter_cost:
                print('\t\tTarget is touched!')

                # Calculating the iteration cost
                iter_costs = calc_costs_mul(iter_traj_states, iter_traj_inputs)
                np.save(data_dir + f'\\data_cost_{j}.npy', iter_costs, allow_pickle=True)

                # Updating the safe set with the resulted trajectory
                ss_tup.append((j+1, iter_traj_states, iter_costs))
                for t, x in enumerate(iter_traj_states):
                    try:
                        index = safe_set.index(x)
                        continue
                    except ValueError:
                        safe_set.append(x)
                break

            # Check if any of the agents collide with the obstacle
            if collision_check(agent_pos, obs_pos_list):
                print('\t\tCollision happened!')
                num_collision += 1
                break

            # Start steering the agents toward their targets
            try:
                # The control inputs are computed separately because of resource limitations (the results are
                # identically same with the centralized computations)
                in_act, obj_value1 = new_dist_l1_single_agent(agent_pos[:4], list_decompose(safe_set, [0, 1, 2, 3]),
                                                              q_func, observations, theta, x_f_mul[:4])
                in_act2, obj_value2 = new_dist_l1_single_agent(agent_pos[4:], list_decompose(safe_set, [4, 5, 6, 7]),
                                                               q_func, observations, theta, x_f_mul[4:])
            except Exception as e:
                print(e)
                break

            # Store respective information of the current step when it ends
            obj_value = obj_value1 + obj_value2
            iter_obj.append(obj_value)
            in_act.extend(in_act2)
            agent_pos = sys(agent_pos, in_act)
            iter_traj_states.append(list(np.floor(copy.copy(agent_pos)*10)/10))
            iter_traj_inputs.append(np.array(in_act))
            old_obs_poses = copy.copy(obs_pos_list)
            obser_list, obs_pos_list = get_multi_obs_pos(obses_init_pos_list)
            iter_observations.extend(obser_list)

        # Store the respective information of the iteration when it ends
        objectives.append(iter_obj)
        np.save(data_dir + f'\\data_trajs_{j}.npy', iter_traj_states, allow_pickle=True)
        np.save(data_dir + f'\\data_obs_{j}.npy', iter_observations, allow_pickle=True)
        np.save(data_dir + f'\\data_objs_{j}.npy', iter_obj, allow_pickle=True)
        observations.extend(iter_observations)

        # Start the process of eliminating the unsafe trajectories from the safe set w.r.t the updated ambiguity set
        temp_ss = copy.copy(safe_set)
        to_elim = []

        # Remove the states that are part of the robust trajectory with which the safe set is initialized (exempt these
        # steps from safety check)
        for state in s0_states_multi:
            temp_ss.remove(state)

        # Calculate the risk value of all states in the safe set w.r.t the ambiguity set which is updated with the
        # updated dateset
        for i_temp in range(1, len(temp_ss)):
            risk_val = calc_cvar_new_dist_multi_agent(np.array(temp_ss[i_temp]), observations, theta)

            # Check if the risk value of a state is above the acceptable threshold (if the state is unsafe)
            if risk_val > 1e-3:
                print("\t\t CVaR value:", risk_val)

                # Store the index of the trajectories that contain an unsafe state
                for succ_iter in range(len(ss_tup)):
                    try:
                        if temp_ss[i_temp] in ss_tup[succ_iter][1]:
                            to_elim.append(succ_iter)
                    except:
                        print(i_temp, len(temp_ss), succ_iter, len(ss_tup), ss_tup)
                        raise IndexError

        # Remove the redundant trajectory indices that contains unsafe state(
        to_elim = list(set(to_elim))

        # Remove the trajectories that contain unsafe state(s)
        while len(to_elim) > 0:
            try:
                ss_tup.pop(max(to_elim))
                to_elim.remove(max(to_elim))
            except:
                raise IndexError

        # calculate the cost of each state with the updated trajectories within the safe set
        all_trajs = []
        all_costs = []
        for succ_iter in range(len(ss_tup)):
            all_trajs.extend(ss_tup[succ_iter][1])
            all_costs.extend(ss_tup[succ_iter][2])
        ss_dict = defaultdict(lambda: 10000)
        for i, traj in enumerate(all_trajs):
            if ss_dict[tuple(traj)] > all_costs[i]:
                ss_dict[tuple(traj)] = all_costs[i]
        safe_set.clear()
        safe_set = [list(x) for x in ss_dict.keys()]
        q_func = [x for x in ss_dict.values()]

    # Store the number of collisions happened during iterations simulated for a specific ambiguity set radius
    np.save(data_dir + '\\data_collision.npy', num_collision, allow_pickle=True)
