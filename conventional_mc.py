# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:45:58 2017

@author: Alexandra
"""

import numpy as np, subprocess, itertools
import matplotlib.pyplot as plt
import timeit
from copy import deepcopy

###################   global parameters   ############################
# TODO: modify these parameters accordingly since I just put random values for now
R_channel = 0.7
L_channel = 5
charge_hydrogen = +0.4
charge_oxygen = -0.8
bond_length_water = 0.1
bond_angle_water = 1.88496
I_water = 1        # moment of inertia of water molecule
mass_water = 1
k_electric = 10    # electric constant, equal to (1 / (4 \pi \epsilon_0))
n_neural_particles = 10  # number of neural particles on each side
epsilon_LJ = 2.5; sigma_LJ = 0.3  # they should be different for different particles
random_force_strength = 2.0
const_force = 2.0  # constant force that drives water molecule towards the right

#######################################################################
# these are helper global variables
horizontal_coords_particles = np.linspace(0, L_channel, n_neural_particles + 2)

coords_particles = np.array(list(zip(horizontal_coords_particles, np.zeros(n_neural_particles + 2))) \
                 + list(zip(horizontal_coords_particles, R_channel * np.ones(n_neural_particles + 2))))

coords_negative_particles = np.array([
    [0,0], [L_channel, 0], [0, R_channel], [L_channel, R_channel]
])

sigma_6 = sigma_LJ ** 6
sigma_12 = sigma_LJ ** 12
mass_vector = np.array([mass_water, mass_water, I_water])
#######################################################################

def init_config(temp_x=None, temp_y=None, temp_theta=None):
    temp_x = np.random.uniform(low=0 * L_channel, high=0.1 * L_channel) if temp_x is None else temp_x
    temp_y = np.random.uniform(low=0.4 * R_channel, high=0.6 * R_channel) if temp_y is None else temp_y
    temp_theta = np.random.uniform(high= 2 * np.pi) if temp_theta is None else temp_theta
    v_x, v_y, omega = np.random.normal(scale=1.0, size=3)
    return np.array([temp_x, temp_y, temp_theta]), np.array([v_x, v_y, omega])

def VerletNextR(r_t,v_t,a_t,h):
    r_t_plus_h = r_t + v_t*h + 0.5*a_t*h*h
    return r_t_plus_h

def VerletNextV(v_t,a_t,a_t_plus_h,h):
    v_t_plus_h = v_t + 0.5*(a_t + a_t_plus_h)*h
    return v_t_plus_h

def get_net_force_for_a_single_particle(coord, charge):
    # TODO: 1. add random force, 2. add constant fixed force pointing to the right
    # LJ forces
    displacement_vectors = coord - coords_particles

    distances = np.linalg.norm(displacement_vectors, axis=1)

    #print (np.max(distances), np.min(distances))

    temp_LJ_Force = epsilon_LJ * np.sum(
        np.dot(np.diag(48 * sigma_12 / np.power(distances, 14) - 24 * sigma_6 / np.power(distances, 8)),
            displacement_vectors),
        axis=0)
    # electric forces

    displacement_vectors_electric = coord - coords_negative_particles

    distances_electric = np.linalg.norm(displacement_vectors_electric, axis=1)

    assert (len(distances_electric) == 4)

    temp_electric_Force = k_electric * np.sum(
        np.dot(np.diag(charge / np.power(distances_electric, 3)),
            displacement_vectors_electric),
        axis=0)

    # print charge, temp_LJ_Force, temp_electric_Force
    return temp_LJ_Force + temp_electric_Force + random_force_strength * np.random.normal(size=2) + np.array([const_force, 0]) \
            + np.array([1000000, 0]) * charge


def get_force_and_torque(d_water, r_water, theta_water):  # params: x, y, theta (define configurations)
    relative_coords_hydrogen_oxygen = np.array([
        [np.cos(theta_water + bond_angle_water / 2.0), np.sin(theta_water + bond_angle_water / 2.0)],
        [np.cos(theta_water - bond_angle_water / 2.0), np.sin(theta_water - bond_angle_water / 2.0)],
        [0, 0]
    ])

    coords_hydrogen_oxygen = np.array([d_water, r_water]) + bond_length_water * relative_coords_hydrogen_oxygen

    charge_hydrogen_oxygen = [charge_hydrogen, charge_hydrogen, charge_oxygen]

    temp_forces = [get_net_force_for_a_single_particle(item[0], item[1])
                       for item in zip(coords_hydrogen_oxygen, charge_hydrogen_oxygen)]

    net_force = np.sum(temp_forces, axis=0)

    net_torque = np.dot(np.array([temp_forces[0][1], -temp_forces[0][0]]), relative_coords_hydrogen_oxygen[0]) \
        + np.dot(np.array([temp_forces[1][1], -temp_forces[1][0]]), relative_coords_hydrogen_oxygen[1])  # assume center of mass is at Oxygen

    net_torque *= bond_length_water

    return np.array([net_force[0], net_force[1], net_torque])

def simulate(num_steps, h_stepsize=0.001, starting_config=(None,None,None)):

    starttime = timeit.default_timer() # timer
    configs, velocities = init_config(*starting_config)
    # if not starting_config[0] is None: configs = starting_config
    configs_list, velocities_list = [], []
    for item in range(num_steps):
        configs_list.append(configs.copy())
        velocities_list.append(velocities.copy())
        temp_acceleration = get_force_and_torque(configs[0], configs[1], configs[2]) / mass_vector
        temp_next_configs = VerletNextR(configs, velocities, temp_acceleration, h_stepsize)
        temp_next_acceleration = get_force_and_torque(
            temp_next_configs[0], temp_next_configs[1], temp_next_configs[2]) / mass_vector

        temp_next_v = VerletNextV(velocities, temp_acceleration, temp_next_acceleration, h_stepsize)

        if 0 < temp_next_configs[0] < L_channel and 0 < temp_next_configs[1] < R_channel:
            configs, velocities = temp_next_configs, temp_next_v
        else:  # restart when it goes out of the channel
            configs, velocities = init_config(*starting_config)

    endtime = timeit.default_timer() # timer
    processtime = endtime - starttime # timer

    return np.array(configs_list), np.array(velocities_list), processtime
#Qiangqiang Huang,qh4@illinois.edu
def visual(coords_negative_particles,coords_particles,my_positions,my_velocities,dpivalue = 100):
    """
    coords_particles denotes coordinates of the particles and charges of the membrane
    my_positions is a np.array. The shape is N*3. N denotes the simulation step.
    my_velocities is a np.array. The shape is N*3. N denotes the simulation step.
    dpivalue is the default dpi for all the figures. Reduce it to shrink all the figures
    """
    step = len(my_positions)
    fig1 = plt.figure(dpi = dpivalue)
    #draw particles and charges of the membrane
    plt.subplot(121)
    plt.scatter(coords_particles[:,0], coords_particles[:,1])
    plt.scatter(coords_negative_particles[:,0], coords_negative_particles[:,1],s=30,c='r',marker='X')
    #draw the water's positions
    plt.scatter(my_positions[:,0], my_positions[:,1],s=10)
    #draw the water's velocity vector
    #plt.quiver(my_positions[:,0], my_positions[:,1],my_velocities[:,0], my_velocities[:,1],alpha=.5)
    #annotate the first time step
    plt.annotate('1',[my_positions[0,0],my_positions[0,1]])
    plt.scatter(my_positions[0,0],my_positions[0,1],s=50,c='g')
    #annotate the last time step
    plt.annotate(str(step),[my_positions[-1,0],my_positions[-1,1]])
    plt.scatter(my_positions[-1,0],my_positions[-1,1],s=50,c='g')
    plt.xlabel(r'$X$')
    plt.ylabel(r'$Y$')
    plt.subplot(122)
    plt.scatter(my_positions[:,1], np.cos(my_positions[:,2]),s=10)
    #draw the water's velocity vector
    #plt.quiver(my_positions[:,0], my_positions[:,1],my_velocities[:,0], my_velocities[:,1],alpha=.5)
    #annotate the first time step
    plt.annotate('1',[my_positions[0,1],np.cos(my_positions[0,2])])
    plt.scatter(my_positions[0,1],np.cos(my_positions[0,2]),s=50,c='g')
    #annotate the last time step
    plt.annotate(str(step),[my_positions[-1,1],np.cos(my_positions[-1,2])])
    plt.scatter(my_positions[-1,1],np.cos(my_positions[-1,2]),s=50,c='g')
    plt.xlabel(r'$Y$')
    plt.ylabel(r'$cos(\theta)$')
    fig1.tight_layout()
    fig1.set_size_inches(7, 3.5)

    fig2 = plt.figure(dpi = dpivalue)
    #draw histodiagrams
    plt.subplot(231)
    plt.hist(my_positions[:,0], 100, weights=1.0/step*np.ones(shape=(step,1)), alpha=0.5)
    plt.xlabel(r'$X$')
    plt.xlim(-0.1,7.1)
    plt.ylabel(r'$Probability$')
    plt.subplot(232)
    plt.hist(my_positions[:,1], 100, weights=1.0/step*np.ones(shape=(step,1)), alpha=0.5)
#    print('nnnnnnn',n)
    plt.xlabel(r'$Y$')
    plt.xlim(-0.1,0.8)
    #plt.ylabel(r'$Probability$')
    plt.subplot(233)
    plt.hist(np.cos(my_positions[:,2]), 100, weights=1.0/step*np.ones(shape=(step,1)), alpha=0.5)
#    print('nnnnnnn',n)
    plt.xlabel(r'$cos(\theta)$')
    plt.xlim(-1.1,1.1)
    fig2.tight_layout()
    #plt.ylabel(r'$Probability$')



    fig3 = plt.figure(dpi = dpivalue)
    #draw trace
    plt.subplot(131)
    plt.plot(my_positions[:,0], alpha=0.5)
    plt.xlabel(r'$step$')
    plt.ylabel(r'$X$')
    plt.subplot(132)
    plt.plot(my_positions[:,1], alpha=0.5)
    plt.xlabel(r'$step$')
    plt.ylabel(r'$Y$')
    plt.subplot(133)
    plt.plot(np.cos(my_positions[:,2]), alpha=0.5)
    plt.xlabel(r'$step$')
    plt.ylabel(r'$cos(\theta)$')
    # plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.3,
    #                 wspace=0.3)
    fig3.tight_layout()
    fig3.set_size_inches(7, 2.5)
    plt.show()
    return 0


if __name__ == '__main__':
    my_positions, my_velocities = simulate(500)

    plt.figure(1)
    plt.scatter(coords_particles[:,0], coords_particles[:,1])
    #plt.scatter(coords_negative_particles[:,0], coords_negative_particles[:,1])
    plt.scatter(my_positions[:,0], my_positions[:,1])
    plt.show()
    print(my_positions)


def clustering(positions, n_clusters):
    from sklearn import cluster, preprocessing
    temp_positions = np.array([positions[:, 0], positions[:,1], np.cos(positions[:,2])]).T
    min_max_scaler = preprocessing.MinMaxScaler()
    temp_positions = min_max_scaler.fit_transform(temp_positions)
    kmeans = cluster.KMeans(n_clusters=n_clusters).fit(temp_positions)
    return kmeans.labels_, min_max_scaler.inverse_transform(kmeans.cluster_centers_)

def sort_one_list_based_on_another(one_list, another_list):
    return np.array([x for _, x in sorted(zip(another_list, one_list), key=lambda x: x[0])])

def get_new_starting_configs(cluster_labels, cluster_centers,
                            num_clusters_for_consideration):
    temp_sums = [np.sum(cluster_labels == item) for item in np.unique(cluster_labels)]
    sorted_cluster_centers = sort_one_list_based_on_another(cluster_centers, temp_sums)
    unweighted_reward_components = np.abs((sorted_cluster_centers - np.mean(sorted_cluster_centers,
        axis=0)) / np.std(sorted_cluster_centers, axis=0))
    unweighted_reward_components = unweighted_reward_components[:num_clusters_for_consideration]
    temp_component_sum = unweighted_reward_components.sum(axis=0)
    temp_component_sum /= np.linalg.norm(temp_component_sum)
    weighted_reward = (temp_component_sum * unweighted_reward_components).sum(axis=1)
    sorted_cluster_centers_based_on_reward = sort_one_list_based_on_another(
        sorted_cluster_centers[:num_clusters_for_consideration], weighted_reward)
    return sorted_cluster_centers_based_on_reward[::-1], temp_component_sum

# enhanced sampling with reinforcement learning
def RL_simulation(init_simulation_steps = 1000, num_rounds=100, num_steps_each_round=1000,
                num_clusters = 50, num_clusters_for_consideration=10, num_starting_points=1):
    my_positions_list, weights_list, starting_configs_list = [], [], []
    my_positions, _, _ = simulate(init_simulation_steps)
    my_positions_list.append(my_positions.copy())
    for item in range(num_rounds):
        # print (item)
        cluster_labels, cluster_centers = clustering(np.concatenate(my_positions_list, axis=0), num_clusters)
        starting_configs, weights = get_new_starting_configs(cluster_labels, cluster_centers,
            num_clusters_for_consideration)
        starting_configs = starting_configs[:num_starting_points]
        weights_list.append(weights.copy())
        starting_configs_list.append(starting_configs.copy())
        temp_pos_list = []
        for item_config in starting_configs:
            item_config[2] = np.arccos(item_config[2])
            temp_my_pos, _, _ = simulate(num_steps_each_round, starting_config=item_config)
            temp_pos_list.append(temp_my_pos.copy())
        my_positions_list.append(np.concatenate(temp_pos_list).copy())
    return my_positions_list, weights_list, starting_configs_list

from sklearn import cluster, preprocessing
from sklearn.metrics.pairwise import euclidean_distances
def get_num_states_vs_time(temp_positions, threshold):
    min_max_scaler = preprocessing.MinMaxScaler()
    temp_positions[:, 2] = np.cos(temp_positions[:, 2])
    temp_positions = min_max_scaler.fit_transform(temp_positions)
    explored_states_list, index_list = [temp_positions[0]], [0]
    for index, item in enumerate(temp_positions[1:]):
        distances_to_previous_explore_states = euclidean_distances([item], explored_states_list).flatten()
        if np.all(distances_to_previous_explore_states > threshold):
            explored_states_list.append(item)
            index_list.append(index + 1)
    return explored_states_list, index_list

def plot_num_states_vs_time(multiple_index_list, labels_list):
    for item_index, item_label in zip(multiple_index_list, labels_list):
        plt.plot(item_index, range(len(item_index)), label=item_label)
        plt.xlabel('time step')
        plt.ylabel('number of explored states')
    plt.legend()
    return

def getNewPos(Pos,nbins=10,steps=1000):
    Pos = list(zip(*Pos)) #make it a list
    hist_matrix, edges = np.histogramdd(Pos, bins= nbins,normed=0) #hist 3D
    maxnumber = hist_matrix.max()+1
    newhist_matrix = np.copy(hist_matrix)
    newhist_matrix -= maxnumber
    newhist_matrix = -newhist_matrix
    summn = newhist_matrix.sum()
    prob = newhist_matrix
    prob = prob/prob.sum() # normalized
    #building prob as n_i/all_n
    edges = np.array(edges)
    x_edge = edges[0][2]-edges[0][1]
    y_edge = edges[1][2]-edges[1][1]
    theta_edge = edges[2][2]-edges[2][1]

    nowPos = [Pos[0][len(Pos)],Pos[1][len(Pos)],Pos[2][len(Pos)]]
    nowX = int(nowPos[0]/x_edge)
    nowY = int(nowPos[1]/y_edge)
    nowTheta = int(nowPos[2]/theta_edge)

    for i in range(steps):
        # metropolis scan with prob[x][y][theta] as acceptance rate
        # numbers need to be altered, to see how many "jumps" we need during the simulation
        x = np.random.randint(0,nbins)
        y = np.random.randint(0,nbins)
        theta = np.random.randint(0,nbins)
        newPos_x = edges[0][x]+0.5*x_edge
        newPos_y = edges[1][y]+0.5*y_edge
        newPos_theta = edges[2][theta]+0.5*theta_edge

        newPos = np.array([newPos_x,newPos_y,newPos_theta])

        if min(1,(prob[x][y][theta]/prob[nowX][nowY][nowTheta])) > np.random.uniform():
            if newPos[0] <= 0:
                newPos[0] += 0.5*x_edge
            if newPos[0] >= 5:
                newPos[0] -= 0.5*x_edge
            if newPos[1] <= 0:
                newPos[1] += 0.5*y_edge
            if newPos[1] >= 0:
                newPos[1] -= 0.5*y_edge
            return newPos
        if i == steps-1:
            return nowPos

### MD and MC sweep
def simulate2(num_steps, h_stepsize=0.001):

    starttime = timeit.default_timer() # timer

    configs, velocities = init_config()

    configs_list, velocities_list = [], []

    for item in range(num_steps):
        configs_list.append(configs)

        velocities_list.append(velocities)

        temp_acceleration = get_force_and_torque(configs[0], configs[1], configs[2]) / mass_vector

        temp_next_configs = VerletNextR(configs, velocities, temp_acceleration, h_stepsize)

        temp_next_acceleration = get_force_and_torque(
            temp_next_configs[0], temp_next_configs[1], temp_next_configs[2]) / mass_vector

        temp_next_v = VerletNextV(velocities, temp_acceleration, temp_next_acceleration, h_stepsize)

        if 0 < temp_next_configs[0] < L_channel and 0 < temp_next_configs[1] < R_channel:
            configs, velocities = temp_next_configs, temp_next_v
        else:  # restart when it goes out of the channel
            configs, velocities = init_config()

        if item != 0 and item%1000 == 0:
            configs = getNewPos(configs_list,10)

    endtime = timeit.default_timer() # timer
    processtime = endtime - starttime # timer

    return np.array(configs_list), np.array(velocities_list), processtime
