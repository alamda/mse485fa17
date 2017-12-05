# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 15:45:58 2017

@author: Alexandra
"""

import numpy as np, subprocess, itertools
import matplotlib.pyplot as plt

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
k_electric = 1    # electric constant, equal to (1 / (4 \pi \epsilon_0))
n_neural_particles = 10  # number of neural particles on each side
epsilon_LJ = 2.5; sigma_LJ = 0.25  # they should be different for different particles
random_force_strength = 2.0
const_force = 0.0  # constant force that drives water molecule towards the right

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

def init_config():
    # generate random (x,y,theta, v_x, v_y, omega)
    temp_x = np.random.uniform(low=0.4 * L_channel, high=0.6 * L_channel) #I made it closer to the center
    
    temp_y = np.random.uniform(low=0.4 * R_channel, high=0.6 * R_channel) #I made it closer to the center
    
    temp_theta = np.random.uniform(high= 2 * np.pi)
    
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
    
    #print (temp_LJ_Force, temp_electric_Force)
    
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

def simulate(num_steps, h_stepsize=0.01):
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
    
    return np.array(configs_list), np.array(velocities_list)

def get_poorly_sampled_points(list_of_points, range_of_each_dim,
                            num_of_bins = 10,
                            num_of_boundary_points = 10,
                            preprocessing = True,
                            auto_range_for_histogram = False,   # set the range of histogram based on min,max values in each dimension
                            ):
    """
    this function is copied from https://github.com/weiHelloWorld/accelerated_sampling_with_autoencoder/blob/master/MD_simulation_on_alanine_dipeptide/current_work/src/molecule_spec_sutils.py#L360
    for finding poorly sampled region
    :param preprocessing: if True, then more weight is not linear, this would be better based on experience
    """
    dimensionality = len(list_of_points[0])
    list_of_points = list(zip(*list_of_points))
    assert (len(list_of_points) == dimensionality)
    hist_matrix, edges = np.histogramdd(list_of_points, bins= num_of_bins * np.ones(dimensionality), range = range_of_each_dim)

    # following is the main algorithm to find boundary and holes
    # simply find the points that are lower than average of its 4 neighbors
    if preprocessing:
        hist_matrix = np.array(map(lambda x: map(lambda y: - np.exp(- y), x), hist_matrix))   # preprocessing process

    diff_with_neighbors = np.zeros(hist_matrix.shape)
    temp_1 = [list(range(item)) for item in hist_matrix.shape]
    for grid_index in itertools.product(*temp_1):
        neighbor_index_list = [(np.array(grid_index) + temp_2).astype(int) for temp_2 in np.eye(dimensionality)]
        neighbor_index_list += [(np.array(grid_index) - temp_2).astype(int) for temp_2 in np.eye(dimensionality)]
        neighbor_index_list = filter(lambda x: np.all(x >= 0) and np.all(x < num_of_bins), neighbor_index_list)
        # print "grid_index = %s" % str(grid_index)
        # print "neighbor_index_list = %s" % str(neighbor_index_list)
        diff_with_neighbors[tuple(grid_index)] = hist_matrix[tuple(grid_index)] - np.average(
            [hist_matrix[tuple(temp_2)] for temp_2 in neighbor_index_list]
        )

    # get grid centers
    edge_centers = map(lambda x: 0.5 * (np.array(x[1:]) + np.array(x[:-1])), edges)
    grid_centers = np.array(list(itertools.product(*edge_centers)))  # "itertools.product" gives Cartesian/direct product of several lists
    grid_centers = np.reshape(grid_centers, np.append(num_of_bins * np.ones(dimensionality), dimensionality).astype(int))

    potential_centers = []
    # now sort these grids (that has no points in it)
    # based on total number of points in its neighbors
    
    temp_seperate_index = []

    for _ in range(dimensionality):
        temp_seperate_index.append(list(range(num_of_bins)))

    index_of_grids = list(itertools.product(
                    *temp_seperate_index
                    ))

    index_of_grids =  filter(lambda x: diff_with_neighbors[x] < 0, index_of_grids)     # only apply to grids with diff_with_neighbors value < 0
    sorted_index_of_grids = sorted(index_of_grids, key = lambda x: diff_with_neighbors[x]) # sort based on histogram, return index values

    for index in sorted_index_of_grids[:num_of_boundary_points]:  # note index can be of dimension >= 2
        temp_potential_center = map(lambda x: round(x, 2), grid_centers[index])
        potential_centers.append(temp_potential_center)

    return potential_centers

if __name__ == '__main__':
    my_positions, my_velocities = simulate(500)

    plt.figure(1)
    plt.scatter(coords_particles[:,0], coords_particles[:,1])
    #plt.scatter(coords_negative_particles[:,0], coords_negative_particles[:,1])
    plt.scatter(my_positions[:,0], my_positions[:,1])
    plt.show()

    print(my_positions)
    

def clustering():
    pass

def get_new_starting_configs():
    pass

def get_rewards():
    pass

def get_volumn_of_explored_region():
    pass