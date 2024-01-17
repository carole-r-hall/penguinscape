import csv
import math
import numpy as np
import os
import random
import time

from itertools import product
import numba
from numba import jit, njit


T = 36000000.0 # total amount of time allotted for simulation

shifts = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]])

# CONSTANTS
dt = 0.005 # s -- larger time steps can be tried, but may produce unrealistic dynamics if the time step is too large (i.e., faster to run but problematic results)
N = int(T/dt)

pot_e = []
empirical_sigma = 0.685 # sigma parameter found from parameter fitting protocol (describes distance at which potential energy equals zero)
sigma = 0.685 # distance where potential energy equals zero
pref_dist = sigma*(2.0**(1.0/6.0)) # preferred nesting distance derived from sigma
epsilon = 0.05 # well depth
cutoff = 2.5*sigma # cutoff distance

def generate_points_equal_spacing(n, center, rez):
    # print("generate points")
    n_sqrt = math.ceil(np.sqrt(n))
    coords = np.zeros((n,2))
    window_space = pref_dist/2.0
    rows = np.linspace(center[0] - (rez/2-window_space-0.5*pref_dist), center[0] + (rez/2-window_space-0.5*pref_dist), n_sqrt)
    cols = np.linspace(center[1] - (rez/2-window_space-0.5*pref_dist), center[1] + (rez/2-window_space-0.5*pref_dist), n_sqrt)

    all_combinations = list(product(rows,cols))
    
    # don't bias towards the left -- populate the assigned spaces in a random order
    np.random.shuffle(all_combinations)

    # only take the points needed from the grid in a shuffled order
    counter = 0
    for combo in all_combinations:
        if counter < n:
            coords[counter, 0] = combo[0]
            coords[counter, 1] = combo[1]
        counter += 1
    return coords

@njit(parallel = True)
def compute_forces(index_organizer, flat_indexes, nest_position, nest_forces, nest_potential_energy, n):
    for index in numba.prange(len(index_organizer)):
        [i,j] = index_organizer[index]
        for p in range(flat_indexes[i][j][0], flat_indexes[i][j][1]):
            # same-cell dynamics
            for q in range(flat_indexes[i][j][0], flat_indexes[i][j][1]):
                if q < p:
                    dx = nest_position[n,p,0] - nest_position[n,q,0]
                    dy = nest_position[n,p,1] - nest_position[n,q,1]
                    rsqr = dx*dx + dy*dy
                    sigma_sq = sigma*sigma
                    rsqrt = math.sqrt(rsqr)
                    if rsqrt < cutoff:
                        attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                        repel = attract*attract
                        potential_E = 4*epsilon*(repel - attract)
                        f_over_r = 24*epsilon*((2*repel) - attract)/rsqr
                        fx = f_over_r*dx
                        fy = f_over_r*dy
                        nest_forces[n,p,0] += fx
                        nest_forces[n,p,1] += fy
                        nest_forces[n,q,0] -= fx
                        nest_forces[n,q,1] -= fy
                        nest_potential_energy[n] += potential_E

            # for shift_index in range(len(shifts)):
            #     shift = shifts[shift_index]
            #     focus_i = i + shift[0]
            #     focus_j = j + shift[1]
            #     if coverages[focus_i][focus_j] != 0:
            #         for q in range(flat_indexes[focus_i][focus_j][0], flat_indexes[focus_i][focus_j][1]):
            #             dx = nest_position[n,p,0] - nest_position[n,q,0]
            #             dy = nest_position[n,p,1] - nest_position[n,q,1]
            #             rsqr = dx*dx + dy*dy
            #             sigma_sq = sigma*sigma
            #             rsqrt = math.sqrt(rsqr)
            #             if rsqrt < cutoff:
            #                  attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
            #                  repel = attract*attract
            #                  potential_E = 4 *(repel - attract)
            #                  f_over_r = 24*((2*repel) - attract)/rsqr
            #                  fx = f_over_r*dx
            #                  fy = f_over_r*dy
            #                  nest_forces[n,p,0] += fx
            #                  nest_forces[n,p,1] += fy
            #                  nest_potential_energy[n] += potential_E
            #         # Remove break to make it so that particles can interact with multiple neighbors
            #         break
            
            # removing conditional from  loop
            for shift_index in range(len(shifts)):
                shift = shifts[shift_index]
                focus_i = i + shift[0]
                focus_j = j + shift[1]
                if num_nests[focus_i][focus_j] > 0:
                    for q in range(flat_indexes[focus_i][focus_j][0], flat_indexes[focus_i][focus_j][1]):
                        dx = nest_position[n,p,0] - nest_position[n,q,0]
                        dy = nest_position[n,p,1] - nest_position[n,q,1]
                        rsqr = dx*dx + dy*dy
                        sigma_sq = sigma*sigma
                        rsqrt = math.sqrt(rsqr)
                        if rsqrt < cutoff:
                                attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                                repel = attract*attract
                                potential_E = 4*epsilon*(repel - attract)
                                f_over_r = 24*epsilon*((2*repel) - attract)/rsqr
                                fx = f_over_r*dx
                                fy = f_over_r*dy
                                nest_forces[n,p,0] += fx
                                nest_forces[n,p,1] += fy
                                nest_potential_energy[n] += potential_E
                # Remove break to make it so that particles can interact with multiple neighbors
                
@jit(nopython = True)
def other_slow_function(nest_velocities, nest_position, index_organizer, flat_indexes, r, v, lons, lats, n):
    # make sure particles don't leave box
    for pair_index in range(len(index_organizer)):
        [i,j] = index_organizer[pair_index]
        for nest in range(flat_indexes[i][j][0], flat_indexes[i][j][1]):
            if (nest_position[n,nest,0] <= lons[i][j]-(15-0.5*sigma) and nest_velocities[n,nest,0] < 0) or (nest_position[n,nest,0]
                                                                    >= lons[i][j]+(15-0.5*sigma) and nest_velocities[n,nest,0] > 0):
                nest_velocities[n,nest,0] *= -0.25
                
                
            if (nest_position[n,nest,1] <= lats[i][j]-(15.0-0.5*sigma) and nest_velocities[n,nest,1] < 0) or (nest_position[n,nest,1]
                                                                        >= lats[i][j]+(15.0-0.5*sigma) and nest_velocities[n,nest,1] > 0):
                nest_velocities[n,nest,1] *= -0.25

                

if __name__ == "__main__":
    
    counter = 0
    
    with open('./data/height_width.csv', 'r') as file_obj: 
      
        # Create reader object by passing the file 
        # object to DictReader method 
        reader_obj = csv.reader(file_obj) 

        # Iterate over each row in the csv file 
        # using reader object 
        for row in reader_obj: 
            if counter == 0:
                height = int(row[0])
                width = int(row[1])
            else:
                break
            counter += 1
    
    index_organizer_i = []
    index_organizer_j = []
    
    lons = np.zeros((height, width))
    lats = np.zeros((height, width))
    coverage_percentages = np.zeros((height, width))
    
    coordinate_organizer = [[[[]] for i in range(width)] for j in range(height)]
    
    with open('./data/lons_data.csv', 'r') as file_lons: 
      
        # Create reader object by passing the file 
        # object to DictReader method 
        reader_lons = csv.reader(file_lons) 

        # Iterate over each row in the csv file 
        # using reader object 
        i_1 = 0
        for row in reader_lons: 
            for j_1 in range(len(row)):
                lons[i_1,j_1] = float(row[j_1])
            i_1 += 1
            
    with open('./data/lats_data.csv') as file_lats: 
      
        # Create reader object by passing the file 
        # object to DictReader method 
        reader_lats = csv.reader(file_lats) 

        # Iterate over each row in the csv file 
        # using reader object 
        i_2 = 0
        for row in reader_lats: 
            for j_2 in range(len(row)):
                lats[i_2,j_2] = float(row[j_2])
            i_2 += 1
        
    with open('./data/coverage_percentages.csv', 'r') as file_covs:
        reader_covs = csv.reader(file_covs)
        i_3 = 0
        for row in reader_covs:
            for j_3 in range(len(row)):
                coverage_percentages[i_3,j_3] = float(row[j_3])
            i_3 += 1
        
    print("coverage_percentages_max = ", np.amax(coverage_percentages))
    coverage_percentages = (coverage_percentages*900.0)/(pref_dist**2.0)
    num_nests = coverage_percentages.astype(np.int32)
    total_num_nests = num_nests.sum()
    print("total num nests: ", total_num_nests)
    coverage_mask = np.copy(num_nests.astype(np.int8))
    flat_indexes = np.zeros((height,width,2), dtype=np.int32)
    index_tracker = 0
    
    flat_indexes = np.zeros((height,width,2), dtype=np.int32)
    index_tracker = 0
    
    #num_nests = coverages.astype(np.int32)
    total_num_nests = num_nests.sum()

    for i in range(height):
        for j in range(width):
            coordinate_organizer[i][j] = generate_points_equal_spacing(int(num_nests[i][j]), center = [lons[i][j], lats[i][j]], rez = 29)
            index_organizer_i.append(i)
            index_organizer_j.append(j)
            flat_indexes[i][j][0] = index_tracker
            index_tracker += num_nests[i][j]
            flat_indexes[i][j][1] = index_tracker-1
                
    index_organizer_i = np.array(index_organizer_i, dtype=np.int32)
    index_organizer_j = np.array(index_organizer_j, dtype=np.int32)
    index_organizer = np.array(list(zip(index_organizer_i, index_organizer_j)), dtype=np.int32)

    nest_velocities = np.zeros((3, total_num_nests, 2)) # velocities of particles
    nest_position = np.zeros((3, total_num_nests, 2)) # positions of particles
    nest_forces = np.zeros((3, total_num_nests, 2)) # forces exerted on particles
    nest_potential_energy = np.zeros((3)) # store potential energy for all time steps
    
    nest_position = np.zeros((3, total_num_nests, 2)) # positions of particles
    
    ''''----P4: DEFINING INITIAL CONDITIONS FOR NESTS----'''
    idx = 0
    
    for pair in index_organizer:
        (i,j) = pair
        nest_position[0,idx:idx+num_nests[i][j]] = coordinate_organizer[i][j][0:num_nests[i][j]]
        idx += num_nests[i][j]
    
    coords_x = open('./output_files/coords_list_x.csv', 'a+', newline ='')
    # writing the data into the file
    with coords_x:    
        write = csv.writer(coords_x)
        write.writerow(nest_position[0,:,0])
    coords_x.close()
    coords_y = open('./output_files/coords_list_y.csv', 'a+', newline ='')
    # writing the data into the file
    with coords_y:    
        write = csv.writer(coords_y)
        write.writerow(nest_position[0,:,1])
    coords_y.close()

    energies_list = open('./output_files/energies_list.csv', 'a+', newline = '')
    with energies_list:
        write = csv.writer(energies_list)
        write.writerow([nest_potential_energy[0]])
    energies_list.close()

    timesteps_file = open('./output_files/timesteps.csv', 'a+', newline = '')
    with timesteps_file:
        write = csv.writer(timesteps_file)
        write.writerow(['time'])
    timesteps_file.close()
    
        
    

    pot_e.append(nest_potential_energy[0])
    true_n = 0 # timesteps to keep track of the potential energy, other values aren't stored except most recent time steps

    for pair_index in range(len(index_organizer)):
        pair = index_organizer[pair_index]
        [i,j] = pair
        for p in range(flat_indexes[i][j][0], flat_indexes[i][j][1]):
            nest_velocities[0,p] = [random.choice([-1,1])*1+random.uniform(-0.25,0.25), random.choice([-1,1])*1+random.uniform(-0.25,0.25)]
    
    for pair_index in range(len(index_organizer)):
        pair = index_organizer[pair_index]
        [i,j] = pair
        for nest in range(flat_indexes[i][j][0], flat_indexes[i][j][1]):
            if (nest_position[0,nest,0] <= lons[i][j]-(15-0.5*sigma) and nest_velocities[0,nest,0] < 0) or (nest_position[0,nest,0]
                                                                    >= lons[i][j]+(15-0.5*sigma) and nest_velocities[0,nest,0] > 0):
                nest_velocities[0,nest,0] *= -0.25
                
            if (nest_position[0,nest,1] <= lats[i][j]-(15.0-0.5*sigma) and nest_velocities[0,nest,1] < 0) or (nest_position[0,nest,1]
                                                                        >= lats[i][j]+(15.0-0.5*sigma) and nest_velocities[0,nest,1] > 0):
                nest_velocities[0,nest,1] *= -0.25
    
    # coverage = coverage.where
    # Need to call other function (i think it PE here)
    compute_forces(index_organizer, flat_indexes, nest_position, nest_forces, nest_potential_energy, 0)
    # update position
    nest_position[1] = nest_position[0] + dt*nest_velocities[0] + 0.5*dt*dt*nest_forces[0]
    compute_forces(index_organizer, flat_indexes, nest_position, nest_forces, nest_potential_energy, 1)

    # update velocity
    nest_velocities[1] = nest_velocities[0] + 0.5*dt*(nest_forces[0]*nest_forces[1])
    other_slow_function(nest_velocities, nest_position, index_organizer, flat_indexes, nest_position, nest_velocities, lons, lats,1)

    pot_e.append(nest_potential_energy[1])
    
    n = 2
    while True == True:
        time1 = time.perf_counter()
        # update position
        nest_position[2] = nest_position[1] + dt*nest_velocities[1] + 0.5*dt*dt*nest_forces[1]

        # update forces/energies
        # start = time.time()
        compute_forces(index_organizer, flat_indexes, nest_position, nest_forces, nest_potential_energy, 2)
        # end = time.time()
        # print("compute forces time = ", end-start)
        # update velocity
        nest_velocities[2] = nest_velocities[1] + 0.5*dt*(nest_forces[1] + nest_forces[2])

        # make sure particles don't leave box
        other_slow_function(nest_velocities, nest_position, index_organizer, flat_indexes, nest_position, nest_velocities, lons, lats,2)

        time2 = time.perf_counter()

        if n % 10000 == 1:
            
            coords_x = open('./output_files/coords_list_x.csv', 'a+', newline ='')

            # writing the data into the file
            with coords_x:    
                write = csv.writer(coords_x)
                write.writerow(nest_position[1,:,0])
            coords_x.close()


            coords_y = open('./output_files/coords_list_y.csv', 'a+', newline ='')

            # writing the data into the file
            with coords_y:    
                write = csv.writer(coords_y)
                write.writerow(nest_position[1,:,1])
            coords_y.close()



            energies_file = open('./output_files/energies_list.csv', 'a+', newline = '')

            with energies_file:
                write = csv.writer(energies_file)
                write.writerow([nest_potential_energy[1]])
            energies_file.close()

            timesteps_file = open('./output_files/timesteps.csv', 'a+', newline = '')

            with timesteps_file:
                write = csv.writer(timesteps_file)
                write.writerow([time2 - time1])
            timesteps_file.close()


            # switch so that the current is in index 1 and we'll put all the new stuff in position 2

        nest_velocities[1] = nest_velocities[2,:]
        nest_position[1] = nest_position[2,:]
        nest_forces[1] = nest_forces[2,:]
        nest_potential_energy[1] = nest_potential_energy[2]
        nest_forces[2] = 0
        nest_potential_energy[2] = 0
        pot_e.append(nest_potential_energy[1])
        
        n += 1

        #print("One step takes {t} seconds".format(t = time2 - time1))
    
    
    
