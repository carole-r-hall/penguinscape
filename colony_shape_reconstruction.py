# Author: Carole Hall
# Description: this program performs a reconstruction procedure grounded in molecular dynamics
# to obtain high-resolution penguin colony shape information from low-resolution remote sensing 
# data. Any questions about this code, or related inquiries can be sent to carole.hall@stonybrook.edu. 

import numpy as np
import math
import random
import pandas as pd
import rasterio
import rasterio.features
import rasterio.warp
import os
from itertools import product
import csv

# NOTE: this code as-is pertains to the reconstruction of the Cape Adare colony (ADAR);
# this code may be easily modified to reconstruct the included data for BERK, BSON, DEVI, INEX, and POSS.
# coarsening factors are provided in the text related to this code, however any coarsening factor can be
# used in practice to either get finer-resolution results (smaller coarsening factor) or to speed up the
# simulation (larger coarsening factor). 

# DEFINE WHERE PLOTS SHOULD BE SAVED

output_folder = "./output_files_adar"

# DEFINE LJ PARAMETERS

# parameters of the simulation
T = 36000.0 # total amount of time allotted for simulation
empirical_sigma = 0.685 # sigma parameter found from parameter fitting protocol (describes distance at which potential energy equals zero)
coarse_f = 5.0 # coarsening factor varying by colony
sigma = coarse_f*0.685 # distance where potential energy equals zero - scaled according to coarsening factor
pref_dist = sigma*(2.0**(1.0/6.0)) # preferred nesting distance derived from sigma
epsilon = 1.0 # well depth
cutoff = 2.5*sigma # cutoff distance


'''----METHODS DEFINITION----'''

# nest initialization on a pseudo-random grid

def generate_points_equal_spacing(n, center, rez):
    n_sqrt = math.ceil(np.sqrt(n))
    coords = np.zeros((n,2))
    window_space = 1.1*pref_dist/2.0
    rows = np.linspace(center[0] - (rez/2-window_space), center[0] + (rez/2-window_space), n_sqrt)
    cols = np.linspace(center[1] - (rez/2-window_space), center[1] + (rez/2-window_space), n_sqrt)

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


'''----P1: READING IN THE DATA----'''

# create a raster mask for colony vals
colony_raster = rasterio.open('adar_raster.tif') # raster where each pixel value is %-guano cover

array = colony_raster.read()

print(array.shape) # dimensions of raster + band


# extract the values in the same shape as the raster
coverages = array[0] #only one band (% guano) in this raster, hence the [0] index
        

# don't deal with fractions of nests
for r in range(len(coverages)):
    for c in range(len(coverages[0])):
        if coverages[r][c]*900.0/(1.5*pref_dist**2.0) < 1.0: # cant have a fraction of a nest
            coverages[r][c] = 0

# get coordinates for the ``coordinate organizer'' -- this is a matrix where each value is the CENTER COORDINATE of each pixel. i.e., coordinate_organizer[i][j] gives the pixel center coord of the pixel in the i-th row and j-th col of raster
with rasterio.open('adar_raster.tif') as dataset:
    mask = dataset.dataset_mask()
    band = dataset.read(1)
    height = band.shape[0]
    width = band.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(dataset.transform, rows, cols)
    lons = np.array(xs) # coordinates of centers of each pixel
    lats = np.array(ys) # coordinates of centers of each pixel

coordinate_organizer = [[[[]] for i in range(width)] for j in range(height)]
print("FIRST LONS = ", lats[0][0])
print("SHAPE = ", lons.shape)

'''----P2: GENERATION/ORGANIZATION OF GRID POINTS----'''

rez = 30.0
pot_e = []

# point generation -- coordinates of each pixel are kept in coordinate_organizer
# to access a set of coordinates in pixel in row i and column j, call:
# coordinate_organizer[i][j]
cutoff_for_movement = 1.1 #SET CUTOFF TO 0.8 TO NOT CONSIDERED PIXELS WITH OVER 80% COVERAGE
total_num_nests = 0
index_tracker = 0
flat_indexes = [[[] for i in range(width)] for j in range(height)]
num_particles = [[0 for i in range(width)] for j in range(height)]
index_organizer_i = []
index_organizer_j = []
neg_index_organizer_i = []
neg_index_organizer_j = []
for i in range(height):
    for j in range(width):
        if coverages[i][j] > 0.0:
            n = int((900.0*coverages[i][j])/(1.1*pref_dist**2.0)) # get num nests from cov val - divide by spacing you want for nests
            num_particles[i][j] = n
            coords = generate_points_equal_spacing(n = n, center = [lons[i][j], lats[i][j]], rez = 30.0)
            coordinate_organizer[i][j] = coords
            total_num_nests += n
            index_organizer_i.append(i)
            index_organizer_j.append(j)
            flat_indexes[i][j] = range(index_tracker,index_tracker+n)
            index_tracker += n
            
index_organizer = list(zip(index_organizer_i, index_organizer_j))
neg_index_organizer = list(zip(neg_index_organizer_i, neg_index_organizer_j))

print("total num nests= ", total_num_nests)

# index_organizer is a list, where each element is a pair (i,j) corresponding to a cell we'd be moving in the simulation--over-populated and empty pixels are ignored and not stored here to save time
# neg_index_organizer is a list, where each element is a pair (i,j) corresponding to the overly-populated cells we want to keep constant. Empty pixels are not stored here


'''----P3: DEFINING THE ENVIRONMENT----'''

# setting a timestep
dt = 0.005 # s -- larger time steps can be tried, but may produce unrealistic dynamics if the time step is too large (i.e., faster to run but problematic results)
N = int(T/dt) # num timesteps in the sim


'''----P3a: ALLOCATION OF ARRAYS: FIRST AXIS: TIME, SECOND: PARTICLE'S INDEX, THIRD: VALUE----'''

v = np.zeros((3, total_num_nests, 2)) # velocities of particles
r = np.zeros((3, total_num_nests, 2)) # positions of particles
f = np.zeros((3, total_num_nests, 2)) # forces exerted on particles
e = np.zeros((3)) # store potential energy for all time steps


'''----P4: DEFINING INITIAL CONDITIONS FOR NESTS----'''

idx = 0
for pair in index_organizer:
    (i,j) = pair
    r[0,idx:idx+num_particles[i][j]] = coordinate_organizer[i][j][0:num_particles[i][j]]
    idx += num_particles[i][j]


coords_x = open('./output_files_adar/coords_list_x.csv', 'a+', newline ='')
  
# writing the data into the file
with coords_x:    
    write = csv.writer(coords_x)
    write.writerow(r[0,:,0])
coords_x.close()


coords_y = open('./output_files_adar/coords_list_y.csv', 'a+', newline ='')
  
# writing the data into the file
with coords_y:    
    write = csv.writer(coords_y)
    write.writerow(r[0,:,1])
coords_y.close()


velocities_x = open('./output_files_adar/velocities_list_x.csv', 'a+', newline ='')
  
# writing the data into the file
with velocities_x:    
    write = csv.writer(velocities_x)
    write.writerow(v[0,:,0])
velocities_x.close()


velocities_y = open('./output_files_adar/velocities_list_y.csv', 'a+', newline ='')
  
# writing the data into the file
with velocities_y:    
    write = csv.writer(velocities_y)
    write.writerow(v[0,:,1])
velocities_y.close()

with open('./output_files_adar/energies_list.csv', mode='w') as csv_file_e:
    fieldnames = ['E']
    writer_e = csv.DictWriter(csv_file_e, fieldnames=fieldnames)
    writer_e.writerow({'E': str(e[0])})
csv_file_e.close()

pot_e.append(e[0])
true_n = 0 # timesteps to keep track of the potential energy, other values aren't stored except most recent time steps

'''----P4a: COMPUTING FORCES----'''

def compute_forces(n,true_n):
    for pair in index_organizer:
        (i,j) = pair
        for p in flat_indexes[i][j]:
            # same-cell dynamics
            for q in flat_indexes[i][j]:
                if q < p:
                    dx = r[n,p,0] - r[n,q,0]
                    dy = r[n,p,1] - r[n,q,1]
                    rsqr = dx*dx + dy*dy
                    sigma_sq = sigma*sigma
                    rsqrt = math.sqrt(rsqr)
                    if rsqrt < cutoff:
                         attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                         repel = attract*attract
                         potential_E = 4 *(repel - attract)
                         f_over_r = 24*((2*repel) - attract)/rsqr
                         fx = f_over_r*dx
                         fy = f_over_r*dy
                         f[n,p,0] += fx
                         f[n,p,1] += fy
                         f[n,q,0] -= fx
                         f[n,q,1] -= fy
                         e[n] += potential_E
            
            if coverages[i-1][j-1] > 0.0: # upper left neighbor
                for q in flat_indexes[i-1][j-1]:
                    dx = r[n,p,0] - r[n,q,0]
                    dy = r[n,p,1] - r[n,q,1]
                    rsqr = dx*dx + dy*dy
                    sigma_sq = sigma*sigma
                    rsqrt = math.sqrt(rsqr)
                    if rsqrt < cutoff:
                         attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                         repel = attract*attract
                         potential_E = 4 *(repel - attract)
                         f_over_r = 24*((2*repel) - attract)/rsqr
                         fx = f_over_r*dx
                         fy = f_over_r*dy
                         f[n,p,0] += fx
                         f[n,p,1] += fy
                         e[n] += potential_E
            
            elif coverages[i-1][j] > 0.0: # upper middle neighbor
                for q in flat_indexes[i-1][j]:
                    dx = r[n,p,0] - r[n,q,0]
                    dy = r[n,p,1] - r[n,q,1]
                    rsqr = dx*dx + dy*dy
                    sigma_sq = sigma*sigma
                    rsqrt = math.sqrt(rsqr)
                    if rsqrt < cutoff:
                         attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                         repel = attract*attract
                         potential_E = 4 *(repel - attract)
                         f_over_r = 24*((2*repel) - attract)/rsqr
                         fx = f_over_r*dx
                         fy = f_over_r*dy
                         f[n,p,0] += fx
                         f[n,p,1] += fy
                         e[n] += potential_E
            
            elif coverages[i-1][j+1] > 0.0: # upper right neighbor
                for q in flat_indexes[i-1][j+1]:
                    dx = r[n,p,0] - r[n,q,0]
                    dy = r[n,p,1] - r[n,q,1]
                    rsqr = dx*dx + dy*dy
                    sigma_sq = sigma*sigma
                    rsqrt = math.sqrt(rsqr)
                    if rsqrt < cutoff:
                         attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                         repel = attract*attract
                         potential_E = 4 * (repel - attract)
                         f_over_r = 24*((2*repel) - attract)/rsqr
                         fx = f_over_r*dx
                         fy = f_over_r*dy
                         f[n,p,0] += fx
                         f[n,p,1] += fy
                         e[n] += potential_E
            
            elif coverages[i][j-1] > 0.0: # left neighbor
                for q in flat_indexes[i][j-1]:
                    dx = r[n,p,0] - r[n,q,0]
                    dy = r[n,p,1] - r[n,q,1]
                    rsqr = dx*dx + dy*dy
                    sigma_sq = sigma*sigma
                    rsqrt = math.sqrt(rsqr)
                    if rsqrt < cutoff:
                         attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                         repel = attract*attract
                         potential_E = 4 *(repel - attract)
                         f_over_r = 24*((2*repel) - attract)/rsqr
                         fx = f_over_r*dx
                         fy = f_over_r*dy
                         f[n,p,0] += fx
                         f[n,p,1] += fy
                         e[n] += potential_E
            
            elif coverages[i][j+1] > 0.0: # right neighbor
                for q in flat_indexes[i][j+1]:
                    dx = r[n,p,0] - r[n,q,0]
                    dy = r[n,p,1] - r[n,q,1]
                    rsqr = dx*dx + dy*dy
                    sigma_sq = sigma*sigma
                    rsqrt = math.sqrt(rsqr)
                    if rsqrt < cutoff:
                         attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                         repel = attract*attract
                         potential_E = 4 *(repel - attract)
                         f_over_r = 24*((2*repel) - attract)/rsqr
                         fx = f_over_r*dx
                         fy = f_over_r*dy
                         f[n,p,0] += fx
                         f[n,p,1] += fy
                         e[n] += potential_E
            
            elif coverages[i+1][j-1] > 0.0: # lower left neighbor
                for q in flat_indexes[i+1][j-1]:
                    dx = r[n,p,0] - r[n,q,0]
                    dy = r[n,p,1] - r[n,q,1]
                    rsqr = dx*dx + dy*dy
                    sigma_sq = sigma*sigma
                    rsqrt = math.sqrt(rsqr)
                    if rsqrt < cutoff:
                         attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                         repel = attract*attract
                         potential_E = 4 * (repel - attract)
                         f_over_r = 24*((2*repel) - attract)/rsqr
                         fx = f_over_r*dx
                         fy = f_over_r*dy
                         f[n,p,0] += fx
                         f[n,p,1] += fy
                         e[n] += potential_E
                         
            elif coverages[i+1][j] > 0.0: # lower middle neighbor
                for q in flat_indexes[i+1][j]:
                    dx = r[n,p,0] - r[n,q,0]
                    dy = r[n,p,1] - r[n,q,1]
                    rsqr = dx*dx + dy*dy
                    sigma_sq = sigma*sigma
                    rsqrt = math.sqrt(rsqr)
                    if rsqrt < cutoff:
                         attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                         repel = attract*attract
                         potential_E = 4 * (repel - attract)
                         f_over_r = 24*((2*repel) - attract)/rsqr
                         fx = f_over_r*dx
                         fy = f_over_r*dy
                         f[n,p,0] += fx
                         f[n,p,1] += fy
                         e[n] += potential_E

            elif coverages[i+1][j+1] > 0.0: # lower right neighbor
                for q in flat_indexes[i+1][j+1]:
                    dx = r[n,p,0] - r[n,q,0]
                    dy = r[n,p,1] - r[n,q,1]
                    rsqr = dx*dx + dy*dy
                    sigma_sq = sigma*sigma
                    rsqrt = math.sqrt(rsqr)
                    if rsqrt < cutoff:
                         attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                         repel = attract*attract
                         potential_E = 4*(repel - attract)
                         f_over_r = 24*((2*repel) - attract)/rsqr
                         fx = f_over_r*dx
                         fy = f_over_r*dy
                         f[n,p,0] += fx
                         f[n,p,1] += fy
                         e[n] += potential_E

for pair in index_organizer:
    (i,j) = pair
    for p in flat_indexes[i][j]:
        v[0,p] = [random.choice([-1,1])*1+random.uniform(-0.25,0.25), random.choice([-1,1])*1+random.uniform(-0.25,0.25)]

for pair in index_organizer:
    (i,j) = pair
    for nest in flat_indexes[i][j]:
        if (r[0,nest,0] <= lons[i][j]-15.0 and v[0,nest,0] < 0) or (r[0,nest,0]
                                                                  >= lons[i][j]+15.0 and v[0,nest,0] > 0) or (r[0,nest,1]
                                                                    <= lats[i][j]-15.0 and v[0,nest,1] < 0) or (r[0,nest,1]
                                                                    >= lats[i][j]+15.0 and v[0,nest,1] > 0):
            v[1,nest,0] *= -0.25
            v[1,nest,1] *= -0.25

'''----P4b: COMPUTING ENERGETICS AT CURRENT POSITION/ UPDATES TO v/r----'''

for pair in index_organizer:
    (i,j) = pair
    for p in flat_indexes[i][j]:
        # same-cell dynamics
        for q in flat_indexes[i][j]:
            if q < p:
                dx = r[0,p,0] - r[0,q,0]
                dy = r[0,p,1] - r[0,q,1]
                rsqr = dx*dx + dy*dy
                sigma_sq = sigma*sigma
                rsqrt = math.sqrt(rsqr)
                if rsqrt < cutoff:
                     attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                     repel = attract*attract
                     potential_E = 4 * (repel - attract)
                     f_over_r = 24*((2*repel) - attract)/rsqr
                     fx = f_over_r*dx
                     fy = f_over_r*dy
                     f[0,p,0] += fx
                     f[0,p,1] += fy
                     f[0,q,0] -= fx
                     f[0,q,1] -= fy
                     e[0] += potential_E
                     
        if coverages[i-1][j-1] > 0.0: # upper left neighbor
            for q in flat_indexes[i-1][j-1]:
                dx = r[0,p,0] - r[0,q,0]
                dy = r[0,p,1] - r[0,q,1]
                rsqr = dx*dx + dy*dy
                sigma_sq = sigma*sigma
                rsqrt = math.sqrt(rsqr)
                if rsqrt < cutoff:
                     attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                     repel = attract*attract
                     potential_E = 4 * (repel - attract)
                     f_over_r = 24*((2*repel) - attract)/rsqr
                     fx = f_over_r*dx
                     fy = f_over_r*dy
                     f[0,p,0] += fx
                     f[0,p,1] += fy
                     e[0] += potential_E
                     
        elif coverages[i-1][j] > 0.0: # upper middle neighbor
            for q in flat_indexes[i-1][j]:
                dx = r[0,p,0] - r[0,q,0]
                dy = r[0,p,1] - r[0,q,1]
                rsqr = dx*dx + dy*dy
                sigma_sq = sigma*sigma
                rsqrt = math.sqrt(rsqr)
                if rsqrt < cutoff:
                     attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                     repel = attract*attract
                     potential_E = 4 * (repel - attract)
                     f_over_r = 24*((2*repel) - attract)/rsqr
                     fx = f_over_r*dx
                     fy = f_over_r*dy
                     f[0,p,0] += fx
                     f[0,p,1] += fy
                     e[0] += potential_E                
                    
        elif coverages[i-1][j+1] > 0.0: # upper right neighbor
            for q in flat_indexes[i-1][j+1]:
                dx = r[0,p,0] - r[0,q,0]
                dy = r[0,p,1] - r[0,q,1]
                rsqr = dx*dx + dy*dy
                sigma_sq = sigma*sigma
                rsqrt = math.sqrt(rsqr)
                if rsqrt < cutoff:
                     attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                     repel = attract*attract
                     potential_E = 4 * (repel - attract)
                     f_over_r = 24*((2*repel) - attract)/rsqr
                     fx = f_over_r*dx
                     fy = f_over_r*dy
                     f[0,p,0] += fx
                     f[0,p,1] += fy
                     e[0] += potential_E          

        elif coverages[i][j-1] > 0.0: # left neighbor
            for q in flat_indexes[i][j-1]:
                dx = r[0,p,0] - r[0,q,0]
                dy = r[0,p,1] - r[0,q,1]
                rsqr = dx*dx + dy*dy
                sigma_sq = sigma*sigma
                rsqrt = math.sqrt(rsqr)
                if rsqrt < cutoff:
                     attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                     repel = attract*attract
                     potential_E = 4 * (repel - attract)
                     f_over_r = 24*((2*repel) - attract)/rsqr
                     fx = f_over_r*dx
                     fy = f_over_r*dy
                     f[0,p,0] += fx
                     f[0,p,1] += fy
                     e[0] += potential_E      

        elif coverages[i][j+1] > 0.0: # right neighbor
            for q in flat_indexes[i][j+1]:
                dx = r[0,p,0] - r[0,q,0]
                dy = r[0,p,1] - r[0,q,1]
                rsqr = dx*dx + dy*dy
                sigma_sq = sigma*sigma
                rsqrt = math.sqrt(rsqr)
                if rsqrt < cutoff:
                     attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                     repel = attract*attract
                     potential_E = 4 * (repel - attract)
                     f_over_r = 24*((2*repel) - attract)/rsqr
                     fx = f_over_r*dx
                     fy = f_over_r*dy
                     f[0,p,0] += fx
                     f[0,p,1] += fy
                     e[0] += potential_E      

        elif coverages[i+1][j-1] > 0.0: # lower left neighbor
            for q in flat_indexes[i+1][j-1]:
                dx = r[0,p,0] - r[0,q,0]
                dy = r[0,p,1] - r[0,q,1]
                rsqr = dx*dx + dy*dy
                sigma_sq = sigma*sigma
                rsqrt = math.sqrt(rsqr)
                if rsqrt < cutoff:
                     attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                     repel = attract*attract
                     potential_E = 4 *(repel - attract)
                     f_over_r = 24 *((2*repel) - attract)/rsqr
                     fx = f_over_r*dx
                     fy = f_over_r*dy
                     f[0,p,0] += fx
                     f[0,p,1] += fy
                     e[0] += potential_E      

        elif coverages[i+1][j] > 0.0: # lower middle neighbor
            for q in flat_indexes[i+1][j]:
                dx = r[0,p,0] - r[0,q,0]
                dy = r[0,p,1] - r[0,q,1]
                rsqr = dx*dx + dy*dy
                sigma_sq = sigma*sigma
                rsqrt = math.sqrt(rsqr)
                if rsqrt < cutoff:
                     attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                     repel = attract*attract
                     potential_E = 4 * (repel - attract)
                     f_over_r = 24*((2*repel) - attract)/rsqr
                     fx = f_over_r*dx
                     fy = f_over_r*dy
                     f[0,p,0] += fx
                     f[0,p,1] += fy
                     e[0] += potential_E


        elif coverages[i+1][j+1] > 0.0: # lower right neighbor
            for q in flat_indexes[i+1][j+1]:
                dx = r[0,p,0] - r[0,q,0]
                dy = r[0,p,1] - r[0,q,1]
                rsqr = dx*dx + dy*dy
                sigma_sq = sigma*sigma
                rsqrt = math.sqrt(rsqr)
                if rsqrt < cutoff:
                     attract = (sigma_sq*sigma_sq*sigma_sq)/(rsqr*rsqr*rsqr)
                     repel = attract*attract
                     potential_E = 4 * (repel - attract)
                     f_over_r = 24*((2*repel) - attract)/rsqr
                     fx = f_over_r*dx
                     fy = f_over_r*dy
                     f[0,p,0] += fx
                     f[0,p,1] += fy
                     e[0] += potential_E

# update position

r[1] = r[0] + dt*v[0] + 0.5*dt*dt*f[0]

# update energy and force

compute_forces(1,1)

# update velocity

v[1] = v[0] + 0.5*dt*(f[0]*f[1])

# make sure particles don't leave box
for pair in index_organizer:
    (i,j) = pair
    for nest in flat_indexes[i][j]:
        if (r[1,nest,0] <= lons[i][j]-15.0 and v[1,nest,0] < 0) or (r[1,nest,0]
                                                                  >= lons[i][j]+15.0 and v[1,nest,0] > 0) or (r[1,nest,1]
                                                                    <= lats[i][j]-15.0 and v[1,nest,1] < 0) or (r[1,nest,1]
                                                                    >= lats[i][j]+15.0 and v[1,nest,1] > 0):
            v[1,nest,0] *= -0.25
            v[1,nest,1] *= -0.25
            
pot_e.append(e[1])

for n in range(2,N):
    # update position
    r[2] = r[1] + dt*v[1] + 0.5*dt*dt*f[1]

    # update forces/energies
    compute_forces(2,n)

    # update velocity
    v[2] = v[1] + 0.5*dt*(f[1] + f[2])

    # make sure particles don't leave box

    for pair in index_organizer:
        (i,j) = pair
        for nest in flat_indexes[i][j]:
            if (r[2,nest,0] <= lons[i][j]-15.0 and v[2,nest,0] < 0) or (r[2,nest,0]
                                                                      >= lons[i][j]+15.0 and v[2,nest,0] > 0) or (r[2,nest,1]
                                                                        <= lats[i][j]-15.0 and v[2,nest,1] < 0) or (r[2,nest,1]
                                                                        >= lats[i][j]+15.0 and v[2,nest,1] > 0):
                v[2,nest,0] *= -0.25
                v[2,nest,1] *= -0.25

# write results to the CSV files every 1000 timesteps

    if n % 1000 == 1:

        coords_x = open('./output_files_adar/coords_list_x.csv', 'a+', newline ='')
          
        # writing the data into the file
        with coords_x:    
            write = csv.writer(coords_x)
            write.writerow(r[1,:,0])
        coords_x.close()


        coords_y = open('./output_files_adar/coords_list_y.csv', 'a+', newline ='')
          
        # writing the data into the file
        with coords_y:    
            write = csv.writer(coords_y)
            write.writerow(r[1,:,1])
        coords_y.close()

        velocities_x = open('./output_files_adar/velocities_list_x.csv', 'a+', newline ='')
          
        # writing the data into the file
        with velocities_x:    
            write = csv.writer(velocities_x)
            write.writerow(v[1,:,0])
        velocities_x.close()


        velocities_y = open('./output_files_adar/velocities_list_y.csv', 'a+', newline ='')
          
        # writing the data into the file
        with velocities_y:    
            write = csv.writer(velocities_y)
            write.writerow(v[1,:,1])
        velocities_y.close()

    with open('./output_files_adar/energies_list.csv', mode='a') as csv_file_e:
        writer_e = csv.DictWriter(csv_file_e, fieldnames = fieldnames)
        writer_e.writerow({'E': str(e[1])})
    csv_file_e.close()
    
    # switch so that the current is in index 1 and we'll put all the new stuff in position 2

    v[1] = v[2,:]
    r[1] = r[2,:]
    f[1] = f[2,:]
    e[1] = e[2]
    f[2] = 0
    e[2] = 0



