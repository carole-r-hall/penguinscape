# Author: Carole Hall
# Description: this program takes a given radial distribution function (RDF) curve in the form of a CSV
# where x and y sample coordinates of the curve are stored. This given RDF is assumed to be that cooresponding
# to a real distribution of points (or penguin nests in our application). This code then runs a molecular
# dynamics simulation (under a Lennard-Jones potential) over a grid of distance/temperature parameters
# to find example distributions, and the average RDF of the simulated distribution is compared with that
# from the CSV given to the program. Least squared distances between the curves are then saved to a CSV file.
#
# NOTE: the input RDF curve is assumed to be from a 12x12 2-D box of particles. In this program, the box size
# depends on the particle density provided (empirical_density). When comparing RDF curves of different x-value
# length, the x-values are truncated in the larger case to allow easier comparison between the curves.
#
# NOTE 2: the MD simulation in this program was written to allow easy implementation of animating results.
# credit is given to a blog found at https://scipython.com/blog/the-maxwellboltzmann-distribution-in-two-dimensions/
# for helpful tips on incorporating animation into MD simulations. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
import math
import time
import csv
import itertools
from scipy.interpolate import interp1d # optional for cosmetic smoothing of RDF graphs, not used in actual analysis
import matplotlib
from matplotlib.patches import Rectangle

empirical_density = 1.66736 # empirical density for HERO; modify this depending on colony's average density

# create csv for least squares RDF minimization
with open('least_squares.csv', 'w', newline='') as csvfile:
    fieldnames = ['sigma', 'T*', 'LSD_RDF']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

# figure stuff
DPI = 100
width, height = 1000, 500

'''---------------IMPORT RDF DATA-----------'''


empirical_rdf = []
empirical_dists = []

# extract colony RDF DATA from CSV - calculated in separate program
with open('heroina_RDF.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        empirical_rdf.append(float(row['rdf']))
        empirical_dists.append(float(row['dists']))

# make sure data are in the form of np arrays for efficiency
empirical_rdf = np.array(empirical_rdf)
empirical_dists = np.array(empirical_dists)

class simulate_nests:

    def __init__(self, xy, v, a, pot,tot_steps, epsilon, sigma, box_size):
       
        # allocate structures for xy position, velocity, acceleration, potential energy
         
        self.xy = np.asarray(xy, dtype=float) # coordinates
        self.v = np.asarray(v, dtype=float) # velocities
        self.a = np.asarray(a, dtype=float) # accelerations
        self.pe = np.asarray(pe, dtype=float) # potential energies
        self.n = self.xy.shape[0] # number of particles
        self.tot_steps = tot_steps # total timesteps
        self.nsteps = 0 # current step
        self.all_pe = np.zeros(tot_steps) # potential energy sum
        self.temperature = np.zeros(tot_steps) 
        self.epsilon = epsilon # well depth
        self.sigma = sigma # distance at which PE is zero
        self.boxlength = boxlength # side length of 2-D box inside which the particles interact

    def forces(self):
        sd = np.zeros(2) # scaled units - box becomes length 1 here
        rd = np.zeros(2) # real units
        self.pe = self.pe*0.0
        self.a = self.a*0.0
        cutoff = 2.5*self.sigma
        pe_cutoff = 4.0/(self.sigma**12/cutoff**12)-4.0/(self.sigma**6/cutoff**6) # distances greater than cutoff make no contribution to energy - this is "shifted LJ potential"
        for i in range(self.n-1):
            for j in range(i+1,self.n): # don't deal with redundancy 
                sd = self.xy[i,:]-self.xy[j,:] # scaled distance 
                for d in range(2): # for periodic boundary conditions
                    if (np.abs(sd[d])> 0.5):
                        sd[d] = sd[d] - np.copysign(1.0,sd[d]) # wraparound for periodic bounadry
                
                rd = self.boxlength*sd # convert to real space
                rsq = np.dot(rd,rd) # dist squared for use in LJ equations
                
                if(rsq < cutoff**2):
                    # ignore everything after cutoff distance 
                    rm2 = self.sigma**2/rsq # sigma^2/r^2
                    rm6 = rm2**3.0 # sigma^6/r^6
                    rm12 = rm6**2.0 # sigma^12/r^12
                    pe_eq = self.epsilon*(4.0*(rm12-rm6)-pe_cutoff) # shiftted LJ potential 
                    # force = negative gradient of potential energy 
                    force = 4*self.epsilon*(12*(self.sigma**12/(rsq**6*rsq)) - 6*(self.sigma**6/(rsq**3*rsq)))
                    self.pe[i] = self.pe[i]+0.5*pe_eq # add energy
                    self.pe[j] = self.pe[j]+0.5*pe_eq # add energy 2x because pe(i,j) = pe(j,i)
                    self.a[i,:] = self.a[i,:]+force*sd # add forces
                    self.a[j,:] = self.a[j,:]-force*sd # force(i,j) = -force(j,i)
                
        return self.a, np.sum(self.pe)  #return acceleration, total potential energy


    def calculate_temperature(self):
        ke = 0.0 # kinetic energy used in calculating temperature  

        for i in range(self.n):
            real_v = self.boxlength*self.v[i,:]
            ke = ke + 0.5*np.dot(real_v,real_v)
        ke_avg = 1.0*ke/self.n 
        temperature = 2.0*ke_avg/2.0

        return temperature
        
    # move forward by a timestep (deltat) using Leap Frog
    def timestep(self, deltat, t_star, volume):
        self.nsteps += 1

        # periodic boundary conditions
        for i in range(2):
            period = np.where(self.xy[:,i] > 0.5)
            self.xy[period,i] = self.xy[period,i]-1.0
            period = np.where(self.xy[:,i] < -0.5)
            self.xy[period,i] = self.xy[period,i]+1.0
           
        # Update the coordinates specified by velocity
        self.xy = self.xy + deltat*self.v + 0.5*(deltat**2.0)*self.a 

        # periodic boundary conditions -> wraparound
        for i in range(2):
            period = np.where(self.xy[:,i] > 0.5)
            self.xy[period,i] = self.xy[period,i]-1.0
            period = np.where(self.xy[:,i] < -0.5)
            self.xy[period,i] = self.xy[period,i]+1.0

        # Calculate temperature
        self.temperature[self.nsteps] = self.calculate_temperature()
        
        # Calculate temperature
        self.temperature[self.nsteps] = self.calculate_temperature()

        if self.nsteps % 10 == 1 and self.nsteps < avg_start:
            # rescale velocities periodically -> every 10 timesteps here (this helps us maintain t_star thermostat)
            scaling = np.sqrt(t_star/self.temperature[self.nsteps])
            self.v = scaling*self.v + 0.5*deltat*self.a # rescale
        else:
            self.v = self.v + 0.5*deltat*self.a # don't rescale in between

        # potential energy and forces
        self.a, self.all_pe[self.nsteps] = self.forces()
        
        # update velocity to finish integration
        self.v = self.v + 0.5*deltat*self.a

# generate points with sufficient space to not cause energy spiking
# there may occasionally be configurations where noise causes points
# to be too close.

def generate_points_with_min_distance(n, shape, min_dist):
    # compute grid shape based on number of points
    width_ratio = shape[1] / shape[0]
    num_y = np.int32(np.sqrt(n))
    num_x = np.int32(np.sqrt(n))

    # create regularly spaced points
    x = np.linspace(0., shape[1]-(sigma*(2**(1/6))), num_x, dtype=np.float32)
    y = np.linspace(0., shape[0]-(sigma*(2**(1/6))), num_y, dtype=np.float32)
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1,2)

    # compute spacing
    init_dist = np.min((x[1]-x[0], y[1]-y[0]))


    return coords

# get the pairwise distances of all points in a configuration
def compute_distances(coords, boxlength):
    dists = []
    num_particles = coords.shape[0]
    for i in range(num_particles):
        for j in range(num_particles):
            # don't count distances of particles to themselves
            if i == j:
                continue
            coordi = coords[i]
            coordj = coords[j]
            # compute the dist between i and j 
            dr = coordj-coordi
            dr = dr - boxlength*np.floor(dr/boxlength+0.5)
            
            dr2 = dr*dr 
            dist = np.sqrt(dr2.sum()) # add up squares of x,y components, take sqrt after
            
            dists.append(dist)
    return np.array(dists)        

# used in creation of the RDF 
def hist_d(dists, max_d, step_length):
    # this is the list of bins in which to calculate
    xs = np.arange(0, max_d+step_length, step_length)
    hist, xs_d = np.histogram(dists, bins=xs)
    return hist, xs_d

def get_rdf(hist,xs_d,num_particles, boxlength):
    density = num_particles/boxlength/boxlength
    xs = (xs_d[1:]+xs_d[:-1])/2.0
    dr = xs_d[1]-xs_d[0]
    denominator = 2.*np.pi*xs*dr*density*(num_particles)
    rdf = hist/denominator
    
    return rdf, xs


#timesteps
NSteps = 4000
tot_steps = NSteps

# start the averaging process here
avg_start = 1000

sbar = 0.25 # starting mean speed
# time step size
dt = 0.005

# number of particles.
n = 100

# grid over which to try a new simulation on each parameter combination
# see which combination results in the best RDF to the given one 
epsilons = [1.0]
sigmas = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.755, 0.76, 0.765, 0.77, 0.775, 0.78, 0.79, 0.80, 0.85] # when you find a value that works, make the grid finer around there to optimize
Ts = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 13.0] # note: high temperatures cause greater speeds in the beginning which can cause issues with energy spiking and the simulation not working correctly

'''-----------------GRID SEARCH SIMULATION-----------------'''

for epsilon, sigma, t_star in list(itertools.product(epsilons, sigmas, Ts)):
    print("-------------BEGIN SIM----------------")
    print("epsilon = ", epsilon)
    print("sigma = ", sigma)
    pe_tot = 0.0
    
    # van der waals radius 
    r = sigma

    # initialize particles + box 
    boxlength = math.sqrt(n/empirical_density)

    volume  = boxlength**2
    density = n / volume
    print("volume = ", volume, " density = ", density)

            
    xy = generate_points_with_min_distance(n, [boxlength,boxlength], sigma*(2**(1/6)))

    xy = np.array(xy)

    xy = xy[:,:2]/boxlength

    cent = np.sum(xy,axis=0)/n

    for i in range(2):
        xy[:,i] = xy[:,i]-cent[i]

    # initialize points, randomize velocites/accelerations around some speed sbar
    theta = np.random.random(n) * 2 * np.pi # theta = angle for orientation
    s0 = sbar * np.random.random(n)
    v = (s0 * np.array((np.cos(theta), np.sin(theta)))).T
    a = (s0 * np.array((np.cos(theta), np.sin(theta)))).T #initialize accelerations in the same method as velocities
    pe = np.zeros(n)
    all_pe = np.zeros(NSteps)
    times = np.array([i*dt for i in range(NSteps)])

    sim = simulate_nests(xy, v, a, pe,tot_steps, epsilon, sigma, boxlength)

    tic = time.perf_counter()

    '''-----------FOR RDF CALCULATION-------------'''
    box = np.array([[-boxlength/2.0,boxlength/2.0],[boxlength/2.0,boxlength/2.0]])
    test_boxlength = boxlength
    test_x_size = 0.1
    rdf_list = []
    test_num_particles = n
    
    for i in range(1,NSteps):
        sim.timestep(dt, t_star, volume)
        
        # once the simulation has approached equilibrium a bit, start averaging
        # the pot_e distribution, periodically get the rdf to average 
        if i >= avg_start:
            pe_tot += sum(sim.all_pe) / (i - avg_start + 1)
            '''---------------GET AVERAGE RDF PERIODICALLY--------------------'''
            if i % 10 == 1:
                coords = [[sim.xy[j,0], sim.xy[j,1]] for j in range(len(sim.xy))]
            
                coords = np.array(coords)

                coords = coords*boxlength
                
                distance_list_i = compute_distances(coords, boxlength = test_boxlength)
                
                #histogram
                dist_hist_i, xs_d_i = hist_d(distance_list_i, max_d=test_boxlength/2.0, step_length=test_x_size)
                
                rdf, dists = get_rdf(dist_hist_i, xs_d_i, test_num_particles, test_boxlength)
                rdf = np.array(rdf)
                rdf_list.append(rdf/empirical_density)
                dists = np.array(dists)
                
    avg_rdf = np.mean(rdf_list, axis=0)
    avg_rdf_norm = (avg_rdf-np.min(avg_rdf))/(np.max(avg_rdf)-np.min(avg_rdf))
    toc = time.perf_counter()
    timed = toc - tic

    # find sum of squares between empirical and experimental for distance length of the min of them:

    sum_sq = 0
    for k in range(min(len(avg_rdf), len(empirical_rdf))):
        diff = avg_rdf[k] - empirical_rdf[k]
        sum_sq += diff*diff

    print("sum squared difference is ", sum_sq, " for epsilon = ", epsilon, " and sigma = ", sigma, ".")

    # append to a csv file
    data = [str(epsilon), str(sigma), str(t_star), str(sum_sq)]
    with open(r'least_squares_RDFs.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)
    
    print("Running simulation took ", timed, " seconds." )

    
    '''---------------VISUALIZATIONS----------------------'''

    ####RDF####

    # bunch of stylistic things for the plot
    fig= plt.figure(figsize=(width/DPI, height/DPI), dpi=DPI)
    matplotlib.rc('xtick', labelsize=7.5) 
    matplotlib.rc('ytick', labelsize=7.5)
    plt.xticks([1,3,5])
    plt.yticks([1])
    plt.axhline(1.0,linestyle='--',color='black')
    plt.xlim(0,max(dists))
    cubic_interp = interp1d(dists, avg_rdf, kind = 'cubic')
    tempx = np.linspace(dists.min(), dists.max(), 500)
    tempy = cubic_interp(tempx)
    plt.plot(tempx, tempy, c = 'palevioletred', linewidth = 2.0)
    cubic_interp2 = interp1d(empirical_dists, empirical_rdf, kind = 'cubic')
    tempx2 = np.linspace(dists.min(), dists.max(), 500)
    tempy2 = cubic_interp2(tempx2)
    plt.plot(tempx2, tempy2, c = 'darkslateblue', linewidth = 2.0)
    plt.title("Empirical vs. Experimental RDF for Heroina Island at sigma = {s}, epsilon = {e}, T* = {t}".format(s = sigma, e = epsilon, t = t_star))
    plt.savefig("rdf_s_{s}_e_{e}_T_{t}.png".format(s = sigma, e = epsilon, t = t_star), dpi=300)

    ####ENERGETICS####
    
    # figure set up
    fig2 = plt.figure(figsize=(width/DPI, height/DPI), dpi=DPI)
    plt.axis('off')
    sim_ax = fig2.add_subplot(121, aspect='equal', autoscale_on=True)
    sim_ax.add_patch(Rectangle((-0.5,-0.5), 1, 1, edgecolor = 'red', facecolor = 'grey', alpha = 0.25))
    sim_ax.set_xlim(-0.65, 0.65)
    sim_ax.set_ylim(-0.65, 0.65)
    pot_ax = fig2.add_subplot(122)
    pot_ax.set_xlabel('Time')
    pot_ax.set_ylabel('Potential Energy')
    sim_ax.axis('off')
    pot_ax.set_xlim(0, NSteps*dt)
    fig2.tight_layout()
    
    sim_ax.scatter(sim.xy[:,0], sim.xy[:,1])
    pot_ax.plot(times[1:], sim.all_pe[1:], c = 'red')

    plt.savefig("energetics_s_{s}_e_{e}_T_{t}.png".format(s = sigma, e = epsilon, t = t_star), dpi=300)   

