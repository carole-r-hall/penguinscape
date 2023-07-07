# Author: Carole Hall
# Description: some code dedicated to the calculation of radial distributio  functions (RDF) for multiple samples taken over flat areas of an individual Adelie penguin colony in Antarctica.
# This code is written to correspond to Beagle Island (BEAG), but can be modified to fit any of the other 3 Danger Island colonies (BRAS, EARL, HERO) to find RDFs correspomding to the nest
# distributions of those colonies.

import shapefile
import matplotlib.pyplot as plt
import matplotlib
import scipy as sp
import subprocess
import sys
import geopandas as gp
import numpy as np
import matplotlib.image as mpimg
import PIL
import rasterio
from affine import Affine

from rasterio.plot import show
from osgeo import ogr, osr
from scipy.interpolate import interp1d

PIL.Image.MAX_IMAGE_PIXELS = 487063040


def histogram_distances(distance_list, max_dist, bin_size):
    bins = np.arange(0, max_dist+bin_size, bin_size)
    hist, bin_edges = np.histogram( distance_list, bins=bins )
    return hist, bin_edges


def plot_configuration_pbc(configuration,box_L):
    for shift_x in range(-1,2):
        for shift_y in range(-1,2):
            if shift_x == 0 and shift_y == 0: 
                #this is for plotting transparency
                alpha = 1
            else:
                alpha = 0.3
            plt.scatter(shift_x*box_L+configuration[:,0],shift_y*box_L+configuration[:,1],alpha=alpha)

def compute_distances( configuration, box_size ):
    distance_list = []
    num_particles = configuration.shape[0]
    for i in range(num_particles):
        for j in range(num_particles):
            if i == j: continue
            posi = configuration[i]
            posj = configuration[j]
            
            dr = posj-posi
            
            dr = dr - box_size*np.floor(dr/box_size+0.5)
            
            dr2 = dr*dr
            
            dist = np.sqrt( dr2.sum() )            
            distance_list.append(dist)
            
    return np.array(distance_list)

def plot_rdf(gofr,bin_centers):
    plt.plot(bin_centers,gofr,marker='o')
    plt.ylabel("RDF")
    plt.xlabel("r")
    
def get_gofr(hist,bin_edges,num_particles, box_size):
    rho = num_particles/box_size/box_size
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2.0
    dr = bin_edges[1]-bin_edges[0]
    denominator = 2.*np.pi*bin_centers*dr*rho*( num_particles )
    gofr = hist/denominator
    
    return gofr, bin_centers
test_box_L = 12.0
test_bin_size = 0.1
num_configurations = 10
gofr_list = []
densities = []
for i in range(10):
    shape = gp.read_file("beagle_{}.shp".format(i+1))
    xmin, xmax, ymin, ymax = plt.axis()
    x_coords = shape.geometry.x
    test_num_particles = len(x_coords)
    y_coords = shape.geometry.y
    np.transpose([x_coords, y_coords])
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    
    density = len(x_coords)/144.0
    densities.append(density)
    box = np.array([[min_x,max_x],[min_y,max_y]])

    particles = [[x_coords[i], y_coords[i]] for i in range(len(x_coords))]
    particles = np.array(particles)
    
    distance_list_i_pbc = compute_distances(particles, box_size = test_box_L)
    
    # record distances 
    dist_hist_i_pbc, bin_edges_i_pbc = histogram_distances(distance_list_i_pbc,max_dist=test_box_L/2., bin_size=test_bin_size)
    gofr, bin_centers = get_gofr( dist_hist_i_pbc, bin_edges_i_pbc, test_num_particles, test_box_L )
    gofr = np.array(gofr)
    gofr = gofr/density
    gofr_list.append(gofr)

densities = np.array(densities)
print("AVERAGE DENSITY = ", np.mean(densities))
plt.figure(dpi = 250)
matplotlib.rc('xtick', labelsize=7.5) 
matplotlib.rc('ytick', labelsize=7.5)
plt.xticks([1,3,5])
plt.yticks([1])
plt.xlim(0.1, 5.9)
for gofr in gofr_list:
    cubic_interp = interp1d(bin_centers, gofr, kind = 'cubic')
    tempx = np.linspace(bin_centers.min(), bin_centers.max(), 500)
    tempy = cubic_interp(tempx)
    plt.plot(tempx, tempy, c = 'lightsteelblue', alpha = 0.70, linewidth = 1.5)
avg_gofr = np.mean(gofr_list, axis=0)
#plot_rdf(avg_gofr, bin_centers)
cubic_interpolation = interp1d(bin_centers, avg_gofr, kind='cubic')
x = np.linspace(bin_centers.min(), bin_centers.max(), 500)
y = cubic_interpolation(x)
plt.plot(x,y, c = "darkslateblue", linewidth = "2.0")
plt.axhline(1.0,linestyle='--',color='grey')
plt.xlabel("r")
plt.ylabel("RDF")
plt.title("Beagle Radial Distribution Function")
plt.show()

