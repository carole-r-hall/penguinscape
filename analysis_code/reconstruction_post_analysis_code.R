# Author: Carole Hall
# Description: this code will find the intersection over union for raster data that's been resampled to be comparable to the same resolution.

library(terra)
library(rgdal)
library(raster)
library(sf)
library(stars)
library(rgeos)
library(ggplot2)
library(sp)
library(dplyr)
library(maptools)
library(gridExtra)
# normalize the rasters
raster01 = function(r){
  
  # get the min max values
  minmax_r = range(values(r), na.rm=TRUE) 
  
  # rescale 
  return( (r-minmax_r[1]) / (diff(minmax_r)))
}

for (i in c(4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 30.0))
{
  tmp_ground_truth <- "adar_downsampled_annotations_for_validation" # these were resampled into comparable rasters using disagg() or agg() from terra -- also remember to crop the rasters to be the same extent during comparison
  tmp_simulation <- "adar_downsampled_simulation_validation"
  tmp_landsat <- "adar_landsat"
  ground_truth <- raster(file.path(tmp_ground_truth, sprintf("adar_raster_%sm_resolution.tif", i)))
  ground_truth_n <- ground_truth > 0
  simulation <- raster(file.path(tmp_simulation, sprintf("adar_raster_%sm_resolution.tif", i)))
  simulation <- resample(simulation, ground_truth)
  simulation_n <- simulation > 0
  landsat <- raster(file.path(tmp_landsat, sprintf("adar_raster_%sm_resolution.tif", i)))
  landsat_n <- raster01(landsat)
  landsat <- resample(landsat, ground_truth)
  landsat_n <- landsat > 0
  print(simulation_n)
  print(ground_truth_n)
  print(landsat_n)
  union_vhr_landsat <- landsat_n + ground_truth_n
  union_vhr_sim <- simulation_n + ground_truth_n
  union_vhr_landsat[union_vhr_landsat > 0] <- 1
  union_vhr_sim[union_vhr_sim > 0] <- 1
  ground_truth_n[ground_truth_n == 0] <- NA
  simulation_n[simulation_n == 0] <- NA
  landsat_n[landsat_n == 0] <- NA
  intersection_vhr_landsat <- mask(landsat_n, ground_truth_n)
  intersection_vhr_sim <- mask(simulation_n, ground_truth_n)
  intersection_vhr_landsat[is.na(intersection_vhr_landsat)] <- 0
  intersection_vhr_sim[is.na(intersection_vhr_sim)] <- 0
  int_over_union_vhr_landsat <- sum(intersection_vhr_landsat[intersection_vhr_landsat>0])/sum(union_vhr_landsat[union_vhr_landsat>0])
  int_over_union_vhr_sim <- sum(intersection_vhr_sim[intersection_vhr_sim>0])/sum(union_vhr_sim[union_vhr_sim>0])
  print(int_over_union_vhr_landsat)
  print(int_over_union_vhr_sim)
  plot(intersection_vhr_sim)

  tmp_comparison <- "adar_comparison"
  png(filename = file.path(tmp_comparison, sprintf("badar_comparison_%sm_resolution.png", i)),width = 6000, height =  2000, units = 'px', bg = "white")
  m <- rbind(c(1,2,3))
  layout(m)
  xminamount <- 0
  yminamount <- 0
  xmaxamount <- 0
  ymaxamount <- 0
  image(landsat_n, col = "brown", axes = FALSE)
  image(ground_truth_n, col = "brown", axes = FALSE)
  image(simulation_n, col = "brown", axes = FALSE)
  mtext(sprintf("Resolution = %s, int_over_union VHR/Simulation = %.5s, int_over_union VHR/Landsat = %.5s", i, int_over_union_vhr_sim, int_over_union_vhr_landsat), side = 3, line = -5, outer = TRUE, cex = 5)
  dev.off()
}


