# Downsampled raster data

There are 6 different folders corresponding to each colony: Cape Adare (ADAR), 
Berkeley/Cameron Island (BERK), Brownson Islands (BSON), Devil island (DEVI, 
Inexpressible Island (INEX), and Possession Island (POSS). These are downsamples of 
the colony shape files in each of those locations, meant for validation for using our 
reconstruction procedure to see if we can reconstruct a high-resolution shape from
this low resolution raster data. The raster data can be thought of as colony coverage 
percentages on a per-pixel basis, where the spatial resolution of each pixel is 30 meters
(meant to mimic Landsat satellite imagery). The reconstruction code requires CSVs 
to access the raster data, and these are stored as csvs for the longitude and latitude
per center of each pixel, the coverage percentages for each pixel, and the height (number
of rows) and width (number of columns) of the raster.
