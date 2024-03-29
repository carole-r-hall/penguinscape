# penguinscape
Reconstruction of penguin colony shape from remotely-sensed data. 

## Introduction to materials
This repository is a collection of Python and R code, along with data related to reconstruction of Adélie penguin colony shape from low-resolution remotely-sensed data (in this case, the application would be 30 meter-resolution satellite imagery).

One can find the following procedures:

- Code to downsample annotations taken from very high-resolution (VHR) imagery of penguin colony shapes to 30 m resolution. This creates example data on which to test our reconstruction protocol.
- Code to fit parameters of a Lennard-Jones model from empirical data on penguin nesting distributions. 
- Code to reconstruct high-resolution colony shapes from low-resolution data using a molecular dynamics-based approach. Case studies on six colonies included: Cape Adare (ADAR), Berkley/Cameron Island (BERK), Brownson Island (BSON), Devil Island (DEVI), Inexpressible Island (INEX), and Possession Island (POSS).

One can find the following relevant data:

- 12 m x 12 m snapshots of annotated UAV data showing nesting coordinates. There are ten Adélie-only samples over relatively flat locations for each of four Danger Islands colonies: Beagle Island (BEAG), Brash Island (BRAS), Earle Island (EARL), and Heroina Island (HERO).
- Very High-Resolution (VHR) annotations showing high-resolution colony shapes for six colonies: Cape Adare (ADAR), Berkley/Cameron Island (BERK), Brownson Island (BSON), Devil Island (DEVI), Inexpressible Island (INEX), and Possession Island (POSS).
- The average radial distribution functions corresponding to each colony: BEAG, BRAS, EARL, HERO, taken from the 12 m x 12 m snapshots of annotated nesting coordinates taken from UAV. These are in CSV files where one column is x-values (distance values, or dists), and one colum is y-values (radial distribution function evaluated at each distance, or rdf).
- The 30 m raster data found from downsampling VHR annotation data (on which the reconstruction method is tested) for each colony: ADAR, BERK, BSON, DEVI, INEX, POSS.
- The crop extents of each colony location: ADAR, BERK, BSON, DEVI, INEX, POSS.

Please note: additional information regarding the background behind this repository's contents is currently in review and will be publically available soon. In the meantime, any inquiries about the enclosed code and/or data can be made to carole.hall@stonybrook.edu. DOI: 10.5281/zenodo.10547724.
