# penguinscape
Code related to reconstruction of penguin colony shape from remotely-sensed data. 

## Introduction to materials
This repository is a collection of Python and R code, along with data related to reconstruction of Adélie penguin colony shape from low-resolution remotely-sensed data (in this case, the application would be 30 meter-resolution satellite imagery).

One can find the following procedures:

- Code to downsample annotations taken from very high-resolution (VHR) imagery of penguin colony shapes to 30 m resolution. This creates example data on which to test our reconstruction protocol.
- Code to fit parameters of a Lennard-Jones model from empirical data on penguin nesting distributions. 
- Code to reconstruct high-resolution colony shapes from low-resolution data using a molecular dynamics-based approach. Case studies on six colonies included: Cape Adare (ADAR), Berkley/Cameron Island (BERK), Brownson Island (BSON), Devil Island (DEVI), Inexpressible Island (INEX), and Possession Island (POSS).

One can find the following relevant data:

- 12 m x 12 m snapshots of annotated UAV data showing nesting coordinates. There are ten Adélie-only samples over relatively flat locations for each of four Danger Islands colonies: Beagle Island (BEAG), Brash Island (BRAS), Earle Island (EARL), and Heroina Island (HERO).
- VHR annotations showing high-resolution colony shapes for six colonies: Cape Adare (ADAR), Berkley/Cameron Island (BERK), Brownson Island (BSON), Devil Island (DEVI), Inexpressible Island (INEX), and Possession Island (POSS).
