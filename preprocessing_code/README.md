# data preprocessing code
### Author: Carole Hall

This program finds the average radial distribution function for a group of files each containing a 2-D distribution of points. The data provided in this repository represent distributions 
of Adélie penguin nests over flat areas taken from colonies located in the Danger Islands, Antarctica. 

For more detailed information on calculating the radial distribution function, I recommend checking out this really great tutorial from Glen Hocky in the Chemistry Department at New York University
(off of which much of the code for calculating these radial distribution functions is based):

https://hockygroup.hosting.nyu.edu/exercise/rdf-2020.html

For a summary: the radial distribution function is a way to summarize the order of a set of points in space. The radial distribution function often has characteristic traits depending
on the nature of the point set considered if one is dealing with a set of atoms. For example, a liquid has an archetypal shape which is different from that of a solid, and both the shapes
of solids and liquids are distinct from that of a gas. Additionally, the radial distribution function has been used in certain applications to define the amount of randomness contained in 
a distribution of points--a more flat radial distribution function implies less order or more randomness (more similar to that of a gas), while a very spikey curve can imply more order or less 
randomness (and solids generally have more spikey radial distribution functions). 
  
For a little more detail: calculate all distances between particles included in the distribution, then put those distances into a histogram. Normalize the histogram by multiplying by the density
along with the area of the circular shell at radius r from a reference particle. More detail and good visualization of this can be found at:

 https://en.wikipedia.org/wiki/Radial_distribution_function

 
