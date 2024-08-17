library(orthogonalsplinebasis)
library(Splinets)

##### O-splines #####
knots = c(0,0,0,0,1,1,1,1)
obases = OBasis(knots = knots, order = 4)
plot(obases)

bases = SplineBasis(knots = knots, order = 4)
plot(bases)


##### Splinets ##### 
k = 3 # order
n_knots = 8 # number of knots
xi = seq(0, 1, length.out = n_knots)
so = splinet(knots = xi, smorder = k) 
plot(so$bs) # Plotting B-splines
plot(so$os) # Plotting Splinets
