from scipy import sparse
import time
import numpy as np 
import random

xmin = 1.8
xmax = 30.0
nx   = 8
x    = np.linspace(xmin, xmax, nx) 
dx   = (x[1]-x[0])
# fx
p    = np.fft.fft(x, norm = 'ortho') 
print len(x)
print "dx = ", dx
print "dk = ", 2 * np.pi / (xmax - xmin)
print np.fft.fftfreq(nx, dx)
print  7/(8 * (xmax - xmin) )
print (nx-1)/nx 