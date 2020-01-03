import math
import time
import numpy as np
import scipy as sp
from tools import Diag 

#-------------------------------------
def polariton(R, nf, red = 0):
  Hpl = Help(R, nf, red) 
  nR  = len(R)
  nState = Hpl.shape[1]
  vectors = np.zeros( (nR, nState, nState), dtype=np.float32) 
  Ep = np.zeros( (nR, nState), dtype=np.float32) 
  # Interpolation of data
  for ri in range(nR):
    E,V = Diag(Hpl[ ri, :, :] )  
    #--- Phase Fix -------------
    #if ri>0:
    #	for ei in range(2*nf) :
    #	  sign = np.dot(Vold[:,ei],V[:,ei])
    #       sign = sign/abs(sign)
    #	  V = V*sign
    #---------------------------
    Vold = V 
    vectors[ri,:,:] = V
    Ep[ ri, :] = E  
  return Ep, vectors

#-------------------------------------
# ENERGY FUNCTION
def Ex(x,fname):
   dat = np.loadtxt(fname) 
   eg = np.interp(x,dat[:,0],dat[:,1])
   return eg
#-------------------------------------
#------------------------------------- 

def suplement(x): 
   a               = 0.113932
   b               = 0.0103053
   c               = -0.000203488
   fx = np.zeros(len(x)) 
   for xi in range(len(x)):
    r = x[xi] 
    if r>25.0:
	r = 25.0 
    fx[xi] = a+ b*r + c*r**2 
   return fx 
    
#----------------------------------------
# Data of the diabatic states

def Help(R, nf = 25, red = 22):
 wc = 7.5/27.2114
 xi = 0.0
 nR = len(R)
 nState = (2*nf-red)
 He = np.zeros((nR, nState, nState), dtype = np.float32)
 shift = 107.30974086754172 - 0.001129
 #-------------------------------------------
 # Electronic Data
 #----------------------------------------------
 H11 =  Ex(R,'dat/DIABATIC_ENERGY_IONIC_STATE.dat')
 H22 =  Ex(R,'dat/DIABATIC_ENERGY_ATOMIC_STATE.dat')
 H12 =  Ex(R, 'dat/DIABATIC_COUPLING_CONSTANT.dat') 
 H11f = suplement(R) 
 #-------------------------------------------
 mu11 = Ex(R,'dat/DIABATIC_DIPOLE_IONIC_STATE.dat') 
 mu22 = Ex(R,'dat/DIABATIC_DIPOLE_ATOMIC_STATE.dat') 
 #----------------------------------------------
 
 # Interpolation of data
 for ri in range(nR):
  s = (1+math.tanh(2.0*(R[ri]-17.2)))*0.5
  ED = np.zeros((2,2))
  ED[0,0] = (H11[ri] + shift) * (1-s) + s*(H11f[ri]) #ionic_D
  ED[1,1] = H22[ri]  + shift # atomic_D
  ED[1,1] += 0.5*(0.259495 - 0.217292)*(1.0+math.tanh(0.5*(R[ri]-30.0)))*0.5 # step
  ED[0,1] = H12[ri] # coupling_D
  ED[1,0] = ED[0,1]

  u_ab = np.zeros((2,2))
  u_ab[0,0] = (1-s) * mu11[ri] + s * (R[ri]*1.03022 - 0.508982)
  u_ab[1,1] = (1-s) * mu22[ri] 
  dse = np.matmul(u_ab,u_ab)
  for i in range(nState):
    a = int(i/nf)
    m = i%nf
    for j in range(nState):
     b = int(j/nf)
     o = j%nf
     He[ri, i, j] = ED[a,b]*float((m==o)) + (o+0.5)*(wc)*float((a==b)*(m==o))
     He[ri, i, j] += u_ab[a,b]*xi*(np.sqrt(m)*float((m-1)==o)+np.sqrt(m+1)*float((m+1)==o))
     He[ri, i, j] += (xi**2.0)*(dse[a,b]/wc)*float(m==o) 
 return He
#--------------------------------------------------------

#----------------------------------------

if __name__ == "__main__":
 potential = open('Potential.txt', 'w+')
 #dipol = open('dipol.txt', 'w')
 Rmin = 2.0
 Rmax = 40
 n_steps = 1000
 nf = 5
 red = 2
 step = (Rmax-Rmin)/n_steps
 rDist = np.arange(Rmin,Rmax,step)
 He =  Help(Rmin,Rmax,n_steps,nf,red) 
 for r in range(n_steps):
  potential.write(str(rDist[r]) + ' ' + str(He[r,r]))
  potential.write(' ' + str(He[r+n_steps,r+n_steps]))
  potential.write(' ' + str(He[r+(nf)*n_steps,r+nf*n_steps]))
  potential.write(' ' + str(He[r+(nf+1)*n_steps,r+(nf+1)*n_steps]))
  potential.write('\n')

 potential.close()
