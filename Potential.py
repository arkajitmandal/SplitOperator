import math
import time
import numpy as np
import scipy as sp
from numpy import linalg as LA

#----------------------------------------
# MATRIX DIAGONALIZATION

def Diag(H):
    E,V = LA.eigh(H) # E corresponds to the eigenvalues and V corresponds to the eigenvectors
    return E,V
#----------------------------------------

#-------------------------------------
# ENERGY FUNCTION
def Ex(x,fname):
   dat = np.loadtxt(fname) 
   eg = np.interp(x,dat[:,0],dat[:,1])
   return eg
#-------------------------------------

#----------------------------------------
#----------------------------------------
# Data of the diabatic states
def adiabat(Rmin,Rmax,n,nf):
 Hpl = Help(Rmin,Rmax,n,nf)
 vectors = np.zeros((n,nf*2,nf*2))
 # Interpolation of data
 for ri in range(n):
  Hplr = np.zeros((nf,nf))
  for row in range(nf):
   for col in range(nf):
        Hplr[row,col] = Hpl[ row * n * 2 + ri , col * n * 2 + ri]
  E,V = Diag(Hplr)
  #--- Phase Fix -------------
  #if ri>0:
  #     for ei in range(2*nf) :
  #       sign = np.dot(Vold[:,ei],V[:,ei])
  #       sign = sign/abs(sign)
  #       V = V*sign
  #---------------------------
  # G0 E0
  vectors[ri,0,0] = V[0,0]
  vectors[ri,0,2] = V[0,1]
  vectors[ri,2,0] = V[1,0]
  vectors[ri,2,2] = V[1,1]
  # G1 E1
  vectors[ri,1,1] = V[0,0]
  vectors[ri,1,3] = V[0,1]
  vectors[ri,3,1] = V[1,0]
  vectors[ri,3,3] = V[1,1]
 return vectors




def polariton(Rmin,Rmax,n,nf,red):
 Hpl = Help(Rmin,Rmax,n,nf)[:(nf*2-red)*n,:(nf*2-red)*n]
 vectors = np.zeros((n,(nf*2-red),(nf*2-red)),dtype=np.float32) 
 Ep = vectors = np.zeros((n,(nf*2-red)),dtype=np.float32) 
 # Interpolation of data
 for ri in range(n):
  Hplr = np.zeros((nf*2-red,nf*2-red)) 
  for row in range(2*nf-red):
   for col in range(2*nf-red):
  	Hplr[row,col] = Hpl[ row * n + ri , col * n + ri] 
  E,V = Diag(Hplr)  
  #--- Phase Fix -------------
  #if ri>0:
  #	for ei in range(2*nf) :
  #	  sign = np.dot(Vold[:,ei],V[:,ei])
  #       sign = sign/abs(sign)
  #	  V = V*sign
  #---------------------------
  Vold = V 
  vectors[ri,:,:] = V
 return vectors
#--------------------------------------------------------
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

def Help(Rmin,Rmax,n,nf,red=0):

 wc = 7.5/27.2114
 xi = 0.04
 He = np.zeros((n,(2*nf-red),(2*nf-red)),dtype = np.float32)
 Rmin = float(Rmin)
 Rmax = float(Rmax)
 step = float((Rmax-Rmin)/n)
 rDist = np.arange(Rmin,Rmax,step)
 shift = 107.30974086754172 - 0.001129

 #-------------------------------------------
 # Electronic Data
 #----------------------------------------------
 H11 =  Ex(rDist,'DIABATIC_ENERGY_IONIC_STATE.dat')
 H22 =  Ex(rDist,'DIABATIC_ENERGY_ATOMIC_STATE.dat')
 H12 =  Ex(rDist, 'DIABATIC_COUPLING_CONSTANT.dat') 
 H11f = suplement(rDist) 
 #-------------------------------------------
 mu11 = Ex(rDist,'DIABATIC_DIPOLE_IONIC_STATE.dat') 
 mu22 = Ex(rDist, 'DIABATIC_DIPOLE_ATOMIC_STATE.dat') 
 #----------------------------------------------
 
 # Interpolation of data
 for ri in range(n):
  s = (1+math.tanh(2.0*(rDist[ri]-17.2)))*0.5
  ED = np.zeros((2,2))
  ED[0,0] = (H11[ri] + shift) * (1-s) + s*(H11f[ri]) #ionic_D
  ED[1,1] = H22[ri]  + shift # atomic_D
  ED[1,1] += 0.5*(0.259495 - 0.217292)*(1.0+math.tanh(0.5*(rDist[ri]-30.0)))*0.5 # step
  ED[0,1] = H12[ri] # coupling_D
  ED[1,0] = ED[0,1]

  u_ab = np.zeros((2,2))
  u_ab[0,0] = (1-s) * mu11[ri] + s * (rDist[ri]*1.03022 - 0.508982)
  u_ab[1,1] = (1-s) * mu22[ri] 
  self = np.matmul(u_ab,u_ab)
  for i in range(2*nf-red):
    a = int(i/nf)
    m = i%nf
    for j in range(2*nf-red):
     b = int(j/nf)
     o = j%nf
     He[ri, i, j] = ED[a,b]*float((m==o)) + (o+0.5)*(wc)*float((a==b)*(m==o))
     He[ri, i, j] += u_ab[a,b]*xi*(np.sqrt(m)*float((m-1)==o)+np.sqrt(m+1)*float((m+1)==o))
     He[ri, i, j] += (xi**2.0)*(self[a,b]/wc)*float(m==o) 
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
