import math
import time
import numpy as np
import scipy as sp
import Potential as Pt
from numpy import linalg as LA
import sys
from scipy.sparse import linalg as sLa
from scipy import sparse as sprs
from tools import *

dt = 8.0
tsteps = 1200
Time = range(tsteps)
print "Defauls"

#--------------------------------
# CONVERSION VALUE FROM PS TO AU
ps = 41341.37
#--------------------------------
nf = 2
red = 0
#-------------------------------
# Intial parameters
Rmin = 1.8
Rmax = 30.0
nR = 1024
aniskip = 250
#---------------------------------
dR = float((Rmax-Rmin)/nR)
R = np.arange(Rmin,Rmax,dR)
#---------------------------------

#---------------------------------
# exponential of V in adiabatic eigenrepresentation
def expV(ei, dt):
  return np.exp(-1j * dt * ei, dtype=np.complex64) 
# exponential of T in momentum (FFT) representation
def expT(dR, dt, nR, mass = 9267.5654):
  p = np.fft.fftfreq(nR) * (2.0 * np.pi/ dR)
  return np.exp(-1j * dt * (p * p)/(2 * mass), dtype=np.complex64) 


#---------------------------------
#    MAIN CODE
#---------------------------------

#ve = Pt.electronic(Rmin,Rmax,n_steps,nf)
Ep, Up = Pt.polariton(R, nf, red)
nState = Up.shape[1]
UV = expV(Ep, dt/2)
UT = expT(dR, dt, nR)
#ve = Pt.adiabat(Rmin,Rmax,n_steps,nf,red)

# Initial Wf
cD0 = psi(R, Up, nState)
np.savetxt("psi0.txt",cD0.real) 
#---------------------------------

# files
popP = open("popP.txt","w+")
popD = open("popD.txt","w+")
dis =  open("dis.txt","w+")
wf =   open("psi.txt","w+")
cDt = cD0
#-----------------------------

for t in Time:
  print t 
  # population in diabatic representation
  rhoD = population(cDt, nState)
  popD.write(str(t*dt/ps) + " " + " ".join(rhoD.astype(str)) + "\n" )
  
  # population in polaritonic representation
  cPt = DtoA(cDt, nR, nState, Up)  
  rhoP = population(cPt, nState) 
  popP.write(str(t*dt/ps) + " " + " ".join(rhoP.astype(str)) + "\n" ) 
  
  # write wavefunction
  if (t%aniskip == 0):
    for i in range(nR):
      density = np.zeros(nState,dtype=np.float32) 
      for j in range(nState):
        density[j] =  (cPt[j*nR + i].conjugate() * cPt[j*nR + i]).real
      wf.write( str( Rmin +  dR * i)  + " "  + " ".join(density.astype(str))  + "\n" )
    wf.write("\n\n")	  
  #--------------------

  # evolution 1st step 
  cPt = UV * cPt 
  cDt = AtoD(cPt, nR, nState, Up) 

  # evolution 2nd step
  fDt = np.zeros(len(cDt), dtype=np.complex64) 
  for i in range(nState):
    # FFT
    fDt[ nR * i: nR * (i + 1) ]  = np.fft.fft(cDt[ nR * i: nR * (i + 1) ], norm = 'ortho')
    # evolution in FFT
    fDt[ nR * i: nR * (i + 1) ] = UT * fDt[ nR * i: nR * (i + 1) ] 

  # iFFT
  for i in range(nState):
    cDt[ nR * i: nR * (i + 1) ]  = np.fft.ifft(fDt[ nR * i: nR * (i + 1) ], norm = 'ortho')
 
  # evolution 3rd step
  cPt = DtoA(cDt, nR, nState, Up)  
  cPt = UV * cPt 
  cDt = AtoD(cPt, nR, nState, Up) 

  pDis = dissociation(cDt, R, nState, nf)
  dis.write(str(t*dt/ps) + " "  + str(pDis) + "\n") 
   

np.savetxt("psi%s.txt"%(Time[-1]),cDt) 
wf.close()
popD.close()       
popP.close()            
dis.close()
