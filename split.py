import math
import time
import numpy as np
import scipy as sp
import Potential as Pt
from numpy import linalg as LA
import sys
from scipy.sparse import linalg as sLa
from scipy import sparse as sprs


dt = 1.0
tsteps = 8400
Time = range(tsteps)
print "Defauls"

#--------------------------------
# CONVERSION VALUE FROM PS TO AU
ps = 41341.37
#--------------------------------

#-------------------------------
# Intial parameters
nf = 25
red = 22
Rmin = 1.8
Rmax = 25.0
nR = 1500
aniskip = 100000
#---------------------------------
dR = float((Rmax-Rmin)/nR)
nState = nf*2-red

#---------------------------------
#---------------------------------
# INITIAL WAVE-FUNCTION
def psi(Rmin,Rmax,n,nf,vp,red):
 p = np.zeros((2*n*nf - n*red),dtype=np.complex64)
 Rmin = float(Rmin)
 Rmax = float(Rmax)
 step = float((Rmax-Rmin)/n)
 rDist = np.arange(Rmin,Rmax,step)
 Norm = 0
 #CP0  =  DtoP(CD0,Rmin,Rmax,n,nf,vp)
 for ri in range(n):
  A = 0.260778 
  B = 19.1221  
  C = 3.01125  
  p[ri+n] = A*np.exp(-B*(rDist[ri] -C)**2.0)  
  Norm += abs(p[ri+n] **2)
 #return p/Norm**0.5
 return PtoD(p/np.sqrt(Norm),Rmin,Rmax,n,nf,vp,red)

#---------------------------------
# ROTATION FROM D TO P

def DtoP(cD,rmin,rmax,xstep,nf,vp,red):
 N = len(cD)
 dr = (rmax-rmin)/xstep
 cP = np.zeros((N),dtype= np.complex64)
 for ri in range(xstep):
  U = vp[ri,:,:]
  r = ri*dr + rmin
  for i in range(2*nf-red):
   for j in range(2*nf-red):
    k = i*xstep + ri
    l = j*xstep + ri
    cP[k] += U[j,i] * cD[l]
 return cP
#---------------------------------
def PtoD(cP,rmin,rmax,xstep,nf,vp,red):
 N = len(cP)
 dr = (rmax-rmin)/xstep
 cD = np.zeros((N),dtype= np.complex64)
 for ri in range(xstep):
  U = vp[ri,:,:]
  r = ri*dr + rmin
  for i in range(2*nf-red):
   for j in range(2*nf-red):
    k = i*xstep + ri
    l = j*xstep + ri
    cD[k] += U[i,j] * cP[l]
 return cD
#       print size(Adt) + size(Apow)
#---------------------------------
# sparse matrix size 	
def size(m):
	return m.data.nbytes + m.indptr.nbytes + m.indices.nbytes

#---------------------------------
def population(Ci,N) :
  p = np.zeros(N,dtype=np.float32) 
  nx = len(Ci)/N
  for i in range(N):
	p[i] = Ci[i*nx:(i+1)*nx].conjugate().dot(Ci[i*nx:(i+1)*nx]).real 
  return p


#---------------------------------
def dissociation(cDt,Rmin,Rmax,n_steps,nf,red):
 Rmin = float(Rmin)
 Rmax = float(Rmax)
 step = float((Rmax-Rmin)/n_steps)
 ref = int((15-Rmin)/step) 
 Prob = 0
 for i in range(nf,2*nf-red) :
  start = (n_steps * i) + ref
  end = (i+1) * n_steps
  Prob += (np.matmul(cDt.T[start:end].conjugate(),cDt[start:end])).real
 return Prob
#---------------------------------
# exponential of V in adiabatic eigenrepresentation
def expV(ep, dt):
  return np.exp(-1j * dt * ep, dtype=np.complex64) 
# exponential of T in momentum (FFT) representation
def expT(dx, dt, n_steps, mass = 9267.5654):
  p = np.fft.fftfreq(n_steps, dx)
  return np.exp(-1j * dt * (p*p)/(2*mass), dtype=np.complex64) 


#---------------------------------
#    MAIN CODE
#---------------------------------

#ve = Pt.electronic(Rmin,Rmax,n_steps,nf)
ep, vp = Pt.polariton(Rmin,Rmax,nR,nf,red)
UV = expV(ep, dt/2)
UT = expT(dR, dt, nR)
#ve = Pt.adiabat(Rmin,Rmax,n_steps,nf,red)

# Initial Wf
cD0 = psi(Rmin, Rmax, nR, nf, vp, red)
np.savetxt("psi0.txt",cD0.real) 
#---------------------------------
#---------------------------------
# Evolve
popP = open("popP.txt","w+")
popD = open("popD.txt","w+")
#popA = open("popA.txt","w+",buffering=0)
dis =  open("dis.txt","w+")
wf =   open("psi.txt","w+")
cDt = cD0
#-----------------------------
for t in Time:
 
 # psi 
 if (t%aniskip==0):
  for i in range(nR):
   density = np.zeros(nState,dtype=np.complex64) 
   for j in range(nState):
    density[j] =  (cPt[j*nR + i].conjugate() * cPt[j*nR + i]).real
   wf.write( str( Rmin +  step*i)  + " "  + " ".join(density.astype(str))  + "\n" )
  wf.write("\n\n")	  


 print t 
 # population in diabatic representation
 rhoD = population(cDt, nState)
 popD.write(str(t*dt/ps) + " " + " ".join(rhoD.astype(str)) + "\n" )
 # population in polaritonic representation
 cPt = DtoP(cDt,Rmin,Rmax, nR,nf,vp,red)  
 rhoP = population(cPt, nState) 
 popP.write(str(t*dt/ps) + " " + " ".join(rhoP.astype(str)) + "\n" ) 
 # evolution 1st step 
 cPt = UV.dot(cPt, dtype=np.complex64)
 cDt = PtoD(cDt,Rmin, Rmax, nR, nf, vp, red)  
 # evolution 2nd step
 fDt = np.zeros(len(cDt), dtype=np.complex64) 
 for iState in range(len(nState)):
  # FFT
  fDt[ nR * iState: nR * (iState + 1) ]  = np.fft.fft(cDt[ nR * iState: nR * (iState + 1) ], dtype=np.complex64)
  # evolution in FFT
  fDt[ nR * iState: nR * (iState + 1) ] = UT.dot(fDt[ nR * iState: nR * (iState + 1) ], dtype=np.complex64)

 # iFFT
 for iState in range(len(nState)):
  cDt[ nR * iState: nR * (iState + 1) ]  = np.fft.ifft(fDt[ nR * iState: nR * (iState + 1) ], dtype=np.complex64)
 # evolution 2nd step
 cPt = DtoP(cDt,Rmin,Rmax, nR,nf,vp,red)  
 cPt = UV.dot(cPt, dtype=np.complex64)
 cDt = PtoD(cDt,Rmin, Rmax, nR, nf, vp, red)  

 pDis = dissociation(cDt,Rmin,Rmax,nR,nf,red)
 dis.write(str(t*dt/ps) + " "  + str(pDis) + "\n") 
   

np.savetxt("psi%s.txt"%(Time[-1]),cDt) 
wf.close()
popD.close()       
popP.close()            
dis.close()
