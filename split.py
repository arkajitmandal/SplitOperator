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
n_steps = 1500
aniskip = 100000
#---------------------------------


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



step = float((Rmax-Rmin)/n_steps)
nState = nf*2-red
#---------------------------------
# Energies
#---------------------------------

#ve = Pt.electronic(Rmin,Rmax,n_steps,nf)
ep, vp = Pt.polariton(Rmin,Rmax,n_steps,nf,red)
UV = expV(ep, dt/2)
#ve = Pt.adiabat(Rmin,Rmax,n_steps,nf,red)

# Initial Wf
cD0 = psi(Rmin,Rmax,n_steps,nf,vp,red)
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
 print t 
 # population in diabatic representation
 rhoD = population(cDt, nState)
 popD.write(str(t*dt/ps) + " " + " ".join(rhoD.astype(str)) + "\n" )
 # population in polaritonic representation
 cPt = DtoP(cDt,Rmin,Rmax,n_steps,nf,vp,red)  
 rhoP = population(cPt, nState) 
 popP.write(str(t*dt/ps) + " " + " ".join(rhoP.astype(str)) + "\n" ) 
 # evolution 1st step 
 cPt = UV.dot(cPt)
 cDt = PtoD(cDt,Rmin,Rmax,n_steps,nf,vp,red)  
 # FFT
 fDt = np.zeros(len(cDt), dtype=np.complex64) 
 for iState in range(len(nState)):
  fDt[ n_steps * iState: n_steps * (iState + 1) ]  = np.fft.fft(cDt[ n_steps * iState: n_steps * (iState + 1) ], dtype=np.complex64)
 # evolution 2nd step

  
 #print cDt 

 rhoD = population(cDt,nf*2-red)
 # rotation from D to A
 cPt = DtoP(cDt,Rmin,Rmax,n_steps,nf,vp,red)  
 #cAt = DtoP(cDt,Rmin,Rmax,n_steps,nf,ve)  
 rhoP = population(cPt,nf*2-red) 
 #rhoA = population(cAt,nf*2) 
 pDis = dissociation(cDt,Rmin,Rmax,n_steps,nf,red)
 popD.write(str(t*dt/ps) + " " + " ".join(rhoD.astype(str)) + "\n" )
 #popA.write(str(t/ps) + " " + " ".join(rhoA.astype(str)) + "\n" )
 popP.write(str(t*dt/ps) + " " + " ".join(rhoP.astype(str)) + "\n" ) 
 dis.write(str(t*dt/ps) + " "  + str(pDis) + "\n") 
 # psi20
 if (t%aniskip==0):
  for i in range(n_steps):
   density = np.zeros(2*nf-red,dtype=np.complex64) 
   for j in range(2*nf-red):
    density[j] =  (cPt[j*n_steps + i].conjugate() * cPt[j*n_steps + i]).real
   wf.write( str( Rmin +  step*i)  + " "  + " ".join(density.astype(str))  + "\n" )
  wf.write("\n\n")	     

np.savetxt("psi%s.txt"%(Time[-1]),cDt) 
wf.close()
popD.close()       
popP.close()       
#popA.close()       
dis.close()
