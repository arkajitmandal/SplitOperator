import math
import time
import numpy as np
import scipy as sp
import Potential as Pt
from numpy import linalg as LA
import sys
from scipy.sparse import linalg as sLa
from scipy import sparse as sprs
dt = 0.2
tsteps = 8400
Time = range(tsteps)
print "Defauls"



def T(Rmin,Rmax,n_steps,nf,mass=9267.5654):
 N = n_steps*nf*2
 Te = np.zeros((N,N),dtype = np.float32)
 Rmin = float(Rmin)
 Rmax = float(Rmax)
 step = float((Rmax-Rmin)/n_steps)
 rDist = np.arange(Rmin,Rmax,step)
 K = np.pi/step

 for ri in range(n_steps):
  for rj in range(ri,n_steps):
   if ri == rj:  
    T = (0.5/mass)*K**2/3*(1+(2.0/n_steps**2)) 
   else:    
    T = (0.5/mass)*(2*K**2/(n_steps**2))*((-1)**(rj-ri)/(np.sin(np.pi*(rj-ri)/n_steps)**2)) 
   for i in range(2*nf-red):
    row = ri + i*n_steps
    col = rj + i*n_steps
    if (abs(T)>1E-12):
    	Te[row,col] = T
    	Te[col,row] = Te[row,col]
 return Te
#---------------------------------

#---------------------------------
# CREATION OF THE HAMILTONIAN

def H(Rmin,Rmax,n_steps,nf):
 #H = np.zeros((n_steps*2*nf,n_steps*2*nf))
 Rmin = float(Rmin)
 Rmax = float(Rmax)
 step = float((Rmax-Rmin)/n_steps)
 Vij = Pt.Help(Rmin,Rmax,n_steps,nf)
 Tij = T(Rmin,Rmax,n_steps,nf)
 H = Vij + Tij 

 return H


def Ut(Rmin,Rmax,n_steps,nf,dt,red):	
 Rmin = float(Rmin)
 Rmax = float(Rmax)
 step = float((Rmax-Rmin)/n_steps)
 Vij = sprs.csr_matrix(Pt.Help(Rmin,Rmax,n_steps,nf,red))
 Tij = sprs.csr_matrix(T(Rmin,Rmax,n_steps,nf)[:(2*nf-red)*n_steps,:(2*nf-red)*n_steps])
 Hij = Tij + Vij 
 Tij = 0
 Vij = 0 
 print size(Hij) 
 #Hij = sprs.csr_matrix(Hij)
 exH = exdt( Hij, dt,(2*nf-red) * n_steps) #exdt(Hij,dt,N)
 return exH



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

def psi2(Rmin,Rmax,n,nf,V,vp):
 p = np.zeros((2*n*nf),dtype=complex)
 Rmin = float(Rmin)
 Rmax = float(Rmax)
 step = float((Rmax-Rmin)/n)
 rDist = np.arange(Rmin,Rmax,step)
 Norm = 0
 CD0  =  V[:,0] 
 CP0  =  DtoP(CD0,Rmin,Rmax,n,nf,vp) 
 for ri in range(n):
  for k in range(1):
    p[ri+n] += abs(CP0[ri+k*n])
  Norm += p[ri+n]**2
 return PtoD(p/math.sqrt(Norm),Rmin,Rmax,n,nf,vp)
#---------------------------------
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
# exponential of matrix 
def exdt(A,dt,N,terms=14):
	Apow = sprs.csr_matrix(np.identity(N,dtype=np.complex64))
	Adt = sprs.csr_matrix(np.identity(N,dtype=np.complex64))
	for i in range(1,terms):
	    Apow = Apow.dot(A)  
	    Adt  = Adt + Apow *  (( -1j * dt )**i) /math.factorial(i)
	return Adt 
	
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
step = float((Rmax-Rmin)/n_steps)
#---------------------------------
Udt = Ut(Rmin,Rmax,n_steps,nf,dt,red) 

#---------------------------------
# Energies
#---------------------------------

#ve = Pt.electronic(Rmin,Rmax,n_steps,nf)
vp = Pt.polariton(Rmin,Rmax,n_steps,nf,red)
#ve = Pt.adiabat(Rmin,Rmax,n_steps,nf,red)

# Initial Wf
cD0 = psi(Rmin,Rmax,n_steps,nf,vp,red)
np.savetxt("psi0.txt",cD0.real) 
#---------------------------------
#---------------------------------
# Evolve
popP = open("popP.txt","w+",buffering=0)
popD = open("popD.txt","w+",buffering=0)
#popA = open("popA.txt","w+",buffering=0)
dis =  open("dis.txt","w+",buffering=0)
wf =   open("psi.txt","w+",buffering=0)
cDt = cD0
#-----------------------------
for t in Time:
 print t 
 # evolution
 cDt = Udt.dot(cDt)

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
