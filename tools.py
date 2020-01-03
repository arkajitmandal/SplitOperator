# INITIAL WAVE-FUNCTION
def psi(R, Up, nState):
  nR = len(R) 
  p = np.zeros( nState * nR, dtype=np.complex64)
  istate = 1
  A = 0.260778 
  B = 19.1221  
  C = 3.01125 
  p[istate * nR: (istate + 1) * nR] = A*np.exp(-B * R -C)**2.0)  
  norm = np.sqrt(p.dot( p.conjugate())) 
  p = p/norm
  #return p
  return AtoD(p, nR, nState, Up)

# rotating electronic representation
def AtoD(cP, nR, nState, Up):
  N = len(cP)
  cD = np.zeros((N),dtype= np.complex64)
  for ri in range(nR):
    U = Up[ri,:,:]
    for i in range(nState):
      for j in range(nState):
        k = i * nR + ri
        l = j * nR + ri
        cD[k] += U[i, j] * cP[l]
  return cD

# rotating electronic representation
def DtoA(cD, nR, nState, Up):
  N = len(cD)
  cP = np.zeros((N),dtype= np.complex64)
  for ri in range(nR):
    U = Up[ri,:,:]
    for i in range(nState):
      for j in range(nState):
        k = i * nR + ri
        l = j * nR + ri
        cP[k] += U[j, i] * cD[l]
  return cP

# computing populations
def population(Ci, nState) :
  p = np.zeros(nState, dtype=np.float32) 
  nR = int(len(Ci) / nState)
  for i in range(nState):
	p[i] = Ci[i * nR: (i + 1) * nR].conjugate().dot( Ci[ i * nR: (i + 1) * nR]).real 
  return p

# computing dissociation probability
def dissociation(cDt, R, nState):
  Rmin = float(R[0])
  Rmax = float(R[-1])
  dR = float(R[1] - R[0])
  R0 = 15.0
  ref = int((R0 - Rmin) / dR) 
  Prob = 0
  for i in range(nState) :
    start = (nR * i) + ref
    end = (i + 1) * nR
    Prob += (np.matmul(cDt.T[start:end].conjugate(),cDt[start:end])).real
  return Prob


def polariton(R, nf, red = 0):
  Hpl = Help(R, nf, red) 
  nR  = len(R)
  nState = len(Hpl)/nR
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

#----------------------------------------
# MATRIX DIAGONALIZATION

def Diag(H):
    E,V = LA.eigh(H) # E corresponds to the eigenvalues and V corresponds to the eigenvectors
    return E,V
#----------------------------------------


# sparse matrix size 	
def size(m):
	return m.data.nbytes + m.indptr.nbytes + m.indices.nbytes