import numpy as np
import numba as nb

@nb.jit(nb.types.Tuple((nb.int8[:,:],nb.int64[:]))(nb.int8[:,:],nb.int64,nb.int64,nb.int64,nb.int64))
def HowZ2(A,tB,nB,nC,r0):
    pivots = []
    B = A.copy()
    if np.sum(A) == 0:
        return B,np.array(pivots,dtype=nb.int64)
    m = len(B)
    r = r0
    for j in range(nC):
        for t in range(tB):
            ## c is the column of B we are currently looking at
            c = j + t * nB
            iList = [i for i in range(r,m) if B[i,c] > 0]
            if len(iList) > 0:
                i = iList.pop(0)
                pivots.append(c)
                ## found j: if j > r, swap row j with row r
                if i > r:  
                    ## swap using bitflips - more elegant than array indexing
                    B[r] = B[r] ^ B[i]
                    B[i] = B[r] ^ B[i]
                    B[r] = B[r] ^ B[i]
                ## eliminate non-zero entries in column c apart from row r
                for i in [i for i in range(r) if B[i,c] > 0] + iList:     
                    B[i] = B[i] ^ B[r]
                r +=1
    return B,np.array(pivots,dtype=nb.int64)
    
def get_CNOT_circ(A,tB,nB,nC,r0):
    qc = []
    pivots = []
    B = A.copy()
    if np.sum(A) == 0:
        return B,pivots,qc
    m = len(B)
    r = r0
    for j in range(nC):
        for t in range(tB):
            ## c is the column of B we are currently looking at
            c = j + t * nB
            iList = [i for i in range(r,m) if B[i,c] > 0]
            if len(iList) > 0:
                i = iList.pop(0)
                pivots.append(c)
                ## found j: if j > r, swap row j with row r
                if i > r:  
                    ## swap using bitflips - more elegant than array indexing
                    B[r] = B[r] ^ B[i]
                    B[i] = B[r] ^ B[i]
                    B[r] = B[r] ^ B[i]
                    qc.append(('SWAP',(r+1,i+1)))
                ## eliminate non-zero entries in column c apart from row r
                for i in [i for i in range(r) if B[i,c] > 0] + iList:     
                    B[i] = B[i] ^ B[r]
                    qc.append(('CNOT',(r+1,i+1)))
                r +=1
    return B,pivots,qc

# @nb.jit
def blockDims(n,nA=0,tB=1,nC=-1):
    nA = min(n,nA)
    nB = (n - nA) // tB
    if nC < 0 or nC > nB:
        nC = nB 
    return nA,nB,nC

def invRange(n,S):
    '''return list of elements of range(n) NOT in S'''
    return sorted(set(range(n)) - set(S))

def ix_to_perm_mat(ix):
    m = len(ix)
    P = np.zeros((m,m),dtype=int)
    for i in range(m):
        P[ix[i],i] = 1
    return P

def rref_mod2(A,CNOTs=False):
    '''Return Howell matrix form modulo N plus transformation matrix U such that H = U @ A mod N'''
    m,n = A.shape
    A = np.array(A,dtype=np.int8)
    B = np.hstack([A,np.eye(m,dtype=np.int8)])
    nA=0
    tB=1
    nC=-1
    r0=0
    nA,nB,nC = blockDims(n,nA,tB,nC)

    if CNOTs == False:
        HU, pivots = HowZ2(B,tB,nB,nC,r0)
        ix = list(pivots) + invRange(n,pivots)

        H, U = HU[:,:n],HU[:,n:]
        H = H[:,ix]
        P = ix_to_perm_mat(ix)
        
        return H, pivots, U, P
    elif CNOTs: 
        HU, pivots, qc = get_CNOT_circ(B,tB,nB,nC,r0)
        ix = list(pivots) + invRange(n,pivots)
        H, U = HU[:,:n],HU[:,n:]
        H = H[:,ix]

        return qc, H


def rank_mod2(mat):
    """Return rank of binary matrix."""
    _, pivots, _, _ = rref_mod2(mat)
    return len(pivots) 

def is_matrix_full_rank(mat):
    """Checks if matrix is full rank."""
    if mat.ndim == 1:
        return True
    m = mat.shape[0]
    k = rank_mod2(mat)
    return m == k

def inv_mod2(mat):
    _, _, U, P = rref_mod2(mat)
    return (P@U)%2

def is_identity_matrix(mat):
    """Checks if matrix is k x k identity."""
    if not mat.shape[0] == mat.shape[1]:
        return False
    return np.all(np.eye(mat.shape[0]) == mat)


