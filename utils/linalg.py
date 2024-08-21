import numpy as np
from scipy import sparse
# from ldpc.mod2 import rref_mod2, rank
import numba as nb

def is_identity_matrix(matrix):
    """Checks if matrix is k x k identity."""
    if not matrix.shape[0] == matrix.shape[1]:
        return False
    return np.all(np.eye(matrix.shape[0]) == matrix)

def rank_mod2(matrix):
     _, k, _, _ = rref_mod2(matrix)
     return k 

def is_matrix_full_rank(matrix):
    """Checks if matrix is full rank."""
    if matrix.ndim == 1:
        return True
    m = matrix.shape[0]
    k = rank_mod2(matrix)
    # k = rank(matrix)
    return m == k

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

def rref_mod2(A,nA=0,tB=1,nC=-1,r0=0):
    '''Return Howell matrix form modulo N plus transformation matrix U such that H = U @ A mod N'''
    m,n = A.shape
    A = np.array(A,dtype=np.int8)
    B = np.hstack([A,np.eye(m,dtype=np.int8)])
    nA,nB,nC = blockDims(n,nA,tB,nC)
    HU, pivots = HowZ2(B,tB,nB,nC,r0)
    ix = list(pivots) + invRange(n,pivots)
    
    # [rref_mod2_form, matrix_rank, transform_matrix_rows, transform_matrix_columns] 
    H, U = HU[:,:n],HU[:,n:]
    H = H[:,ix]
    P = ix_to_perm_mat(ix)

    return H, len(pivots), U, P

def inv_mod2(mat):
    H, r, U, P = rref_mod2(mat)
    return (P@U)%2

def CNOT_circ_from_GL_mat(matrix, full=True):
    """
    Taken and modified from Joschka Roffe's library: 
    LDPC: Python tools for low density parity check codes
    https://pypi.org/project/ldpc/. 

    
    Converts a binary matrix to row echelon form via Gaussian Elimination.

    Parameters
    ----------
    matrix : numpy.ndarry or scipy.sparse
        A binary matrix in either numpy.ndarray format or scipy.sparse
    full: bool, optional
        If set to `True', Gaussian elimination is only performed on the rows below
        the pivot. If set to `False' Gaussian eliminatin is performed on rows above
        and below the pivot. 
    
    Returns
    -------
        row_ech_form: numpy.ndarray
            The row echelon form of input matrix
        rank: int
            The rank of the matrix
        transform_matrix: numpy.ndarray
            The transformation matrix such that (transform_matrix@matrix)=row_ech_form
        pivot_cols: list
            List of the indices of pivot num_cols found during Gaussian elimination

    Examples
    --------
    >>> H=np.array([[1, 1, 1],[1, 1, 1],[0, 1, 0]])
    >>> re_matrix=row_echelon(H)[0]
    >>> print(re_matrix)
    [[1 1 1]
     [0 1 0]
     [0 0 0]]

    >>> re_matrix=row_echelon(H,full=True)[0]
    >>> print(re_matrix)
    [[1 0 1]
     [0 1 0]
     [0 0 0]]

    """
    num_rows, num_cols = np.shape(matrix)
    
    # Take copy of matrix if numpy (why?) and initialise transform matrix to identity
    if isinstance(matrix, np.ndarray):
        the_matrix = np.copy(matrix)
        transform_matrix = np.identity(num_rows).astype(int)
    elif isinstance(matrix, sparse.csr.csr_matrix):
        the_matrix = matrix
        transform_matrix = sparse.eye(num_rows, dtype="int", format="csr")
    else:
        raise ValueError('Unrecognised matrix type')

    pivot_row = 0
    pivot_cols = []

    reverse_quantum_circuit = []
    
    # Iterate over cols, for each col find a pivot (if it exists)
    for col in range(num_cols):

        # Select the pivot - if not in this row, swap rows to bring a 1 to this row, if possible
        if the_matrix[pivot_row, col] != 1:

            # Find a row with a 1 in this col
            swap_row_index = pivot_row + np.argmax(the_matrix[pivot_row:num_rows, col])

            # If an appropriate row is found, swap it with the pivot. Otherwise, all zeroes - will loop to next col
            if the_matrix[swap_row_index, col] == 1:

                # Swap rows
                the_matrix[[swap_row_index, pivot_row]] = the_matrix[[pivot_row, swap_row_index]]
                reverse_quantum_circuit.append(('SWAP',(swap_row_index+1,pivot_row+1)))
                # Transformation matrix update to reflect this row swap
                transform_matrix[[swap_row_index, pivot_row]] = transform_matrix[[pivot_row, swap_row_index]]

        # If we have got a pivot, now let's ensure values below that pivot are zeros
        if the_matrix[pivot_row, col]:

            if not full:  
                elimination_range = [k for k in range(pivot_row + 1, num_rows)]
            else:
                elimination_range = [k for k in range(num_rows) if k != pivot_row]

            # Let's zero those values below the pivot by adding our current row to their row
            for j in elimination_range:

                if the_matrix[j, col] != 0 and pivot_row != j:    ### Do we need second condition?

                    the_matrix[j] = (the_matrix[j] + the_matrix[pivot_row]) % 2
                    reverse_quantum_circuit.append(('CNOT',(pivot_row+1,j+1))) # control is pivot row
                    # Update transformation matrix to reflect this op
                    transform_matrix[j] = (transform_matrix[j] + transform_matrix[pivot_row]) % 2

            pivot_row += 1
            pivot_cols.append(col)

        # Exit loop once there are no more rows to search
        if pivot_row >= num_rows:
            break

    row_esch_matrix = the_matrix

    return [reverse_quantum_circuit, row_esch_matrix]



