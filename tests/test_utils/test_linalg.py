from utils.linalg import *
from utils.symplectic import *

assert is_identity_matrix(np.eye(5))
assert is_matrix_full_rank(np.array([[1,0],[0,1]])) == True
assert is_matrix_full_rank(np.array([[1,0],[0,1],[0,0]])) == False

M = np.array([[0,0,0,1,1,1,1],
              [0,1,1,0,0,1,1],
              [1,0,1,0,1,0,1]])
M_rref, r, R, C = rref_mod2(M)
print(M_rref)
assert np.allclose(M_rref,R@M@C%2)
