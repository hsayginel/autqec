from utils.linalg import *
from utils.symplectic import *
from automorphisms import symplectic_mat_to_logical_circ

assert is_identity_matrix(np.eye(5))
assert is_matrix_full_rank(np.array([[1,0],[0,1]])) == True
assert is_matrix_full_rank(np.array([[1,0],[0,1],[0,0]])) == False

M = np.array([[0,0,0,1,1,1,1],
              [0,1,1,0,0,1,1],
              [1,0,1,0,1,0,1]])
M_rref, r, R, C = rref_mod2(M)
print(M_rref)
assert np.allclose(M_rref,R@M@C%2)

# k = 2
# cnot_circ_in = (CNOT_gate(1,2,k)@CNOT_gate(2,1,k))%2
# GL_part = cnot_circ_in[:k,:k]

# gates, _ = CNOT_circ_from_GL_mat(GL_part)
# cnot_circ_out = symp_mat_prods(gates,k)

# assert np.allclose(cnot_circ_in,cnot_circ_out) 