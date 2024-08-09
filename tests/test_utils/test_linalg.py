from utils.linalg import *
from utils.symplectic import *
from automorphisms import symplectic_mat_to_logical_circ

assert is_identity_matrix(np.eye(5))

k = 2
cnot_circ_in = (CNOT_gate(1,2,k)@CNOT_gate(2,1,k))%2
GL_part = cnot_circ_in[:k,:k]

gates, _ = CNOT_circ_from_GL_mat(GL_part)
cnot_circ_out = symp_mat_prods(gates,k)

assert np.allclose(cnot_circ_in,cnot_circ_out) 

assert is_matrix_full_rank(np.array([[1,0],[0,1]])) == True
assert is_matrix_full_rank(np.array([[1,0],[0,1],[0,0]])) == False

