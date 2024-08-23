from utils.symplectic import *

H = H_gate(1,1)

S = S_gate(1,1)

gamma_XYZ = gamma_XYZ_gate(1,1)

gamma_XZY = gamma_XZY_gate(1,1)

Xsq = Xsqrt_gate(1,1)

CNOT_1_2 = CNOT_gate(1,2,2)

CNOT_2_1 = CNOT_gate(2,1,2)

CZ = CZ_gate(1,2,2)

CXX = CX_XX_gate(1,2,2)

SWAP = SWAP_gate(1,2,2)

clifford_circ_stab_update
op_2bit_to_op_3bit_and_phase
symp_mat_prods
symp_prod
is_symplectic

from automorphisms import circ_from_symp_mat
import numpy as np
from utils.symplectic import *

# print(H_gate(1,1)@S_gate(1,1))
# print(symp_mat_prods([('H',1),('S',1)],1))

#########################################################################################################
k = 4
M_in = (CZ_gate(1,2,k) @ CZ_gate(3,4,k) @ CZ_gate(1,3,k))%2
gates = circ_from_symp_mat(M_in).run()
M_out = symp_mat_prods(gates,k)
# print(gates)
assert np.allclose(M_in,M_out) 

#########################################################################################################
k = 2
M_in = (S_gate(1,2)@Xsqrt_gate(1,2))%2
gates = circ_from_symp_mat(M_in).run()
M_out = symp_mat_prods(gates,k)
assert np.allclose(M_in,M_out)

#########################################################################################################
k = 2
M_in = (Xsqrt_gate(1,2)@S_gate(1,2))%2
gates = circ_from_symp_mat(M_in).run()
M_out = symp_mat_prods(gates,k)
# print(gates)
assert np.allclose(M_in,M_out)

#########################################################################################################
k=4
M_in = (CNOT_gate(1,2,k))%2
gates = circ_from_symp_mat(M_in).run()
M_out = symp_mat_prods(gates,k)
# print(gates)
assert np.allclose(M_in,M_out)

#########################################################################################################
k=2
M_in = (CNOT_gate(1,2,k) @ Xsqrt_gate(1,k) @ S_gate(1,k) @ CZ_gate(1,2,k))%2
assert is_symplectic(M_in)
gates = circ_from_symp_mat(M_in).run()
# print(gates)
M_out = symp_mat_prods(gates,k)
assert np.allclose(M_in,M_out)

#########################################################################################################
k=4
M_in = (Xsqrt_gate(1,k) @ S_gate(1,k) @ CZ_gate(1,2,k))%2
assert is_symplectic(M_in)
gates = circ_from_symp_mat(M_in).run()
M_out = symp_mat_prods(gates,k)
# print(gates)
assert np.allclose(M_in,M_out)

#########################################################################################################
k=4
M_in = (H_gate(2,k) @ S_gate(2,k))%2
assert is_symplectic(M_in)
gates = circ_from_symp_mat(M_in).run()
M_out = symp_mat_prods(gates,k)
assert np.allclose(M_in,M_out)

#########################################################################################################
k = 3
M_in = (SWAP_gate(1,3,3)@SWAP_gate(2,3,3))%2
assert is_symplectic(M_in)
gates = circ_from_symp_mat(M_in).run()
M_out = symp_mat_prods(gates,k)
assert np.allclose(M_in,M_out)

#########################################################################################################
k = 3
M_in = (SWAP_gate(1,3,k)@CNOT_gate(1,3,k)@SWAP_gate(2,3,k)@H_gate(1,k))%2
assert is_symplectic(M_in)
gates = circ_from_symp_mat(M_in).run()
M_out = symp_mat_prods(gates,k)
assert np.allclose(M_in,M_out)

#########################################################################################################
k = 3
M_in = (SWAP_gate(1,2,k)@SWAP_gate(2,3,k)@S_gate(1,k)@S_gate(2,k)@H_gate(2,k)@Xsqrt_gate(3,k))
assert is_symplectic(M_in)
gates = circ_from_symp_mat(M_in).run()
M_out = symp_mat_prods(gates,k)
assert np.allclose(M_in,M_out)

#########################################################################################################
k = 2
M_in = (CNOT_gate(1,2,k)@CNOT_gate(2,1,k)@S_gate(1,k)@CZ_gate(1,2,k))%2
assert is_symplectic(M_in)
gates = circ_from_symp_mat(M_in).run()
M_out = symp_mat_prods(gates,k)
assert np.allclose(M_in,M_out)

#########################################################################################################
M_in = np.array([[1,1,0,1],[1,0,1,1],[0,0,0,1],[0,0,1,1]])
assert is_symplectic(M_in)
gates = circ_from_symp_mat(M_in).run()
M_out = symp_mat_prods(gates,k)
assert np.allclose(M_in,M_out)

M_in = np.array([[1,1,0,1],[1,0,1,1],[0,0,0,1],[0,0,1,1]])
omega = np.eye(2*k,dtype=int)
omega[:,:k], omega[:,k:] = omega[:,k:].copy(), omega[:,:k].copy()
M_in_inv = omega@M_in.T@omega
assert is_symplectic(M_in_inv)
gates = circ_from_symp_mat(M_in_inv).run()
M_out = symp_mat_prods(gates[::-1],k)
assert np.allclose(M_in,M_out)

#########################################################################################################
k = 2
M_in = (CNOT_gate(1,2,2)@CNOT_gate(2,1,2))%2
assert is_symplectic(M_in)
gates = circ_from_symp_mat(M_in).run()
M_out = symp_mat_prods(gates,k)
assert np.allclose(M_in,M_out)

M_in = (H_gate(1,1)@S_gate(1,1))
omega = np.eye(2,dtype=int)
k=1
omega[:,:k], omega[:,k:] = omega[:,k:].copy(), omega[:,:k].copy()
M_in_inv = omega@M_in.T@omega
gates = circ_from_symp_mat(M_in).run()
# print(gates)
M_out = symp_mat_prods(gates,k)
M_out_inv = symp_mat_prods(gates[::-1],k)
assert np.allclose(M_in,M_out)
assert np.allclose(M_in_inv,M_out_inv)

gates_inv = circ_from_symp_mat(M_in_inv).run()
M_out_inv2 = symp_mat_prods(gates_inv,k)
M_out2 = symp_mat_prods(gates_inv[::-1],k)
assert np.allclose(M_in_inv,M_out_inv)
assert np.allclose(M_in,M_out2)

#########################################################################################################
k = 4
M_in = (H_gate(1,k)@H_gate(3,k))%2
gates = circ_from_symp_mat(M_in).run()
M_out = symp_mat_prods(gates,k)
assert np.allclose(M_in,M_out)