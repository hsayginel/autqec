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

from automorphisms import symplectic_mat_to_logical_circ
import numpy as np
from utils.symplectic import *

#########################################################################################################
k = 4
M_in = (CZ_gate(1,2,k) @ CZ_gate(3,4,k) @ CZ_gate(1,3,k))%2
gates = symplectic_mat_to_logical_circ(M_in).run()
M_out = symp_mat_prods(gates[::-1],k)
# print(gates)
assert np.allclose(M_in,M_out) 

#########################################################################################################
k = 2
M_in = (S_gate(1,2)@Xsqrt_gate(1,2))%2
gates = symplectic_mat_to_logical_circ(M_in).run()
M_out = symp_mat_prods(gates[::-1],k)
assert np.allclose(M_in,M_out)

#########################################################################################################
k = 2
M_in = (Xsqrt_gate(1,2)@S_gate(1,2))%2
gates = symplectic_mat_to_logical_circ(M_in).run()
M_out = symp_mat_prods(gates[::-1],k)
# print(gates)
assert np.allclose(M_in,M_out)

#########################################################################################################
k=4
M_in = (CNOT_gate(1,2,k))%2
gates = symplectic_mat_to_logical_circ(M_in).run()
M_out = symp_mat_prods(gates[::-1],k)
# print(gates)
assert np.allclose(M_in,M_out)

#########################################################################################################
k=2
M_in = (CNOT_gate(1,2,k) @ Xsqrt_gate(1,k) @ S_gate(1,k) @ CZ_gate(1,2,k))%2
assert is_symplectic(M_in)
gates = symplectic_mat_to_logical_circ(M_in).run()
M_out = symp_mat_prods(gates[::-1],k)
assert np.allclose(M_in,M_out)

#########################################################################################################
k=4
M_in = (Xsqrt_gate(1,k) @ S_gate(1,k) @ CZ_gate(1,2,k))%2
assert is_symplectic(M_in)
gates = symplectic_mat_to_logical_circ(M_in).run()
M_out = symp_mat_prods(gates[::-1],k)
# print(gates)
assert np.allclose(M_in,M_out)

#########################################################################################################
k=4
M_in = (H_gate(2,k) @ S_gate(2,k))%2
assert is_symplectic(M_in)
gates = symplectic_mat_to_logical_circ(M_in).run()
M_out = symp_mat_prods(gates[::-1],k)
assert np.allclose(M_in,M_out)

#########################################################################################################
k = 3
M_in = (SWAP_gate(1,3,3)@SWAP_gate(2,3,3))%2
assert is_symplectic(M_in)
gates = symplectic_mat_to_logical_circ(M_in).run()
M_out = symp_mat_prods(gates[::-1],k)
assert np.allclose(M_in,M_out)

#########################################################################################################
k = 3
M_in = (SWAP_gate(1,3,k)@CNOT_gate(1,3,k)@SWAP_gate(2,3,k)@H_gate(1,k))%2
assert is_symplectic(M_in)
gates = symplectic_mat_to_logical_circ(M_in).run()
M_out = symp_mat_prods(gates[::-1],k)
assert np.allclose(M_in,M_out)

#########################################################################################################
k = 3
M_in = (SWAP_gate(1,2,k)@SWAP_gate(2,3,k)@S_gate(1,k)@S_gate(2,k)@H_gate(2,k)@Xsqrt_gate(3,k))
assert is_symplectic(M_in)
gates = symplectic_mat_to_logical_circ(M_in).run()
M_out = symp_mat_prods(gates[::-1],k)
assert np.allclose(M_in,M_out)

#########################################################################################################
k = 2
M_in = (CNOT_gate(1,2,k)@CNOT_gate(2,1,k)@S_gate(1,k)@CZ_gate(1,2,k))%2
assert is_symplectic(M_in)
gates = symplectic_mat_to_logical_circ(M_in).run()
M_out = symp_mat_prods(gates[::-1],k)
assert np.allclose(M_in,M_out)

#########################################################################################################
M_in = np.array([[1,1,0,1],[1,0,1,1],[0,0,0,1],[0,0,1,1]])
assert is_symplectic(M_in)
gates = symplectic_mat_to_logical_circ(M_in).run()
M_out = symp_mat_prods(gates[::-1],k)
assert np.allclose(M_in,M_out)

#########################################################################################################
k = 2
M_in = (CNOT_gate(1,2,2)@CNOT_gate(2,1,2))%2
assert is_symplectic(M_in)
gates = symplectic_mat_to_logical_circ(M_in).run()
M_out = symp_mat_prods(gates[::-1],k)
assert np.allclose(M_in,M_out)