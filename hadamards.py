from automorphisms import symplectic_mat_to_logical_circ
import numpy as np
from utils.symplectic import *
from utils.linalg import inv_mod2, invRange,rref_mod2

#########################################################################################################
# k = 4
# M_in = (H_gate(1,k)@H_gate(3,k))%2
# gates = symplectic_mat_to_logical_circ(M_in).find_H_gates()
# M_out = symp_mat_prods(gates[::-1],k)
# assert np.allclose(M_in,M_out)


# #########################################################################################################
# print()
# k = 2
# M_in = (S_gate(1,2)@Xsqrt_gate(1,2))%2
# gates = symplectic_mat_to_logical_circ(M_in).run()
# M_out = symp_mat_prods(gates,k)
# assert np.allclose(M_in,M_out)

# #########################################################################################################
print()
k = 3
M_in = (SWAP_gate(1,3,k)@CNOT_gate(1,3,k)@SWAP_gate(2,3,k)@H_gate(1,k))%2
# print(M_in)
gates = symplectic_mat_to_logical_circ(M_in).run()
print(gates)
M_out = symp_mat_prods(gates,k)
assert np.allclose(M_in,M_out)

# #########################################################################################################
# print()
# k=4
# M_in = (Xsqrt_gate(1,k) @ S_gate(1,k) @ CZ_gate(1,2,k))%2
# gates = symplectic_mat_to_logical_circ(M_in).run()
# print(gates)
# M_out = symp_mat_prods(gates,k)
# assert np.allclose(M_in,M_out)
