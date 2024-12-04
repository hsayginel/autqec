from autqec.utils.pauli import *


mat = np.array([[1,0,1,1]]) #YZ
assert binary_vecs_to_paulis(mat,phase_bit = False)[0][0] == 'Y'
assert binary_vecs_to_paulis(mat,phase_bit = False)[0][1] == 'Z'

mat = np.array([[1,0,1,1]]) #YZ
assert binary_vecs_to_paulis(mat,phase_bit = True)[0] == 1
assert binary_vecs_to_paulis(mat,phase_bit = True)[1][0][0] == 'XZ'
assert binary_vecs_to_paulis(mat,phase_bit = True)[1][0][1] == 'Z'

pauli1 = ['XZ','Z']
phase1 = 1
pauli2 = ['X','Z']
phase2 = 0
pauli_product = multiply_pauli_strings(pauli1, phase1, pauli2, phase2)

assert pauli_product[0][0] == 'Z'
assert pauli_product[0][1] == 'I'
assert pauli_product[1] == 3 # phase (power of i)