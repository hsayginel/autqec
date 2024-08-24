from utils.qec import stabs_to_H_symp, compute_standard_form
from utils.pauli import *

n = 5
k = 1 
d = 3
stabs = ['XZZXI','IXZZX','XIXZZ','ZXIXZ']
H_symp = stabs_to_H_symp(stabs)
G,LX,LZ,D = compute_standard_form(H_symp)

print(binary_vecs_to_paulis(G))
print(binary_vecs_to_paulis(LX))
print(binary_vecs_to_paulis(LZ))
print(binary_vecs_to_paulis(D))