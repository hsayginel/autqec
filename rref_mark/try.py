import numpy as np 


from ldpc.mod2 import rref_mod2
from time import time

A = np.load('examples/bivariant_bicycle_codes/code_data/HX_n144k12d12.npy')
t0 = time()
H, r, U, P = rref_mod2(A)
t1 = time()
print(t1-t0)
t1 = time()
H2, r2, U2, P2 = rref_mod2(A)
t2 = time()
print(t2-t1)

print(np.allclose(H,H2))
print(np.allclose(r,r2))
print(np.allclose(U,U2))
print(np.allclose(P,P2))

print(np.allclose(U@A@P%2,H))

# print(rref)

# HX = np.load('examples/bivariant_bicycle_codes/code_data/HX_n144k12d12.npy')
# HX = np.array(HX,dtype=np.int8)
# HZ = np.load('examples/bivariant_bicycle_codes/code_data/HZ_n144k12d12.npy')
# HZ = np.array(HZ,dtype=np.int8)
# rref, U, pivots = rref_mod2(HZ)
# print(len(pivots))

