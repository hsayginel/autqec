import numpy as np
from utils.linalg import inv_mod2, invRange,rref_mod2, rank_mod2

M = np.array([[0,0,0,0,1,1,0,0],
              [1,1,0,0,1,0,0,0],
              [0,0,1,0,0,0,0,0],
              [0,0,0,1,0,0,0,0],
              [1,0,0,0,1,0,0,0],
              [0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,1]])

k = 4
XX = M[:k,:k].copy()
ZX = M[k:,:k].copy()
ZZ = M[k:,k:].copy()
XZ = M[:k,k:].copy()
_, p_XX, _, _ = rref_mod2(XX) # [0 2 3]
_, p_ZZ, _, _ = rref_mod2(ZZ) # [0 1 2 3]
_, p_XZ, _, _ = rref_mod2(XZ) # [0 1]
_, p_ZX, _, _ = rref_mod2(ZX) # [0]

H_ix_try = list(set(p_XZ) & set(p_ZX))

M_og = M.copy()
for i in H_ix_try:
    M[:,[i,i+k]] = M[:,[i+k,i]]
    if rank_mod2(M[:k,:k]) > rank_mod2(XX) and :
        print(('H',i+1))


print(p_XX)
print(p_ZZ)
print(p_XZ)
print(p_ZX)