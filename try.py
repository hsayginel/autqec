from utils.qec import stabs_to_H_symp, compute_standard_form
from utils.pauli import *
from utils.symplectic import *

n = 5
k = 1 
d = 3
stabs = ['XZZXI','IXZZX','XIXZZ','ZXIXZ']
H_symp = stabs_to_H_symp(stabs)
G,LX,LZ,D = compute_standard_form(H_symp)
tableux = np.vstack([G,LX,D,LZ]) 
T_symp_prod, omega = symp_prod(tableux,tableux,return_omega=True)
if np.allclose(T_symp_prod,omega) == False:
    raise AssertionError("Check stabilizer/destabilizer tableux.")

# print(binary_vecs_to_paulis(G))
# print(binary_vecs_to_paulis(LX))
# print(binary_vecs_to_paulis(LZ))
# print(binary_vecs_to_paulis(D))

H_new_tableux = stabs_to_H_symp(['YIYXX','IXZZX','XZZXI','XXYIY','XIZIX','XXXXX'])

print(np.mod(H_new_tableux @ omega @ tableux.T @ omega,2))

S_new_tableux = stabs_to_H_symp(['XIXZZ','IZYYZ','ZYYZI','ZZXIX','ZIYIZ','ZZZZZ'])

print(np.mod(S_new_tableux @ omega @ tableux.T @ omega,2))

HS_new_tableux = stabs_to_H_symp(['ZXIXZ','IYXXY','XXYIY','XIXZZ','XIIXY','XXXXX'])

print(np.mod(HS_new_tableux @ omega @ tableux.T @ omega,2))

print()
print()
k = 4
U_ACT = (S_gate(1,4) @  H_gate(1,4) @ S_gate(4,4)@ CNOT_gate(1,4,4) @ CZ_gate(1,2,4)@CNOT_gate(2,1,4))%2
print(U_ACT)

# phases of U_act
def i_phases(U_ACT):
    k = len(U_ACT)//2
    p = np.zeros(2*k,dtype=int)
    for row in range(2*k):
        for i in range(k):
            if U_ACT[row,i] == 1 and U_ACT[row,i+k] == 1:
                p[row] = (p[row] + 1)%4
    return p

U_ACT = S_gate(1,1) @ H_gate(1,1)
print(i_phases(U_ACT))