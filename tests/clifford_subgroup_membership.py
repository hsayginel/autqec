from utils.linalg import *
import pickle
from math import comb

with open('codetables/logical_gates/gates_n20k18.pkl','rb') as f:
    circs = pickle.load(f)
log_circs = circs['logical']

def aut_gens_as_binary_mat(log_circs,k):
    """
    Binary representation: | H | SWAP | CNOT | CZ | S | C(X,X) | Xsqrt |

    Args:
        log_circs (list): list of logical circuits
        k (int): number of logical qubits
    
    Returns:
        binary_vecs_4_auts (np.array): binary representation of logical group generators
    
    """ 
    no_of_gens = len(log_circs)
    kc2 = comb(k,2)
    no_of_cols = 3*k + 4*kc2
    binary_vecs_4_auts = np.zeros((no_of_gens,no_of_cols),dtype=int)
    for g, row in enumerate(binary_vecs_4_auts):
        circ = log_circs[g]
        for op in circ: 
            gate_type, qubits = op
            if gate_type == 'H':
                i = qubits - 1
                col_ind = i
            elif gate_type == 'SWAP':
                i,j = qubits 
                col_ind = comb(k, 2) - comb(k - i + 1, 2) + (j - i - 1) + k 
            elif gate_type == 'CNOT':
                i,j = qubits 
                col_ind = comb(k, 2) - comb(k - i + 1, 2) + (j - i - 1) + k + kc2
            elif gate_type == 'CZ':
                i,j = qubits
                col_ind = comb(k, 2) - comb(k - i + 1, 2) + (j - i - 1) + k + 2*kc2
            elif gate_type == 'S':
                i = qubits - 1
                col_ind = i + k + 3*kc2
            elif gate_type == 'C(X,X)':
                i,j = qubits
                col_ind = comb(k, 2) - comb(k - i + 1, 2) + (j - i - 1) + 2*k + 3*kc2
            elif gate_type == 'Xsqrt':
                i = qubits - 1
                col_ind = i + 2*k + 4*kc2
            binary_vecs_4_auts[g, col_ind] = 1
    return binary_vecs_4_auts


M = aut_gens_as_binary_mat(log_circs,k=18)

print(len(rref_mod2(M)[1]))

