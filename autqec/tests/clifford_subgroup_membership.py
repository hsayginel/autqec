from autqec.utils.linalg import *
import pickle
from math import comb
from itertools import combinations

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
                binary_vecs_4_auts[g, col_ind] = 1
            elif gate_type == 'SWAP':
                i,j = qubits 
                col_ind = comb(k, 2) - comb(k - i + 1, 2) + (j - i - 1) + k 
                binary_vecs_4_auts[g, col_ind] = 1
            elif gate_type == 'CNOT':
                i,j = qubits 
                col_ind = comb(k, 2) - comb(k - i + 1, 2) + (j - i - 1) + k + kc2
                binary_vecs_4_auts[g, col_ind] = 1
            elif gate_type == 'CZ':
                i,j = qubits
                col_ind = comb(k, 2) - comb(k - i + 1, 2) + (j - i - 1) + k + 2*kc2
                binary_vecs_4_auts[g, col_ind] = 1
            elif gate_type == 'S':
                i = qubits - 1
                col_ind = i + k + 3*kc2
                binary_vecs_4_auts[g, col_ind] = 1
            elif gate_type == 'C(X,X)':
                i,j = qubits
                col_ind = comb(k, 2) - comb(k - i + 1, 2) + (j - i - 1) + 2*k + 3*kc2
                binary_vecs_4_auts[g, col_ind] = 1
            elif gate_type == 'Xsqrt':
                i = qubits - 1
                col_ind = i + 2*k + 4*kc2
                binary_vecs_4_auts[g, col_ind] = 1
            else: 
                pass
           
    return binary_vecs_4_auts


def cliff_binary_to_circ(M,k):
    kc2 = comb(k,2)
    no_of_cols = 3*k + 4*kc2
    circs = []
    pairs = list(combinations(range(1, k + 1), 2))
    for g in M: 
        circ = []
        for i in range(no_of_cols): 
            x = g[i]
            if i<k: 
                if x == 1: 
                    circ.append(('H',i+1))
            elif i<k+kc2: 
                if x == 1: 
                    ind = i - k 
                    m,n = pairs[ind]
                    circ.append(('SWAP',(m,n)))
            elif i<k+2*kc2: 
                if x == 1: 
                    ind = i - (k+kc2) 
                    m,n = pairs[ind]
                    circ.append(('CNOT',(m,n)))
            elif i<k+3*kc2:
                if x == 1: 
                    ind = i - (k+2*kc2) 
                    m,n = pairs[ind]
                    circ.append(('CZ',(m,n)))
            elif i<2*k+3*kc2: 
                if x == 1: 
                    ind = i - (k+3*kc2) 
                    circ.append(('S',ind+1))
            elif i<2*k+4*kc2:
                if x == 1: 
                    ind = i - (2*k+3*kc2) 
                    m,n = pairs[ind]
                    circ.append(('C(X,X)',(m,n)))
            elif i>2*k+4*kc2: 
                if x == 1: 
                    print(i,2*k+4*kc2)
                    ind = i - (2*k+4*kc2) 
                    circ.append(('Xsqrt',ind+1))
        circs.append(circ)
    return circs

