import numpy as np
def binary_vecs_to_paulis(mat,phase_bit = False):
    """Returns the rows of a binary-symplectic matrix as Pauli operators."""
    
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)

    n = mat.shape[1] // 2
    m = mat.shape[0]

    pauli_mat = []
    phase = np.zeros(m,dtype=int)
    for i in range(m):
        row_result = []
        for j in range(n):
            pair = (mat[i, j], mat[i, j+n])
            if pair == (0, 0):
                row_result.append('I')
            elif pair == (1, 0):
                row_result.append('X')
            elif pair == (0, 1):
                row_result.append('Z')
            elif pair == (1, 1):
                if phase_bit == True:
                    row_result.append('XZ')
                    phase[i] += 1
                else: 
                    row_result.append('Y')
        pauli_mat.append(row_result)
        
    if phase_bit == True: 
        return phase, pauli_mat
    else:
        return pauli_mat


def multiply_pauli_strings(pauli1, phase1, pauli2, phase2):
    """ Returns the product of two pauli strings accounting for anticommutation of X and Z.
        e.g. Y = (['X','Z'],1) 

        Phases: {0,1,2,3} == {+1, +i, -1, -i}
    """
    if len(pauli1) != len(pauli2):
        raise ValueError("Pauli strings must be of the same length")
    
    result = []
    phase = phase1%4 + phase2%4 # initial phase difference  
    
    for p1, p2 in zip(pauli1, pauli2):
        if p1 == 'I':
            result.append(p2)
        elif p2 == 'I':
            result.append(p1)
        elif p1 == 'X':
            if p2 == 'X': 
                result.append('I')
            elif p2 == 'Z':
                result.append('XZ')
            elif p2 == 'XZ':
                result.append('Z')
        elif p1 == 'Z':
            if p2 == 'X': 
                result.append('XZ')
                phase += 2
            elif p2 == 'Z':
                result.append('I')
            elif p2 == 'XZ':
                result.append('X')
                phase += 2
        elif p1 == 'XZ':
            if p2 == 'X': 
                result.append('Z')
                phase += 2
            elif p2 == 'Z':
                result.append('X')
            elif p2 == 'XZ':
                result.append('I')
                phase += 2
    
    return result, phase%4
