import numpy as np

def symp_prod(A,B,return_omega=False):
    """
    Computes the binary symplectic product between A and B.
    """
    A = np.array(A,dtype=int)
    B = np.array(B,dtype=int)
   
    if B.ndim == 1:
        B = B.reshape(1, -1)
        
    m, n = A.shape
    m_b, n_b = B.shape

    assert n == n_b 

    omega = np.eye(n,dtype=int)
    nhalf = n//2
    omega[:,:nhalf], omega[:,nhalf:] = omega[:,nhalf:].copy(), omega[:,:nhalf].copy()

    if return_omega:
        return np.array((A @ omega @ B.T)%2,dtype=int), omega
    else: 
        return np.array((A @ omega @ B.T)%2,dtype=int)

def is_symplectic(M):
    """
    Asserts that matrix M is symplectic.
    """
    n = M.shape[0]
    prod, omega = symp_prod(M,M,return_omega=True)
    return np.allclose(prod,omega)

def H_gate(i,n):
    i = i - 1
    eye = np.eye(2*n,dtype=int)
    eye[:,[i,i+n]] = eye[:,[i+n,i]]
    return eye

def CZ_gate(i,j,n):
    i = i - 1
    j = j - 1
    eye = np.eye(2*n,dtype=int)
    eye[:,i+n] = eye[:,j] + eye[:,i+n]
    eye[:,j+n] = eye[:,i] + eye[:,j+n]
    return eye

def S_gate(i,n):
    i = i - 1
    eye = np.eye(2*n,dtype=int)
    eye[:,i+n] = eye[:,i] + eye[:,i+n]
    return eye

def Xsqrt_gate(i,n):
    i = i - 1
    eye = np.eye(2*n,dtype=int)
    eye[:,i] = eye[:,i] + eye[:,i+n]
    return eye

def CX_XX_gate(i,j,n):
    i = i - 1
    j = j - 1
    eye = np.eye(2*n,dtype=int)
    eye[:,i] = eye[:,i] + eye[:,j+n]
    eye[:,j] = eye[:,j] + eye[:,i+n]
    return eye

def SWAP_gate(i,j,n):
    i = i - 1
    j = j - 1
    eye = np.eye(2*n,dtype=int)
    eye[:,[i,j]] = eye[:,[j,i]]
    eye[:,[i+n,j+n]] = eye[:,[j+n,i+n]]
    return eye

def CNOT_gate(i,j,n):
    i = i - 1
    j = j - 1
    eye = np.eye(2*n,dtype=int)
    eye[:,i] = eye[:,i] + eye[:,j]
    eye[:,j+n] = eye[:,i+n] + eye[:,j+n]
    return eye

def gamma_XZY_gate(i,n):
    return H_gate(i,n)@S_gate(i,n)

def gamma_XYZ_gate(i,n):
    return S_gate(i,n)@H_gate(i,n)

def symp_mat_prods(ops,n):
    """
    Returns symplectic product of quantum gates.
    """
    mat = np.eye(2*n,dtype=int)
    for op in ops:
        gate_type, qubits = op
        # Add the corresponding gate to the circuit
        if gate_type == 'CNOT':
            mat = mat @ CNOT_gate(qubits[0], qubits[1], n) 
        elif gate_type == 'CZ':
            mat = mat @ CZ_gate(qubits[0], qubits[1], n) 
        elif gate_type == 'SWAP':
            mat = mat @ SWAP_gate(qubits[0], qubits[1], n) 
        elif gate_type == 'H':
            mat = mat @ H_gate(qubits, n) 
        elif gate_type == 'S':
            mat = mat @ S_gate(qubits, n) 
        elif gate_type == 'Xsqrt':
            mat = mat @ Xsqrt_gate(qubits, n) 
        elif gate_type == 'C(X,X)':
            mat = mat @ CX_XX_gate(qubits[0], qubits[1], n) 
        elif gate_type == 'GammaXYZ':
            mat = mat @ gamma_XYZ_gate(qubits, n)
        elif gate_type == 'GammaXZY':
            mat = mat @ gamma_XZY_gate(qubits, n) 
        else:
            raise TypeError(f"Unsupported gate type: {gate_type}")
    return mat%2

# def op_2bit_to_op_3bit(op_2bit):
#     m = op_2bit.shape[0]
#     n = op_2bit.shape[1]//2
#     zeros = np.zeros((m,n),dtype=int)
#     op_3bit = np.hstack((zeros,op_2bit))
#     for row in range(m):
#         for i in range(n):
#             op_3bit[row,i] = (op_3bit[row,i+n]+op_3bit[row,i+2*n])%2
    
#     return np.array(op_3bit,dtype=int)

def op_2bit_to_op_3bit_and_phase(op_2bit):
    if op_2bit.ndim == 1:
        op_2bit = op_2bit.reshape(1, -1)
    m = op_2bit.shape[0]
    n = op_2bit.shape[1]//2
    zeros = np.zeros((m,n),dtype=int)
    op_3bit = np.hstack((zeros,op_2bit))
    phases = np.zeros(m,dtype=int)
    for row in range(m):
        p = 0
        for i in range(n):
            if op_3bit[row,i+n] == 1 and op_3bit[row,i+2*n] == 1:
                p += 1
            op_3bit[row,i] = (op_3bit[row,i+n]+op_3bit[row,i+2*n])%2
        phases[row] = p
    
    return phases, op_3bit

def clifford_circ_stab_update(op_3bit_p,physical_circ):
    """ 
        Args: 
            op_3bit_p (tuple): phases + op_3bit 
    """
    phases = op_3bit_p[0].copy()
    op_3bit = op_3bit_p[1].copy()
    if op_3bit.ndim == 1:
        op_3bit = op_3bit.reshape(1, -1)
    n = op_3bit.shape[1] // 3
    for gate in physical_circ:
        gate_type, qubits = gate
        if type(qubits) == int:
            q = qubits - 1
            a = q # x+z
            b = q + n # x 
            c = q + 2*n # z
        elif type(qubits) == tuple: 
            q1, q2 = qubits
            q1 = q1-1
            q2 = q2-1
        if gate_type == 'X':
            phases = (phases + 2*op_3bit[:,c])%4
        elif gate_type == 'Z':
            phases = (phases + 2*op_3bit[:,b])%4
        elif gate_type == 'S':
            phases = (phases + op_3bit[:,b])%4
            op_3bit[:,[a,c]] = op_3bit[:,[c,a]]
        elif gate_type == 'H':
            phases = (phases - op_3bit[:,a] + op_3bit[:,b] + op_3bit[:,c])%4
            op_3bit[:,[b,c]] = op_3bit[:,[c,b]]
        elif gate_type == 'GammaXYZ':
            phases = (phases + op_3bit[:,a] - op_3bit[:,c])%4
            col_a = op_3bit[:,a].copy() # x+z
            col_b = op_3bit[:,b].copy() # x
            col_c = op_3bit[:,c].copy() # z
            op_3bit[:,a] = col_c
            op_3bit[:,b] = col_a
            op_3bit[:,c] = col_b
        elif gate_type == 'GammaXZY':
            phases = (phases + op_3bit[:,a] - op_3bit[:,b])%4
            col_a = op_3bit[:,a].copy() # x+z
            col_b = op_3bit[:,b].copy() # x
            col_c = op_3bit[:,c].copy() # z
            op_3bit[:,a] = col_b
            op_3bit[:,b] = col_c
            op_3bit[:,c] = col_a
        elif gate_type == 'Xsqrt':
            phases = (phases - op_3bit[:,c])%4
            op_3bit[:,[a,b]] = op_3bit[:,[b,a]]
        elif gate_type == 'SWAP':
            op_3bit[:,[q1,q2]] = op_3bit[:,[q2,q1]]
            op_3bit[:,[q1+n,q2+n]] = op_3bit[:,[q2+n,q1+n]]
            op_3bit[:,[q1+2*n,q2+2*n]] = op_3bit[:,[q2+2*n,q1+2*n]]
        elif gate_type == 'CZ':
            x1 = op_3bit[:,q1+n]
            x2 = op_3bit[:,q2+n]
            z1 = op_3bit[:,q1+2*n]
            z2 = op_3bit[:,q2+2*n]
            phases = (phases + 2*x1*x2)%4
            op_3bit[:,q1] = (op_3bit[:,q1] + op_3bit[:,q2+n])%2
            op_3bit[:,q2] = (op_3bit[:,q2] + op_3bit[:,q1+n])%2
            op_3bit[:,q1+2*n] = (op_3bit[:,q1+2*n] + op_3bit[:,q2+n])%2
            op_3bit[:,q2+2*n] = (op_3bit[:,q2+2*n] + op_3bit[:,q1+n])%2
        elif gate_type == 'CNOT':
            x1 = op_3bit[:,q1+n]
            x2 = op_3bit[:,q2+n]
            z1 = op_3bit[:,q1+2*n]
            z2 = op_3bit[:,q2+2*n]
            op_3bit[:,q1] =  (op_3bit[:,q1] + op_3bit[:,q2+2*n])%2
            op_3bit[:,q2] = (op_3bit[:,q2] + op_3bit[:,q1+n])%2
            op_3bit[:,q2+n] = (op_3bit[:,q2+n] + op_3bit[:,q1+n])%2
            op_3bit[:,q1+2*n] = (op_3bit[:,q1+2*n] + op_3bit[:,q2+2*n])%2
        elif gate_type == 'C(X,X)':
            x1 = op_3bit[:,q1+n]
            x2 = op_3bit[:,q2+n]
            z1 = op_3bit[:,q1+2*n]
            z2 = op_3bit[:,q2+2*n]
            phases = (phases + 2*z1*z2)%4
            op_3bit[:,q1] = (op_3bit[:,q1] + op_3bit[:,q2+2*n])%2
            op_3bit[:,q2] = (op_3bit[:,q2] + op_3bit[:,q1+2*n])%2
            op_3bit[:,q1+n] = (op_3bit[:,q1+n] + op_3bit[:,q2+2*n])%2
            op_3bit[:,q2+n] = (op_3bit[:,q2+n] + op_3bit[:,q1+2*n])%2
        else: 
            raise TypeError(f'Gate type unknown: {gate_type}')
    return phases, op_3bit