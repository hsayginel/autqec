from utils.linalg import *
from utils.perms import *
from utils.pauli import *
from utils.qec import compute_standard_form
from utils.symplectic import *
from itertools import combinations

class circ_from_aut:
    def __init__(self,H,aut):
        """
        Class for finding the physical qubit circuits of the 
        generators of the automorphism groups of stabilizer QECCs. 

        3-bit representation
        --------------------
        Paulis (Z | X | X+Z): 
        I --> 000   
        X --> 011   
        Z --> 101   
        Y --> 110   
        
        1-qubit Clifford operators: 
        | Operators      |   Permutation   |
        |----------------|-----------------|
        | $H$            | (1,2)           |
        | $S$            | (1,3)           |
        | $\sqrt{X}$     | (2,3)           |
        | $\Gamma_{XZY}$ | (1,2,3)         |
        | $\Gamma_{XYZ}$ | (1,3,2)         |

        Args:
            H (np.array): stabilizer generators of the QEC.
            aut (list): list of cycles representing the automorphism.
        """
        if is_matrix_full_rank(H) == False:
            raise AssertionError("Rows of H should be independent. Use a generating set of stabilizers.")
        if not isinstance(aut, list):
            raise TypeError("Aut must be a list of tuples.")
        n = H.shape[1]//2
        self.n = n
        self.H = H
        self.aut = aut
        self.bits = np.arange(1,3*n+1,1)
        self.qubit_indices = np.arange(1,n+1,1)
        self.bits_image = permute_by_cycles(self.bits,self.aut)

        # 3-bit rep embedding
        E = np.array([[1,1,1],[1,0,1],[1,1,0]],dtype=int)
        EInv = np.mod(np.linalg.inv(E),2)
        EInv = np.array(EInv,dtype=int)
        id_mat = np.eye(n,dtype=int)
        self.E_mat = np.kron(E,id_mat)   
        self.EInv_mat = np.kron(EInv,id_mat)

    def swaps(self): 
        """
        Returns SWAP gates of the automorphism and orders qubits.
        """
        bits_image = self.bits_image
        permuted_qubit_indices = reduce_triplets_to_qubits(bits_image)

        SWAPs_reversed = sort_with_swaps(permuted_qubit_indices)
        physical_SWAP_gates = SWAPs_reversed[::-1]
        
        # Reverse SWAPS.
        qubit_triplets = [tuple(bits_image[i:i+3]) for i in range(0, len(bits_image), 3)]
        ordered_qubit_triplets = apply_swaps(qubit_triplets, SWAPs_reversed)
        self.ordered_qubit_triplets = ordered_qubit_triplets

        return physical_SWAP_gates, ordered_qubit_triplets
    
    def single_qubit_cliffords(self,ordered_qubit_triplets):
        """
        Returns the 1-qubit Clifford gates of the automorphism.
        """
        single_qubit_gates = []
        for i,triplet in enumerate(ordered_qubit_triplets):
            gate_ind = i+1
            reduced_triplet = (triplet[0]%3,triplet[1]%3,triplet[2]%3)
            if reduced_triplet == (1,2,0):
                pass
            elif reduced_triplet == (1,0,2):
                single_qubit_gates.append(("Xsqrt",gate_ind)) 
            elif reduced_triplet == (2,1,0):
                single_qubit_gates.append(("H",gate_ind))
            elif reduced_triplet == (0,2,1):
                single_qubit_gates.append(("S",gate_ind))
            elif reduced_triplet == (0,1,2):
                single_qubit_gates.append(("GammaXZY",gate_ind)) # REVERSE
            elif reduced_triplet == (2,0,1):
                single_qubit_gates.append(("GammaXYZ",gate_ind)) # REVERSE
            else:
                raise AssertionError(f"Unknown triplet: {reduced_triplet}")

        return single_qubit_gates
    
    def auts_to_perm_mat(self):
        """
        Converts a permutation in cyclic notation to a permutation matrix.

        Args:
        perms (list of tuples): List of tuples representing cycles in the permutation.

        Returns:
        perm_matrix (np.array): Permutation matrix.
        """     
        # correct qubit order for 3-bit representation (X+Z | Z | X)
        n = self.n   
        XZ_bits = [i for i in range(3, 3*n + 1, 3)]
        Z_bits = [i for i in range(1, 3*n + 1, 3)]
        X_bits = [i for i in range(2, 3*n + 1, 3)]
        q3bit_order = XZ_bits + Z_bits + X_bits
        new_aut = []
        for cycle in self.aut:
            new_aut.append(tuple(q3bit_order.index(x)+1 for x in cycle))
        
        # Initialize the identity matrix of size n
        perm_matrix = np.eye(3*self.n,dtype=int)
        for cycle in new_aut:
            # Rotate the elements in the cycle
            for i in range(len(cycle)):
                from_idx = cycle[i] - 1  # convert to 0-based index
                to_idx = cycle[(i + 1) % len(cycle)] - 1  # next element in the cycle
                perm_matrix[from_idx, from_idx] = 0
                perm_matrix[from_idx, to_idx] = 1
        
        return perm_matrix
    
    def perm_mat_to_symp_mat(self):
        """Take a list of permutations and convert to symplectic matrix by conjugating by E. """
        return np.mod(self.E_mat @ self.auts_to_perm_mat() @ self.EInv_mat,2) 
    
    def symp_transform(self):
        """Check whether matrix A is a direct sum of a matrix M and a symplectic matrix S"""
        A = self.perm_mat_to_symp_mat()
        n = len(A) // 3
        symp_mat = A[n:,n:]
        if np.sum(A[n:,:n])!=0:
            raise AssertionError('Off-diagonal check: bottom left.')
        elif np.sum(A[:n,n:])!=0:
            raise AssertionError('Off-diagonal check: top right.')
        elif is_symplectic(symp_mat) == False:
            raise AssertionError('Automorphism generator is not a valid symplectic transformation.')
        return symp_mat

    def circ(self):
        """
        Returns the circuit of the automorphism as 
        1-qubit Cliffords + SWAPs.
        """        

        physical_SWAP_gates, ordered_qubit_triplets = self.swaps()
        single_qubit_gates = self.single_qubit_cliffords(ordered_qubit_triplets)
        self.single_qubit_gates = single_qubit_gates
        physical_circuit = single_qubit_gates + physical_SWAP_gates

        return physical_circuit, self.symp_transform()


class logical_circ_and_pauli_correct:
    def __init__(self, H, phys_circ):
        """
        Class for finding appropriate Pauli corrections of 
        Clifford circuits to preserve stabilizers and infer
        the corresponding logical action of the circuit. 

        Args: 
            H (np.array): stabilizer generators of the QEC.
            phys_circ (list): quantum circuit.
        """

        # standard form 
        G, LX, LZ, D = compute_standard_form(H)
        self.G = G
        self.LX = LX
        self.D = D
        self.LZ = LZ

        # code parameters
        n = G.shape[1] // 2
        m = G.shape[0]
        self.n = n 
        self.m = m
        self.k = n-m

        # tableux
        tableux = np.vstack([G,LX,D,LZ]) 
        T_symp_prod, omega = symp_prod(tableux,tableux,return_omega=True)
        self.omega = omega
        if np.allclose(T_symp_prod,omega) == False:
            raise AssertionError("Check stabilizer/destabilizer tableux.")
        self.tableux = tableux
        self.tableux_phases ,self.tableux_pauli = binary_vecs_to_paulis(tableux,phase_bit=True)
        self.show_tableux = '\n'.join([' '.join(x) for x in binary_vecs_to_paulis(tableux)])

        # physical circuit
        self.phys_circ = phys_circ
        
    def new_tableux(self):
        T_G_L = np.vstack([self.G,self.LX,self.LZ])
        T_G_L_4bit = op_2bit_to_op_3bit_and_phase(T_G_L)
        return clifford_circ_stab_update(T_G_L_4bit,self.phys_circ)
    
    def new_tableux_anticomm(self):
        m = self.m
        n = self.n
        k = self.k
        omega = self.omega
        b = np.mod(self.new_tableux()[1][:,n:] @ omega @ self.tableux.T @ omega,2)
        G_comp = b[:m].copy()
        L_comp = b[m:].copy()

        if np.allclose(b[:,n:-k],np.zeros((m+2*k,m),dtype=int)) == False:
            raise AssertionError('Physical circuit maps operators outside of code space.')

        if np.allclose(G_comp[:,m:],np.zeros((m,2*n-m),dtype=int)) == False:
            raise AssertionError('Physical circuit does not preserve the stabilizer group.')
        
        return b, G_comp, L_comp
    
    def U_logical_act(self):
        m = self.m
        k = self.k
        _, G_comp, L_comp = self.new_tableux_anticomm()
        u_act = np.zeros((2*k,2*k),dtype=int)
        u_act[:,:k] = L_comp[:,m:m+k]
        u_act[:,k:] = L_comp[:,-k:]

        return u_act
    
    def i_phases(self,U_ACT):
        """
            Returns a phase vector that operators pick up via U_ACT.
            Phases: {0,1,2,3} == {+1, +i, -1, -i}
        """
        k = len(U_ACT)//2
        p = np.zeros(2*k,dtype=int)
        for row in range(2*k):
            for i in range(k):
                if U_ACT[row,i] == 1 and U_ACT[row,i+k] == 1:
                    p[row] = (p[row] + 1)%4
        return p
    
    def new_tableux_pauli_prod_phases(self):
        n = self.n
        m = self.m
        k = self.k
        tableux = self.tableux
        tableux_pauli = self.tableux_pauli
        tableux_phases = self.tableux_phases

        b,_,_ = self.new_tableux_anticomm()
        tableux_new_phases = np.zeros(len(b),dtype=int)
        for t_ind, t in enumerate(b):
            t_indices = np.where(t==1)[0]
            pauli_multiply = ['I']*self.n
            pauli_phase = 0
            for i in t_indices:
                pauli_multiply, p = multiply_pauli_strings(pauli_multiply,pauli_phase,tableux_pauli[i],tableux_phases[i])
                tableux_new_phases[t_ind] = np.mod(tableux_new_phases[t_ind] + p,4)
        self.i_phases_from_Y_ops = self.i_phases(self.U_logical_act())
        tableux_new_phases[m:] = (tableux_new_phases[m:].copy()+self.i_phases_from_Y_ops)%4
        return tableux_new_phases
                 
    def run(self):
        """ Finds the required Pauli corrections X or Z to fix any -1 phases of stabilizers. """
        U_p = np.zeros(2*self.n,dtype=int)
        pauli_circ = []
        n = self.n
        m = self.m
        k = self.k
        T = self.tableux

        # find phase differences. 
        p = self.new_tableux()[0]
        q = self.new_tableux_pauli_prod_phases()
        phase_diff = np.mod(p - q, 4)
        if np.all(phase_diff % 2 == 0) == False: # check there are no i or -i phases. 
            raise AssertionError("Pauli correction to the physical circuit failed.")

        # correct stabilizer phases. 
        for i in range(self.m):
            if phase_diff[i] != 0:
                h = (i+n)%(2*n)
                U_p = np.mod(U_p + T[h],2)

        # multiply X/Y/Z to logical circuit to match the phase to logical act. 
        ## X-logicals of the code
        for i in range(m,m+k):
            if phase_diff[i] != 0:
                h = (i+n)%(2*n)
                U_p = np.mod(U_p + T[h],2)
        ## Z-logicals of the code
        for i in range(m+k,m+2*k):
            if phase_diff[i] != 0:
                h = (i+m+n)%(2*n)
                U_p = np.mod(U_p + T[h],2)

        # convert U_p into quantum circuit
        for i in range(n):
            if U_p[i] == 1 and U_p[i+n] == 0:
                pauli_circ.append(('X',i+1))
            elif U_p[i] == 0 and U_p[i+n] == 1:
                pauli_circ.append(('Z',i+1))
            elif U_p[i] == 1 and U_p[i+n] == 1:
                pauli_circ.append(('Y',i+1))

        U_ACT = self.U_logical_act()
        circ_logical_act = circ_from_symp_mat(U_ACT).run()
        if np.allclose(symp_mat_prods(circ_logical_act,self.k),U_ACT) == False:
            raise AssertionError("Logical circuit is wrong.")
        return circ_logical_act, pauli_circ + self.phys_circ
    

class circ_from_symp_mat:
    def __init__(self,symplectic_mat):
        """
        Class to find the quantum circuit implemented by a symplectic transformation 
        up to a logical Pauli correction. 

        Gates: {H, CZ, S, Xsqrt, CX(X,X), CX}
        """
        assert is_symplectic(symplectic_mat)

        k = symplectic_mat.shape[0] // 2

        # invert symplectic matrix
        omega = np.eye(2*k,dtype=int)
        omega[:,:k], omega[:,k:] = omega[:,k:].copy(), omega[:,:k].copy()
        self.k = k
        self.symplectic_mat = omega@symplectic_mat.T@omega

    def find_H_gates(self):
        """ Return H gates. """
        H_circ = []
        
        symplectic_mat_og = self.symplectic_mat.copy()
        symplectic_mat = self.symplectic_mat.copy()
        k = self.k

        X_part = symplectic_mat_og[:,:k]
        Z_part = symplectic_mat_og[:,k:]
        XX_part = symplectic_mat_og[:k,:k]
        ZZ_part = symplectic_mat_og[k:,k:]
        XZ_part = symplectic_mat_og[:k,k:]
        ZX_part = symplectic_mat_og[k:,:k]
        
        if rank_mod2(XX_part) == k and rank_mod2(ZZ_part) == k:
            return None
        elif rank_mod2(XZ_part) == k and rank_mod2(ZX_part) == k:
            symplectic_mat[:,k:] = X_part
            symplectic_mat[:,:k] = Z_part
            for i in range(k):
                H_circ.append(("H",i+1))
            self.symplectic_mat = symplectic_mat.copy()
            return H_circ
        elif rank_mod2(XX_part) != k or rank_mod2(ZZ_part) != k:
            qubit_indices = np.arange(k,dtype=int)
            for r in range(1, k + 1):  # r is the length of combinations
                for combo in combinations(qubit_indices, r):
                    mat_copy = symplectic_mat_og.copy()
                    for i in combo:
                        mat_copy[:,[i,i+k]] = mat_copy[:,[i+k,i]]
                    XX = mat_copy[:k,:k]
                    ZZ = mat_copy[k:,k:]
                    if rank_mod2(XX) == k and rank_mod2(ZZ) == k:
                        self.symplectic_mat = mat_copy.copy()
                        for i in combo:
                            H_circ.append(("H",int(i+1)))
                        return H_circ
            raise AssertionError('Individual H gates failed.')
        return H_circ
    
    def find_phase_type_gates(self,gate_type):
        """ Returns CZ, S, Xsqrt, C(X,X) gates. """
        symplectic_mat_og = self.symplectic_mat.copy()
        k = self.k
        XX_part = symplectic_mat_og[:k,:k]
        ZZ_part = symplectic_mat_og[k:,k:]
        XZ_part = symplectic_mat_og[:k,k:]
        ZX_part = symplectic_mat_og[k:,:k]

        if gate_type == 'Z':
            gate_1 = 'S'
            gate_2 = 'CZ'
            block_A = XX_part
            block_B = XZ_part
        elif gate_type == 'X':
            gate_1 = 'Xsqrt'
            gate_2 = 'C(X,X)'
            block_A = ZZ_part
            block_B = ZX_part

        A_rref, pivots, A_transform_rows, A_transform_cols = rref_mod2(block_A)
        assert np.allclose(A_rref,np.eye(k))
        assert len(pivots) == k

        B_new = (A_transform_rows@block_B@A_transform_cols)%2
        is_symmetric = np.array_equal(B_new, B_new.T)
        assert is_symmetric

        circ = set()
        for i in range(k):
            if B_new[i, i] == 1:
                circ.add((gate_1,i+1))
            for j in range(i+1,k):
                if i != j and B_new[i, j] == 1:
                    circ.add((gate_2,(i+1,j+1)))

        circ = sorted(circ)
        return circ
    
    def find_CNOT_circuits(self):
        symplectic_mat = self.symplectic_mat.copy()
        XX_part_GL_matrix = symplectic_mat[:self.k,:self.k] 
        CNOT_circ, reduced_mat = rref_mod2(XX_part_GL_matrix,CNOTs=True)
        assert is_identity_matrix(reduced_mat)

        return CNOT_circ[::-1]
        
    def run(self):
        """ Full algorithm for finding the quantum circuit of the 
        logical action represented by the symplectic matrix.
        """
        k = self.k

        # STEP 1: Logical H until XX and ZZ are full rank.
        H_circ = self.find_H_gates()
        XX_part = self.symplectic_mat[:k,:k]
        ZZ_part = self.symplectic_mat[k:,k:]
        assert is_symplectic(self.symplectic_mat)
        if rank_mod2(XX_part) != k or rank_mod2(ZZ_part) != k or is_symplectic(self.symplectic_mat) == False:
            raise AssertionError("Problem with Hadamards.")

        # STEP 2: Logical S,CZ,Xsqrt,CX_(X,X) until XZ and ZX are zero rank.
        ##    2a: XZ part
        XZ_circ = None
        XZ_part = self.symplectic_mat[:k,k:]
        if rank_mod2(XZ_part) != 0:
            XZ_circ = self.find_phase_type_gates(gate_type='Z')
            for item in XZ_circ:
                if 'CZ' in item:  
                    i, j = item[1] 
                    self.symplectic_mat = (self.symplectic_mat @ CZ_gate(i,j,k))%2
                elif 'S' in item:  
                    i = item[1]
                    self.symplectic_mat = (self.symplectic_mat @ S_gate(i,k))%2
        XZ_part = self.symplectic_mat[:k,k:]
        if rank_mod2(XZ_part) != 0:
            raise AssertionError("S and CZ gates did not bring the rank to 0.")
        if is_symplectic(self.symplectic_mat) == False:
            raise AssertionError("Problem with S and CZ gates.")
        ##    2b: ZX part
        ZX_circ = None
        ZX_part = self.symplectic_mat[k:,:k]
        if rank_mod2(ZX_part) != 0:
            ZX_circ = self.find_phase_type_gates(gate_type='X')
            for item in ZX_circ:
                if 'C(X,X)' in item:  
                    i, j = item[1]  
                    self.symplectic_mat = (self.symplectic_mat @ CX_XX_gate(i,j,k))%2
                elif 'Xsqrt' in item:  
                    i = item[1]
                    self.symplectic_mat = (self.symplectic_mat @ Xsqrt_gate(i,k))%2
        ZX_part = self.symplectic_mat[k:,:k]
        if rank_mod2(ZX_part) != 0:
            raise AssertionError("Xsqrt and CX(X,X) gates did not bring the rank to 0.")
        if is_symplectic(self.symplectic_mat) == False:
            raise AssertionError("Problem with Xsqrt and CX(X,X) gates.")


        XX_part = self.symplectic_mat[:k,:k]
        ZZ_part = self.symplectic_mat[k:,k:]
        XZ_part = self.symplectic_mat[:k,k:]
        ZX_part = self.symplectic_mat[k:,:k]

        assert rank_mod2(XX_part) == k
        assert rank_mod2(ZZ_part) == k
        assert rank_mod2(XZ_part) == 0
        assert rank_mod2(ZX_part) == 0
        assert is_symplectic(self.symplectic_mat)

        # STEP 3: CNOT circuits via Gaussian Elimination.
        CNOT_circ = self.find_CNOT_circuits()

        logical_circ = []
        logical_circs = [H_circ,XZ_circ,ZX_circ,CNOT_circ]

        for circ in logical_circs:
            if circ:
                logical_circ.extend(circ)
        return logical_circ