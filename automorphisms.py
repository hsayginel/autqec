from utils.linalg import *
from utils.perms import *
from utils.pauli import *
from utils.qec import compute_standard_form
from utils.symplectic import *
from itertools import combinations

class physical_circ_of_aut:
    def __init__(self,H_symp,aut):
        """
        Class for finding the physical qubit circuits of the 
        generators of the automorphism groups of stabilizer QECCs
        including appropriate Pauli corrections. 

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
            H_symp (np.array): stabilizer generators of the QEC.
            aut (list): list of cycles representing the automorphism.
        """
        if not isinstance(aut, list):
            raise TypeError("Aut must be a list of tuples.")
        if is_matrix_full_rank(H_symp) == False:
            raise AssertionError("Rows of H_symp should be independent. Use a generating set of stabilizers.")
        
        n = H_symp.shape[1]//2
        self.n = n
        self.H_symp = H_symp
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
        np.ndarray: Permutation matrix.
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
        self.pauli_correct_check = False

        physical_SWAP_gates, ordered_qubit_triplets = self.swaps()
        single_qubit_gates = self.single_qubit_cliffords(ordered_qubit_triplets)
        self.single_qubit_gates = single_qubit_gates
        physical_circuit = single_qubit_gates + physical_SWAP_gates

        if single_qubit_gates:
            self.pauli_correct_check = True

        return physical_circuit, self.symp_transform()
    
    def circ_w_pauli_correction(self):
        """
        Prepends Pauli corrections to the physical circuit.
        """
        physical_circuit, symp_mat = self.circ()
        if self.pauli_correct_check:
            G, LX, LZ, D = compute_standard_form(self.H_symp)
            pauli_circ = pauli_correction(G, LX, D, LZ).run(symp_mat,physical_circuit)
            physical_circuit = pauli_circ + physical_circuit 
        return physical_circuit

class pauli_correction:
    def __init__(self, G, LX, D, LZ):
        """
        Class for finding appropriate Pauli corrections of 
        Clifford gates of stabilizer QECCs. 

        Args: 
            G (np.array): stabilizers.
            LX (np.array): logical X operators.
            D (np.array): destabilizers. 
            LZ (np.array): logical Z operators. 
        """
        n = G.shape[1] // 2
        m = G.shape[0]
        self.n = n 
        self.m = m
        T = np.vstack([G,LX,D,LZ]) # tableux
        
        # tableux check
        T_symp_prod, omega = symp_prod(T,T,return_omega=True)
        self.omega = omega
        if np.allclose(T_symp_prod,omega) == False:
            raise AssertionError("Check stabilizer/destabilizer tableux.")

        self.G = G
        self.LX = LX
        self.D = D
        self.LZ = LZ

        _,self.stabs_og_pauli = binary_vecs_to_paulis(G,phase_bit=True)
        self.H_4bit = op_2bit_to_op_3bit_and_phase(G)
        self.stabs_og_phases, self.stabs_og_3bit = self.H_4bit

    def im_stabs(self,symp_mat):
        """ Mapping of stabilizers via the symplectic transformation. """
        stabs = self.G
        stabs_new = np.mod(stabs@symp_mat,2)
        return stabs_new

    def im_stabs_composition(self,symp_mat):
        """ Original stabilizer composition of new stabilizers. """
        stabs_new = self.im_stabs(symp_mat)
        return symp_prod(stabs_new,self.D)

    def im_stabs_check_phases(self,symp_mat):
        """ Phases of new stabilizers using mod 4 arithmetic. 
                    {0,1,2,3} == {+1, +i, -1, -i}
        (Note that Y operator is represented as XZ with phase +1.)"""
        # old stabs
        stabs_og = self.G
        stabs_og_phases = self.stabs_og_phases.copy()
        stabs_og_pauli = self.stabs_og_pauli
        # new stabs
        stabs_new = self.im_stabs(symp_mat)
        stabs_new_phases = np.zeros_like(stabs_og_phases)
        # new stabs composition
        stabs_new_composition = self.im_stabs_composition(symp_mat)
        # multiply stabs
        s_vec_multiply = np.zeros_like(stabs_og,dtype=int)
        for stab_ind, s in enumerate(stabs_new_composition):
            stabs_og_indices = np.where(s==1)[0]
            s_pauli_multiply = ['I']*self.n
            s_pauli_phase = 0
            for i in stabs_og_indices:
                s_vec_multiply[stab_ind] += stabs_og[i]
                s_pauli_multiply, p = multiply_pauli_strings(s_pauli_multiply,s_pauli_phase,stabs_og_pauli[i],stabs_og_phases[i])
                stabs_new_phases[stab_ind] = np.mod(stabs_new_phases[stab_ind] + p,4)
        assert np.allclose(s_vec_multiply%2,stabs_new)
        return stabs_new_phases
 
    def physical_circ_phases(self,physical_circ):
        """ Phases of stabilizers after the action of Clifford circuits.
                        {0,1,2,3} == {+1, +i, -1, -i}
        """
        H_4bit = self.H_4bit
        return clifford_circ_stab_update(H_4bit,physical_circ)[0]
                
    def run(self,symp_mat,physical_circ):
        """ Finds the required Pauli corrections X or Z to fix any -1 phases. """
        stabs_im_phases = self.im_stabs_check_phases(symp_mat)
        stabs_phys_circ_phases = (self.physical_circ_phases(physical_circ))
        phase_diff = np.mod(stabs_im_phases - stabs_phys_circ_phases, 4)
        
        if np.all(phase_diff % 2 == 0) == True:
            phase_diff = phase_diff/2
            destab_indices = np.where(phase_diff==1)[0]
            pauli_circs = []
            for i in destab_indices:
                pauli_circs += binary_vecs_to_paulis(self.D[i])

            def find_Z_X_positions(lists):
                gates = []
                for _, sublist in enumerate(lists):
                    for i, elem in enumerate(sublist):
                        if elem in ('Z', 'X'):
                            gates.append((elem, i + 1))  
                return gates
            
            return find_Z_X_positions(pauli_circs)

        else:
            print(phase_diff)
            raise AssertionError("Pauli correction failed: multiples of i phases present.")


class logical_circ_of_aut:
    def __init__(self,H_symp,aut):
        """
        Class for finding the logical circuits of the 
        generators of the automorphism groups of QECCs
        including appropriate Pauli corrections.

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
            H_symp (np.array): stabilizer generators of the QEC.
            aut (list): list of cycles representing the automorphism.
        """
        if not isinstance(aut, list):
            raise TypeError("Auts must be list of cycles.")
        if is_matrix_full_rank(H_symp) == False:
            raise AssertionError("Rows of H_symp should be independent. Use a generating set of stabilizers.")
        self.aut = aut
        self.H_symp = H_symp
        
        n = H_symp.shape[1]//2 
        m = H_symp.shape[0] 
        self.n = n # no of physical qubits
        self.m = m # no of independent generators
        self.k = n-m # no of logical qubits

        # standard form 
        G, LX, LZ, D = compute_standard_form(H_symp)
        if G.ndim == 1:
            G = G.reshape(1, -1)
        if LX.ndim == 1:
            LX = LX.reshape(1, -1)
        if LZ.ndim == 1:
            LZ = LZ.reshape(1, -1)
        if D.ndim == 1:
            D = D.reshape(1, -1)
        self.G = G
        self.LX = LX
        self.LZ = LZ
        self.D = D
       
        # automorphism gates
        phys_act = physical_circ_of_aut(H_symp,aut)
        phys_circ, phys_symp_transform = phys_act.circ()
        self.phys_circ_w_pauli_correction = phys_act.circ_w_pauli_correction()
        self.phys_circ = phys_circ
        self.phys_symp_transform = phys_symp_transform
        
        # mapping of logicals
        self.LX_new = LX @ phys_symp_transform
        self.LZ_new = LZ @ phys_symp_transform

    def print_phys_circ(self):
        print("Physical circuit")
        print(self.phys_circ)
 
    def print_physical_act(self):
        LX_p = binary_vecs_to_paulis(self.LX,phase_bit=False)
        LZ_p = binary_vecs_to_paulis(self.LZ,phase_bit=False)
        LX_new_p = binary_vecs_to_paulis(self.LX_new,phase_bit=False)
        LZ_new_p = binary_vecs_to_paulis(self.LZ_new,phase_bit=False)
        print('X logicals')
        print(LX_p,'-->',LX_new_p)
        print('Z logicals')
        print(LZ_p,'-->',LZ_new_p)

    def construct_symplectic_mat(self):
        """
        Returns a 2k x 2k symplectic matrix representing 
        the mapping of logical X and Z operators. 

        S = | XX | XZ |
            |---------|
            | ZX | ZZ |
        """
        LX = self.LX
        LZ = self.LZ
        logicals_og = np.vstack((LZ,LX))
        LX_new = self.LX_new
        LZ_new = self.LZ_new
        logicals_new = np.vstack((LX_new,LZ_new))
        
        symplectic_mat = symp_prod(logicals_new,logicals_og)

        if is_symplectic(symplectic_mat) == False:
            self.print_physical_act()
            print(symplectic_mat)
            raise AssertionError("Invalid representation of logical operators OR the automorphism does not represent a valid transformation.")
        return symplectic_mat
    
    def circ(self):
        """
        Reverse engineers the logical circuit of the automorphism by considering
        the symplectic matrix representation of the logical action. 

        Note: Column operations correspond to Clifford gates. 
        """
        symplectic_mat_og = self.construct_symplectic_mat()
        symplectic_mat = symplectic_mat_og.copy()
        logical_circ = symplectic_mat_to_logical_circ(symplectic_mat).run()

        if np.allclose(symp_mat_prods(logical_circ,self.k),symplectic_mat) == False:
            raise AssertionError("Logical circuit is wrong.")
        
        return logical_circ, symplectic_mat
    
    def circ_w_pauli_correction(self):
        # stabs
        stabs = self.G
        s_phases, s_paulis = binary_vecs_to_paulis(stabs,phase_bit=True)

        # logical circuit phases
        logical_circ, symplectic_mat = self.circ()
        id_mat = np.eye(2*self.k, dtype=int)
        id_mat = op_2bit_to_op_3bit_and_phase(id_mat)
        logical_circ_phases, _ = clifford_circ_stab_update(id_mat,logical_circ)

        # physical circuit phases
        LX_LZ = np.vstack((self.LX,self.LZ))
        LX_LZ_4bit = op_2bit_to_op_3bit_and_phase(LX_LZ)
        LXLZ_phases, LXLZ_3bit = clifford_circ_stab_update(LX_LZ_4bit,self.phys_circ_w_pauli_correction)
        LXLZ_new = LXLZ_3bit[:,self.n:]

        stab_indices_mat = np.mod(symp_prod(LXLZ_new,self.D),2)
        self.LX_LZ = LX_LZ
        self.LXLZ_new = LXLZ_new
        self.stab_indices_mat = stab_indices_mat 
        
        physical_circ_phases = np.zeros(2*self.k,dtype=int)
        for l_ind, l in enumerate(LXLZ_new):
            stab_indices = np.where(stab_indices_mat[l_ind]==1)[0]
            _, l_pauli = binary_vecs_to_paulis(l,phase_bit=True)
            pauli = l_pauli[0]
            phase = LXLZ_phases[l_ind].copy()
            for i in stab_indices:
                pauli, phase = multiply_pauli_strings(pauli,phase,s_paulis[i],s_phases[i])
            physical_circ_phases[l_ind] = phase

        self.physical_circ_phases = physical_circ_phases 
        self.logical_circ_phases = logical_circ_phases
        phase_diff = physical_circ_phases - logical_circ_phases

        pauli_gates = []
        if np.all(phase_diff % 2 == 0) == True:
            phase_diff = phase_diff/2
            X_part = phase_diff[:self.k]
            Z_part = phase_diff[self.k:]
            
            for q in range(self.k):
                if X_part[q] == 1 and Z_part[q] == 0:
                    pauli_gates.append(('Z',q+1))
                elif Z_part[q] == 1 and X_part[q] == 0:
                    pauli_gates.append(('X',q+1))
                elif X_part[q] == 1 and Z_part[q] == 1:
                    X_logical_new = binary_vecs_to_paulis(symplectic_mat[q])[0][0]
                    Z_logical_new = binary_vecs_to_paulis(symplectic_mat[q+self.k])[0][0]
                    if X_logical_new == 'Z' and Z_logical_new == 'Y':
                        pauli_gates.append(('X',q+1))
                    elif X_logical_new == 'Y' and Z_logical_new == 'X':
                        pauli_gates.append(('Z',q+1))
                    else:
                        pauli_gates.append(('X',q+1))
                        pauli_gates.append(('Z',q+1))

            return logical_circ + pauli_gates
        else: 
            self.print_pauli_corrections()
            raise AssertionError("Pauli correction failed: multiples of i phases present.")
    
    def print_pauli_corrections(self):
        print('Stabilizers in new logicals')
        print(self.stab_indices_mat)
        print('Logical circuit phase')
        print(self.logical_circ_phases)
        print('Physical circuit phase')
        print(self.physical_circ_phases)

class symplectic_mat_to_logical_circ:
    def __init__(self,symplectic_mat):
        """
        Class to find the quantum circuit implemented by a symplectic transformation 
        up to a logical Pauli correction. 

        Gates: {H, CZ, S, Xsqrt, CX(X,X), CX}
        """
        assert is_symplectic(symplectic_mat)

        k = symplectic_mat.shape[0] // 2

        self.k = k
        self.symplectic_mat = symplectic_mat.copy()

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
        
        if rank(XX_part) == k and rank(ZZ_part) == k:
            return None
        elif rank(XZ_part) == k and rank(ZX_part) == k:
            symplectic_mat[:,k:] = X_part
            symplectic_mat[:,:k] = Z_part
            for i in range(k):
                H_circ.append(("H",i+1))
            self.symplectic_mat = symplectic_mat.copy()
            return H_circ
        elif rank(XX_part) != k or rank(ZZ_part) != k:
            qubit_indices = np.arange(k,dtype=int)
            for r in range(1, k + 1):  # r is the length of combinations
                for combo in combinations(qubit_indices, r):
                    mat_copy = symplectic_mat_og.copy()
                    for i in combo:
                        mat_copy[:,[i,i+k]] = mat_copy[:,[i+k,i]]
                    XX = mat_copy[:k,:k]
                    ZZ = mat_copy[k:,k:]
                    if rank(XX) == k and rank(ZZ) == k:
                        self.symplectic_mat = mat_copy.copy()
                        for i in combo:
                            H_circ.append(("H",int(i+1)))
                        return H_circ
            raise AssertionError('Individual H gates failed.')
    
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

        A_rref, A_rank, A_transform_rows, A_transform_cols = reduced_row_echelon(block_A)
        assert np.allclose(A_rref,np.eye(k))
        assert A_rank == k

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
        CNOT_circ, reduced_mat = CNOT_circ_from_GL_mat(XX_part_GL_matrix)
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
        if rank(XX_part) != k or rank(ZZ_part) != k or is_symplectic(self.symplectic_mat) == False:
            raise AssertionError("Problem with Hadamards.")

        # STEP 2: Logical S,CZ,Xsqrt,CX_(X,X) until XZ and ZX are zero rank.
        ##    2a: XZ part
        XZ_circ = None
        XZ_part = self.symplectic_mat[:k,k:]
        if rank(XZ_part) != 0:
            XZ_circ = self.find_phase_type_gates(gate_type='Z')
            for item in XZ_circ:
                if 'CZ' in item:  
                    i, j = item[1] 
                    self.symplectic_mat = (self.symplectic_mat @ CZ_gate(i,j,k))%2
                elif 'S' in item:  
                    i = item[1]
                    self.symplectic_mat = (self.symplectic_mat @ S_gate(i,k))%2
        XZ_part = self.symplectic_mat[:k,k:]
        if rank(XZ_part) != 0:
            raise AssertionError("S and CZ gates did not bring the rank to 0.")
        if is_symplectic(self.symplectic_mat) == False:
            raise AssertionError("Problem with S and CZ gates.")
        ##    2b: ZX part
        ZX_circ = None
        ZX_part = self.symplectic_mat[k:,:k]
        if rank(ZX_part) != 0:
            ZX_circ = self.find_phase_type_gates(gate_type='X')
            for item in ZX_circ:
                if 'C(X,X)' in item:  
                    i, j = item[1]  
                    self.symplectic_mat = (self.symplectic_mat @ CX_XX_gate(i,j,k))%2
                elif 'Xsqrt' in item:  
                    i = item[1]
                    self.symplectic_mat = (self.symplectic_mat @ Xsqrt_gate(i,k))%2
        ZX_part = self.symplectic_mat[k:,:k]
        if rank(ZX_part) != 0:
            raise AssertionError("Xsqrt and CX(X,X) gates did not bring the rank to 0.")
        if is_symplectic(self.symplectic_mat) == False:
            raise AssertionError("Problem with Xsqrt and CX(X,X) gates.")


        XX_part = self.symplectic_mat[:k,:k]
        ZZ_part = self.symplectic_mat[k:,k:]
        XZ_part = self.symplectic_mat[:k,k:]
        ZX_part = self.symplectic_mat[k:,:k]

        assert rank(XX_part) == k
        assert rank(ZZ_part) == k
        assert rank(XZ_part) == 0
        assert rank(ZX_part) == 0
        assert is_symplectic(self.symplectic_mat)

        # STEP 3: CNOT circuits via Gaussian Elimination.
        CNOT_circ = self.find_CNOT_circuits()

        logical_circ = []
        logical_circs = [CNOT_circ,ZX_circ,XZ_circ,H_circ]

        for circ in logical_circs:
            if circ:
                logical_circ.extend(circ[::-1])
        return logical_circ