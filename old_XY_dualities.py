from utils.linalg import *
from utils.perms import *
from utils.pauli import *
from utils.qec import compute_standard_form
from utils.symplectic import *
import subprocess
import re
import pickle
from itertools import combinations

class qec_code_XY_dualities_from_magma_with_intersection:
    def __init__(self,n,k,d,H_symp):
        """
        H = (X | X+Z)

        Args:
            n (int): number of physical qubits
            k (int): number of logical qubits
            H_symp (np.array): Stabilizers (in binary symplectic format) H = (SX|SZ)
        """

        self.n = n
        self.k = k
        self.d = d
        self.H_symp = H_symp

        # 2-bit rep of H
        m = H_symp.shape[0]
        X_part = H_symp[:,:n].copy()
        XZ_part = np.zeros((m,n),dtype=int)
        for row_ind in range(m):
            for col_ind in range(n):
                XZ_part[row_ind,col_ind] = (H_symp[row_ind,col_ind] + H_symp[row_ind,col_ind+n])%2
        self.H_2bit = np.hstack((X_part,XZ_part))
        

    def qubits_2bitrep_order(self):
        n = self.n
        qubit_doublets_list = np.arange(1,2*n+1,1) # Qubit triplets
        magma_order = qubit_doublets_list.reshape((n,2)).flatten(order='F') # X | X + Z 
        X_part = magma_order[:n]
        XZ_part = magma_order[n:2*n]
        two_bit_rep_order = np.concatenate((X_part,XZ_part)) # X | X + Z 
        return two_bit_rep_order
    
    def preprocess_H(self):
        n = self.n
        H_rref, _, transform_rows, transform_cols = rref_mod2(self.H_2bit)
        qubit_labels_og = self.qubits_2bitrep_order()
        reordered_qubit_list = qubit_labels_og@transform_cols
        return reordered_qubit_list, H_rref, transform_rows, transform_cols

    def convert_to_magma_mat(self,mat,mat_name='M'):
        """
        Args: 
            mat (np.array): matrix 
        """
        mat = np.array(mat,dtype=int)
        n_rows, n_cols = mat.shape
        mat_str = [','.join(map(str, row)) for row in mat]
        mat_str = ',\n'.join(mat_str)
        magma_code = f"{mat_name} := Matrix(GF(2), {n_rows}, {n_cols},\n" + "[" + mat_str + "]);\n"
        return magma_code 

    def run(self,fileroot,save_auts = True,save_magma_commands = False, save_magma_output = False):
        n = self.n
        k = self.k
        d = self.d

        reordered_qubit_list, H_rref, transform_rows, transform_cols = self.preprocess_H()

        H_rref_MAGMA = self.convert_to_magma_mat(H_rref)
        id_mat = np.eye(n,dtype=int) 
        id_row = np.hstack([id_mat,id_mat])@transform_cols
        I_MAGMA = self.convert_to_magma_mat(id_row,mat_name='I')

        commands_part1 = """
        // Define a function to find prime factors
        PrimeFactors := function(n)
            // Use Factorization function to find prime factors
            F := Factorization(n);
            
            // Initialize an empty list to store prime factors
            prime_factors := [];
            
            // Iterate through the factors and extract the primes
            for pair in F do
                Append(~prime_factors, pair[1]);
            end for;
            
            // Return the list of prime factors
            return prime_factors;
        end function;
        """

        commands_part2 = """
        C1 := LinearCode(M);
        C1;
        C2 := LinearCode(I);
        printf "End\n";
        printf "\n";
        time
        autgroup1 := AutomorphismGroup(C1); 
        autgroup2 := AutomorphismGroup(C2);
        autgroup := autgroup1 meet autgroup2;
        autgroup_order := Order(autgroup);
        printf "\n";
        printf "Order: ";
        autgroup_order;
        printf "\n";
        prime_factors := PrimeFactors(autgroup_order);

        for p in prime_factors do
            printf "Sylow Order: ";
            p;
            for g in Generators(Sylow(autgroup,p)) do
                printf "---\n";
                g;
                printf "---\n";
            end for;
        end for;
        """

        commands = commands_part1 + H_rref_MAGMA + I_MAGMA + commands_part2

        if save_magma_commands == True:
            with open(fileroot + f'magma_commands_n{n}k{k}d{d}.txt', "w") as file:
                file.write(commands)
        
        raw_magma_output = self.magma(commands)

        if save_magma_output == True:
            with open(fileroot + f'magma_output_n{n}k{k}d{d}.txt', "w") as file:
                file.write(raw_magma_output)

        # time
        time_pattern = r"Time:\s*([\d\.]+)"
        match = re.search(time_pattern, raw_magma_output)
        if match:
            time = float(match.group(1))
        else:
            time = 0.0
        
        # automorphism group order
        order_pattern = r"Order:\s*(\d+)"
        match = re.search(order_pattern, raw_magma_output)
        if match:
            order = int(match.group(1))
        else:
            order = 1

        # automorphism group generators and qubit relabelling to original basis
        aut_gens, aut_gens_text = self.parse_magma_output_for_aut_gens(raw_magma_output)
        fixed_auts_gens = []
        for g in aut_gens:
            correct_g = []
            for cycle in g:
                new_cycle = []
                for i in cycle:
                    new_cycle.append(reordered_qubit_list[i-1])
                correct_g.append(tuple(new_cycle))
            fixed_auts_gens.append(correct_g)

        # store in dictionary
        code_auts_dict = {}
        code_auts_dict['order'] = order
        code_auts_dict['auts'] = fixed_auts_gens
        code_auts_dict['time'] = time

        if save_auts == True:
            with open(fileroot + f'auts_n{n}k{k}d{d}.pkl', 'wb') as file:
                pickle.dump(code_auts_dict, file)

        return code_auts_dict

    def magma(self,commands):
        process = subprocess.Popen(['magma'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(commands)
        return stdout

    def parse_magma_output_for_aut_gens(self,raw_magma_output):
        """ Extract automorphism group generators """
        aut_gens_text = re.sub(r'\n','',raw_magma_output)
        aut_gens_text = re.sub(r'\s+', ' ', aut_gens_text).strip()
        aut_gens_text = re.findall(r'---(.*?)---',aut_gens_text,re.DOTALL)
        aut_gens = []
        for g in aut_gens_text:
            cycles = g.split(')(')
            one_aut_gen = []
            for cycle in cycles:
                elements = cycle.strip('()').split(',')
                elements = tuple(int(elem) for elem in elements)
                one_aut_gen.append(elements)
            aut_gens.append(one_aut_gen)
        return aut_gens, aut_gens_text
    
class physical_circ_of_XY_duality:
    def __init__(self,H_symp,aut):
        """
        Class for finding the physical qubit circuits of the 
        generators of the automorphism groups of stabilizer QECCs
        including appropriate Pauli corrections. 

        2-bit representation
        --------------------
        Paulis (X | X+Z): 
        I --> 00   
        X --> 10   
        Y --> 01   
        
        1-qubit Clifford operators: 
        | Operators      |   Permutation   |
        |----------------|-----------------|
        | $S$            | (1,2)           |

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
        self.bits = np.arange(1,2*n+1,1)
        self.qubit_indices = np.arange(1,n+1,1)
        self.bits_image = permute_by_cycles(self.bits,self.aut)

        # 2-bit rep embedding
        id_mat = np.eye(n,dtype=int)
        zeros = np.zeros_like(id_mat)
        self.E_mat = np.vstack((np.hstack((id_mat,id_mat)),np.hstack((zeros,id_mat))))
        self.EInv_mat = self.E_mat.copy()

    def swaps(self): 
        """
        Returns SWAP gates of the automorphism and orders qubits.
        """
        bits_image = self.bits_image

        permuted_qubit_indices = reduce_doublets_to_qubits(bits_image)

        SWAPs_reversed = sort_with_swaps(permuted_qubit_indices)
        physical_SWAP_gates = SWAPs_reversed[::-1]
        
        # Reverse SWAPS.
        qubit_doublets = [(bits_image[i],bits_image[i+1]) for i in range(0, 2*self.n, 2)]
        ordered_qubit_doublets = apply_swaps(qubit_doublets, SWAPs_reversed)
        self.ordered_qubit_triplets = ordered_qubit_doublets

        return physical_SWAP_gates, ordered_qubit_doublets
    
    def single_qubit_cliffords(self,ordered_qubit_doublets):
        """
        Returns the 1-qubit Clifford gates of the automorphism.
        """
        single_qubit_gates = []
        for i,doublet in enumerate(ordered_qubit_doublets):
            gate_ind = i+1
            reduced_triplet = (doublet[0]%2,doublet[1]%2)
            if reduced_triplet == (1,0):
                pass
            elif reduced_triplet == (0,1):
                single_qubit_gates.append(("S",gate_ind))
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
        # correct qubit order for 2-bit representation (X | Z)
        n = self.n   
        X_bits = [i for i in range(1, 2*n + 1, 2)]
        Z_bits = [i for i in range(2, 2*n + 1, 2)]
        qbit_order = X_bits + Z_bits
        new_aut = []
        for cycle in self.aut:
            new_aut.append(tuple(qbit_order.index(x)+1 for x in cycle))
        
        # Initialize the identity matrix of size n
        perm_matrix = np.eye(2*self.n,dtype=int)
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
        symp_mat = self.perm_mat_to_symp_mat()
        if is_symplectic(symp_mat) == False:
            raise AssertionError('Automorphism generator is not a valid symplectic transformation.')
        return symp_mat

    def circ(self):
        """
        Returns the circuit of the automorphism as 
        1-qubit Cliffords + SWAPs.
        """        
        self.pauli_correct_check = False

        physical_SWAP_gates, ordered_qubit_doublets = self.swaps()
        single_qubit_gates = self.single_qubit_cliffords(ordered_qubit_doublets)
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
            pauli_circ = pauli_correction_XY_duality(G, LX, D, LZ).run(symp_mat,physical_circuit)
            physical_circuit = pauli_circ + physical_circuit 
        return physical_circuit

class pauli_correction_XY_duality:
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
            raise AssertionError("Physical Pauli correction failed: multiples of i phases present.")
        

class logical_circ_of_XY_duality:
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
        phys_act = physical_circ_of_XY_duality(H_symp,aut)
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
        phase_diff = np.array(np.abs(physical_circ_phases - logical_circ_phases),dtype=int)

        pauli_gates = []
        if np.all(phase_diff % 2 == 0) == True:
            phase_diff = np.array(phase_diff/2,dtype=int)
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
            return  pauli_gates + logical_circ 
        else: 
            self.print_pauli_corrections()
            raise AssertionError("Logical Pauli correction failed: multiples of i phases present.")
    
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

        # self.k = k
        # self.symplectic_mat = symplectic_mat.copy()

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
        # logical_circs = [CNOT_circ,ZX_circ,XZ_circ,H_circ]

        for circ in logical_circs:
            if circ:
                logical_circ.extend(circ)
        return logical_circ