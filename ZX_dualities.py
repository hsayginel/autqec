from utils.linalg import *
from utils.perms import *
from utils.pauli import *
from utils.qec import compute_standard_form
from utils.symplectic import *
import subprocess
import re
import pickle
from itertools import combinations

class qec_code_ZX_dualities_from_magma_with_intersection:
    def __init__(self,n,k,d,H_symp):
        """
        H = (X | Z)

        Args:
            n (int): number of physical qubits
            k (int): number of logical qubits
            H_symp (np.array): Stabilizers (in binary symplectic format) H = (SX|SZ)
        """

        self.n = n
        self.k = k
        self.d = d
        self.H_symp = H_symp

    def qubits_2bitrep_order(self):
        n = self.n
        qubit_doublets_list = np.arange(1,2*n+1,1) # Qubit triplets
        magma_order = qubit_doublets_list.reshape((n,2)).flatten(order='F') # X | Z 
        X_part = magma_order[:n]
        Z_part = magma_order[n:2*n]
        two_bit_rep_order = np.concatenate((X_part,Z_part)) #  X | Z
        return two_bit_rep_order
    
    def preprocess_H(self):
        n = self.n
        H_rref, _, transform_rows, transform_cols = rref_mod2(self.H_symp)
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
            with open(fileroot + f'ZX_dualities_n{n}k{k}d{d}.pkl', 'wb') as file:
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
    
class circ_from_ZX_duality:
    def __init__(self,H_symp,aut):
        """
        Class for finding the physical qubit circuits of the 
        generators of the automorphism groups of stabilizer QECCs
        including appropriate Pauli corrections. 

        2-bit representation
        --------------------
        Paulis (X | Z): 
        I --> 00   
        X --> 10   
        Z --> 01   
        
        1-qubit Clifford operators: 
        | Operators      |   Permutation   |
        |----------------|-----------------|
        | $H$            | (1,2)           |

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
        id_mat = np.eye(2*n,dtype=int)
        self.E_mat = id_mat
        self.EInv_mat = id_mat

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
                single_qubit_gates.append(("H",gate_ind))
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
        q3bit_order = X_bits + Z_bits
        new_aut = []
        for cycle in self.aut:
            new_aut.append(tuple(q3bit_order.index(x)+1 for x in cycle))
        
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
    

