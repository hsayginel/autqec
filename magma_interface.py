from automorphisms import *
import subprocess
import re
import pickle
import itertools

class qec_code_auts_from_magma:
    def __init__(self,n,k,d,H_symp):
        """
        H = (X+Z | X | Z)

        Args:
            n (int): number of physical qubits
            k (int): number of logical qubits
            H_symp (np.array): Stabilizers (in binary symplectic format) H = (SX|SZ)
        """

        self.n = n
        self.k = k
        self.d = d
        self.H_symp = H_symp

        # 3-bit rep of H
        m = H_symp.shape[0]
        id_mat = np.eye(n,dtype=int)
        row_1 = np.hstack((np.hstack((id_mat,id_mat)),id_mat))
        XZ_part = np.zeros((m,n),dtype=int)
        for row_ind in range(m):
            for col_ind in range(n):
                XZ_part[row_ind,col_ind] = (H_symp[row_ind,col_ind] + H_symp[row_ind,col_ind+n])%2
        row_2 = np.hstack((XZ_part,H_symp))
        H_3bit = np.vstack((row_1,row_2))
        self.H_3bit = np.array(H_3bit,dtype=int)
    
    def qubits_3bitrep_order(self):
        n = self.n
        qubit_triplet_list = np.arange(1,3*n+1,1) # Qubit triplets
        magma_order = qubit_triplet_list.reshape((n,3)).flatten(order='F') # Z | X | X+Z
        Z_part = magma_order[:n]
        X_part = magma_order[n:2*n]
        XZ_part = magma_order[2*n:]
        three_bit_rep_order = np.concatenate((XZ_part,X_part,Z_part)) # X+Z | X | Z
        return three_bit_rep_order

    def preprocess_H_3bit(self):
        H_rref, _, transform_rows, transform_cols = reduced_row_echelon(self.H_3bit)
        qubit_labels_og = self.qubits_3bitrep_order()
        reordered_qubit_list = qubit_labels_og@transform_cols
        return reordered_qubit_list, H_rref, transform_rows, transform_cols

    def convert_to_magma_mat(self,mat):
        """
        Args: 
            mat (np.array): matrix 
        """
        mat = np.array(mat,dtype=int)
        n_rows, n_cols = mat.shape
        mat_str = [','.join(map(str, row)) for row in mat]
        mat_str = ',\n'.join(mat_str)
        magma_code = "M := Matrix(GF(2), {}, {},\n".format(n_rows,n_cols) + "[" + mat_str + "]);\n"
        return magma_code 

    def run(self,fileroot,save_auts = True,save_magma_commands = False, save_magma_output = False):
        n = self.n
        k = self.k
        d = self.d

        reordered_qubit_list, H_rref, transform_rows, transform_cols = self.preprocess_H_3bit()

        H_rref_MAGMA = self.convert_to_magma_mat(H_rref)

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
        C := LinearCode(M);
        C;
        printf "End\n";
        printf "\n";
        time
        autgroup := AutomorphismGroup(C); 
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

        commands = commands_part1 + H_rref_MAGMA + commands_part2

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

class qec_code_auts_from_magma_with_intersection:
    def __init__(self,n,k,d,H_symp):
        """
        H = (X+Z | X | Z)

        Args:
            n (int): number of physical qubits
            k (int): number of logical qubits
            H_symp (np.array): Stabilizers (in binary symplectic format) H = (SX|SZ)
        """

        self.n = n
        self.k = k
        self.d = d
        self.H_symp = H_symp

        # 3-bit rep of H
        m = H_symp.shape[0]
        id_mat = np.eye(n,dtype=int)
        row_1 = np.hstack((np.hstack((id_mat,id_mat)),id_mat))
        XZ_part = np.zeros((m,n),dtype=int)
        for row_ind in range(m):
            for col_ind in range(n):
                XZ_part[row_ind,col_ind] = (H_symp[row_ind,col_ind] + H_symp[row_ind,col_ind+n])%2
        row_2 = np.hstack((XZ_part,H_symp))
        H_3bit = np.vstack((row_1,row_2))
        self.H_3bit = np.array(H_3bit,dtype=int)
    
    def qubits_3bitrep_order(self):
        n = self.n
        qubit_triplet_list = np.arange(1,3*n+1,1) # Qubit triplets
        magma_order = qubit_triplet_list.reshape((n,3)).flatten(order='F') # Z | X | X+Z
        Z_part = magma_order[:n]
        X_part = magma_order[n:2*n]
        XZ_part = magma_order[2*n:]
        three_bit_rep_order = np.concatenate((XZ_part,X_part,Z_part)) # X+Z | X | Z
        return three_bit_rep_order

    def preprocess_H_3bit(self):
        H_rref, _, transform_rows, transform_cols = reduced_row_echelon(self.H_3bit)
        qubit_labels_og = self.qubits_3bitrep_order()
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

        reordered_qubit_list, H_rref, transform_rows, transform_cols = self.preprocess_H_3bit()

        H_rref_MAGMA = self.convert_to_magma_mat(H_rref)
        id_mat = np.eye(n,dtype=int) 
        id_row = np.hstack([id_mat,id_mat,id_mat])@transform_cols
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
    
    
class qec_embed_code:
    def __init__(self, H_symp, embedding = 'two_code_blocks'):
        self.H_symp = H_symp 
        self.embedding = embedding

    def gen_w2_embed_mat(self,n):
        combos = itertools.combinations(range(n), 2)
        matrix = []
        for combo in combos:
            column = [0] * n  
            for index in combo:
                column[index] = 1  
            matrix.append(column)
        matrix = [list(row) for row in zip(*matrix)]
        matrix = np.hstack((np.eye(n),np.matrix(matrix)))
      
        return np.matrix(matrix,dtype=int)

    def embed_mat(self):
        H_symp = self.H_symp
        n = H_symp.shape[1]//2

        option = self.embedding
        if option == 'two_code_blocks':
            id_mat = np.eye(n,dtype=int)
            zeros_mat = np.zeros_like(id_mat,dtype=int)

            block_col1 = np.vstack((id_mat,zeros_mat))
            block_col2 = np.vstack((zeros_mat,id_mat))
            block_col3 = np.vstack((id_mat,id_mat))

            V_T = np.hstack((np.hstack((block_col1,block_col2)),block_col3))

        elif option == "three_code_blocks":
            NotImplementedError

        elif option == 'all_weight_2': 
            NotImplementedError
            # HX_part = H_symp[:,:n]
            # HZ_part = H_symp[:,n:]
            # V_T = self.gen_w2_embed_mat(5)
            # M_T = V_T[:,n:]
            # M = M_T.T
            # A_V = np.hstack((HX_part,(HX_part@M_T)%2))
            # ones = np.eye(M.shape[0])
            # zeros = np.zeros((HZ_part.shape[0],ones.shape[1]))
            # B_V = np.vstack((np.hstack((HZ_part,zeros)),np.hstack((M,ones))))
            # assert A_V.shape[1] == B_V.shape[1]
            # no_of_rows = B_V.shape[0] - A_V.shape[0] 
            # A_V = np.vstack((A_V,np.zeros((no_of_rows,A_V.shape[1]))))

        elif option == 'all_weight_3':
            NotImplementedError

        else:
            raise TypeError("Unknown option for embedding.")
        
        return V_T