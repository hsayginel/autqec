import pickle
import re
import subprocess
from utils.symplectic import *
from itertools import combinations, permutations


def convert_to_magma_mat(mat,mat_name='M'):
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

class clifford_subgroups:
    def __init__(self,k,symplectic_mat_list):
        """
        Args: 
            k (int): number of logical qubits
            symplectic_mat_dict (dict): dictionary of subgroup generators given as symplectic matrices.
        """
        self.k = k
        self.symplectic_mat_list = symplectic_mat_list

    def MAGMA_define_SG(self):
        k = self.k
        symplectic_mat_list = self.symplectic_mat_list
        commands = ''
        listofgens = '['
        for i, sympmat in enumerate(symplectic_mat_list):
            commands += (convert_to_magma_mat(sympmat,'a'+f'{i}'))
            listofgens += f'a{i},'
        listofgens = listofgens[:-1]
        listofgens += ']'
        commands += f"""
listofgens := {listofgens};
G := GeneralLinearGroup({2*k},GF(2));
listminimal := [];
indices := [];
s := 1;
order_previous := 1; // Order of the trivial subgroup
while s le #listofgens do
    x := listofgens[s];
    Append(~listminimal, x);
    S := sub<G | listminimal>;
    order_current := #S;
    if order_current gt order_previous then
        Append(~indices, s);
        order_previous := order_current;
    else
        Prune(~listminimal); // Remove the last element from listminimal
    end if;
    s := s + 1;
end while;
SG := sub<G | listminimal>;
counter := 1;
command := "FG<";
for i in indices do
    if counter lt #indices then
        command cat:= "a";
        command cat:= IntegerToString(indices[counter]);
        command cat:=",";
        counter := counter+1;
    else
        command cat:="a";
        command cat:= IntegerToString(i);
        command cat:= ">, phi := FPGroup(SG); return FG, phi";
    end if;
end for;
FG, phi := eval command;
order:=1;
order:=Order(FG);
printf "Order: ";
order;
if order eq 1 then 
    exit;
end if;
        """
        return commands
    
    def return_order(self):
        commands = self.MAGMA_define_SG()
        process = subprocess.Popen(['magma'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        stdout, stderr = process.communicate(commands)
        lines = stdout.strip().split('\n')

        for line in lines:
            if line.startswith("Order:"):
                # Extract the order
                order = int(line.split(":")[1].strip())
                return order
        else:
            return 1
    
    def individual_SWAPs(self):
        k = self.k
        SWAP_gates = ""
        gates_var_names = "["
        gates_str = "["
        combos = list(combinations(range(k), 2))
        for pair in combos:
            i = pair[0]
            j = pair[1]
            SWAP_gates += convert_to_magma_mat(SWAP_gate(i+1,j+1,k),f"SWAP_{i+1}_{j+1}")
            gates_var_names += f"SWAP_{i+1}_{j+1},"
            gates_str += f"\"SWAP_({i+1},{j+1})\","
        gates_var_names = gates_var_names[:-1]
        gates_var_names += "]"
        gates_str = gates_str[:-1]
        gates_str += "]"
        return SWAP_gates, gates_var_names, gates_str

    def individual_CNOTs(self):
        "Probably inefficient."
        k = self.k
        CNOT_gates = ""
        gates_var_names = "["
        gates_str = "["
        combos = list(permutations(range(k), 2))
        for pair in combos:
            i = pair[0]
            j = pair[1]
            CNOT_gates += convert_to_magma_mat(CNOT_gate(i,j,k),f"CNOT_{i+1}_{j+1}")
            gates_var_names += f"CNOT_{i+1}_{j+1},"
            gates_str += f"\"CNOT_({i+1},{j+1})\","
        gates_var_names = gates_var_names[:-1]
        gates_var_names += "]"
        gates_str = gates_str[:-1]
        gates_str += "]"
        return CNOT_gates, gates_var_names, gates_str

    def individual_CZs(self):
        k = self.k
        CZ_gates = ""
        gates_var_names = "["
        gates_str = "["
        combos = list(combinations(range(k), 2))
        for pair in combos:
            i = pair[0]
            j = pair[1]
            CZ_gates += convert_to_magma_mat(CZ_gate(i+1,j+1,k),f"CZ_{i+1}_{j+1}")
            gates_var_names += f"CZ_{i+1}_{j+1},"
            gates_str += f"\"CZ_({i+1},{j+1})\","
        gates_var_names = gates_var_names[:-1]
        gates_var_names += "]"
        gates_str = gates_str[:-1]
        gates_str += "]"
        return CZ_gates, gates_var_names, gates_str

    def full_single_qubit_gamma_XYZ(self):
        k = self.k
        gamma_gates = ""
        gates_str = "["
        gates_var_names = "["
        for i in range(k):
            gamma_gates += convert_to_magma_mat(gamma_XYZ_gate(i+1,k),f"gammaXYZ{i+1}")
            gates_str += f"\"gammaXYZ_{i+1}\","
            gates_var_names += f"gammaXYZ{i+1},"
        gates_str = gates_str[:-1]
        gates_var_names = gates_var_names[:-1]
        gates_str += "]"
        gates_var_names += "]"
        return gamma_gates, gates_var_names, gates_str
    
    def full_single_qubit_gamma_XZY(self):
        k = self.k
        gamma_gates = ""
        gates_str = "["
        gates_var_names = "["
        for i in range(k):
            gamma_gates += convert_to_magma_mat(gamma_XZY_gate(i+1,k),f"gammaXZY{i+1}")
            gates_str += f"\"gammaXZY_{i+1}\","
            gates_var_names += f"gammaXZY{i+1},"
        gates_str = gates_str[:-1]
        gates_var_names = gates_var_names[:-1]
        gates_str += "]"
        gates_var_names += "]"
        return gamma_gates, gates_var_names, gates_str

    def full_SWAP_network(self):
        k = self.k
        SWAP_network = ""
        gates_var_names = "["
        gates_str = "["
        for i in range(k-1):
            j = i+1
            SWAP_network += convert_to_magma_mat(SWAP_gate(i+1,j+1,k),f"SWAP_{i+1}_{j+1}")
            gates_var_names += f"SWAP_{i+1}_{j+1},"
            gates_str += f"\"SWAP_({i+1},{j+1})\","
        gates_var_names = gates_var_names[:-1]
        gates_var_names += "]"
        gates_str = gates_str[:-1]
        gates_str += "]"
        return SWAP_network, gates_var_names, gates_str

    def full_CNOT_network(self): 
        k = self.k
        if k == 1:
            return "  "
        # check 1 CNOT gate: CNOT_(O,1)
        ctrl = 0
        targ = 1
        one_CNOT_gate = convert_to_magma_mat(CNOT_gate(ctrl+1,targ+1,k),"CNOT_1_2")
        # check full SWAP network
        SWAP_network, SWAP_var_names, SWAP_gates_str = self.full_SWAP_network()

        return one_CNOT_gate + SWAP_network, "[CNOT_1_2," + SWAP_var_names[1:], "[\"CNOT_(1,2)\"," + SWAP_gates_str[1:]
    
    def full_single_qubit_H(self):
        k = self.k
        H_gates = ""
        gates_str = "["
        gates_var_names = "["
        for i in range(k):
            gate = H_gate(i+1,k)
            H_gates += convert_to_magma_mat(gate,f"H{i+1}")
            gates_str += f"\"H_{i+1}\","
            gates_var_names += f"H{i+1},"
        gates_str = gates_str[:-1]
        gates_var_names = gates_var_names[:-1]
        gates_str += "]"
        gates_var_names += "]"
        return H_gates, gates_var_names, gates_str
    
    def full_single_qubit_S(self):
        k = self.k
        S_gates = ""
        gates_str = "["
        gates_var_names = "["
        for i in range(k):
            gate = S_gate(i+1,k)
            S_gates += convert_to_magma_mat(gate,f"S{i+1}")
            gates_str += f"\"S_{i+1}\","
            gates_var_names += f"S{i+1},"
        gates_str = gates_str[:-1]
        gates_var_names = gates_var_names[:-1]
        gates_str += "]"
        gates_var_names += "]"
        return S_gates, gates_var_names, gates_str
        
    def full_clifford(self):
        k = self.k
        H_gates, H_var_names, H_gates_str = self.full_single_qubit_H() 
        S_gates, S_var_names, S_gates_str = self.full_single_qubit_S()
        if k == 1:
            return H_gates + S_gates, H_var_names[:-1] + ',' + S_var_names[1:], H_gates_str[:-1] + ',' + S_gates_str[1:]
        CNOT_network, CNOT_network_var_names, CNOT_network_str = self.full_CNOT_network()
        gates = H_gates + S_gates + CNOT_network
        gates_var_names = H_var_names[:-1] + ',' + S_var_names[1:-1] + ',' + CNOT_network_var_names[1:]
        gates_as_str = H_gates_str[:-1] + ',' + S_gates_str[1:-1] + ',' + CNOT_network_str[1:]
        return gates, gates_var_names, gates_as_str
        
    def MAGMA_check_gates(self,gates, gates_var_names, gates_as_str):
        magma_commands = self.MAGMA_define_SG()
        cliffList = f"cliffList:={gates_var_names};\n"
        cliffListStr = f"cliffListStr:={gates_as_str};"
        check_membership = """
index := 1;
printf "Logical operators: \\n";
for cliff in cliffList do
    if cliff in SG then
        print cliffListStr[index], cliff @@ phi;
    end if;
index := index+1;
end for;
printf "FINISH";
"""
        commands = magma_commands + gates + cliffList + cliffListStr + check_membership 
        process = subprocess.Popen(['magma'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        stdout, stderr = process.communicate(commands)
        lines = stdout.strip().split('\n')

        logical_operators = {}
        logical_ops_flag = False
        for line in lines:
            if line.startswith("Order:"):
                # Extract the order
                order = int(line.split(":")[1].strip())
            elif line.startswith("Logical operators:"):
                # Skip the line that states "Logical operators:"
                logical_ops_flag = True
                continue
            if logical_ops_flag == True:
                # Extract logical operators
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts
                    logical_operators[key] = value
            if line.startswith("FINISH"):
                break
        
        return order, logical_operators
    
    def check_full_clifford(self):
        gates, gates_var_names, gates_as_str = self.full_clifford()
        return self.MAGMA_check_gates(gates, gates_var_names, gates_as_str)
    
    def all_gates(self):
        k = self.k
        H_gates, H_var_names, H_gates_str = self.full_single_qubit_H() 
        S_gates, S_var_names, S_gates_str = self.full_single_qubit_S()
        g_XYZ_gates, g_XYZ_var_names, g_XYZ_str = self.full_single_qubit_gamma_XYZ()
        g_XZY_gates, g_XZY_var_names, g_XZY_str = self.full_single_qubit_gamma_XZY()

        gates_1Q = H_gates + S_gates + g_XYZ_gates + g_XZY_gates
        gates_var_names_1Q = H_var_names[:-1] + ',' + S_var_names[1:-1] + ',' + g_XYZ_var_names[1:-1] + ',' + g_XZY_var_names[1:]
        gates_as_str_1Q = H_gates_str[:-1] + ',' + S_gates_str[1:-1] + ',' + g_XYZ_str[1:-1] + ',' + g_XZY_str[1:]

        if k == 1:
            return gates_1Q, gates_var_names_1Q, gates_as_str_1Q
        
        SWAP_gates, SWAP_var_names, SWAP_gates_str = self.individual_SWAPs()
        CNOT_gates, CNOT_var_names, CNOT_gates_str = self.individual_CNOTs()
        CZ_gates, CZ_var_names, CZ_gates_str = self.individual_CZs()

        gates_2Q = SWAP_gates + CNOT_gates + CZ_gates
        gates_var_names_2Q = SWAP_var_names[:-1] + ',' + CNOT_var_names[1:-1] + ',' + CZ_var_names[1:]
        gates_as_str_2Q = SWAP_gates_str[:-1] + ',' + CNOT_gates_str[1:-1] + ',' + CZ_gates_str[1:]

        gates = gates_1Q + gates_2Q
        gates_var_names = gates_var_names_1Q[:-1] + ',' + gates_var_names_2Q[1:]
        gates_as_str = gates_as_str_1Q[:-1] + ',' + gates_as_str_2Q[1:]

        return gates, gates_var_names, gates_as_str
    
    def check_all_gates_individually(self):
        gates, gates_var_names, gates_as_str = self.all_gates()
        return self.MAGMA_check_gates(gates, gates_var_names, gates_as_str)