import subprocess
import numpy as np

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

def magma(commands):
    process = subprocess.Popen(['magma'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(commands)
    return stdout

def MAGMA_define_SG(symplectic_mat_list,A):
    """Checks if A is in the span of a list of symplectic matrices."""
    k = symplectic_mat_list[0].shape[0]//2
    commands = ''
    listofgens = '['
    for i, sympmat in enumerate(symplectic_mat_list):
        commands += (convert_to_magma_mat(sympmat,'a'+f'{i}'))
        listofgens += f'a{i},'
    listofgens = listofgens[:-1]
    listofgens += ']'
    A_magma = convert_to_magma_mat(A,'A')
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
structure:=GroupName(FG);
printf "Structure: ";
structure;
if order eq 1 then 
    exit;
end if;
{A_magma}
A;
A in SG;
        """
    return commands
