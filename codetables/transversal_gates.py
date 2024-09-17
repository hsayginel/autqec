import subprocess
import re

def magma_check_transversal(commands):
    process = subprocess.Popen(['magma'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(commands)
    matches = re.findall(r'(H|S|Xsq|SH):\s(true|false)', stdout)
    result = {key: value for key, value in matches}
    return result

def h_as_perms(n):
    tuples = [(i*3+1, i*3+2) for i in range(n)]
    result = ''.join(f'({x},{y})' for x, y in tuples)
    return result

def s_as_perms(n):
    tuples = [(i*3+2, i*3+3) for i in range(n)]
    result = ''.join(f'({x},{y})' for x, y in tuples)
    return result

def xsq_as_perms(n):
    tuples = [(i*3+1, i*3+3) for i in range(n)]
    result = ''.join(f'({x},{y})' for x, y in tuples)
    return result

def sh_as_perms(n):
    tuples = [(i*3+1, i*3+2, i*3+3) for i in range(n)]
    result = ''.join(f'({x},{y},{z})' for x, y, z in tuples)
    return result

def QECC(n,k):
    h_perms = h_as_perms(n)
    s_perms = s_as_perms(n)
    xsq_perms = xsq_as_perms(n)
    sh_perms = sh_as_perms(n)
    magma_commands = f"""
    F<w> := GF(4);
    n := {n};
    k := {k};
    Q := QECC(F,n,k);
    autgroup := AutomorphismGroup(Q);
    h := Sym(3*n)!{h_perms};
    s := Sym(3*n)!{s_perms};
    xsq := Sym(3*n)!{xsq_perms};
    sh := Sym(3*n)!{sh_perms};
    printf "H: ";
    h in autgroup;
    printf "S: ";
    s in autgroup;
    printf "Xsq: ";
    xsq in autgroup;
    printf "SH: ";
    sh in autgroup;
    """
    return magma_commands


for n in range(1,30+1):
    code = QECC(n,1)
    print(n, magma_check_transversal(code))


