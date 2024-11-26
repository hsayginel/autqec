from tests.clifford_subgroup_membership import *

log_circs = [[('S',5),('SWAP',(2,3)),('C(X,X)',(4,5))]]
M = aut_gens_as_binary_mat(log_circs,k=5)
print(M)

print(cliff_binary_to_circ(M,k=5))