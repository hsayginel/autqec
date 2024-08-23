import pickle
from automorphisms import *
from utils.qec import compute_standard_form
from magma_interface import *

n = 5
k = 1
d = '?'
# Code Automorphisms
H_symp = np.load(f'codetables/parity_checks/H_symp_n{n}k{k}.npy')
# G, LX, LZ, D = compute_standard_form(H_symp)
code = qec_code_auts_from_magma_with_intersection(n,k,d,H_symp)
auts_data = code.run('./tests/',save_auts=False)
auts = auts_data['auts']

# Physical Circuit
phys_act = circ_from_aut(H_symp,aut=auts[3])
phys_circ,symp_mat= phys_act.circ()
print(phys_circ)

# Logical Circuit
log_act = logical_circ_and_pauli_correct(H_symp,phys_circ)
# print(log_act.show_tableux)
# print(log_act.im_tableux_anticomm()[0])
phases = log_act.run()
print(phases)


# print(pauli_correction(G,LX,D,LZ).im_stabs_check_phases(symp_mat))