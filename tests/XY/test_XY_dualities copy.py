from XY_dualities import *
from utils.qec import stabs_to_H_symp

n = 10
k = 2 
d = '?'
H_symp = np.load(f'codetables/parity_checks/H_symp_n{n}k{k}.npy') 
code = qec_code_XY_dualities_from_magma_with_intersection(n,k,d,H_symp)
auts_data = code.run('./tests/XY/')
auts = auts_data['auts']

#######################################################################
print(auts)
phys_act = physical_circ_of_XY_duality(H_symp,auts[0])
bits_image = phys_act.bits_image
circ, symp_mat_phys = phys_act.circ()
print(symp_mat_phys)
print(circ)
circ_pauli_corr = phys_act.circ_w_pauli_correction()
print(circ_pauli_corr)

#######################################################################
log_act = logical_circ_of_XY_duality(H_symp,auts[0])
log_act.print_phys_circ()
log_act.print_physical_act()
logical_circ = log_act.circ()
logical_circ_pauli_corr = log_act.circ_w_pauli_correction()
print(logical_circ)
print(logical_circ_pauli_corr)