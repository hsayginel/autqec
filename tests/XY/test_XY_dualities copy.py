from XY_dualities import *
from utils.qec import stabs_to_H_symp

n = 5
k = 1 
d = '?'
H_symp = np.load(f'codetables/parity_checks/H_symp_n{n}k{k}.npy') 
code = qec_code_XY_dualities_from_magma_with_intersection(n,k,d,H_symp)
auts_data = code.run('./tests/XY/',save_auts=False)
auts = auts_data['auts']

#######################################################################
for aut in auts: 
    phys_act = physical_circ_of_XY_duality(H_symp,aut)
    bits_image = phys_act.bits_image
    circ, symp_mat_phys = phys_act.circ()
    # print(symp_mat_phys)
    print(circ)
    # circ_pauli_corr = phys_act.circ_w_pauli_correction()
# print(circ_pauli_corr)

#######################################################################
for aut in auts: 
    log_act = logical_circ_of_XY_duality(H_symp,aut)
    logical_circ, symp_mat_logic = log_act.circ()
    print(logical_circ)
    # logical_circ_pauli_corr = log_act.circ_w_pauli_correction()
    # print(logical_circ_pauli_corr)