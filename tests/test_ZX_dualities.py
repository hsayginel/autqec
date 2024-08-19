from ZX_dualities import *

n = 5
k = 1 
d = '?'
H_symp = np.load(f'codetables/parity_checks/H_symp_n{n}k{k}.npy')
# code = qec_code_ZX_dualities_from_magma_with_intersection(n,k,d,H_symp)
# auts_data = code.run('./tests/')
# auts = auts_data['auts']

#######################################################################

auts = [[(4, 6), (3, 5), (8, 10), (7, 9)], [(2, 1), (4, 5, 10, 7), (6, 9, 8, 3)], [(4, 8), (6, 10), (3, 7), (5, 9)]]
phys_act = physical_circ_of_ZX_duality(H_symp,auts[1])
bits_image = phys_act.bits_image
circ, _ = phys_act.circ()
circ_pauli_corr = phys_act.circ_w_pauli_correction()
print(circ)
print(circ_pauli_corr)

#######################################################################
log_act = logical_circ_of_ZX_duality(H_symp,auts[1])
logical_circ = log_act.circ()
logical_circ_pauli_corr = log_act.circ_w_pauli_correction()
print(logical_circ)
print(logical_circ_pauli_corr)