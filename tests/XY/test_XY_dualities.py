from XY_dualities import *
from utils.qec import stabs_to_H_symp

n = 5
k = 1 
d = 3
stabs = ['XZZXI','IXZZX','XIXZZ','ZXIXZ']
H_symp = stabs_to_H_symp(stabs)
G,LX,LZ,D = compute_standard_form(H_symp)
code = qec_code_XY_dualities_from_magma_with_intersection(n,k,d,H_symp)
auts_data = code.run('./tests/XY/')
auts = auts_data['auts']


#######################################################################

phys_act = physical_circ_of_XY_duality(H_symp,auts[1])
bits_image = phys_act.bits_image
circ, symp_mat_phys = phys_act.circ()
print(symp_mat_phys)
# circ_pauli_corr = phys_act.circ_w_pauli_correction()
# print(symp_mat_phys)
# print(circ)
# print(circ_pauli_corr)

# #######################################################################
# log_act = logical_circ_of_XY_duality(H_symp,auts[1])
# log_act.print_phys_circ()
# log_act.print_physical_act()
# logical_circ = log_act.circ()
# logical_circ_pauli_corr = log_act.circ_w_pauli_correction()
# print(logical_circ)
# print(logical_circ_pauli_corr)