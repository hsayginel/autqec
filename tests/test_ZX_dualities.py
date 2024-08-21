from ZX_dualities import *
from utils.qec import stabs_to_H_symp

n = 5
k = 1 
d = 3
stabs = ['XZZXI','IXZZX','XIXZZ','ZXIXZ']
H_symp = stabs_to_H_symp(stabs)
G,LX,LZ,D = compute_standard_form(H_symp)
code = qec_code_ZX_dualities_from_magma_with_intersection(n,k,d,H_symp)
auts_data = code.run('./tests/')
auts = auts_data['auts']
#######################################################################

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