from XY_dualities import *
from utils.qec import stabs_to_H_symp

n = 5
k = 1 
d = 3
stabs = ['XZZXI','IXZZX','XIXZZ','ZXIXZ']
H_symp = stabs_to_H_symp(stabs)
G,LX,LZ,D = compute_standard_form(H_symp)
code = qec_code_XY_dualities_from_magma_with_intersection(n,k,d,H_symp)
auts_data = code.run('./tests/',save_auts=False)
auts = auts_data['auts']
#######################################################################

phys_act = circ_from_XY_duality(H_symp,auts[1])
bits_image = phys_act.bits_image
circ, _ = phys_act.circ()
print(circ)


#######################################################################
from automorphisms import *
circ = logical_circ_and_pauli_correct(H_symp,circ).run()
print(circ)