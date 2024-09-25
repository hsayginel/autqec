from automorphisms import *
from magma_interface import *

n = 5
k = 1 
d = '?'
H_symp = np.load(f'codetables/parity_checks/H_symp_n{n}k{k}.npy')

# without intersection
code_auts_dict = qec_code_auts_from_magma(n,k,d,H_symp).run(fileroot='./',save_auts=False)
auts = code_auts_dict['auts']
G, LX, LZ, D = compute_standard_form(H_symp)
for aut in auts:
    phys_act = circ_from_aut(H_symp,aut)        
    phys_circ,_ = phys_act.circ()
    print(phys_circ)
    logical_act = logical_circ_and_pauli_correct(H_symp,phys_circ)   
    logical_circ = logical_act.run()
    print(logical_circ[0])
print('\n----------------------------\n')
# with intersection
code_auts_dict = qec_code_auts_from_magma_with_intersection(n,k,d,H_symp).run(fileroot='./',save_auts=False)
auts = code_auts_dict['auts']
G, LX, LZ, D = compute_standard_form(H_symp)
for aut in auts:
    phys_act = circ_from_aut(H_symp,aut)        
    phys_circ,_ = phys_act.circ()
    print(phys_circ)
    logical_act = logical_circ_and_pauli_correct(H_symp,phys_circ)   
    logical_circ = logical_act.run()
    print(logical_circ[0])