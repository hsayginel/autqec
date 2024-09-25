from automorphisms import *
from magma_interface import *
n = 28 
k = 2
H_symp = np.load(f'codetables/parity_checks/H_symp_n{n}k{k}.npy')
auts_file = f'codetables/auts_data_w_intersection/auts_n{n}k{k}d?.pkl'
auts_file = f'codetables/auts_data/auts_n{n}k{k}d?.pkl'
with open(auts_file, 'rb') as f:
    code_auts_dict = pickle.load(f)
auts = code_auts_dict['auts']
gates = {}
gates['physical'] = []
gates['logical'] = []
symp_mats = []
for aut in auts:
    phys_act = physical_circ_of_aut(H_symp,aut)        
    phys_circ = phys_act.circ_w_pauli_correction()
    gates['physical'].append(phys_circ)
    logical_act = logical_circ_of_aut(H_symp,aut)   
    logical_circ, symplectic_mat = logical_act.circ()
    logical_circ = logical_act.circ_w_pauli_correction()
    gates['logical'].append(logical_circ)
    symp_mats.append(symplectic_mat)

with open(f'codetables/logical_gates/gates_n{n}k{k}.pkl', 'wb') as f:
    pickle.dump(gates, f)
with open(f'codetables/logical_gates/symp_mats_n{n}k{k}.pkl', 'wb') as f:
    pickle.dump(symp_mats, f)