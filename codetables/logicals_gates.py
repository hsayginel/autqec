from automorphisms import *
from magma_interface import *

d = '?'
n_min = 1
n_max = 15
with open("codetables/error_log/logicals_errors.txt", "w") as file:
    for n in range(n_min,n_max+1):
        for k in range(n+1):
            try: 
                print("Processing code [[{},{}]]".format(n,k))
                H_symp = np.load(f'codetables/parity_checks/H_symp_n{n}k{k}.npy')
                auts_file = f'codetables/auts_data_w_intersection/auts_n{n}k{k}d?.pkl'
                with open(auts_file, 'rb') as f:
                    code_auts_dict = pickle.load(f)
                auts = code_auts_dict['auts']
                gates = {}
                gates['physical'] = []
                gates['logical'] = []
                for aut in auts:
                    phys_act = physical_circ_of_aut(H_symp,aut)        
                    phys_circ = phys_act.circ_w_pauli_correction()
                    gates['physical'].append(phys_circ)
                    logical_act = logical_circ_of_aut(H_symp,aut)   
                    logical_circ = logical_act.circ_w_pauli_correction()
                    gates['logical'].append(logical_circ)
                # with open(f'codetables/logical_gates/gates_n{n}k{k}.pkl', 'wb') as f:
                #     pickle.dump(gates, f)

            except Exception as e:
                error_message = f"Case [[{n,k}]] failed with error: {e}\n"
                file.write(error_message)
                file.flush() 