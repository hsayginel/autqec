from automorphisms import *
from magma_interface import *

d = '?'
n_min = 2
n_max = 20
with open("codetables/error_log/logicals_errors.txt", "w") as file:
    for n in range(n_min,n_max+1):
        for k in range(1,n+1):
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
                symp_mats = []
                for aut in auts:
                    phys_act = circ_from_aut(H_symp,aut)        
                    phys_circ, _ = phys_act.circ()
                    logical_act = logical_circ_and_pauli_correct(H_symp,phys_circ)   
                    symplectic_mat = logical_act.U_logical_act()
                    logical_circ, phys_circ_corr = logical_act.run()
                    gates['physical'].append(phys_circ_corr)
                    gates['logical'].append(logical_circ)
                    symp_mats.append(symplectic_mat)

                with open(f'codetables/logical_gates/gates_n{n}k{k}.pkl', 'wb') as f:
                    pickle.dump(gates, f)
                with open(f'codetables/logical_gates/symp_mats_n{n}k{k}.pkl', 'wb') as f:
                    pickle.dump(symp_mats, f)

            except Exception as e:
                error_message = f"Case [[{n,k}]] failed with error: {e}\n"
                file.write(error_message)
                file.flush() 