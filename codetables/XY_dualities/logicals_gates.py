from XY_dualities import *
from automorphisms import *

d = '?'
n_min = 1
n_max = 15
with open("codetables/XY_dualities/logicals_errors.txt", "w") as file:
    for n in range(n_min,n_max+1):
        for k in range(n+1):
            try: 
                print("Processing code [[{},{}]]".format(n,k))
                H_symp = np.load(f'codetables/parity_checks/H_symp_n{n}k{k}.npy')
                auts_file = f'codetables/XY_dualities/auts_data/auts_n{n}k{k}d?.pkl'
                with open(auts_file, 'rb') as f:
                    code_auts_dict = pickle.load(f)
                auts = code_auts_dict['auts']
                gates = {}
                gates['physical'] = []
                gates['logical'] = []
                symp_mats = []
                for aut in auts:
                    phys_act = circ_from_XY_duality(H_symp,aut)        
                    circ, _ = phys_act.circ()
                    logical_act = logical_circ_and_pauli_correct(H_symp,circ)   
                    symplectic_mat = logical_act.U_logical_act()
                    logical_circ, phys_circ = logical_act.run()
                    gates['physical'].append(phys_circ)
                    gates['logical'].append(logical_circ)
                    symp_mats.append(symplectic_mat)

                with open(f'codetables/XY_dualities/logical_gates/gates_n{n}k{k}.pkl', 'wb') as f:
                    pickle.dump(gates, f)
                with open(f'codetables/XY_dualities/logical_gates/symp_mats_n{n}k{k}.pkl', 'wb') as f:
                    pickle.dump(symp_mats, f)

            except Exception as e:
                error_message = f"Case [[{n,k}]] failed with error: {e}\n"
                file.write(error_message)
                file.flush() 