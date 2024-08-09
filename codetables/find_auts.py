from automorphisms import *
from magma_interface import *

n_min = 1
n_max = 15
d = '?'
with open("codetables/error_log/auts_data_errors.txt", "w") as file:
    for n in range(n_min,n_max+1):
        for k in range(n+1):
            try: 
                print("Processing code [[{},{}]]".format(n,k))
                H_symp = np.load(f'codetables/parity_checks/H_symp_n{n}k{k}.npy')
                code_auts_dict = qec_code_auts_from_magma(n,k,d,H_symp).run(fileroot='codetables/auts_data/')
            except Exception as e:
                error_message = f"Case [[{n,k}]] failed with error: {e}\n"
                file.write(error_message)
                file.flush() 