from automorphisms import *
from magma_interface import *

n_min = 28
n_max = 30
k_min = 2
k_max = n_max 
d = '_'
with open("codetables/error_log/auts_data_w_intersect_errors.txt", "w") as file:
    for n in range(n_min,n_max+1):
        for k in range(k_min,n+1):
            try: 
                print("Processing code [[{},{}]]".format(n,k))
                H_symp = np.load(f'codetables/parity_checks/H_symp_n{n}k{k}.npy')
                code_auts_dict = qec_code_auts_from_magma_with_intersection(n,k,d,H_symp).run(fileroot='codetables/auts_data_w_intersection/')
            except Exception as e:
                error_message = f"Case [[{n,k}]] failed with error: {e}\n"
                file.write(error_message)
                file.flush() 