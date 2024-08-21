from XY_dualities import *
from magma_interface import *

n_min = 1
n_max = 15
k_min = 0
k_max = n_max 
d = '?'
with open("codetables/XY_dualities/errorlog.txt", "w") as file:
    for n in range(n_min,n_max+1):
        for k in range(k_min,n+1):
            try: 
                print("Processing code [[{},{}]]".format(n,k))
                H_symp = np.load(f'codetables/parity_checks/H_symp_n{n}k{k}.npy')
                code_auts_dict = qec_code_XY_dualities_from_magma_with_intersection(n,k,d,H_symp).run(fileroot='codetables/XY_dualities/auts_data/')
            except Exception as e:
                error_message = f"Case [[{n,k}]] failed with error: {e}\n"
                file.write(error_message)
                file.flush() 