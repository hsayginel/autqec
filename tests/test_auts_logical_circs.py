from automorphisms import *

n = 5
k = 1 
d = 3
H_symp = np.array(([[1,0,1,0,1,0,0,1,1,0],
                    [0,0,1,1,0,1,0,0,1,1],
                    [0,1,1,1,1,0,0,0,0,0],
                    [0,0,0,0,0,0,1,1,1,1]]))
auts = [[(2, 3),(4, 6),(7, 15),(8, 14),(9, 13),(10, 12)],
[(4, 13),(5, 14),(6, 15),(7, 10),(8, 11),(9, 12)],
[(1, 10, 6),(2, 11, 4),(3, 12, 5),(7, 8, 9)],
[(1, 3, 2),(4, 6, 5),(7, 9, 8),(10, 12, 11),(13, 15, 14)],
[(1, 5, 13, 10, 8),(2, 6, 14, 11, 9),(3, 4, 15, 12, 7)]]

G, LX, LZ, D = compute_standard_form(H_symp)

for i, aut in enumerate(auts):
    print()
    print('Aut no:',i+1)
    logical_act = logical_circ_of_aut(H_symp,aut)        
    logical_act.print_physical_act()
    symp_mat = logical_act.construct_symplectic_mat()
    logical_circ = logical_act.circ()
    print(logical_circ)