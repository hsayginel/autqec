from automorphisms import *
from utils.symplectic import *

####################################################
def basic_test():
    n = 10
    # SWAPS
    aut1 = [(1,4),(2,5),(3,6),(7,13),(8,14),(9,15)]
    physical_act = physical_circ_of_aut(np.eye(10),aut1)
    print(physical_act.circ())

    # Single-qubit Cliffords
    aut2 = [(1,2,3),(4,5),(8,9),(11,10,12),(13,15)]
    physical_act = physical_circ_of_aut(np.eye(10),aut1)
    print(physical_act.circ())

    # Both
    aut3 = aut1+aut2
    physical_act = physical_circ_of_aut(np.eye(10),aut1)
    print(physical_act.circ())
####################################################

def test_auts_symplectic_mats():
    # Z | X | X+Z
    H = [(1,2)]
    Xsqrt = [(2,3)]
    S = [(1,3)]
    gamma_XZY = [(1,2,3)]
    gamma_XYZ = [(1,3,2)]

    # H
    act = physical_circ_of_aut(np.eye(2),H)
    symp_mat = act.symp_transform()
    assert np.allclose(symp_mat,H_gate(1,1))
    assert act.circ()[0][0][0] == 'H'

    # Xsqrt
    act = physical_circ_of_aut(np.eye(2),Xsqrt)
    symp_mat = act.symp_transform()
    assert np.allclose(symp_mat,Xsqrt_gate(1,1))
    assert act.circ()[0][0][0] == 'Xsqrt'

    # S
    act = physical_circ_of_aut(np.eye(2),S)
    symp_mat = act.symp_transform()
    assert np.allclose(symp_mat,S_gate(1,1))
    assert act.circ()[0][0][0] == 'S'

    # gamma_XYZ
    act = physical_circ_of_aut(np.eye(2),gamma_XYZ)
    symp_mat = act.symp_transform()
    assert np.allclose(symp_mat,np.array([[1,1],[1,0]]))
    assert act.circ()[0][0][0] == 'GammaXYZ'

    # gamma_XZY
    act = physical_circ_of_aut(np.eye(2),gamma_XZY)
    symp_mat = act.symp_transform()
    assert np.allclose(symp_mat,np.array([[0,1],[1,1]]))
    assert act.circ()[0][0][0] == 'GammaXZY'

def n5k1d3():
    n = 5
    H_symp = np.array(([[1,0,1,0,1,0,0,1,1,0],
                        [0,0,1,1,0,1,0,0,1,1],
                        [0,1,1,1,1,0,0,0,0,0],
                        [0,0,0,0,0,0,1,1,1,1]]))
    auts = [[(2, 3),(4, 6),(7, 15),(8, 14),(9, 13),(10, 12)],
    [(4, 13),(5, 14),(6, 15),(7, 10),(8, 11),(9, 12)],
    [(1, 10, 6),(2, 11, 4),(3, 12, 5),(7, 8, 9)],
    [(1, 3, 2),(4, 6, 5),(7, 9, 8),(10, 12, 11),(13, 15, 14)],
    [(1, 5, 13, 10, 8),(2, 6, 14, 11, 9),(3, 4, 15, 12, 7)]]
    
    for aut in auts:
        phys_act = physical_circ_of_aut(H_symp,aut)        
        circ = phys_act.circ_w_pauli_correction()
        print(circ)
        
       
        
def steane():
    n = 7
    H_X = np.array(([1,0,0,1,0,1,1],
                [0,1,0,1,1,1,0],
                [0,0,1,0,1,1,1]))

    H_Z = np.array(([1,0,0,1,0,1,1],
                    [0,1,0,1,1,1,0],
                    [0,0,1,0,1,1,1]))
    H_symp = np.vstack([np.hstack([H_X,np.zeros_like(H_X)]), np.hstack([np.zeros_like(H_Z),H_Z])])

    auts = [[(1, 16, 19, 10),(2, 17, 20, 11),(3, 18, 21, 12),(4, 13),(5, 14),(6, 15)],
            [(2, 3),(5, 6),(8, 9),(11, 12),(14, 15),(17, 18),(20, 21)],
            [(1, 10),(2, 11),(3, 12),(16, 19),(17, 20),(18, 21)],
            [(1, 16),(2, 17),(3, 18),(10, 19),(11, 20),(12, 21)],
            [(1, 5, 9),(2, 6, 7),(3, 4, 8),(10, 14, 21),(11, 15, 19),(12, 13, 20),(16, 17, 18)],
            [(1, 7, 4),(2, 8, 5),(3, 9, 6),(10, 19, 13),(11, 20, 14),(12, 21, 15)],
            [(1, 4, 7, 10, 13, 16, 19),(2, 5, 8, 11, 14, 17, 20),(3, 6, 9, 12, 15, 18, 21)]]
    for aut in auts:
        phys_act = physical_circ_of_aut(H_symp,aut)        
        circ = phys_act.circ_w_pauli_correction()
        print(circ)


######
basic_test()
test_auts_symplectic_mats()
print('[[5,1,3]]')
n5k1d3()
print('[[7,1,3]]')
steane()