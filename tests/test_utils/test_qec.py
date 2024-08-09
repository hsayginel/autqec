from utils.qec import *

# Steane 
def steane():
    print("Steane")
    k = 1
    H_symp = np.vstack((np.array((
                        [0,0,0,1,1,1,1,0,0,0,0,0,0,0],
                        [0,1,1,0,0,1,1,0,0,0,0,0,0,0],
                        [1,0,1,0,1,0,1,0,0,0,0,0,0,0])),
                        np.array((
                        [0,0,0,0,0,0,0,0,0,0,1,1,1,1],
                        [0,0,0,0,0,0,0,0,1,1,0,0,1,1],
                        [0,0,0,0,0,0,0,1,0,1,0,1,0,1]))))
    G, Z_logicals, X_logicals, D = compute_standard_form(H_symp)
    
    # anticommutation of logicals
    assert np.allclose(symp_prod(X_logicals,Z_logicals),np.eye(k))
    # anticommutation of stabilizers & destabilizers
    assert np.allclose(symp_prod(G,D),np.eye(G.shape[0]))
    # commutation of stabilizers with logicals
    for Z in Z_logicals:
        assert np.allclose(symp_prod(G,Z),np.zeros((G.shape[0])))
    for X in X_logicals:
        assert np.allclose(symp_prod(G,X),np.zeros((G.shape[0])))
    # commutation of destabilizers with logicals
    for Z in Z_logicals:
        assert np.allclose(symp_prod(D,Z),np.zeros((D.shape[0])))
    for X in X_logicals:
        assert np.allclose(symp_prod(D,X),np.zeros((D.shape[0])))

# [[5,1,3]]
def n5():
    k=1
    print("5-qubit")
    H_symp = np.array(([[1,0,1,0,1,0,0,1,1,0],
                        [0,0,1,1,0,1,0,0,1,1],
                        [0,1,1,1,1,0,0,0,0,0],
                        [0,0,0,0,0,0,1,1,1,1]]))

    G, Z_logicals, X_logicals, D = compute_standard_form(H_symp)
    
    # anticommutation of logicals
    assert np.allclose(symp_prod(X_logicals,Z_logicals),np.eye(k))
    # anticommutation of stabilizers & destabilizers
    assert np.allclose(symp_prod(G,D),np.eye(G.shape[0]))
    # commutation of stabilizers with logicals
    for Z in Z_logicals:
        assert np.allclose(symp_prod(G,Z),np.zeros((G.shape[0])))
    for X in X_logicals:
        assert np.allclose(symp_prod(G,X),np.zeros((G.shape[0])))
    # commutation of destabilizers with logicals
    for Z in Z_logicals:
        assert np.allclose(symp_prod(D,Z),np.zeros((D.shape[0])))
    for X in X_logicals:
        assert np.allclose(symp_prod(D,X),np.zeros((D.shape[0])))


    # assert np.allclose(symp_prod(S,D),np.eye(S.shape[0]))

# [[15,1,3]]
## REED-MULLER CODE
def reed_muller():
    k=1
    print("Reed-Muller")
    H_X = np.array(([[1,0,0,0,1,1,1,0,0,0,1,1,1,0,1],
                    [0,1,0,0,1,0,0,1,1,0,1,1,0,1,1],
                    [0,0,1,0,0,1,0,1,0,1,1,0,1,1,1],
                    [0,0,0,1,0,0,1,0,1,1,0,1,1,1,1]]))
    H_Z = np.array(([[1,0,0,0,0,0,0,0,0,0,1,1,1,0,0],
                    [0,1,0,0,0,0,0,0,0,0,1,1,0,1,0],
                    [0,0,1,0,0,0,0,0,0,0,1,0,1,1,0],
                    [0,0,0,1,0,0,0,0,0,0,0,1,1,1,0],
                    [0,0,0,0,1,0,0,0,0,0,1,1,0,0,1],
                    [0,0,0,0,0,1,0,0,0,0,1,0,1,0,1],
                    [0,0,0,0,0,0,1,0,0,0,0,1,1,0,1],
                    [0,0,0,0,0,0,0,1,0,0,1,0,0,1,1],
                    [0,0,0,0,0,0,0,0,1,0,0,1,0,1,1],
                    [0,0,0,0,0,0,0,0,0,1,0,0,1,1,1]]))

    H_symp = np.vstack((np.hstack((H_X,np.zeros_like(H_X))),
                np.hstack((np.zeros_like(H_Z),H_Z))))

    G, Z_logicals, X_logicals, D = compute_standard_form(H_symp)
    
    # anticommutation of logicals
    assert np.allclose(symp_prod(X_logicals,Z_logicals),np.eye(k))
    # anticommutation of stabilizers & destabilizers
    assert np.allclose(symp_prod(G,D),np.eye(G.shape[0]))
    # commutation of stabilizers with logicals
    for Z in Z_logicals:
        assert np.allclose(symp_prod(G,Z),np.zeros((G.shape[0])))
    for X in X_logicals:
        assert np.allclose(symp_prod(G,X),np.zeros((G.shape[0])))
    # commutation of destabilizers with logicals
    for Z in Z_logicals:
        assert np.allclose(symp_prod(D,Z),np.zeros((D.shape[0])))
    for X in X_logicals:
        assert np.allclose(symp_prod(D,X),np.zeros((D.shape[0])))

    # print(symp_prod(H_symp,D))


def n5_another():
    k = 1
    print('n5_another')
    H_symp = np.array([
        [1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 1, 1]
    ])
    G, Z_logicals, X_logicals, D = compute_standard_form(H_symp)
    
    # anticommutation of logicals
    assert np.allclose(symp_prod(X_logicals,Z_logicals),np.eye(k))
    # anticommutation of stabilizers & destabilizers
    assert np.allclose(symp_prod(G,D),np.eye(G.shape[0]))
    # commutation of stabilizers with logicals
    for Z in Z_logicals:
        assert np.allclose(symp_prod(G,Z),np.zeros((G.shape[0])))
    for X in X_logicals:
        assert np.allclose(symp_prod(G,X),np.zeros((G.shape[0])))
    # commutation of destabilizers with logicals
    for Z in Z_logicals:
        assert np.allclose(symp_prod(D,Z),np.zeros((D.shape[0])))
    for X in X_logicals:
        assert np.allclose(symp_prod(D,X),np.zeros((D.shape[0])))


steane()
reed_muller()
n5()
n5_another()