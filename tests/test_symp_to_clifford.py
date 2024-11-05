from automorphisms import circ_from_symp_mat
from utils.symplectic import *
from utils.linalg import inv_mod2

def random_symp_mats(h):
    _,n = h.shape
    X_part = h[:,:n//2].copy()
    Z_part = h[:,n//2:].copy()
    h_ZX = np.hstack((Z_part,X_part))
    symp_mat = (np.eye(n,dtype=int) + h_ZX.T@h)%2
    assert is_symplectic(symp_mat)
    return symp_mat

n = 20
for _ in range(100):
    h = np.random.randint(0, 2, n).reshape((1,n))
    S1 = random_symp_mats(h)
    h = np.random.randint(0, 2, n).reshape((1,n))
    S2 = random_symp_mats(h)
    h = np.random.randint(0, 2, n).reshape((1,n))
    S3 = random_symp_mats(h)
    h = np.random.randint(0, 2, n).reshape((1,n))
    S4 = random_symp_mats(h)
    h = np.random.randint(0, 2, n).reshape((1,n))
    S5 = random_symp_mats(h)
    h = np.random.randint(0, 2, n).reshape((1,n))
    S6 = random_symp_mats(h)

    S_in = np.mod(S1 @ S2 @ S3 @ S4 @ S5 @ S6,2)

    gates = circ_from_symp_mat(S_in).run()
    S_out = symp_mat_prods(gates,n//2)
    assert np.allclose(S_in,S_out)
