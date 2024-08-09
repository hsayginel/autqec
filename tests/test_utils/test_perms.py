import numpy as np
from utils.perms import permute_by_cycles, reduce_triplets_to_qubits
import pytest

# permute_by_cycles()
n = 4
ordered_list = np.arange(1,3*n+1,1)
# test q1 and q2 swap 
perm1 = [(1,4),(2,5),(3,6)]
permuted_list = permute_by_cycles(ordered_list,perm1).tolist()
assert permuted_list == [4,5,6,1,2,3,7,8,9,10,11,12]

#############################################
assert reduce_triplets_to_qubits(permuted_list) == [2, 1, 3, 4]

permuted_list = [1,5,3,4,2,6]
def test_wrong_triplets():
    with pytest.raises(AssertionError):
        reduce_triplets_to_qubits(permuted_list)

