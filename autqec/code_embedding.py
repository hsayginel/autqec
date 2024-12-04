import itertools
from math import comb
from autqec.utils.linalg import *

class qec_embed_code:
    def __init__(self, H, embedding = 'two_code_blocks', custom_embedding = None):
        """ Class for embedding stabilizer codes into a binary linear code 
            where extra columns correspond to pairs of qubits.
            
            Action of CNOT, CZ gates can represented as automorphisms. 
            
            User can restrict connectivity by only adding columns for connected 
            pairs of qubits. 

            Args: 
                H (np.array): stabilizer generators of the original QEC.
                embedding (str): options are 'two_code_blocks', 'all_weight_2', 'custom'
                custom_embedding (np.array): binary matrix representing qubit connectivities, e.g. np.array([[1,1,0],[0,1,1]])
            """
        self.H = H 
        n = H.shape[1]//2
        self.n = n
        self.embedding = embedding
        self.custom_embedding = custom_embedding

    def gen_two_block_embed_mat(self):
        nhalf = self.n//2
        id_mat = np.eye(nhalf,dtype=int)
        zeros_mat = np.zeros_like(id_mat,dtype=int)

        block_col1 = np.hstack((id_mat,zeros_mat))
        block_col2 = np.hstack((zeros_mat,id_mat))
        block_col3 = np.hstack((id_mat,id_mat))

        return np.vstack((np.vstack((block_col1,block_col2)),block_col3)).T

    def gen_w2_embed_mat(self):
        n = self.n
        combos = itertools.combinations(range(n), 2)
        w2_embed_mat = np.zeros((comb(n,2),n),dtype=int)
        for row, pair in enumerate(combos):
            w2_embed_mat[row,pair[0]] = 1
            w2_embed_mat[row,pair[1]] = 1
        return np.vstack((np.eye(n,dtype=int),w2_embed_mat)).T

    def embed_mat(self):
        H = self.H
        n = self.n
        option = self.embedding
        
        HX_part = H[:,:n]
        HZ_part = H[:,n:]
        if option == 'two_code_blocks':
            V_T = self.gen_two_block_embed_mat()
        elif option == 'all_weight_2': 
            V_T = self.gen_w2_embed_mat()
        elif option == 'custom':
            V_T = self.custom_embedding.T
        else:
            raise TypeError("Unknown option for embedding.")
        M_T = V_T[:,n:]
        M = M_T.T
        A_V = np.hstack((HX_part,(HX_part@M_T)%2))
        ones = np.eye(M.shape[0])
        zeros = np.zeros((HZ_part.shape[0],ones.shape[1]))
        B_V = np.vstack((np.hstack((HZ_part,zeros)),np.hstack((M,ones))))
        assert A_V.shape[1] == B_V.shape[1]
        no_of_rows = B_V.shape[0] - A_V.shape[0] 
        A_V = np.vstack((A_V,np.zeros((no_of_rows,A_V.shape[1]))))
        return np.array(np.hstack((A_V,B_V)),dtype=int)
    
def embed_circ_to_normal_circ(embed_circ, n_og, embed_codespace_dict):
    new_circuit = []
    for gate in embed_circ:
        gate_type, q_ind = gate
        if gate_type == 'SWAP':
            q1 = q_ind[0]
            q2 = q_ind[1]
            if q1 < n_og + 1 and q2 < n_og + 1:
                new_circuit.append(gate)
            elif q1 > n_og and q2 > n_og:
                embed_col1 = embed_codespace_dict[int(q1 - n_og)]
                embed_col2 = embed_codespace_dict[int(q2 - n_og)]
                q_a, q_b = embed_col1
                q_c, q_d = embed_col2
                new_circuit.extend([('CNOT',(q_b,q_a)),('CNOT',(q_d,q_c)),('SWAP',(q_a,q_c)),
                                    ('CNOT',(q_b,q_a)),('CNOT',(q_d,q_c))]) 
            elif q1 < n_og+1 and q2 > n_og:
                embed_col = embed_codespace_dict[int(q2 - n_og)]
                if q1 in embed_col: 
                    targ = q1
                    ctrl = [x for x in embed_col if x != q1][0]
                    new_circuit.append(('CNOT',(ctrl,targ)))
                else: 
                    q_a,q_b = embed_col
                    new_circuit.extend([('SWAP',(q1,q_a)),('CNOT',(q_b,q_a)),('SWAP',(q1,q_a))])
            elif q1 > n_og and q2 < n_og+1:
                embed_col = embed_codespace_dict[int(q1 - n_og)]
                if q2 in embed_col: 
                    targ = q2
                    ctrl = [x for x in embed_col if x != q2][0]
                    new_circuit.append(('CNOT',(ctrl,targ)))
                else: 
                    q_a,q_b = embed_col
                    new_circuit.extend([('SWAP',(q2,q_a)),('CNOT',(q_b,q_a)),('SWAP',(q2,q_a))])
        else: 
            if q_ind < n_og + 1:
                new_circuit.append(gate)
            else: 
                if gate_type == 'S':
                    pair = embed_codespace_dict[int(q_ind - n_og)]
                    new_circuit.append(('S',pair[0]))
                    new_circuit.append(('S',pair[1]))
                    new_circuit.append(('CZ',(pair[0],pair[1])))
    
    return new_circuit       