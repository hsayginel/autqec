import igraph as ig
import numpy as np
from sympy.combinatorics import Permutation, PermutationGroup
import scipy.sparse as sparse
from time import time

def sparse_pcm_to_tanner_graph(pcm):
    """
    Creates a Tanner graph in igraph format from a sparse parity check matrix.
    Args:
        pcm: A (sparse) parity check matrix.
    Returns:
        An igraph Graph object representing the Tanner graph.
    """
    rows, cols = pcm.shape
    g = ig.Graph()
    g.add_vertices(cols + rows)
    # g.vs["type"] = ["variable"] * cols + ["check"] * rows
    g.vs["color"] = [1] * cols + [2] * rows
    edges = []
    rows_indices, col_indices = pcm.nonzero()
    for i in range(len(rows_indices)):
        edges.append((col_indices[i], cols + rows_indices[i]))
    g.add_edges(edges)
    return g

def graph_aut_group(g, n, print_order=True, print_time=False):
    """
    Calculates and returns the automorphism group of an igraph graph as a sympy PermutationGroup.

    Args:
        g: An igraph Graph object.
        n: number of columns of adjancency matrix
        print_order: A boolean indicating whether to print the order of the automorphism group.
                     Defaults to True.

    Returns:
        A sympy PermutationGroup object representing the automorphism group of the graph,
        or Identity if no automorphism generators are found.
    """
    ti = time()
    automorphism_generators = g.automorphism_group(color=g.vs["color"])
    tf = time()
    if print_time:
        print(f'Time taken for graph auts: {tf-ti}')
    if automorphism_generators:
        sympy_permutations = [Permutation(list(generator)[:n]) for generator in automorphism_generators]
        sympy_group = PermutationGroup(sympy_permutations)
        if print_order:
            print("Automorphism group order:",sympy_group.order())
        return sympy_group
    else:
        print("Cannot create sympy group, no automorphism generators found.")
        return PermutationGroup(Permutation(range(1)))


def B_for_all_cliffords(n,bits_3=True):
    """ Parity-check matrix for classical code B that has 
        all possible Clifford gates as automorphisms.
    """
    
    identity = np.eye(n,dtype=np.uint8)
    if bits_3:
        return np.hstack([identity,identity,identity])
    else:
        v = []
        for i in range(n):
            v.extend([i,n+i])
        return np.hstack([identity,identity])
    
def valid_clifford_auts(pcm,bits_3=True,return_order=True,return_aut_time=False):
    m,pcm_cols = pcm.shape
    
    if bits_3:
        n = pcm_cols//3
        v = np.column_stack([np.arange(n), np.arange(n, 2*n), np.arange(2*n, 3*n)]).flatten()
        B_pcm = B_for_all_cliffords(n,bits_3)[:,v]
    else: 
        n = pcm_cols//2
        v = np.column_stack([np.arange(n), np.arange(n, 2*n)]).flatten()
        B_pcm = B_for_all_cliffords(n,bits_3)[:,v]

    B_tanner_graph = sparse_pcm_to_tanner_graph(B_pcm)
    B_graphauts = graph_aut_group(B_tanner_graph,pcm_cols,False)

    pcm = pcm[:,v]
    tanner_graph = sparse_pcm_to_tanner_graph(pcm)
    code_graph_auts = graph_aut_group(tanner_graph,pcm_cols,False,print_time=return_aut_time)
    clifford_auts = B_graphauts.subgroup_search(code_graph_auts.__contains__)
    if return_order: 
        print(f'Order: {clifford_auts.order()}')
    
    clifford_auts_cyclic = []
    for a in clifford_auts:
        clifford_auts_cyclic.append(a.cyclic_form)
   
    return cycle_indices(clifford_auts_cyclic)

def valid_clifford_auts_B_rows(pcm,bits_3=True,return_order=True,return_aut_time=False):
    m,pcm_cols = pcm.shape
    
    if bits_3:
        n = pcm_cols//3
        v = np.column_stack([np.arange(n), np.arange(n, 2*n), np.arange(2*n, 3*n)]).flatten()
        B_pcm = B_for_all_cliffords(n,bits_3)[:,v]
    else: 
        n = pcm_cols//2
        v = np.column_stack([np.arange(n), np.arange(n, 2*n)]).flatten()
        B_pcm = B_for_all_cliffords(n,bits_3)[:,v]

    pcm = np.vstack([pcm[:,v],B_pcm])
    tanner_graph = sparse_pcm_to_tanner_graph(pcm)
    code_graph_auts = graph_aut_group(tanner_graph,pcm_cols,False,return_aut_time)

    if return_order: 
        print(f'Order: {code_graph_auts.order()}')
    
    auts_cyclic = []
    for a in code_graph_auts:
        auts_cyclic.append(a.cyclic_form)
   
    return cycle_indices(auts_cyclic)

def cycle_indices(lists):
    return [[[x + 1 for x in sublist] for sublist in group] for group in lists]