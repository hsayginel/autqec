from collections import Counter
from math import ceil

def permute_by_cycles(ordered_list, permutations):
    permuted_list = ordered_list.copy()
    for cycle in permutations:
        intermediate_list = permuted_list.copy()
        for i, _ in enumerate(cycle):
            old_index = cycle[i-1] - 1
            new_index = cycle[i] - 1 
            permuted_list[new_index] = intermediate_list[old_index]
    return permuted_list

def reduce_triplets_to_qubits(triplets_list):
    triplets = [ceil(x/3) for x in triplets_list]
    # Count occurrences of each element
    elem_counts = Counter(triplets)

    # Assert that each element is repeated three times and is adjacent to identical neighbors
    for element, count in elem_counts.items():
        if count == 3:
            # Find the index of the first occurrence of the element
            first_index = triplets.index(element)
            
            # Check if the next two elements are identical
            if (triplets[first_index + 1] == element and 
                triplets[first_index + 2] == element):
                continue  # Element repeated 3 times and adjacent to identical neighbors
            else:
                print(triplets)
                raise AssertionError(f"Automorphism does not represent a physical operation. Element {element} is not repeated three times consecutively.")
    
    # Reduce the list by getting rid of redundant elements
    reduced_list = list(elem_counts.keys())
    return reduced_list

def reduce_doublets_to_qubits(doublets_list):
    doublets = [ceil(x/2) for x in doublets_list]
    # Count occurrences of each element
    elem_counts = Counter(doublets)

    # Assert that each element is repeated three times and is adjacent to identical neighbors
    for element, count in elem_counts.items():
        if count == 2:
            # Find the index of the first occurrence of the element
            first_index = doublets.index(element)
            
            # Check if the next element is identical
            if (doublets[first_index + 1] == element):
                continue  
            else:
                print(doublets)
                raise AssertionError(f"Automorphism does not represent a physical operation. Element {element} is not repeated two times consecutively.")
    
    # Reduce the list by getting rid of redundant elements
    reduced_list = list(elem_counts.keys())
    return reduced_list

def sort_with_swaps(lst):
    swaps = []
    n = len(lst)
    
    def swap(i, j):
        if i != j:
            lst[i], lst[j] = lst[j], lst[i]
            swaps.append(('SWAP',(i+1,j+1)))
    
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if lst[j] < lst[min_idx]:
                min_idx = j
        
        swap(i, min_idx)
    
    return swaps

def apply_swaps(list_of_lists, instructions):
    def swap(lst, i, j):
        lst[i], lst[j] = lst[j], lst[i]

    for instruction in instructions:
        pair = instruction[1]
        i, j = int(pair[0]) - 1, int(pair[1]) - 1
        swap(list_of_lists, i, j)

    return list_of_lists