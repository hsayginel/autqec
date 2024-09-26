# autqec
autqec (optionally pronounced as 'oat cake') is a Python package for studying fault-tolerant logical Clifford gates on stabilizer quantum error correcting codes based on their symmetries. Given a set of stabilizer generators of a stabilizer code, it maps the stabilizer code to a related binary linear code, computes its automorphism group (using MAGMA software), and imposes constraints based on the Clifford operators permitted. The allowed permutation automorphisms translate to symplectic matrices which are represented as Clifford unitaries in $6$ layers of gates as $H-CNOT-S-CZ-X-C(X,X)$. The software also computes appropriate Pauli corrections to the physical circuits with a particular logical action by considering the destabilizers of the stabilizer code. 

autqec can identify transversal, *SWAP*-transversal and short-depth arbitrary Clifford circuits that preserve the stabilizer group and has non-trivial logical action on the logical qubits. Outline of the algorithms for finding logical Clifford operators via code automorphisms is given below.

![Algorithm Outline](algorithm_outline.png)

## Installation (from source)
Download a local copy and run:

`pip install -e .`

## Dependencies (for full functionality)
MAGMA V2.28-8: http://magma.maths.usyd.edu.au/magma/. 

## Acknowledgements
This work is supported by various EPSRC grants. 
