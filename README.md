# autqec
autqec (optionally pronounced as 'oat cake') is a Python package for studying fault-tolerant logical Clifford gates on stabilizer quantum error correcting codes based on their symmetries. Given a set of stabilizer generators of a stabilizer code, it maps the stabilizer code to a related binary linear code, computes its automorphism group (using MAGMA software), and imposes constraints based on the Clifford operators permitted. The allowed permutation automorphisms translate to symplectic matrices which are represented as Clifford unitaries in $6$ layers of gates as $H-CNOT-S-CZ-X-C(X,X)$.

## Installation (from source)
Download a local copy and run:

`pip install -e .`

## Dependencies (for full functionality)
MAGMA V2.28-8: http://magma.maths.usyd.edu.au/magma/. 

## Acknowledgements
This work is supported by various EPSRC grants. 
