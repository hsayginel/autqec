from utils.qiskit import *

qc = construct_circuit([('SWAP',(1,2)), ('X',1)],k=2)
qc.draw(output='mpl')