from qiskit import QuantumCircuit, transpile
from math import pi

def construct_circuit(operations, k):
    """
    Constructs a quantum circuit based on a list of gate operations and the specified number of qubits.
    """

    # Initialize the quantum circuit with k qubits
    qc = QuantumCircuit(k)

    # Parse and add each operation to the circuit
    for op in operations:
        gate_type, qubits = op

        # Add the corresponding gate to the circuit
        if gate_type == 'CNOT':
            qc.cx(qubits[0]-1, qubits[1]-1)
        elif gate_type == 'CZ':
            qc.cz(qubits[0]-1, qubits[1]-1)
        elif gate_type == 'SWAP':
            qc.swap(qubits[0]-1, qubits[1]-1)
        elif gate_type == 'H':
            qc.h(qubits-1)
        elif gate_type == 'S':
            qc.s(qubits-1)
        elif gate_type == 'Xsqrt':
            qc.sx(qubits-1)
        elif gate_type == 'C(X,X)':
            qc.rxx(pi/2,qubits[0]-1, qubits[1]-1)
        elif gate_type == 'GammaXYZ':
            qc.sdg(qubits-1)
            qc.h(qubits-1)
        elif gate_type == 'GammaXZY':
            qc.h(qubits-1)
            qc.s(qubits-1)
        elif gate_type == 'Z':
            qc.z(qubits-1)
        elif gate_type == 'X':
            qc.x(qubits-1)
        elif gate_type == 'Y':
            qc.y(qubits-1)
        else:
            print(f"Unsupported gate type: {gate_type}")

    return qc