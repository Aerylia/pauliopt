import numpy as np

from pauliopt.pauli.clifford_gates import CliffordGate, CliffordType, generate_random_clifford
from pauliopt.pauli.clifford_region import CliffordRegion
from pauliopt.pauli.pauli_polynomial import PauliPolynomial
from pauliopt.phase.optimized_circuits import _validate_temp_schedule
from pauliopt.topologies import Topology
from .clifford_gates import CX, CY, CZ, CliffordGate, ControlGate
from .clifford_region import CliffordRegion
from .clifford_tableau import CliffordTableau
from .pauli_polynomial import PauliPolynomial
from ..phase.optimized_circuits import _validate_temp_schedule
from ..topologies import Topology


def pick_random_gate(num_qubits, gate_set=None):
    if gate_set is None:
        gate_set = [CliffordType.CX, CliffordType.CY, CliffordType.CZ]

    gate = np.random.choice(gate_set)

    return generate_random_clifford(gate, num_qubits)


def compute_effect(pp: PauliPolynomial, gate: CliffordGate, topology: Topology,
                   leg_cache=None):
    pp_ = pp.copy()
    pp_.propagate(gate)

    return pp_.two_qubit_count(topology, leg_cache=leg_cache) - pp.two_qubit_count(
        topology, leg_cache=leg_cache)


def anneal(pp: PauliPolynomial, topology, schedule=("geometric", 1.0, 0.1),
           nr_iterations=100):
    leg_cache = {}
    clifford_region = CliffordTableau(n_qubits=pp.num_qubits)

    schedule = _validate_temp_schedule(schedule)
    random_nrs = np.random.uniform(0.0, 1.0, size=(nr_iterations,))
    num_qubits = pp.num_qubits
    for it in range(nr_iterations):
        t = schedule(it, nr_iterations)
        gate = pick_random_gate(num_qubits)
        effect = compute_effect(pp, gate, topology, leg_cache=leg_cache)
        dist = topology.dist(int(gate.control), int(gate.target))
        accept_step = 2*dist + effect < 0 or random_nrs[it] < np.exp(-np.log(2) * effect / t)
        if accept_step:
            clifford_region.append_gate(gate)
            pp.propagate(gate)
    try:
        from qiskit import QuantumCircuit

    except:
        raise Exception("Please install qiskit to export the circuit")

    clifford_circ = clifford_region.to_cifford_circuit_arch_aware(topology)
    qc = QuantumCircuit(pp.num_qubits)
    qc.compose(clifford_circ, inplace=True)
    qc.compose(pp.to_qiskit(topology), inplace=True)
    qc.compose(clifford_circ.inverse(), inplace=True)
    return qc
