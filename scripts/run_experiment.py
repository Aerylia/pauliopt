import time, argparse
import numpy as np
import pandas as pd

from qiskit.providers.fake_provider import FakeSingapore, FakeJohannesburg, FakeYorktown, FakeMelbourne, FakeValencia
from qiskit import transpile, QuantumCircuit

from pytket.architecture import Architecture
from pytket.passes import SequencePass, DecomposeBoxes, FullPeepholeOptimise, CXMappingPass, NaivePlacementPass, KAKDecomposition, CliffordSimp, SynthesiseTK, RebaseTket, RemoveRedundancies, SimplifyInitial
from pytket.predicates import CompilationUnit
from pytket.placement import Placement
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit

from pauliopt.phase import OptimizedPhaseCircuit, iter_anneal, reverse_traversal_anneal, reverse_traversal
from pauliopt.phase import PhaseCircuit
from pauliopt.topologies import Topology


def compile_with_qiskit(circuit, topology):
     return transpile(circuit, 
                      optimization_level=3, 
                      basis_gates=['u1', 'u2', 'u3', 'cx'], 
                      coupling_map=[[v for v in c] for c in topology._couplings])

def compile_with_tket(circuit, topology):
    architecture = Architecture([[v for v in c] for c in topology._couplings])
    placer = Placement(architecture)
    compilation_pass = SequencePass([DecomposeBoxes(), FullPeepholeOptimise(), CXMappingPass(architecture, placer), 
                                    NaivePlacementPass(architecture), KAKDecomposition(), CliffordSimp(allow_swaps=False),
                                    SynthesiseTK(), RebaseTket(), RemoveRedundancies(), SimplifyInitial(allow_classical=False, create_all_qubits=True)])
    tkcirc = qiskit_to_tk(circuit)
    cu = CompilationUnit(tkcirc)
    compilation_pass.apply(cu)
    
    return tk_to_qiskit(cu.circuit)

def compile_anneal(circuit, topology, cx_block, num_RT_iters, num_anneal_iters, experiment_opt_kwargs, anneal_kwargs):
    return reverse_traversal(circuit, topology, cx_block, 1, num_anneal_iters, experiment_opt_kwargs, anneal_kwargs)

def compile_rt(circuit, topology, cx_block, num_RT_iters, num_anneal_iters, experiment_opt_kwargs, anneal_kwargs):
    return reverse_traversal(circuit, topology, cx_block, num_RT_iters, 0, experiment_opt_kwargs, anneal_kwargs)

def count_cnots(circuit):
    if not isinstance(circuit, QuantumCircuit):
        circuit = circuit.to_qiskit()
    ops = circuit.count_ops()
    if "cx" in ops:
        return ops["cx"]
    return 0

def compile_time_and_count(compiler_function, *args, **kwargs):
    start_time = time.process_time()
    compiled_circuit = compiler_function(*args, **kwargs)
    elapsed_time = time.process_time() - start_time
    return count_cnots(compiled_circuit), elapsed_time

def main(filename, device, num_anneal_iters, num_RT_iters, n_gadgets, cx_block, opt_kwargs, anneal_kwargs):
    topology = devices[device]

    data = {"gadget":n_gadgets, "device":device}
    circuit = PhaseCircuit.random(topology.num_qubits, n_gadgets, min_legs=int(np.round(np.sqrt(topology.num_qubits))))

    qiskit_circuit = circuit.to_qiskit(topology, simplified=False)
    data["original"] = count_cnots(qiskit_circuit)
    data["qiskit"], data["qiskit time"] = compile_time_and_count(compile_with_qiskit, qiskit_circuit, topology)
    data["tket"], data["tket time"] = compile_time_and_count(compile_with_tket, qiskit_circuit, topology)

    data["parity"], data["parity time"] = compile_time_and_count(circuit.to_qiskit, topology, method="paritysynth", cx_synth="permrowcol", reallocate=True)
    data["graysynth"], data["graysynth time"] = compile_time_and_count(circuit.to_qiskit, topology, method="steiner-graysynth", cx_synth="permrowcol", reallocate=True)
    for phase_method in ["steiner-graysynth", "paritysynth", "naive"]:
        for cx_method in ["permrowcol", "naive"]:
            for reallocate in [True, False]:
                for opt_name, optimizer in optimizers.items():
                    experiment_opt_kwargs = {"phase_method": phase_method, 
                                            "cx_method": cx_method,
                                            "reallocate": reallocate}
                    experiment_opt_kwargs.update(opt_kwargs)
                    count, compile_time  = compile_time_and_count(optimizer, circuit.copy(), topology, cx_block, num_RT_iters, num_anneal_iters, experiment_opt_kwargs, anneal_kwargs)
                    args_str = "+".join([phase_method, cx_method,str(reallocate), opt_name])
                    time_str = "time"+args_str
                    data[args_str] = count
                    data[time_str] = compile_time
    df = pd.DataFrame([data])
    with open(filename, 'a') as f:
        df.to_csv(f, mode='a', header=f.tell() == 0, index=False) #f.tell() gives the current file pointer location
    

devices = {
    "Singapore": Topology.from_qiskit_backend(FakeSingapore()),
    "Johannesburg": Topology.from_qiskit_backend(FakeJohannesburg()),
    "Melbourne": Topology.from_qiskit_backend(FakeMelbourne()),
    "Yorktown": Topology.from_qiskit_backend(FakeYorktown()),
    "Valencia": Topology.from_qiskit_backend(FakeValencia())
}

optimizers = {
    "annealer": compile_anneal,
    "iter_anneal":iter_anneal,
    "RT": compile_rt,
    "RT->anneal": reverse_traversal,
    "RT*anneal": reverse_traversal_anneal
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("device", help="The name of the target IBM device", choices=["Singapore", "Johannesburg", "Melbourne", "Yorktown", "Valencia"])
    parser.add_argument("gadgets", nargs='+', help="The number of gadgets in the circuit.", type=int)
    args = parser.parse_args()
    
    name = args.device
    filename = "./results/" + name + "_pauliopt.csv"
    for gadget in args.gadgets:
            main(filename, name, 1000, 10, gadget, 5, {}, {"schedule":("linear", 10.0, 0.1)})
