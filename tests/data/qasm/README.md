# Test circuits from MQT Bench

The `mqtbench_*.qasm` files were generated with [MQT Bench](https://github.com/munich-quantum-toolkit/bench)
2.4.0 (`get_benchmark(name, BenchmarkLevel.INDEP, n)`, written out with `qiskit.qasm2.dumps`).
MQT Bench is released under the MIT licence; see its repository's licence file. Each file starts
with a comment naming the benchmark, level and size.

They are used by `tests/test_qasm.py`: every file must parse, run, and match Qiskit Aer's
statevector for the same file.
