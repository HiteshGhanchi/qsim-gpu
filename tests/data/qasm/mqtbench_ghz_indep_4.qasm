// MQT Bench 2.4.0 (MIT licence), benchmark 'ghz', level indep, 4 qubits.
// Generated with get_benchmark(...) and written by qiskit.qasm2.dumps.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg meas[4];
h q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[0];
barrier q[0],q[1],q[2],q[3];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];