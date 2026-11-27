// MQT Bench 2.4.0 (MIT licence), benchmark 'qaoa', level indep, 4 qubits.
// Generated with get_benchmark(...) and written by qiskit.qasm2.dumps.
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg meas[4];
h q[0];
h q[1];
rzz(10.410545461930019) q[0],q[1];
rx(12.013471791158686) q[0];
h q[2];
rzz(10.410545461930019) q[1],q[2];
rx(12.013471791158686) q[1];
rzz(1.8759344847470487) q[0],q[1];
rx(2.609806595315514) q[0];
rx(12.013471791158686) q[2];
rzz(1.8759344847470487) q[1],q[2];
rx(2.609806595315514) q[1];
rx(2.609806595315514) q[2];
u2(0,-1.0846848814747654) q[3];
barrier q[0],q[1],q[2],q[3];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];