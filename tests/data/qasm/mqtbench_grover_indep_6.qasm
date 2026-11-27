// MQT Bench 2.4.0 (MIT licence), benchmark 'grover', level indep, 6 qubits.
// Generated with get_benchmark(...) and written by qiskit.qasm2.dumps.
OPENQASM 2.0;
include "qelib1.inc";
gate mcphase(param0) q0,q1,q2,q3,q4,q5 { cp(pi/2) q4,q5; cp(pi/4) q3,q5; crx(pi/2) q3,q4; cp(pi/8) q2,q5; crx(pi/4) q2,q4; cp(pi/16) q1,q5; crx(pi/2) q2,q3; crx(pi/8) q1,q4; cp(pi/16) q0,q5; crx(pi/4) q1,q3; crx(pi/8) q0,q4; crx(pi/2) q1,q2; crx(pi/4) q0,q3; crx(pi/2) q0,q2; crx(pi) q0,q1; crx(-pi/2) q1,q2; crx(-pi/4) q1,q3; crx(-pi/2) q2,q3; crx(-pi/8) q1,q4; crx(-pi/4) q2,q4; cp(-pi/16) q1,q5; crx(-pi/2) q3,q4; cp(-pi/8) q2,q5; cp(-pi/4) q3,q5; cp(-pi/2) q4,q5; crx(pi/2) q3,q4; crx(pi/4) q2,q4; crx(pi/2) q2,q3; crx(pi/8) q1,q4; crx(pi/4) q1,q3; crx(-pi/8) q0,q4; crx(pi/2) q1,q2; crx(-pi/4) q0,q3; crx(-pi/2) q0,q2; crx(-pi) q0,q1; crx(-pi/2) q1,q2; crx(-pi/4) q1,q3; crx(-pi/2) q2,q3; crx(-pi/8) q1,q4; crx(-pi/4) q2,q4; crx(-pi/2) q3,q4; }
gate mcx q0,q1,q2,q3,q4 { h q4; cp(pi/2) q3,q4; h q4; h q3; t q3; cx q2,q3; tdg q3; h q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; h q3; t q3; cx q2,q3; tdg q3; h q3; h q4; cp(-pi/2) q3,q4; h q4; h q3; t q3; cx q2,q3; tdg q3; h q3; t q3; cx q1,q3; tdg q3; cx q0,q3; t q3; cx q1,q3; tdg q3; cx q0,q3; h q3; t q3; cx q2,q3; tdg q3; h q3; h q4; cp(pi/8) q0,q4; h q4; cx q0,q1; h q4; cp(-pi/8) q1,q4; h q4; cx q0,q1; h q4; cp(pi/8) q1,q4; h q4; cx q1,q2; h q4; cp(-pi/8) q2,q4; h q4; cx q0,q2; h q4; cp(pi/8) q2,q4; h q4; cx q1,q2; h q4; cp(-pi/8) q2,q4; h q4; cx q0,q2; h q4; cp(pi/8) q2,q4; h q4; }
gate gate_Q q0,q1,q2,q3,q4,q5 { mcphase(pi) q0,q1,q2,q3,q4,q5; h q0; h q1; h q2; h q3; h q4; x q0; x q1; x q2; x q3; x q4; h q4; mcx q0,q1,q2,q3,q4; h q4; x q0; x q1; x q2; x q3; x q4; h q0; h q1; h q2; h q3; h q4; }
qreg q[5];
qreg flag[1];
creg meas[6];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
x flag[0];
gate_Q q[0],q[1],q[2],q[3],q[4],flag[0];
gate_Q q[0],q[1],q[2],q[3],q[4],flag[0];
gate_Q q[0],q[1],q[2],q[3],q[4],flag[0];
gate_Q q[0],q[1],q[2],q[3],q[4],flag[0];
barrier q[0],q[1],q[2],q[3],q[4],flag[0];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure flag[0] -> meas[5];