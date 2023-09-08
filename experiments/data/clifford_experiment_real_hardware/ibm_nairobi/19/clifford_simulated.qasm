OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg meas[7];
s q[4];
h q[5];
swap q[5],q[0];
cx q[1],q[5];
cx q[5],q[0];
h q[0];
cx q[0],q[5];
h q[0];
cx q[1],q[0];
s q[2];
h q[2];
h q[3];
h q[6];
swap q[3],q[1];
cx q[3],q[6];
cx q[2],q[3];
cx q[1],q[3];
h q[1];
h q[2];
cx q[1],q[2];
s q[2];
cx q[2],q[6];
h q[6];
cx q[6],q[2];
h q[6];
x q[0];
x q[1];
x q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
x q[5];
x q[1];
x q[0];
h q[6];
cx q[6],q[2];
h q[6];
cx q[2],q[6];
sdg q[2];
cx q[1],q[2];
h q[2];
h q[1];
cx q[1],q[3];
cx q[2],q[3];
cx q[3],q[6];
swap q[3],q[1];
h q[6];
h q[3];
h q[2];
sdg q[2];
cx q[1],q[0];
h q[0];
cx q[0],q[5];
h q[0];
cx q[5],q[0];
cx q[1],q[5];
swap q[5],q[0];
h q[5];
sdg q[4];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
