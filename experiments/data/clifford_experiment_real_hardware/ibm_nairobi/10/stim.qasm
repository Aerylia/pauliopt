OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg meas[7];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[0];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[3];
cx q[1],q[3];
cx q[1],q[2];
x q[1];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/2) q[4];
rz(pi/2) q[5];
cx q[5],q[4];
rz(-pi) q[4];
sx q[4];
rz(-pi) q[4];
cx q[5],q[3];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
cx q[1],q[3];
rz(-pi) q[3];
sx q[3];
rz(-pi) q[3];
cx q[1],q[3];
cx q[3],q[1];
cx q[1],q[3];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[5],q[6];
rz(-pi) q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[5],q[4];
rz(-pi) q[4];
x q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
cx q[3],q[5];
cx q[1],q[3];
cx q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
cx q[1],q[3];
cx q[3],q[1];
cx q[1],q[3];
rz(pi/2) q[5];
sx q[5];
rz(-pi) q[5];
cx q[5],q[3];
rz(-pi) q[3];
sx q[3];
rz(-pi) q[3];
rz(-pi) q[5];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi) q[5];
rz(pi) q[3];
sxdg q[3];
rz(pi) q[3];
cx q[5],q[3];
rz(pi) q[5];
sxdg q[5];
rz(-pi/2) q[5];
cx q[1],q[3];
cx q[3],q[1];
cx q[1],q[3];
rz(-pi/2) q[1];
sxdg q[1];
rz(-pi/2) q[1];
cx q[1],q[2];
cx q[1],q[3];
cx q[3],q[5];
rz(-pi/2) q[5];
sxdg q[5];
rz(-pi/2) q[5];
x q[4];
rz(pi) q[4];
cx q[5],q[4];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
rz(-pi/2) q[6];
sxdg q[6];
rz(-pi/2) q[6];
rz(pi) q[5];
cx q[5],q[6];
rz(-pi/2) q[6];
sxdg q[6];
rz(-pi/2) q[6];
cx q[2],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[1],q[3];
cx q[3],q[1];
cx q[1],q[3];
rz(pi) q[3];
sxdg q[3];
rz(pi) q[3];
cx q[1],q[3];
rz(-pi/2) q[3];
sxdg q[3];
rz(-pi/2) q[3];
cx q[5],q[3];
rz(pi) q[4];
sxdg q[4];
rz(pi) q[4];
cx q[5],q[4];
rz(-pi/2) q[5];
rz(-pi/2) q[4];
rz(-pi/2) q[2];
sxdg q[2];
rz(-pi/2) q[2];
cx q[0],q[1];
cx q[1],q[0];
cx q[0],q[1];
x q[1];
cx q[1],q[2];
cx q[1],q[3];
rz(-pi/2) q[3];
rz(-pi/2) q[0];
sxdg q[0];
rz(-pi/2) q[0];
cx q[1],q[0];
rz(-pi/2) q[1];
sxdg q[1];
rz(pi) q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
