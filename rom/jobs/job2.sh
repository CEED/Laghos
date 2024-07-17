#!/bin/bash
srun laghos -p 4 -m data/square_gresho.mesh -rs 4 -ok 3 -ot 2 -cfl 0.35 -tf 0.4 -s 7 -offline -romsvds -writesol -ef 0.9999 -romos -rostype load --no-romsns -hrsamptype eqp -nwinsamp 10 -sample-stages > beqp_2a.out

srun laghos -p 4 -m data/square_gresho.mesh -rs 4 -ok 3 -ot 2 -cfl 0.35 -tf 0.4 -s 7 -online -romos -rostype load -romhr --no-romsns -hrsamptype eqp -nwin 213 > beqp_2b.out

srun laghos -p 4 -m data/square_gresho.mesh -rs 4 -ok 3 -ot 2 -cfl 0.35 -tf 0.4 -s 7 -restore -soldiff -romos -rostype load --no-romsns -hrsamptype eqp -nwin 213 > beqp_2c.out

