#!/bin/bash
srun laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.3 -s 7 -offline -romsvds -writesol -ef 0.9999 -romos -rostype load --no-romsns -hrsamptype eqp -nwinsamp 10 -sample-stages > beqp_1a.out

srun laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.3 -s 7 -online -romos -rostype load -romhr --no-romsns -hrsamptype eqp -nwin 91 > beqp_1b.out

srun laghos -p 1 -m data/cube01_hex.mesh -pt 211 -tf 0.3 -s 7 -restore -soldiff -romos -rostype load --no-romsns -hrsamptype eqp -nwin 91 > beqp_1c.out 
