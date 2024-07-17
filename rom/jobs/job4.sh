#!/bin/bash
srun laghos -p 0 -m data/cube01_hex.mesh -cfl 0.1 -tf 0.4 -s 7 -offline -romsvds -writesol -ef 0.9999 -romos -rostype load --no-romsns -hrsamptype eqp -nwinsamp 10 -sample-stages > beqp_4a.out

srun laghos -p 0 -m data/cube01_hex.mesh -cfl 0.1 -tf 0.4 -s 7 -online -romhr -romos -rostype load --no-romsns -hrsamptype eqp -nwin 418 > beqp_4b.out

srun laghos -p 0 -m data/cube01_hex.mesh -cfl 0.1 -tf 0.4 -s 7 -restore -soldiff -romos -rostype load --no-romsns -hrsamptype eqp -nwin 418 > beqp_4c.out
