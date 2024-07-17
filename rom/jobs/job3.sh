#!/bin/bash
srun laghos -p 3 -m data/box01_hex.mesh -tf 0.8 -s 7 -offline -romsvds -writesol -ef 0.9999 -romos -rostype load --no-romsns -hrsamptype eqp -nwinsamp 10 -sample-stages > beqp_3a.out

srun laghos -p 3 -m data/box01_hex.mesh -tf 0.8 -s 7 -online -romhr -romos -rostype load --no-romsns -hrsamptype eqp -nwin 39 > beqp_3b.out

srun laghos -p 3 -m data/box01_hex.mesh -tf 0.8 -s 7 -restore -soldiff -romos -rostype load --no-romsns -hrsamptype eqp -nwin 39 > beqp_3c.out
