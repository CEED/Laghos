NUM_PARALLEL_PROCESSORS=4
testNames=(offline romhr restore)
runAbsoluteFOM="true"
absoluteFOMOptions="-p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -pa -print"
absoluteFOMTol="1e-6"
absoluteFOMTolParallel="1e-1"
absoluteRelErrorTol="1e-1"
absoluteRelErrorTolParallel="1e-1"
speedupPercentTol="200"
case $subTestNum in
  1)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -pa -offline -writesol -romsvds -romos -romsrhs -romvx -efx 0.999999
    ;;
  2)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -pa -online -rdimx 19 -rdime 30 -rdimfv 29 -rdimfe 33 -romhr -romgs -sfacx 10 -sfacv 15 -sface 15 -soldiff -romos -romsrhs -romvx
    ;;
  3)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -pa -restore -rdimx 19 -rdime 30 -romsrhs -romvx
    ;;
esac
