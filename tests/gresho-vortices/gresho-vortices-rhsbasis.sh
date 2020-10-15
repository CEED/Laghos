NUM_PARALLEL_PROCESSORS=8
testNames=(offline romhr restore)
runAbsoluteFOM="true"
absoluteFOMOptions="-p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -pa -print"
absoluteFOMTol="1e-14"
absoluteFOMTolParallel="1"
absoluteRelErrorTol="1"
absoluteRelErrorTolParallel="1"
speedupPercentTol="200"
case $subTestNum in
  1)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -pa -offline -writesol -romsvds -romos -romsrhs
    ;;
  2)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -pa -online -soldiff -romhr -romsvds -sfacx 10 -sfacv 10 -sface 10 -romos -romsrhs -romgs -rdimx 9 -rdimv 20 -rdime 30 -rdimfv 29 -rdimfe 33
    ;;
  3)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -pa -restore -romos -romsrhs -rdimx 9 -rdimv 20 -rdime 30 -rdimfv 29 -rdimfe 33
    ;;
esac
