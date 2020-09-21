NUM_PARALLEL_PROCESSORS=8
testNames=(offline romhr restore)
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
