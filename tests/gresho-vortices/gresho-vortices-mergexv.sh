NUM_PARALLEL_PROCESSORS=4
testNames=(offline romhr restore)
case $subTestNum in
  1)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -pa -offline -writesol -romsvds -romos -romsrhs -romxandv
    ;;
  2)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -pa -online -rdimx 9 -rdimv 18 -rdime 30 -rdimfv 29 -rdimfe 33 -romhr -romgs -sfacx 10 -sfacv 15 -sface 15 -soldiff -romos -romsrhs -romxandv -efx 0.9999
    ;;
  3)
    $LAGHOS -p 4 -m data/square_gresho.mesh -rs 3 -ok 3 -ot 2 -tf 0.1 -s 7 -pa -restore -rdimx 9 -rdimv 18 -rdime 30 -romsrhs -romxandv -efx 0.9999
    ;;
esac
