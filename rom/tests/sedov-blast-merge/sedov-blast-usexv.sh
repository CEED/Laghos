NUM_PARALLEL_PROCESSORS=4
testNames=(offline romhr restore)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.05 -offline -ef 0.9999 -writesol -romsvds -romos -rostype load -romsrhs -romxv
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.05 -online -romsvds -romos -rostype load -sfacx 10 -sfacv 10 -sface 10 -romsrhs -romgs -romhrprep -rdimv 60 -rdime 20 -rdimfv 111 -rdimfe 40 -romxv
    $LAGHOS_SERIAL -m data/cube01_hex.mesh -pt 211 -tf 0.05 -online -romsvds -romos -rostype load -sfacx 10 -sfacv 10 -sface 10 -romsrhs -romgs -romhr -rdimv 60 -rdime 20 -rdimfv 111 -rdimfe 40 -romxv
    ;;
  3)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -restore -romsrhs -rdimv 60 -rdime 20 -romxv -soldiff -romos -rostype load
    ;;
esac
