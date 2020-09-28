NUM_PARALLEL_PROCESSORS=8
testNames=(fom online)
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -offline -romsvds -romos -romsrhs -bef 1.0 -rpar 0
    $MERGE -nset 1 -rhs
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -bef 0.5 -writesol -visit
    ;;
  2)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.01 -online -rdimx 7 -rdimv 12 -rdime 6 -rdimfv 15 -rdimfe 9 -romhr -romos -sfacx 1 -sfacv 32 -sface 32 -soldiff -romgs -romsrhs -bef 1.0
    ;;
esac
