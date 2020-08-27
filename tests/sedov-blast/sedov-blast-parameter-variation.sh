NUM_PARALLEL_PROCESSORS=8
testNames=(fom online)
case $subTestNum in
  1)
    $HEADER laghos -m data/cube01_hex.mesh -pt 211 -tf 0.1 -offline -romsvds -romos -romsrhs -bef 1.0 -rpar 0
    $HEADER ./merge -nset 1 -rhs
    $HEADER laghos -m data/cube01_hex.mesh -pt 211 -tf 0.1 -bef 0.5 -writesol -visit
    ;;
  2)
    $HEADER laghos -m data/cube01_hex.mesh -pt 211 -tf 0.1 -online -rdimx 32 -rdimv 92 -rdime 26 -rdimfv 200 -rdimfe 60 -romhr -romos -sfacx 1 -sfacv 32 -sface 32 -soldiff -romrmass -romsrhs -bef 1.0
    ;;
esac
