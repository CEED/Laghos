NUM_PARALLEL_PROCESSORS=8
testNames=(offline romhr)
absoluteFOMOptions="-m data/cube01_hex.mesh -pt 211 -tf 0.4"
absoluteFOMTol="1e-14"
absoluteFOMTolParallel="1e-14"
absoluteRelErrorTol="1e-1"
absoluteRelErrorTolParallel="1e-1"
speedupTol="1.5"
case $subTestNum in
  1)
    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.4 -offline -writesol -romsvds
    ;;
  2)

    $LAGHOS -m data/cube01_hex.mesh -pt 211 -tf 0.4 -online -rdimx 12 -rdimv 108 -rdime 27 -romhr -soldiff
    ;;
esac
