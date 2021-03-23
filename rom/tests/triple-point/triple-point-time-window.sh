NUM_PARALLEL_PROCESSORS=8
testNames=(offline online romhr restore)
case $subTestNum in
  1)
    $LAGHOS -p 3 -m data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -offline -writesol -romsvds -ef 0.9999 -nwin 4 -tw "$BASE_DIR"/tests/triple-point/triple-point-time-window.csv -romsrhs
    ;;
  2)
    $LAGHOS -p 3 -m data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -online -romgs -soldiff -nwin 4
    ;;
  3)
    $LAGHOS -p 3 -m data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -online -romgs -nwin 4 -romhrprep
    $LAGHOS_SERIAL -p 3 -m data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -online -romgs -nwin 4 -romhr
    ;;
  4)
    $LAGHOS -p 3 -m data/box01_hex.mesh -rs 1 -tf 0.04 -cfl 0.05 -pa -restore -nwin 4 -soldiff
    ;;
esac
